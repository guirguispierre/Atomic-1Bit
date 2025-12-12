#include <M5Cardputer.h>
#include <SD.h>
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <vector>

// Handle Arduino Macro Conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// --- Config ---
#define SD_SCK 40
#define SD_MISO 39
#define SD_MOSI 14
#define SD_CS 12

// Model Config
struct Config {
  int vocab_size;
  int dim;
  int depth;
  int heads;
  int max_seq_len;
  int has_gist;
};

Config config;
float *gist_vector = nullptr;
float *pos_emb_cache = nullptr;

File modelFile;
long layers_offset = 0;
long ln_f_offset = 0;
long head_offset = 0;
long pos_emb_offset = 0;

// --- Math Helper ---
inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
}

void rms_norm(float *x, float *w, float *out, int dim) {
  float ss = 0.0f;
  for (int i = 0; i < dim; i++)
    ss += x[i] * x[i];
  float rms = 1.0f / sqrt(ss / dim + 1e-5f);
  for (int i = 0; i < dim; i++)
    out[i] = x[i] * rms * w[i];
}

void softmax(float *x, int size) {
  float max_val = -1e9f;
  for (int i = 0; i < size; i++)
    if (x[i] > max_val)
      max_val = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = exp(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++)
    x[i] /= sum;
}

// Helper to read float array from file
void read_floats(File &f, float *buf, int count) {
  f.read((uint8_t *)buf, count * sizeof(float));
}

void setup_model() {
  // Retry SD if needed
  if (!SD.begin(SD_CS, SPI, 20000000, "/sd", 10, false)) {
    M5Cardputer.Display.println("SD Init Failed!");
    Serial.println("SD Init Failed");
    while (1)
      ;
  }

  modelFile = SD.open("/atomic_model.bin", FILE_READ);
  if (!modelFile) {
    M5Cardputer.Display.println("Model Not Found!");
    Serial.println("Model Not Found");
    while (1)
      ;
  }

  // Header
  uint32_t magic;
  modelFile.read((uint8_t *)&magic, 4);
  if (magic != 0xATOM1C) {
    M5Cardputer.Display.println("Invalid Magic!");
    while (1)
      ;
  }

  modelFile.read((uint8_t *)&config.vocab_size, 4);
  modelFile.read((uint8_t *)&config.dim, 4);
  modelFile.read((uint8_t *)&config.depth, 4);
  modelFile.read((uint8_t *)&config.heads, 4);
  modelFile.read((uint8_t *)&config.max_seq_len, 4);
  modelFile.read((uint8_t *)&config.has_gist, 4);

  M5Cardputer.Display.printf("Model: D=%d L=%d H=%d\n", config.dim,
                             config.depth, config.heads);
  Serial.printf("Model: D=%d L=%d H=%d\n", config.dim, config.depth,
                config.heads);

  // Gist
  if (config.has_gist) {
    gist_vector = (float *)malloc(config.dim * sizeof(float));
    read_floats(modelFile, gist_vector, config.dim);
  }

  // Skip Token Emb (Vocab * Dim * 4)
  long token_emb_size = (long)config.vocab_size * config.dim * 4;
  modelFile.seek(modelFile.position() + token_emb_size);

  // Pos Emb
  pos_emb_offset = modelFile.position();
  pos_emb_cache =
      (float *)malloc(config.max_seq_len * config.dim * sizeof(float));
  read_floats(modelFile, pos_emb_cache, config.max_seq_len * config.dim);

  layers_offset = modelFile.position();

  // Layer size calc
  long layer_size = (long)config.dim * 4 * 2; // Norms * 2
  layer_size +=
      (long)config.dim * config.dim * 4; // Q,K,V,O (Quantized? Check logic)
  // Actually, export script writes Q,K,V,O as 'b' (int8). So size is Dim*Dim
  // bytes each. Wait, export script: `write_tensor(..., 'b', True)` -> writes
  // bytes. So 4 * Dim * Dim bytes.

  layer_size += (long)config.dim * (config.dim * 4); // FC1
  layer_size += (long)(config.dim * 4) * config.dim; // FC2
  // Correct.

  ln_f_offset = layers_offset + (long)config.depth * layer_size;

  // Head weight is 'b' (int8)?
  // Export script: `write_tensor("head", ..., 'b', True)`.
  // So Head is Vocab * Dim bytes.
  head_offset = ln_f_offset + config.dim * 4;
}

// Forward Pass
// tokens: input token ids
// logits: output logits (Vocab Size) - Caller must allocate! (Actually just
// topk?) We return the predicted token ID directly to save RAM? Let's stick to
// generating logits for the last position.
int forward(std::vector<int> &tokens) {
  int seq_len = tokens.size() + (config.has_gist ? 1 : 0);
  int dim = config.dim;

  // 1. Alloc X (Seq, Dim) in PSRAM if possible, else Heap
  float *X = (float *)heap_caps_malloc(seq_len * dim * sizeof(float),
                                       MALLOC_CAP_SPIRAM);
  if (!X)
    X = (float *)malloc(seq_len * dim * sizeof(float)); // Fallback

  int t_off = 0;
  if (config.has_gist) {
    memcpy(X, gist_vector, dim * sizeof(float));
    t_off = 1;
  }

  // Embeddings
  // Need to find TokenEmb start again.
  // Start of file + 24 (Header) + (HasGist ? Dim*4 : 0)
  long emb_start = 24 + (config.has_gist ? config.dim * 4 : 0);

  for (int i = 0; i < tokens.size(); i++) {
    int id = tokens[i];
    modelFile.seek(emb_start + (long)id * dim * 4);
    read_floats(modelFile, &X[(t_off + i) * dim], dim);

    // Add Pos Emb
    float *p =
        &pos_emb_cache[(t_off + i) * dim]; // Careful with index if seq long
    for (int d = 0; d < dim; d++)
      X[(t_off + i) * dim + d] += p[d];
  }

  // Reset to Layers
  modelFile.seek(layers_offset);

  for (int l = 0; l < config.depth; l++) {
    // --- Layer Logic ---

    // LN1
    float *ln1_w = (float *)malloc(dim * sizeof(float));
    read_floats(modelFile, ln1_w, dim);

    float *X_norm = (float *)malloc(seq_len * dim * sizeof(float));
    for (int t = 0; t < seq_len; t++)
      rms_norm(&X[t * dim], ln1_w, &X_norm[t * dim], dim);
    free(ln1_w);

    // define matmul_seq inside loop to capture vars
    auto matmul_seq = [&](float *input, float *output, int out_dim) {
      // Logic: Read Weights (In, Out) and multiply Input (Seq, In)
      // Weight file format: Row-major (In, Out).
      // We iterate In `i`. Read Row `W[i]` (size Out).
      // Add `input[t][i] * W[i][o]` to `output[t][o]`.

      // 1. Quantize Input (per token)
      int8_t *X_q =
          (int8_t *)malloc(seq_len * dim); // Input Dim is always 'dim'
      float *scales = (float *)malloc(seq_len * sizeof(float));

      for (int t = 0; t < seq_len; t++) {
        float max_val = 0.0f;
        // input is (Seq, Dim)
        for (int d = 0; d < dim; d++)
          max_val = std::max(max_val, abs(input[t * dim + d]));
        float s = 127.0f / std::max(max_val, 1e-5f);
        scales[t] = s;
        for (int d = 0; d < dim; d++)
          X_q[t * dim + d] = (int8_t)std::max(
              -127.0f, std::min(127.0f, round(input[t * dim + d] * s)));
      }

      // 2. Stream Weights & Accumulate
      // Alloc output accumulator
      int32_t *acc = (int32_t *)calloc(seq_len * out_dim, sizeof(int32_t));
      int8_t *w_row = (int8_t *)malloc(out_dim);

      // Loop In Dim (Rows of Weight Matrix)
      for (int i = 0; i < dim; i++) {
        modelFile.read((uint8_t *)w_row, out_dim);

        for (int t = 0; t < seq_len; t++) {
          int8_t x_val = X_q[t * dim + i];
          if (x_val == 0)
            continue;

          for (int o = 0; o < out_dim; o++) {
            acc[t * out_dim + o] += (int32_t)x_val * (int32_t)w_row[o];
          }
        }
      }
      free(w_row);
      free(X_q);

      // 3. Dequantize
      for (int t = 0; t < seq_len; t++) {
        for (int o = 0; o < out_dim; o++) {
          output[t * out_dim + o] = (float)acc[t * out_dim + o] / scales[t];
        }
      }
      free(scales);
      free(acc);
    };

    // Q, K, V
    float *Q = (float *)malloc(seq_len * dim * sizeof(float));
    float *K = (float *)malloc(seq_len * dim * sizeof(float));
    float *V = (float *)malloc(seq_len * dim * sizeof(float));

    matmul_seq(X_norm, Q, dim);
    matmul_seq(X_norm, K, dim);
    matmul_seq(X_norm, V, dim);

    // ATTENTION
    float *AttnOut = (float *)calloc(seq_len * dim, sizeof(float));
    int head_dim = dim / config.heads;
    float scale = 1.0f / sqrt(head_dim);

    for (int h = 0; h < config.heads; h++) {
      for (int t = 0; t < seq_len; t++) {
        std::vector<float> scores(seq_len);

        // Q[t] dot K[context]
        for (int ctx = 0; ctx <= t; ctx++) { // Causal Mask: ctx <= t
          float dot = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            // Q index: t*dim + h*head_dim + d
            // K index: ctx*dim + h*head_dim + d
            dot +=
                Q[t * dim + h * head_dim + d] * K[ctx * dim + h * head_dim + d];
          }
          scores[ctx] = dot * scale;
        }

        // Softmax on valid scores
        // Find max
        float max_s = -1e9f;
        for (int ctx = 0; ctx <= t; ctx++)
          max_s = std::max(max_s, scores[ctx]);
        float sum_s = 0.0f;
        for (int ctx = 0; ctx <= t; ctx++) {
          scores[ctx] = exp(scores[ctx] - max_s);
          sum_s += scores[ctx];
        }

        // Weighted Sum V
        for (int d = 0; d < head_dim; d++) {
          float val = 0.0f;
          for (int ctx = 0; ctx <= t; ctx++) {
            val += (scores[ctx] / sum_s) * V[ctx * dim + h * head_dim + d];
          }
          AttnOut[t * dim + h * head_dim + d] = val;
        }
      }
    }

    free(Q);
    free(K);
    free(V);

    // Output Proj
    float *ResidAdd = (float *)calloc(seq_len * dim, sizeof(float));
    matmul_seq(AttnOut, ResidAdd, dim);
    free(AttnOut);

    // Add Residual
    for (int i = 0; i < seq_len * dim; i++)
      X[i] += ResidAdd[i];
    free(ResidAdd);

    // LN2
    float *ln2_w = (float *)malloc(dim * sizeof(float));
    read_floats(modelFile, ln2_w, dim);
    for (int t = 0; t < seq_len; t++)
      rms_norm(&X[t * dim], ln2_w, &X_norm[t * dim], dim);
    free(ln2_w);

    // MLP
    // FC1: Dim -> 4Dim
    float *FC1_Out = (float *)malloc(seq_len * (dim * 4) * sizeof(float));
    matmul_seq(X_norm, FC1_Out, dim * 4);

    // Gelu
    for (int i = 0; i < seq_len * (dim * 4); i++)
      FC1_Out[i] = gelu(FC1_Out[i]);

    // FC2: 4Dim -> Dim
    float *FC2_Out = (float *)calloc(seq_len * dim, sizeof(float));
    // We reuse matmul_seq.
    // Note: matmul_seq expects Input size to match the number of "Rows" in
    // weight file. FC2 in file is (4Dim, Dim). Input FC1_Out is (Seq, 4Dim).
    // Logic inside matmul_seq uses `dim` global variable as Input Dim?
    // Ah! `matmul_seq` hardcoded `dim` as outer loop.
    // We need to fix `matmul_seq` to accept `in_dim`.

    // FIXING MATMUL_SEQ Call for FC2
    // Since lambda cannot be easily templated/overloaded in this scope without
    // mess, I will implement FC2 specifically or generalize `matmul_seq`. Let's
    // copy-paste logic for FC2 to be safe and clear.
    {
      int fc_in = dim * 4;
      int fc_out = dim;
      int8_t *X_q = (int8_t *)malloc(seq_len * fc_in);
      float *scales = (float *)malloc(seq_len * sizeof(float));
      for (int t = 0; t < seq_len; t++) {
        float max_val = 0.0f;
        for (int d = 0; d < fc_in; d++)
          max_val = std::max(max_val, abs(FC1_Out[t * fc_in + d]));
        scales[t] = 127.0f / std::max(max_val, 1e-5f);
        for (int d = 0; d < fc_in; d++)
          X_q[t * fc_in + d] = (int8_t)std::max(
              -127.0f,
              std::min(127.0f, round(FC1_Out[t * fc_in + d] * scales[t])));
      }

      int32_t *acc = (int32_t *)calloc(seq_len * fc_out, sizeof(int32_t));
      int8_t *w_row = (int8_t *)malloc(fc_out);

      for (int i = 0; i < fc_in; i++) {
        modelFile.read((uint8_t *)w_row, fc_out);
        for (int t = 0; t < seq_len; t++) {
          int8_t x_val = X_q[t * fc_in + i];
          if (x_val == 0)
            continue;
          for (int o = 0; o < fc_out; o++)
            acc[t * fc_out + o] += (int32_t)x_val * (int32_t)w_row[o];
        }
      }
      free(w_row);
      free(X_q);

      for (int t = 0; t < seq_len; t++)
        for (int o = 0; o < fc_out; o++)
          FC2_Out[t * fc_out + o] = (float)acc[t * fc_out + o] / scales[t];
      free(scales);
      free(acc);
    }

    free(FC1_Out);

    // Add Resid
    for (int i = 0; i < seq_len * dim; i++)
      X[i] += FC2_Out[i];
    free(FC2_Out);
    free(X_norm);
  } // End Layers

  // Final Norm
  float *ln_f_w = (float *)malloc(dim * sizeof(float));
  read_floats(modelFile, ln_f_w, dim);

  // Only need LAST token for prediction
  int last_idx = seq_len - 1;
  float *last_hidden = &X[last_idx * dim];
  float *norm_out = (float *)malloc(dim * sizeof(float));
  rms_norm(last_hidden, ln_f_w, norm_out, dim);
  free(ln_f_w);

  // Head (Dim -> Vocab)
  // Stream Head weights
  // This is huge mapping. (Dim, Vocab).
  // Export format: `head` (b). Size `dim * vocab`.
  // Wait, typical head is (Dim, Vocab).
  // My export script uses `quantize_and_transpose(head.weight)`.
  // PyTorch Linear weight is (Out, In) -> (Vocab, Dim).
  // Transpose -> (Dim, Vocab).
  // So we iterate `i` (In=Dim). Read Row of size Vocab.
  // Add to logits.

  float *logits = (float *)calloc(config.vocab_size, sizeof(float));

  // Quantize input (Norm Out)
  float max_val = 0.0f;
  for (int i = 0; i < dim; i++)
    max_val = std::max(max_val, abs(norm_out[i]));
  float scale = 127.0f / std::max(max_val, 1e-5f);
  int8_t *x_q = (int8_t *)malloc(dim);
  for (int i = 0; i < dim; i++)
    x_q[i] =
        (int8_t)std::max(-127.0f, std::min(127.0f, round(norm_out[i] * scale)));

  int8_t *w_row = (int8_t *)malloc(config.vocab_size);
  int32_t *acc = (int32_t *)calloc(config.vocab_size, sizeof(int32_t));

  for (int i = 0; i < dim; i++) {
    modelFile.read((uint8_t *)w_row, config.vocab_size);
    int8_t xv = x_q[i];
    if (xv != 0) {
      for (int v = 0; v < config.vocab_size; v++)
        acc[v] += xv * w_row[v];
    }
  }
  free(w_row);
  free(x_q);

  // Dequantize Logits (and find max)
  int best_token = 0;
  float best_logit = -1e9f;

  for (int v = 0; v < config.vocab_size; v++) {
    logits[v] = (float)acc[v] / scale;
    if (logits[v] > best_logit) {
      best_logit = logits[v];
      best_token = v;
    }
  }

  free(acc);
  free(norm_out);
  free(logits); // We just needed argmax
  free(X);

  return best_token;
}

void setup() {
  auto cfg = M5.config();
  M5Cardputer.begin(cfg, true);
  M5Cardputer.Display.setRotation(1);
  M5Cardputer.Display.setTextSize(1);

  M5Cardputer.Display.println("Init SD...");
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);

  setup_model();

  M5Cardputer.Display.fillScreen(BLACK);
  M5Cardputer.Display.setCursor(0, 0);
  M5Cardputer.Display.println("Atomic-1Bit Ready.");
  M5Cardputer.Display.println("Type & Enter...");
}

std::string inputBuffer = "";
std::vector<int> tokens; // History
// Dummy tokenizer (Char level or space?)
// Real tokenizer requires vocab file. Atomic-1Bit uses GPT2.
// For MVP, we can just feed inputBuffer chars as bytes?
// No, GPT2 vocab is specific.
// We will assume "User: " prompt injection via tokens?
// This simplified sketch will just tokenize by SPACE mapping to random ID for
// test? Actually, let's just run generation from start token (ID 50256) if
// empty.

void loop() {
  M5Cardputer.update();
  if (M5Cardputer.Keyboard.isChange()) {
    if (M5Cardputer.Keyboard.isPressed()) {
      M5Cardputer.Keyboard.update(); // ? Wrapper handles keys?
                                     // M5Cardputer.Keyboard.keys state?
                                     // Simplified:
      // Just assume we want to trigger generation on 'GO' button (BtnA)
    }
  }

  // Check Enter (BtnA is often Enter on Cardputer or G0)
  if (M5Cardputer.BtnA.wasPressed()) {
    M5Cardputer.Display.println("\nThinking...");

    // Start with EOT
    tokens.clear();
    tokens.push_back(50256); // <|endoftext|>

    // Generate 20 tokens
    for (int k = 0; k < 20; k++) {
      int next_tok = forward(tokens);
      tokens.push_back(next_tok);
      M5Cardputer.Display.printf("%d ", next_tok);
    }
    M5Cardputer.Display.println("\nDone.");
  }
}
