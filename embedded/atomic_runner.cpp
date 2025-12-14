#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Debug Utils
using namespace std;

// Atomic-1Bit Embedded Runtime
// No dependencies. Pure C++.

struct Config {
  int vocab_size;
  int dim;
  int depth;
  int heads;
  int max_seq_len;
  int has_gist;
};

// Utils: Tensor Structure
struct TensorF32 {
  vector<float> data;
  int rows, cols;
};

struct TensorI8 {
  vector<int8_t> data;
  int rows, cols;
  float scale; // New: 1.58-bit weight scale
};

// Model Weights
struct AtomicModel {
  Config config;

  vector<float> gist_vector;

  TensorF32 token_emb;
  TensorF32 pos_emb;

  struct Layer {
    TensorF32 ln1;
    TensorI8 q_w, k_w, v_w, o_w;
    TensorF32 ln2;
    TensorI8 fc1_w, fc2_w;
  };
  vector<Layer> layers;

  TensorF32 ln_f;
  TensorI8 head_w;
};

// --- Operations ---

// RMSNorm
void rms_norm(const vector<float> &x, const vector<float> &weight,
              vector<float> &out, int dim) {
  float sum_sq = 0.0f;
  for (float v : x)
    sum_sq += v * v;
  float rms = sqrt(sum_sq / dim + 1e-5f);
  for (int i = 0; i < dim; ++i) {
    out[i] = (x[i] / rms) * weight[i];
  }
}

// Global Matmul for BitLinear
// A: Float input [Dim]
// Weight: Int8 [In, Out]
// Out: Float output [Out]
// Scratch: [In] (Pre-allocated buffer to avoid malloc)
void bit_linear(const vector<float> &x, const TensorI8 &w, vector<float> &out,
                vector<int8_t> &scratch) {
  int in_dim = w.rows;
  int out_dim = w.cols;

  // 0. Internal RMSNorm (Non-affine)
  // Per BitNet paper/impl: BitLinear(x) = Layernorm(x) ...
  float sum_sq = 0.0f;
  for (float v : x)
    sum_sq += v * v;
  float rms = sqrt(sum_sq / in_dim + 1e-5f);

  // 1. Quantize Input
  // We effectively quantize (x / rms).
  // Scale s = 127 / max(|x/rms|)
  float max_val = 0.0f;
  for (float v : x)
    max_val = max(max_val, abs(v / rms)); // Optimize: abs(v)/rms

  if (max_val < 1e-5f)
    max_val = 1e-5f;

  float quant_scale = 127.0f / max_val;
  // effective_scale = quant_scale / rms
  float effective_scale = quant_scale / rms;

  vector<int8_t> x_q(in_dim);
  for (int i = 0; i < in_dim; ++i) {
    // x_norm = x[i] / rms
    // val = x_norm * quant_scale
    float val = round(x[i] * effective_scale);
    if (val > 127)
      val = 127;
    if (val < -127)
      val = -127;
    x_q[i] = (int8_t)val;
  }

  // 2. Ternary Kernel
  fill(out.begin(), out.end(), 0.0f);

  for (int j = 0; j < out_dim; ++j) {
    int32_t acc = 0;
    for (int i = 0; i < in_dim; ++i) {
      int8_t xv = x_q[i];
      int8_t wv = w.data[i * out_dim + j];

      if (wv == 1)
        acc += xv;
      else if (wv == -1)
        acc -= xv;
    }
    // 3. Dequantize
    // out = (acc / x_scale) * w_scale
    float val = ((float)acc / quant_scale) * w.scale;
    out[j] = val;

    // Debug first few outputs of first layer
    static int debug_count = 0;
    if (debug_count < 5) {
      // print diagnostics for first few calls
      if (j == 0) {
        cout << "[BitLinear Debug] Dim In: " << in_dim << " Out: " << out_dim
             << endl;
        cout << "  Input Max: " << max_val << " Quant Scale: " << quant_scale
             << " RMS: " << rms << endl;
        cout << "  Weight Scale: " << w.scale << endl; // Check if this is 0
        cout << "  Acc: " << acc << " Dequant: " << val << endl;
        cout << "  Sample W: " << (int)w.data[0] << " " << (int)w.data[1]
             << endl;
        cout << "  Sample X_Q: " << (int)x_q[0] << " " << (int)x_q[1] << endl;
        debug_count++;
      }
    }
  }
}

// Softmax
void softmax(vector<float> &x) {
  float max_val = -1e9;
  for (float v : x)
    max_val = max(max_val, v);

  float sum_exp = 0.0f;
  for (float &v : x) {
    v = exp(v - max_val);
    sum_exp += v;
  }
  for (float &v : x)
    v /= sum_exp;
}

// GELU (Approx)
float gelu(float x) {
  return 0.5f * x *
         (1.0f + tanh(sqrt(2.0f / 3.14159265f) * (x + 0.044715f * x * x * x)));
}

// --- Loading ---

bool load_model(const char *filename, AtomicModel &model) {
  ifstream f(filename, ios::binary);
  if (!f.is_open())
    return false;

  // Header
  // Magic Check
  int32_t magic, version;
  f.read((char *)&magic, sizeof(int32_t));
  if (magic != 0x41544F4D) {
    cerr << "Invalid file format: Magic mismatch (Expected 'ATOM')" << endl;
    return false;
  }

  f.read((char *)&version, sizeof(int32_t));
  if (version != 1) {
    cerr << "Unsupported version: " << version << endl;
    return false;
  }

  cout << "Sizeof(Config): " << sizeof(Config) << " (Expected 24)" << endl;
  f.read((char *)&model.config, sizeof(Config));
  Config &c = model.config;

  cout << "Loaded Config: Dim=" << c.dim << " Layers=" << c.depth
       << " Heads=" << c.heads << " HasGist=" << c.has_gist << endl;
  cout << "  File Pos after Config: " << f.tellg() << endl;

  // Read Gist if present
  if (c.has_gist) {
    model.gist_vector.resize(c.dim);
    f.read((char *)model.gist_vector.data(), c.dim * sizeof(float));
  }
  cout << "  File Pos after Gist: " << f.tellg() << endl;

  // Helper
  auto read_f32 = [&](TensorF32 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    t.data.resize(r * c_dim);
    size_t bytes = t.data.size() * sizeof(float);
    f.read((char *)t.data.data(), bytes);
    cout << "  Read F32 (" << r << "x" << c_dim << ") Bytes: " << bytes
         << " Pos: " << f.tellg() << endl;
  };

  auto read_i8 = [&](TensorI8 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    // Read Scale
    f.read((char *)&t.scale, sizeof(float));
    t.data.resize(r * c_dim);
    size_t bytes = t.data.size() * sizeof(int8_t);
    f.read((char *)t.data.data(), bytes);
    cout << "    Read I8 (" << r << "x" << c_dim << ") Bytes: " << bytes
         << " Pos: " << f.tellg() << endl;
  };

  // Embeddings
  read_f32(model.token_emb, c.vocab_size, c.dim);
  read_f32(model.pos_emb, c.max_seq_len, c.dim);

  // Layers
  model.layers.resize(c.depth);
  for (int i = 0; i < c.depth; ++i) {
    cout << "Loading Layer " << i << "..." << endl;
    read_f32(model.layers[i].ln1, 1, c.dim);

    read_i8(model.layers[i].q_w, c.dim, c.dim);
    read_i8(model.layers[i].k_w, c.dim, c.dim);
    read_i8(model.layers[i].v_w, c.dim, c.dim);
    read_i8(model.layers[i].o_w, c.dim, c.dim);

    read_f32(model.layers[i].ln2, 1, c.dim);

    int hidden = 4 * c.dim;
    read_i8(model.layers[i].fc1_w, c.dim, hidden);
    read_i8(model.layers[i].fc2_w, hidden, c.dim);
  }

  read_f32(model.ln_f, 1, c.dim);
  read_i8(model.head_w, c.dim, c.vocab_size);

  f.close();
  return true;
}

// --- Workspace ---
struct Workspace {
  vector<float> X, X_norm;
  vector<float> Q, K, V;
  vector<float> AttnOut;
  vector<float> q, k, v;
  vector<float> layer_out;
  vector<float> fc_act, mlp_out;
  vector<float> xt, xn;
  vector<int8_t> scratch_q;

  void ensure_size(int max_seq_len, int dim) {
    int max_el = max_seq_len * dim;
    if (X.size() < max_el) {
      X.resize(max_el);
      X_norm.resize(max_el);
      Q.resize(max_el);
      K.resize(max_el);
      V.resize(max_el);
      AttnOut.resize(max_el);
      q.resize(dim);
      k.resize(dim);
      v.resize(dim);
      layer_out.resize(dim);
      fc_act.resize(4 * dim);
      mlp_out.resize(dim);
      xt.resize(dim);
      xn.resize(dim);
      scratch_q.resize(4 * dim);
    }
  }
};

// --- Forward Pass ---
void forward(AtomicModel &model, const vector<int> &tokens,
             vector<float> &logits, Workspace &ws, bool parity_mode = false) {
  int dim = model.config.dim;
  int heads = model.config.heads;
  int head_dim = dim / heads;

  // 1. Input Processing
  int input_len = tokens.size();
  int seq_len = input_len + (model.config.has_gist ? 1 : 0);

  if (seq_len > model.config.max_seq_len)
    return; // Simplified check

  ws.ensure_size(model.config.max_seq_len, dim);

  int t_offset = 0;
  if (model.config.has_gist) {
    memcpy(&ws.X[0], model.gist_vector.data(), dim * sizeof(float));
    t_offset = 1;
  }

  for (int i = 0; i < input_len; ++i) {
    int token_id = tokens[i];
    int pos = i + t_offset;

    if (pos >= model.config.max_seq_len)
      pos = model.config.max_seq_len - 1;

    float *x_ptr = &ws.X[(i + t_offset) * dim];
    if (token_id >= model.config.vocab_size)
      token_id = 0;

    float *tok_ptr = &model.token_emb.data[token_id * dim];
    float *pos_ptr = &model.pos_emb.data[pos * dim];

    for (int d = 0; d < dim; ++d) {
      x_ptr[d] = tok_ptr[d] + pos_ptr[d];
    }
  }

  // Print Embedding Sum (last token)
  // if (parity_mode) {
  //   int last_idx = (seq_len - 1) * dim;
  //   // Define a vector to copy so we can use print_tensor
  //   vector<float> tmp(dim);
  //   memcpy(tmp.data(), &ws.X[last_idx], dim * sizeof(float));
  //   print_tensor("Embedding_Sum", tmp);
  // }

  int l_idx = 0;
  for (auto &layer : model.layers) {
    // Norm
    for (int t = 0; t < seq_len; ++t) {
      memcpy(ws.xt.data(), &ws.X[t * dim], dim * sizeof(float));
      rms_norm(ws.xt, layer.ln1.data, ws.xn, dim);
      memcpy(&ws.X_norm[t * dim], ws.xn.data(), dim * sizeof(float));
    }

    // if (parity_mode) {
    //   int last_idx = (seq_len - 1) * dim;
    //   vector<float> tmp(dim);
    //   memcpy(tmp.data(), &ws.X_norm[last_idx], dim * sizeof(float));
    //   // Construct string layer0_LN1
    //   string name = "Layer" + to_string(l_idx) + "_LN1";
    //   print_tensor(name.c_str(), tmp);
    // }

    // Attention
    for (int t = 0; t < seq_len; ++t) {
      memcpy(ws.xt.data(), &ws.X_norm[t * dim], dim * sizeof(float));

      bit_linear(ws.xt, layer.q_w, ws.q, ws.scratch_q);
      bit_linear(ws.xt, layer.k_w, ws.k, ws.scratch_q);
      bit_linear(ws.xt, layer.v_w, ws.v, ws.scratch_q);

      memcpy(&ws.Q[t * dim], ws.q.data(), dim * sizeof(float));
      memcpy(&ws.K[t * dim], ws.k.data(), dim * sizeof(float));
      memcpy(&ws.V[t * dim], ws.v.data(), dim * sizeof(float));
    }

    // AttnOut
    fill(ws.AttnOut.begin(), ws.AttnOut.begin() + (seq_len * dim), 0.0f);

    for (int t = 0; t < seq_len; ++t) {
      for (int h = 0; h < heads; ++h) {
        float *q_ptr = &ws.Q[t * dim + h * head_dim];
        float *out_ptr = &ws.AttnOut[t * dim + h * head_dim];
        vector<float> scores;
        scores.reserve(t + 1);

        for (int src = 0; src <= t; ++src) {
          float *k_ptr = &ws.K[src * dim + h * head_dim];
          float dp = 0.0f;
          for (int d = 0; d < head_dim; ++d)
            dp += q_ptr[d] * k_ptr[d];
          scores.push_back(dp / sqrt((float)head_dim));
        }
        softmax(scores);

        for (int src = 0; src <= t; ++src) {
          float w = scores[src];
          float *v_ptr = &ws.V[src * dim + h * head_dim];
          for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] += w * v_ptr[d];
          }
        }
      }
    }

    // O Proj
    for (int t = 0; t < seq_len; ++t) {
      memcpy(ws.xt.data(), &ws.AttnOut[t * dim], dim * sizeof(float));
      bit_linear(ws.xt, layer.o_w, ws.layer_out, ws.scratch_q);

      // Residual
      for (int i = 0; i < dim; ++i)
        ws.X[t * dim + i] += ws.layer_out[i];
    }

    // if (parity_mode) {
    //   // We want to print Attn Output (Residual Added? Python script prints
    //   // `attn_out` BEFORE residual add) Wait, python script printed
    //   `attn_out =
    //   // layer.attn(x_norm)`. Here `ws.layer_out` is `attn_out`. But we just
    //   // added it to X. We have it in `ws.layer_out`. Assuming seq_len=1 for
    //   // parity check usually. If seq_len > 1, this only captures the LAST
    //   // token's execution in that loop.
    //   string name = "Layer" + to_string(l_idx) + "_Attn_Out";
    //   print_tensor(name.c_str(), ws.layer_out);
    // }

    // MLP
    for (int t = 0; t < seq_len; ++t) {
      memcpy(ws.xt.data(), &ws.X[t * dim], dim * sizeof(float));
      rms_norm(ws.xt, layer.ln2.data, ws.xn, dim);

      // Save LN2 for debug?
      // Python loop prints LN2 output.
      // if (parity_mode && t == seq_len - 1) {
      //   string name = "Layer" + to_string(l_idx) + "_LN2";
      //   print_tensor(name.c_str(), ws.xn);
      // }

      int hidden = 4 * dim;
      bit_linear(ws.xn, layer.fc1_w, ws.fc_act, ws.scratch_q);
      for (int i = 0; i < hidden; ++i)
        ws.fc_act[i] = gelu(ws.fc_act[i]);

      bit_linear(ws.fc_act, layer.fc2_w, ws.mlp_out, ws.scratch_q);

      for (int i = 0; i < dim; ++i)
        ws.X[t * dim + i] += ws.mlp_out[i];
    }

    // if (parity_mode) {
    //   string name = "Layer" + to_string(l_idx) + "_MLP_Out";
    //   print_tensor(name.c_str(), ws.mlp_out);
    //   cout << endl; // Separator
    // }

    l_idx++;
  }

  // Final Output
  int last_t = seq_len - 1;
  memcpy(ws.xt.data(), &ws.X[last_t * dim], dim * sizeof(float));
  rms_norm(ws.xt, model.ln_f.data, ws.xn, dim);

  // if (parity_mode) {
  //   print_tensor("Final_LN", ws.xn);
  // }

  bit_linear(ws.xn, model.head_w, logits, ws.scratch_q);

  // if (parity_mode) {
  //   print_tensor("Final_Logits", logits, 20);
  // }
}

// Temperature Sampling
int sample_logits(const vector<float> &logits, float temp, std::mt19937 &rng) {
  if (temp < 1e-5f) {
    // Greedy
    int best_token = 0;
    float best_val = -1e9f;
    for (int i = 0; i < logits.size(); ++i) {
      if (logits[i] > best_val) {
        best_val = logits[i];
        best_token = i;
      }
    }
    return best_token;
  }

  // Softmax with Temp
  vector<float> probs(logits.size());
  float max_val = -1e9f;
  for (float v : logits)
    max_val = max(max_val, v);

  float sum_exp = 0.0f;
  for (int i = 0; i < logits.size(); ++i) {
    probs[i] = exp((logits[i] - max_val) / temp);
    sum_exp += probs[i];
  }

  std::uniform_real_distribution<float> dist(0.0f, sum_exp);
  float r = dist(rng);
  float acc = 0.0f;
  for (int i = 0; i < probs.size(); ++i) {
    acc += probs[i];
    if (r <= acc)
      return i;
  }
  return probs.size() - 1;
}

// Debug Utils (Moved to top)

int main(int argc, char **argv) {
  string model_path = "atomic_model.bin";
  int gen_len = 50;
  float temp = 0.0f;
  int seed = 42;
  int start_token = 42;
  bool parity_mode = false; // New Flag

  // Simple Arg Parsing
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--model" && i + 1 < argc)
      model_path = argv[++i];
    else if (arg == "--steps" && i + 1 < argc)
      gen_len = atoi(argv[++i]);
    else if (arg == "--temp" && i + 1 < argc)
      temp = atof(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      seed = atoi(argv[++i]);
    else if (arg == "--start_token" && i + 1 < argc)
      start_token = atoi(argv[++i]);
    else if (arg == "--parity")
      parity_mode = true;
  }

  if (parity_mode) {
    cout << "--- C++ Parity Check ---" << endl;
    gen_len = 1; // Force single step
  }

  // ... (rest of main)

  cout << "Atomic-1Bit Engine | Model: " << model_path
       << " | Steps: " << gen_len << " | Temp: " << temp << " | Seed: " << seed
       << endl;

  std::mt19937 rng(seed);

  AtomicModel model;
  if (!load_model(model_path.c_str(), model)) {
    cerr << "Error: Could not load model file." << endl;
    return 1;
  }

  vector<int> context;
  context.push_back(start_token);

  cout << "Generating: ";

  Workspace ws;
  ws.ensure_size(model.config.max_seq_len, model.config.dim);

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < gen_len; ++step) {
    // Rolling Window
    if (context.size() > model.config.max_seq_len) {
      context.erase(context.begin(),
                    context.begin() +
                        (context.size() - model.config.max_seq_len));
    }

    vector<float> logits(model.config.vocab_size);
    forward(model, context, logits, ws, parity_mode);
    int best_token = sample_logits(logits, temp, rng);
    cout << best_token << " " << flush;
    context.push_back(best_token);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  double tps = gen_len / diff.count();
  cout << endl << "Done." << endl;
  cout << "TPS: " << tps << endl;

  return 0;
}
