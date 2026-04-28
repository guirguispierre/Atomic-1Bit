/**
 * Atomic-1Bit WebAssembly Inference Runtime
 *
 * Compiles the C++ inference engine to WebAssembly using Emscripten.
 * Runs the Pocket model (4096 vocab, 256 dim, 4 layers) in the browser.
 *
 * Build:
 *   emcc atomic_wasm.cpp -O3 -s WASM=1 \
 *     -s EXPORTED_FUNCTIONS='["_init_model","_generate_token","_get_logits","_free_model"]' \
 *     -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32"]' \
 *     -s ALLOW_MEMORY_GROWTH=1 \
 *     -s INITIAL_MEMORY=67108864 \
 *     -o atomic.js
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// --- Model Config ---
// Pocket model defaults; overridden at load time from binary header
static int VOCAB_SIZE = 4096;
static int DIM = 256;
static int DEPTH = 4;
static int HEADS = 4;
static int HEAD_DIM = 64;
static int CONTEXT_LEN = 128;
static int HIDDEN_DIM = 1024;
static int HAS_GIST = 0;

// --- Memory Buffers ---
// All allocated dynamically after config is read from model binary

static float *token_emb = nullptr;
static float *pos_emb = nullptr;
static float *gist_vector = nullptr;

struct TensorI8 {
    int8_t *data;
    float scale;
    int rows, cols;
};

struct TensorF32 {
    float *data;
    int size;
};

struct Layer {
    TensorF32 ln1;
    TensorI8 q_w, k_w, v_w, o_w;
    TensorF32 ln2;
    TensorI8 fc1_w, fc2_w;
};

static Layer *layers = nullptr;
static TensorF32 ln_f = {nullptr, 0};
static TensorI8 head_w = {nullptr, 0.0f, 0, 0};

// Working memory
static float *x_buf = nullptr;    // [CONTEXT_LEN * DIM]
static float *x_norm = nullptr;   // [DIM]
static float *q_buf = nullptr;    // [DIM]
static float *k_buf = nullptr;    // [DIM]
static float *v_buf = nullptr;    // [DIM]
static float *o_buf = nullptr;    // [DIM]
static int8_t *x_q = nullptr;    // [DIM]
static float *fc_act = nullptr;  // [HIDDEN_DIM]
static float *mlp_out = nullptr; // [DIM]
static float *logits_buf = nullptr; // [VOCAB_SIZE]

// Attention buffers
static float *Q_buf = nullptr;   // [CONTEXT_LEN * DIM]
static float *K_buf = nullptr;   // [CONTEXT_LEN * DIM]
static float *V_buf = nullptr;   // [CONTEXT_LEN * DIM]
static float *attn_out = nullptr; // [CONTEXT_LEN * DIM]
static float *scores = nullptr;  // [CONTEXT_LEN]

// Context state
static int *context_tokens = nullptr;
static int context_len = 0;

// --- Core Operations ---

static inline float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

static void rms_norm(const float *x, const float *w, float *out, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / dim + 1e-5f);
    for (int i = 0; i < dim; i++)
        out[i] = x[i] * rms * w[i];
}

static void softmax(float *x, int len) {
    float max_val = -1e9f;
    for (int i = 0; i < len; i++)
        if (x[i] > max_val)
            max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < len; i++)
        x[i] /= sum;
}

static void quantize_input(const float *x, int8_t *xq, int dim, float *scale_out) {
    float max_val = 0.0f;
    for (int i = 0; i < dim; i++) {
        float av = fabsf(x[i]);
        if (av > max_val)
            max_val = av;
    }
    if (max_val < 1e-5f)
        max_val = 1e-5f;
    float scale = 127.0f / max_val;
    *scale_out = scale;
    for (int i = 0; i < dim; i++) {
        float val = roundf(x[i] * scale);
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        xq[i] = (int8_t)val;
    }
}

static void bit_linear(const float *input, const TensorI8 *w,
                        float *output, int in_dim, int out_dim) {
    // RMSNorm (non-affine)
    float sum_sq = 0.0f;
    for (int i = 0; i < in_dim; i++)
        sum_sq += input[i] * input[i];
    float rms = sqrtf(sum_sq / in_dim + 1e-5f);

    // Normalize and quantize
    float normed[1024]; // Stack buffer, max dim
    for (int i = 0; i < in_dim; i++)
        normed[i] = input[i] / rms;

    float quant_scale;
    quantize_input(normed, x_q, in_dim, &quant_scale);

    // Ternary matmul
    for (int j = 0; j < out_dim; j++) {
        int32_t acc = 0;
        for (int i = 0; i < in_dim; i++) {
            int8_t wv = w->data[i * out_dim + j];
            if (wv == 1)
                acc += x_q[i];
            else if (wv == -1)
                acc -= x_q[i];
        }
        output[j] = ((float)acc / quant_scale) * w->scale;
    }
}

// --- Model Loading ---

static void alloc_tensor_f32(TensorF32 *t, int size) {
    t->data = (float *)malloc(size * sizeof(float));
    t->size = size;
}

static void alloc_tensor_i8(TensorI8 *t, int rows, int cols) {
    t->rows = rows;
    t->cols = cols;
    t->data = (int8_t *)malloc(rows * cols);
}

// Exported C API for JavaScript
extern "C" {

/**
 * Initialize model from a binary buffer.
 * Returns 0 on success, -1 on error.
 *
 * The buffer should contain the ATOM format binary:
 *   [magic:4] [version:4] [config:24] [embeddings] [layers] [head]
 */
int init_model(const uint8_t *data, int data_len) {
    int offset = 0;

    // Read header
    int32_t magic, version;
    memcpy(&magic, data + offset, 4); offset += 4;
    memcpy(&version, data + offset, 4); offset += 4;

    if (magic != 0x41544F4D)
        return -1; // Bad magic

    // Read config (6 ints = 24 bytes)
    int32_t config[6];
    memcpy(config, data + offset, 24); offset += 24;

    VOCAB_SIZE = config[0];
    DIM = config[1];
    DEPTH = config[2];
    HEADS = config[3];
    CONTEXT_LEN = config[4];
    HAS_GIST = config[5];
    HEAD_DIM = DIM / HEADS;
    HIDDEN_DIM = 4 * DIM;

    // Gist vector
    if (HAS_GIST) {
        gist_vector = (float *)malloc(DIM * sizeof(float));
        memcpy(gist_vector, data + offset, DIM * sizeof(float));
        offset += DIM * sizeof(float);
    }

    // Token embeddings
    int emb_size = VOCAB_SIZE * DIM;
    token_emb = (float *)malloc(emb_size * sizeof(float));
    memcpy(token_emb, data + offset, emb_size * sizeof(float));
    offset += emb_size * sizeof(float);

    // Positional embeddings
    int pos_size = CONTEXT_LEN * DIM;
    pos_emb = (float *)malloc(pos_size * sizeof(float));
    memcpy(pos_emb, data + offset, pos_size * sizeof(float));
    offset += pos_size * sizeof(float);

    // Helper lambdas
    auto read_f32 = [&](TensorF32 *t, int size) {
        alloc_tensor_f32(t, size);
        memcpy(t->data, data + offset, size * sizeof(float));
        offset += size * sizeof(float);
    };

    auto read_i8 = [&](TensorI8 *t, int rows, int cols) {
        alloc_tensor_i8(t, rows, cols);
        memcpy(&t->scale, data + offset, sizeof(float));
        offset += sizeof(float);
        memcpy(t->data, data + offset, rows * cols);
        offset += rows * cols;
    };

    // Layers
    layers = (Layer *)malloc(DEPTH * sizeof(Layer));
    for (int l = 0; l < DEPTH; l++) {
        read_f32(&layers[l].ln1, DIM);
        read_i8(&layers[l].q_w, DIM, DIM);
        read_i8(&layers[l].k_w, DIM, DIM);
        read_i8(&layers[l].v_w, DIM, DIM);
        read_i8(&layers[l].o_w, DIM, DIM);
        read_f32(&layers[l].ln2, DIM);
        read_i8(&layers[l].fc1_w, DIM, HIDDEN_DIM);
        read_i8(&layers[l].fc2_w, HIDDEN_DIM, DIM);
    }

    // Final norm + head
    read_f32(&ln_f, DIM);
    read_i8(&head_w, DIM, VOCAB_SIZE);

    // Allocate working memory
    x_buf = (float *)calloc(CONTEXT_LEN * DIM, sizeof(float));
    x_norm = (float *)malloc(DIM * sizeof(float));
    q_buf = (float *)malloc(DIM * sizeof(float));
    k_buf = (float *)malloc(DIM * sizeof(float));
    v_buf = (float *)malloc(DIM * sizeof(float));
    o_buf = (float *)malloc(DIM * sizeof(float));
    x_q = (int8_t *)malloc(HIDDEN_DIM * sizeof(int8_t));
    fc_act = (float *)malloc(HIDDEN_DIM * sizeof(float));
    mlp_out = (float *)malloc(DIM * sizeof(float));
    logits_buf = (float *)malloc(VOCAB_SIZE * sizeof(float));
    Q_buf = (float *)calloc(CONTEXT_LEN * DIM, sizeof(float));
    K_buf = (float *)calloc(CONTEXT_LEN * DIM, sizeof(float));
    V_buf = (float *)calloc(CONTEXT_LEN * DIM, sizeof(float));
    attn_out = (float *)calloc(CONTEXT_LEN * DIM, sizeof(float));
    scores = (float *)malloc(CONTEXT_LEN * sizeof(float));
    context_tokens = (int *)malloc(CONTEXT_LEN * sizeof(int));
    context_len = 0;

    return 0;
}

/**
 * Run forward pass on the current context and return the greedy next token.
 * Also populates logits_buf with the full logit vector.
 */
int generate_token(int input_token) {
    if (context_len >= CONTEXT_LEN) {
        // Shift context window
        memmove(context_tokens, context_tokens + 1,
                (CONTEXT_LEN - 1) * sizeof(int));
        context_len = CONTEXT_LEN - 1;
    }
    context_tokens[context_len] = input_token;
    context_len++;

    int seq_len = context_len;
    int gist_offset = HAS_GIST ? 1 : 0;
    int total_len = seq_len + gist_offset;

    // Embed
    if (HAS_GIST) {
        memcpy(x_buf, gist_vector, DIM * sizeof(float));
    }
    for (int t = 0; t < seq_len; t++) {
        int tok = context_tokens[t];
        if (tok >= VOCAB_SIZE) tok = 0;
        int pos = t + gist_offset;
        if (pos >= CONTEXT_LEN) pos = CONTEXT_LEN - 1;
        float *dst = x_buf + (t + gist_offset) * DIM;
        for (int d = 0; d < DIM; d++)
            dst[d] = token_emb[tok * DIM + d] + pos_emb[pos * DIM + d];
    }

    // Transformer layers
    for (int l = 0; l < DEPTH; l++) {
        Layer *layer = &layers[l];

        // Pre-norm + attention
        for (int t = 0; t < total_len; t++) {
            float *xt = x_buf + t * DIM;
            rms_norm(xt, layer->ln1.data, x_norm, DIM);

            bit_linear(x_norm, &layer->q_w, Q_buf + t * DIM, DIM, DIM);
            bit_linear(x_norm, &layer->k_w, K_buf + t * DIM, DIM, DIM);
            bit_linear(x_norm, &layer->v_w, V_buf + t * DIM, DIM, DIM);
        }

        // Multi-head attention
        memset(attn_out, 0, total_len * DIM * sizeof(float));
        for (int t = 0; t < total_len; t++) {
            for (int h = 0; h < HEADS; h++) {
                float *q_ptr = Q_buf + t * DIM + h * HEAD_DIM;
                float *out_ptr = attn_out + t * DIM + h * HEAD_DIM;

                // Compute scores (causal: only attend to <= t)
                for (int src = 0; src <= t; src++) {
                    float *k_ptr = K_buf + src * DIM + h * HEAD_DIM;
                    float dp = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++)
                        dp += q_ptr[d] * k_ptr[d];
                    scores[src] = dp / sqrtf((float)HEAD_DIM);
                }
                softmax(scores, t + 1);

                // Weighted sum of values
                for (int src = 0; src <= t; src++) {
                    float w = scores[src];
                    float *v_ptr = V_buf + src * DIM + h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++)
                        out_ptr[d] += w * v_ptr[d];
                }
            }
        }

        // Output projection + residual
        for (int t = 0; t < total_len; t++) {
            bit_linear(attn_out + t * DIM, &layer->o_w, o_buf, DIM, DIM);
            float *xt = x_buf + t * DIM;
            for (int d = 0; d < DIM; d++)
                xt[d] += o_buf[d];
        }

        // MLP
        for (int t = 0; t < total_len; t++) {
            float *xt = x_buf + t * DIM;
            rms_norm(xt, layer->ln2.data, x_norm, DIM);

            bit_linear(x_norm, &layer->fc1_w, fc_act, DIM, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                fc_act[i] = gelu_approx(fc_act[i]);

            bit_linear(fc_act, &layer->fc2_w, mlp_out, HIDDEN_DIM, DIM);
            for (int d = 0; d < DIM; d++)
                xt[d] += mlp_out[d];
        }
    }

    // Final norm + head projection (last token only)
    int last_t = total_len - 1;
    rms_norm(x_buf + last_t * DIM, ln_f.data, x_norm, DIM);
    bit_linear(x_norm, &head_w, logits_buf, DIM, VOCAB_SIZE);

    // Greedy argmax
    int best = 0;
    float best_val = logits_buf[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits_buf[i] > best_val) {
            best_val = logits_buf[i];
            best = i;
        }
    }

    return best;
}

/**
 * Get pointer to logits buffer (for temperature sampling in JS).
 */
float *get_logits() {
    return logits_buf;
}

/**
 * Get vocabulary size.
 */
int get_vocab_size() {
    return VOCAB_SIZE;
}

/**
 * Reset the context window.
 */
void reset_context() {
    context_len = 0;
}

/**
 * Free all model memory.
 */
void free_model() {
    free(token_emb); token_emb = nullptr;
    free(pos_emb); pos_emb = nullptr;
    free(gist_vector); gist_vector = nullptr;

    if (layers) {
        for (int l = 0; l < DEPTH; l++) {
            free(layers[l].ln1.data);
            free(layers[l].q_w.data);
            free(layers[l].k_w.data);
            free(layers[l].v_w.data);
            free(layers[l].o_w.data);
            free(layers[l].ln2.data);
            free(layers[l].fc1_w.data);
            free(layers[l].fc2_w.data);
        }
        free(layers); layers = nullptr;
    }

    free(ln_f.data); ln_f.data = nullptr;
    free(head_w.data); head_w.data = nullptr;

    free(x_buf); free(x_norm);
    free(q_buf); free(k_buf); free(v_buf); free(o_buf);
    free(x_q); free(fc_act); free(mlp_out); free(logits_buf);
    free(Q_buf); free(K_buf); free(V_buf); free(attn_out);
    free(scores); free(context_tokens);
}

} // extern "C"
