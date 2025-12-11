#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

#include <iostream>
#include <vector>

using namespace std;

// Atomic-1Bit Embedded Runtime
// No dependencies. Pure C++.

struct Config {
  int vocab_size;
  int dim;
  int depth;
  int heads;
  int max_seq_len;
  int has_gist; // New field
};

// Utils: Tensor Structure
struct TensorF32 {
  vector<float> data;
  int rows, cols;
};

struct TensorI8 {
  vector<int8_t> data;
  int rows, cols;
};

// Model Weights
struct AtomicModel {
  Config config;

  vector<float> gist_vector; // New

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
void bit_linear(const vector<float> &x, const TensorI8 &w, vector<float> &out) {
  int in_dim = w.rows;
  int out_dim = w.cols;

  // 1. Quantize Input
  // Scale s = 127 / max(|x|)
  float max_val = 0.0f;
  for (float v : x)
    max_val = max(max_val, abs(v));
  if (max_val < 1e-5f)
    max_val = 1e-5f;

  float scale = 127.0f / max_val;
  vector<int8_t> x_q(in_dim);
  for (int i = 0; i < in_dim; ++i) {
    float val = round(x[i] * scale);
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
    out[j] = (float)acc / scale;
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
  f.read((char *)&model.config, sizeof(Config));
  Config &c = model.config;

  cout << "Loaded Config: Dim=" << c.dim << " Layers=" << c.depth
       << " Heads=" << c.heads << " HasGist=" << c.has_gist << endl;

  // Read Gist if present
  if (c.has_gist) {
    model.gist_vector.resize(c.dim);
    f.read((char *)model.gist_vector.data(), c.dim * sizeof(float));
  }

  // Helper
  auto read_f32 = [&](TensorF32 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    t.data.resize(r * c_dim);
    f.read((char *)t.data.data(), t.data.size() * sizeof(float));
  };

  auto read_i8 = [&](TensorI8 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    t.data.resize(r * c_dim);
    f.read((char *)t.data.data(), t.data.size() * sizeof(int8_t));
  };

  // Embeddings
  read_f32(model.token_emb, c.vocab_size, c.dim);
  read_f32(model.pos_emb, c.max_seq_len, c.dim);

  // Layers
  model.layers.resize(c.depth);
  for (int i = 0; i < c.depth; ++i) {
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

// --- Forward Pass ---

void forward(AtomicModel &model, int token_id, vector<float> &logits) {
  int dim = model.config.dim;
  int heads = model.config.heads;
  int head_dim = dim / heads;

  // 1. Input Processing
  // If Gist: X has 2 tokens: [Gist, UserToken]
  // Else: X has 1 token: [UserToken]

  int seq_len = model.config.has_gist ? 2 : 1;
  vector<float> X(seq_len * dim); // Flattened (Seq, Dim)

  if (model.config.has_gist) {
    // Tok 0: Gist
    memcpy(X.data(), model.gist_vector.data(), dim * sizeof(float));
    // Tok 1: User (with Pos Emb 0) - Wait, should User be pos 0 or 1?
    // In Python Transformer: `pos = arange(0, T)`, and gist prepended.
    // So Gist is conceptually "before time", or we shifted prompt.
    // In my Python code: `x = cat([gist, x])`. `x` had pos emb added.
    // `gist` had NO pos emb.
    // So: Gist is raw. User is Token + Pos[0].
    for (int i = 0; i < dim; ++i) {
      X[dim + i] = model.token_emb.data[token_id * dim + i] +
                   model.pos_emb.data[0 * dim + i]; // Pos 0
    }
  } else {
    // Tok 0: User (Pos 0)
    for (int i = 0; i < dim; ++i) {
      X[i] = model.token_emb.data[token_id * dim + i] +
             model.pos_emb.data[0 * dim + i];
    }
  }

  // Scratch buffers
  vector<float> X_norm(seq_len * dim);

  for (auto &layer : model.layers) {
    // Norm
    for (int t = 0; t < seq_len; ++t) {
      // Slice X[t]
      vector<float> xt(dim), xn(dim);
      memcpy(xt.data(), &X[t * dim], dim * sizeof(float));
      rms_norm(xt, layer.ln1.data, xn, dim);
      memcpy(&X_norm[t * dim], xn.data(), dim * sizeof(float));
    }

    // Attention (Lazy/Simple: We only care about Last Token output usually,
    // but for deep layers, we need Last Token to attend to Gist Token)

    // Q, K, V for all tokens
    vector<float> Q(seq_len * dim), K(seq_len * dim), V(seq_len * dim);

    for (int t = 0; t < seq_len; ++t) {
      vector<float> xt(dim);
      memcpy(xt.data(), &X_norm[t * dim], dim * sizeof(float));

      vector<float> q(dim), k(dim), v(dim);
      bit_linear(xt, layer.q_w, q);
      bit_linear(xt, layer.k_w, k);
      bit_linear(xt, layer.v_w, v);

      memcpy(&Q[t * dim], q.data(), dim * sizeof(float));
      memcpy(&K[t * dim], k.data(), dim * sizeof(float));
      memcpy(&V[t * dim], v.data(), dim * sizeof(float));
    }

    // We only really need to compute output for the LAST token (User Token)
    // because that's what we predict next with.
    // But in multi-layer, we need output for all tokens?
    // Yes, because next layer Gist token representation changes too?
    // Yes, transformer updates all positions.

    vector<float> AttnOut(seq_len * dim);

    // RE-DO Attention Loop Correctly
    fill(AttnOut.begin(), AttnOut.end(), 0.0f);

    for (int t = 0; t < seq_len; ++t) {
      for (int h = 0; h < heads; ++h) {
        float *q_ptr = &Q[t * dim + h * head_dim];
        float *out_ptr = &AttnOut[t * dim + h * head_dim];

        vector<float> scores;
        // Attend to seq 0..t
        for (int src = 0; src <= t; ++src) {
          float *k_ptr = &K[src * dim + h * head_dim];
          float dp = 0.0f;
          for (int d = 0; d < head_dim; ++d)
            dp += q_ptr[d] * k_ptr[d];
          scores.push_back(dp / sqrt((float)head_dim));
        }
        softmax(scores);

        // Weighted V
        for (int src = 0; src <= t; ++src) {
          float w = scores[src];
          float *v_ptr = &V[src * dim + h * head_dim];
          for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] += w * v_ptr[d];
          }
        }
      }
    }

    // O Proj and Residual
    for (int t = 0; t < seq_len; ++t) {
      vector<float> attn_vec(dim), o_vec(dim);
      memcpy(attn_vec.data(), &AttnOut[t * dim], dim * sizeof(float));
      bit_linear(attn_vec, layer.o_w, o_vec);

      // Add Resid
      for (int i = 0; i < dim; ++i)
        X[t * dim + i] += o_vec[i];
    }

    // MLP
    for (int t = 0; t < seq_len; ++t) {
      vector<float> xt(dim), xn(dim);
      memcpy(xt.data(), &X[t * dim], dim * sizeof(float));
      rms_norm(xt, layer.ln2.data, xn, dim);

      int hidden = 4 * dim;
      vector<float> h_act(hidden);
      bit_linear(xn, layer.fc1_w, h_act);
      for (int i = 0; i < hidden; ++i)
        h_act[i] = gelu(h_act[i]);

      vector<float> mlp_out(dim);
      bit_linear(h_act, layer.fc2_w, mlp_out);

      for (int i = 0; i < dim; ++i)
        X[t * dim + i] += mlp_out[i];
    }
  }

  // Final Output (Last Token Only)
  int last_t = seq_len - 1;
  vector<float> xt(dim), xn(dim);
  memcpy(xt.data(), &X[last_t * dim], dim * sizeof(float));
  rms_norm(xt, model.ln_f.data, xn, dim);
  bit_linear(xn, model.head_w, logits);
}

int main(int argc, char **argv) {
  cout << "--- Atomic-1Bit Bare Metal Runner (Gist Enabled) ---" << endl;

  AtomicModel model;
  if (!load_model("atomic_model.bin", model)) {
    cerr << "Failed to load atomic_model.bin!" << endl;
    return 1;
  }

  int dummy_token = 42;
  if (argc > 1)
    dummy_token = atoi(argv[1]);

  cout << "Running Forward Pass for Token ID: " << dummy_token << endl;
  if (model.config.has_gist)
    cout << ">> Gist Token Detected and Injected." << endl;

  vector<float> logits(model.config.vocab_size);
  forward(model, dummy_token, logits);

  // Print Top 5
  cout << "Logits (First 5): ";
  for (int i = 0; i < 5; ++i)
    cout << logits[i] << " ";
  cout << endl;

  return 0;
}
