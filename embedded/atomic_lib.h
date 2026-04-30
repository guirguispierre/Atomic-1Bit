#ifndef ATOMIC_LIB_H
#define ATOMIC_LIB_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Binary format magic ("ATOM" packed as little-endian int32) and version.
// Must match atomic_1bit/utils/export_to_cpp.py.
static constexpr uint32_t ATOMIC_MAGIC = 0x41544F4Du;
static constexpr uint32_t ATOMIC_VERSION = 1u;

struct Config {
  int vocab_size;
  int dim;
  int depth;
  int heads;
  int max_seq_len;
  int has_gist;
};

inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
}

inline void softmax(vector<float> &x) {
  float max_val = -1e9f;
  for (float f : x)
    if (f > max_val)
      max_val = f;

  float sum = 0.0f;
  for (float &f : x) {
    f = exp(f - max_val);
    sum += f;
  }
  for (float &f : x)
    f /= sum;
}

// Affine RMSNorm (used for the layer's ln1 / ln2 / ln_f).
inline void rms_norm(const vector<float> &x, const vector<float> &w,
                     vector<float> &out, int dim) {
  float ss = 0.0f;
  for (float f : x)
    ss += f * f;
  ss /= dim;
  float rms = 1.0f / sqrt(ss + 1e-5f);

  for (int i = 0; i < dim; ++i) {
    out[i] = x[i] * rms * w[i];
  }
}

// BitLinear with internal SubLN (per BitNet b1.58 spec) and ternary kernel.
//
// Mirrors the Python forward in atomic_1bit/nn/layers.py:
//   1) x_norm   = x / rms(x)
//   2) scale_x  = 127 / max(|x_norm|)
//   3) x_q      = round(x_norm * scale_x)
//   4) acc[o]   = sum_i x_q[i] * w_q[i, o]
//   5) y[o]     = acc * scale_w / scale_x
//
// `scale_w` is the per-tensor float written by export_to_cpp.py
// (= mean(|w|), the inverse of the per-weight quantization scale).
//
// w layout: row-major [in_dim, out_dim] (the exporter calls .t() before
// writing).
inline void bit_linear(const vector<float> &x, const vector<int8_t> &w,
                       float scale_w, vector<float> &out) {
  int in_dim = (int)x.size();
  int out_dim = (int)out.size();

  // SubLN
  float ss = 0.0f;
  for (float v : x)
    ss += v * v;
  float rms = sqrt(ss / in_dim + 1e-5f);

  // Activation absmax over x/rms = max(|x|)/rms
  float max_abs = 0.0f;
  for (float v : x)
    max_abs = max(max_abs, fabsf(v));
  float max_val = max_abs / rms;
  if (max_val < 1e-5f)
    max_val = 1e-5f;

  float scale_x = 127.0f / max_val;
  float effective_scale = scale_x / rms;

  vector<int8_t> x_q(in_dim);
  for (int i = 0; i < in_dim; ++i) {
    float q = roundf(x[i] * effective_scale);
    if (q > 127.0f)
      q = 127.0f;
    if (q < -127.0f)
      q = -127.0f;
    x_q[i] = (int8_t)q;
  }

  float dequant = scale_w / scale_x;
  for (int o = 0; o < out_dim; ++o) {
    int32_t acc = 0;
    for (int i = 0; i < in_dim; ++i) {
      int8_t wv = w[i * out_dim + o];
      if (wv == 1)
        acc += x_q[i];
      else if (wv == -1)
        acc -= x_q[i];
    }
    out[o] = (float)acc * dequant;
  }
}

struct AtomicLayer {
  vector<float> ln1;
  vector<int8_t> q_w, k_w, v_w, o_w;
  float q_s, k_s, v_s, o_s;
  vector<float> ln2;
  vector<int8_t> fc1_w, fc2_w;
  float fc1_s, fc2_s;
};

struct AtomicModel {
  Config config;
  vector<float> token_emb;
  vector<float> pos_emb;
  vector<float> ln_f;
  vector<int8_t> head_w;
  float head_s;
  vector<AtomicLayer> layers;

  vector<float> gist_vector;
};

inline bool read_scale(ifstream &f, float &out) {
  return (bool)f.read((char *)&out, sizeof(float));
}

inline bool load_model(const string &filename, AtomicModel &model) {
  ifstream f(filename, ios::binary);
  if (!f.is_open()) {
    cerr << "load_model: cannot open " << filename << endl;
    return false;
  }

  uint32_t magic = 0, version = 0;
  f.read((char *)&magic, 4);
  f.read((char *)&version, 4);
  if (magic != ATOMIC_MAGIC) {
    cerr << "load_model: bad magic 0x" << hex << magic
         << " (expected 0x" << ATOMIC_MAGIC << ")" << dec << endl;
    return false;
  }
  if (version != ATOMIC_VERSION) {
    cerr << "load_model: unsupported version " << version
         << " (expected " << ATOMIC_VERSION << ")" << endl;
    return false;
  }

  f.read((char *)&model.config.vocab_size, 4);
  f.read((char *)&model.config.dim, 4);
  f.read((char *)&model.config.depth, 4);
  f.read((char *)&model.config.heads, 4);
  f.read((char *)&model.config.max_seq_len, 4);
  f.read((char *)&model.config.has_gist, 4);

  int dim = model.config.dim;
  int hidden = 4 * dim;

  if (model.config.has_gist) {
    model.gist_vector.resize(dim);
    f.read((char *)model.gist_vector.data(), dim * sizeof(float));
  }

  model.token_emb.resize((size_t)model.config.vocab_size * dim);
  f.read((char *)model.token_emb.data(),
         model.token_emb.size() * sizeof(float));

  model.pos_emb.resize((size_t)model.config.max_seq_len * dim);
  f.read((char *)model.pos_emb.data(), model.pos_emb.size() * sizeof(float));

  for (int i = 0; i < model.config.depth; ++i) {
    AtomicLayer layer;

    layer.ln1.resize(dim);
    f.read((char *)layer.ln1.data(), dim * sizeof(float));

    int attn_size = dim * dim;
    read_scale(f, layer.q_s);
    layer.q_w.resize(attn_size);
    f.read((char *)layer.q_w.data(), attn_size);

    read_scale(f, layer.k_s);
    layer.k_w.resize(attn_size);
    f.read((char *)layer.k_w.data(), attn_size);

    read_scale(f, layer.v_s);
    layer.v_w.resize(attn_size);
    f.read((char *)layer.v_w.data(), attn_size);

    read_scale(f, layer.o_s);
    layer.o_w.resize(attn_size);
    f.read((char *)layer.o_w.data(), attn_size);

    layer.ln2.resize(dim);
    f.read((char *)layer.ln2.data(), dim * sizeof(float));

    int size_fc1 = dim * hidden;
    int size_fc2 = hidden * dim;
    read_scale(f, layer.fc1_s);
    layer.fc1_w.resize(size_fc1);
    f.read((char *)layer.fc1_w.data(), size_fc1);

    read_scale(f, layer.fc2_s);
    layer.fc2_w.resize(size_fc2);
    f.read((char *)layer.fc2_w.data(), size_fc2);

    model.layers.push_back(layer);
  }

  model.ln_f.resize(dim);
  f.read((char *)model.ln_f.data(), dim * sizeof(float));

  read_scale(f, model.head_s);
  model.head_w.resize((size_t)dim * model.config.vocab_size);
  f.read((char *)model.head_w.data(), model.head_w.size());

  if (!f.good() && !f.eof()) {
    cerr << "load_model: stream error after reading model" << endl;
    return false;
  }
  return true;
}

inline void forward(AtomicModel &model, const vector<int> &tokens,
                    vector<float> &logits) {
  int dim = model.config.dim;
  int heads = model.config.heads;
  int head_dim = dim / heads;

  int input_len = (int)tokens.size();
  int seq_len = input_len + (model.config.has_gist ? 1 : 0);

  if (seq_len > model.config.max_seq_len)
    return;

  vector<float> X(seq_len * dim);

  int t_offset = 0;
  if (model.config.has_gist) {
    memcpy(X.data(), model.gist_vector.data(), dim * sizeof(float));
    t_offset = 1;
  }

  for (int i = 0; i < input_len; ++i) {
    int token_id = tokens[i];
    if (token_id < 0 || token_id >= model.config.vocab_size) {
      cerr << "forward: token " << token_id << " out of vocab range" << endl;
      return;
    }
    float *x_ptr = &X[(t_offset + i) * dim];
    float *tok_ptr = &model.token_emb.data()[token_id * dim];
    float *pos_ptr = &model.pos_emb.data()[i * dim];

    for (int d = 0; d < dim; ++d) {
      x_ptr[d] = tok_ptr[d] + pos_ptr[d];
    }
  }

  vector<float> X_norm(seq_len * dim);

  for (auto &layer : model.layers) {
    for (int t = 0; t < seq_len; ++t) {
      vector<float> xt(dim);
      memcpy(xt.data(), &X[t * dim], dim * sizeof(float));
      vector<float> xn(dim);
      rms_norm(xt, layer.ln1, xn, dim);
      memcpy(&X_norm[t * dim], xn.data(), dim * sizeof(float));
    }

    vector<float> Q(seq_len * dim), K(seq_len * dim), V(seq_len * dim);
    for (int t = 0; t < seq_len; ++t) {
      vector<float> xt(dim);
      memcpy(xt.data(), &X_norm[t * dim], dim * sizeof(float));
      vector<float> q(dim), k(dim), v(dim);
      bit_linear(xt, layer.q_w, layer.q_s, q);
      bit_linear(xt, layer.k_w, layer.k_s, k);
      bit_linear(xt, layer.v_w, layer.v_s, v);
      memcpy(&Q[t * dim], q.data(), dim * sizeof(float));
      memcpy(&K[t * dim], k.data(), dim * sizeof(float));
      memcpy(&V[t * dim], v.data(), dim * sizeof(float));
    }

    vector<float> AttnOut(seq_len * dim, 0.0f);

    for (int t = 0; t < seq_len; ++t) {
      for (int h = 0; h < heads; ++h) {
        float *q_ptr = &Q[t * dim + h * head_dim];
        vector<float> scores;
        for (int src = 0; src <= t; ++src) {
          float *k_ptr = &K[src * dim + h * head_dim];
          float dp = 0.0f;
          for (int d = 0; d < head_dim; ++d)
            dp += q_ptr[d] * k_ptr[d];
          scores.push_back(dp / sqrt((float)head_dim));
        }
        softmax(scores);

        float *out_ptr = &AttnOut[t * dim + h * head_dim];
        for (int src = 0; src <= t; ++src) {
          float wgt = scores[src];
          float *v_ptr = &V[src * dim + h * head_dim];
          for (int d = 0; d < head_dim; ++d)
            out_ptr[d] += wgt * v_ptr[d];
        }
      }
    }

    for (int t = 0; t < seq_len; ++t) {
      vector<float> attn_vec(dim);
      memcpy(attn_vec.data(), &AttnOut[t * dim], dim * sizeof(float));
      vector<float> o_vec(dim);
      bit_linear(attn_vec, layer.o_w, layer.o_s, o_vec);
      for (int i = 0; i < dim; ++i)
        X[t * dim + i] += o_vec[i];
    }

    int hidden = 4 * dim;
    for (int t = 0; t < seq_len; ++t) {
      vector<float> xt(dim);
      memcpy(xt.data(), &X[t * dim], dim * sizeof(float));
      vector<float> xn(dim);
      rms_norm(xt, layer.ln2, xn, dim);

      vector<float> h_act(hidden);
      bit_linear(xn, layer.fc1_w, layer.fc1_s, h_act);
      for (int i = 0; i < hidden; ++i)
        h_act[i] = gelu(h_act[i]);

      vector<float> mlp_out(dim);
      bit_linear(h_act, layer.fc2_w, layer.fc2_s, mlp_out);
      for (int i = 0; i < dim; ++i)
        X[t * dim + i] += mlp_out[i];
    }
  }

  vector<float> xt(dim), xn(dim);
  memcpy(xt.data(), &X[(seq_len - 1) * dim], dim * sizeof(float));
  rms_norm(xt, model.ln_f, xn, dim);
  bit_linear(xn, model.head_w, model.head_s, logits);
}

#endif
