#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
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
  // guirguispierre 2026-08-10 - float32 summation drifts from torch's pairwise
  // sum enough to shift the int8 quant_scale downstream
  double sum_sq = 0.0;
  for (float v : x)
    sum_sq += (double)v * (double)v;
  float rms = sqrt((float)(sum_sq / dim) + 1e-5f);
  for (int i = 0; i < dim; ++i) {
    out[i] = (x[i] / rms) * weight[i];
  }
}

// --- Decode thread pool ---
//
// --threads N > 1 splits a matmul job's OUTPUT columns across N-1 persistent
// workers plus the calling thread. A job is one part (bit_linear) or several
// parts sharing one quantized input (fused Q/K/V: one sync instead of three).
// Each acc[j] is an independent int32 sum over the whole input dim, so any
// partition of columns is bit-identical to the serial loop: integer adds are
// exact under partitioning (and each claim even keeps the serial i-order).
// The float dequant runs once per output element after accumulation, exactly
// as in the serial path.
//
// Work is claimed dynamically in CHUNK-column pieces off an atomic counter:
// on heterogeneous cores (P+E) a slow worker simply takes fewer chunks, so
// the job finishes when the fast cores run out of work rather than when the
// slowest static slice finishes.
//
// Sync: jobs alternate between two slots. A slot's claim counter `next`
// packs (tag << 20 | chunk index); a pull is only honored when its tag
// matches the slot's committed tag. The caller commits tag -> fields ->
// next in that order, so a tag-matching pull is proof the whole job
// descriptor is visible; stale pulls (old tag, or index beyond the current
// job) are discarded. Workers are purely opportunistic: run() waits only
// for chunks_done == nchunks, never for worker acks, so a preempted or
// parked worker can never stall a call -- the caller just does the chunks
// itself. run() only returns once every claimed chunk is done, so no worker
// can still be inside run_range when its slot's fields are rewritten two
// jobs later.
//
// Worker idle sync is a spin-then-park hybrid sized for microsecond tasks:
// workers spin on the job sequence (decode gaps between jobs are
// single-digit us), then park on a condvar so idle stretches burn nothing.
// The pool exists only when --threads > 1; the serial path keeps zero
// atomics in its inner loop.
struct LinearPool {
  int nthreads = 1; // total participants including the calling thread
  vector<thread> workers;

  // Columns per work claim. 256 int32 accs = 1KB per claim; acc is
  // 128B-aligned so a claim never shares a cache line with its neighbor
  // (see bit_linear). Measured better than 64/128: fewer atomic pulls and
  // longer vectorized runs per claim, and each fused part still splits.
  static constexpr int CHUNK = 256;
  static constexpr int MAX_PARTS = 20; // fused projections + in-dim splits
  static constexpr int IDX_MASK = 0xfffff; // 20-bit claim index in `next`

  // A part is one output tensor; its columns live at [col0, col0+width) in
  // job space and acc[] is indexed in the same job space. A part may also
  // cover only rows [in0, in0+in_len) of the weight (input-dim split): then
  // out == nullptr and its acc range holds a partial sum, reduced by the
  // caller after the join -- still exact int32 math.
  struct Part {
    const int8_t *w;
    float *out;
    float w_scale;
    int width;
    int col0;
    int in0;
    int in_len;
  };

  struct alignas(128) Slot {
    // `tag` is written first and `next` committed last, so a pull carrying
    // the current tag proves the fields below are fully published.
    atomic<uint32_t> tag{0};
    atomic<int> chunks_done{0};
    Part parts[MAX_PARTS];
    int nparts = 0;
    const int8_t *x_q = nullptr;
    int32_t *acc = nullptr;
    int in_dim = 0, total_cols = 0;
    float quant_scale = 1.0f;
    int nchunks = 0;
    // Set for input-dim-split jobs: after the join the caller reduces the
    // nparts partial acc regions into out (exact integer add).
    float *reduce_out = nullptr;
    int reduce_width = 0;
    alignas(128) atomic<uint64_t> next{0};
  };
  Slot slots[2];

  // The wake signal gets its own line: workers spin-read it constantly.
  alignas(128) atomic<uint64_t> seq{0}; // jobs published
  atomic<int> parked{0};
  atomic<bool> stop{false};
  mutex mtx;
  condition_variable cv;

  // Idle spin before parking (~1ms): decode gaps between jobs are us-scale.
  static constexpr int SPIN_ITERS = 400000;
  // Bound a drain's wasted pulls: while a slot is mid-publish its counter
  // can briefly hand out stale-tag indices.
  static constexpr int MAX_PULLS = 1 << 20;

  // Column-count gate for taking a job parallel: below this the serial
  // kernel is only a few us and waking the pool costs more than it saves.
  // 512 measured as the crossover on Apple Silicon at any thread count.
  int min_cols = 512;
  bool in_split_ok = false;

  void start(int n) {
    nthreads = n;
    min_cols = 512;
    in_split_ok = (n >= 2);
    for (int i = 1; i < n; ++i)
      workers.emplace_back([this] { worker_loop(); });
  }

  ~LinearPool() {
    stop.store(true);
    {
      lock_guard<mutex> lk(mtx);
      cv.notify_all();
    }
    for (auto &t : workers)
      if (t.joinable())
        t.join();
  }

  // Kernel for one claimed job-column range: identical arithmetic to the
  // serial kernel below, restricted to [c0, c1) and routed per part.
  void run_range(const Slot &S, int c0, int c1) {
    // Snapshot every field to a local first: acc[] stores are int32_t and
    // may alias the int members of the job, so member loads in the inner
    // loop can't be hoisted -- locals can.
    int32_t *acc_g = S.acc;
    const int8_t *xq = S.x_q;
    int np = S.nparts;
    float qs = S.quant_scale;
    for (int p = 0; p < np; ++p) {
      const int8_t *pw = S.parts[p].w;
      float *pout = S.parts[p].out;
      float ws = S.parts[p].w_scale;
      int width = S.parts[p].width;
      int col0 = S.parts[p].col0;
      int j0 = max(c0, col0), j1 = min(c1, col0 + width);
      if (j0 >= j1)
        continue;
      int i0 = S.parts[p].in0, ilen = S.parts[p].in_len;
      int32_t *acc_p = acc_g + col0;
      int lj0 = j0 - col0, lj1 = j1 - col0;
      for (int lj = lj0; lj < lj1; ++lj)
        acc_p[lj] = 0;
      for (int i = i0; i < i0 + ilen; ++i) {
        int32_t xv = xq[i];
        if (xv == 0)
          continue;
        const int8_t *wrow = pw + (size_t)i * width;
        for (int lj = lj0; lj < lj1; ++lj)
          acc_p[lj] += wrow[lj] * xv;
      }
      if (pout == nullptr)
        continue; // partial acc region: caller reduces after the join
      for (int lj = lj0; lj < lj1; ++lj)
        pout[lj] = ((float)acc_p[lj] / qs) * ws;
    }
  }

  // Caller and workers both pull chunks until the slot's counter is
  // exhausted. Stale-tag pulls are discarded; they can only come from the
  // previous (already complete) job on this slot.
  void drain(Slot &S) {
    uint32_t stag;
    for (int pulls = 0; pulls < MAX_PULLS; ++pulls) {
      if (stop.load(memory_order_acquire))
        return;
      uint64_t v = S.next.fetch_add(1, memory_order_acquire);
      uint32_t vt = (uint32_t)(v >> 20);
      stag = S.tag.load(memory_order_acquire);
      if (vt != stag)
        continue;
      int idx = (int)(v & IDX_MASK);
      if (idx >= S.nchunks)
        break;
      run_range(S, idx * CHUNK, min(idx * CHUNK + CHUNK, S.total_cols));
      S.chunks_done.fetch_add(1, memory_order_release);
    }
  }

  void worker_loop() {
    uint64_t seen = 0;
    int spins = 0;
    for (;;) {
      uint64_t e = seq.load(memory_order_acquire);
      if (e != seen) {
        seen = e;
        drain(slots[e & 1]);
        spins = 0;
        continue;
      }
      if (stop.load(memory_order_acquire))
        return;
      if (++spins < SPIN_ITERS)
        continue;
      // Park. The predicate re-checks seq under the lock and the caller
      // notifies while holding the same lock whenever parked>0, so a wake
      // can never be missed.
      {
        unique_lock<mutex> lk(mtx);
        parked.fetch_add(1);
        cv.wait(lk, [&] {
          return seq.load(memory_order_acquire) != seen ||
                 stop.load(memory_order_acquire);
        });
        parked.fetch_sub(1);
      }
      spins = 0;
    }
  }

  void run(const Part *ps, int np, const int8_t *xq, int32_t *acc_, int in_d,
           int cols, float qs, float *reduce_out = nullptr,
           int reduce_width = 0) {
    uint64_t e = seq.load(memory_order_relaxed) + 1;
    Slot &S = slots[e & 1];
    // Commit order is the safety invariant: tag first, then the fields the
    // tag vouches for, then `next` (release) which arms the job.
    uint32_t tag = (uint32_t)e;
    S.tag.store(tag, memory_order_relaxed);
    for (int p = 0; p < np; ++p)
      S.parts[p] = ps[p];
    S.nparts = np;
    S.x_q = xq;
    S.acc = acc_;
    S.in_dim = in_d;
    S.total_cols = cols;
    S.quant_scale = qs;
    S.reduce_out = reduce_out;
    S.reduce_width = reduce_width;
    S.nchunks = (cols + CHUNK - 1) / CHUNK;
    S.chunks_done.store(0, memory_order_relaxed);
    S.next.store(((uint64_t)tag << 20), memory_order_release); // commit
    seq.store(e, memory_order_release);                        // wake hint
    if (parked.load() > 0) {
      lock_guard<mutex> lk(mtx);
      cv.notify_all();
    }
    drain(S); // caller works too
    int spins = 0;
    while (S.chunks_done.load(memory_order_acquire) != S.nchunks) {
      if (++spins > SPIN_ITERS)
        this_thread::yield();
    }
    // Input-dim-split jobs end in partial acc regions, one per part; the
    // integer reduce is exact, then the usual dequant per element.
    if (S.reduce_out != nullptr) {
      int width = S.reduce_width;
      for (int j = 0; j < width; ++j) {
        int32_t s = 0;
        for (int k = 0; k < np; ++k)
          s += S.acc[k * width + j];
        S.reduce_out[j] = ((float)s / qs) * S.parts[0].w_scale;
      }
    }
  }
};

// Set in main() when --threads > 1; bit_linear checks one pointer per call.
static LinearPool *g_pool = nullptr;
static LinearPool g_linear_pool;

// Column-count gate used when the pool is absent (never taken in that case;
// kept so the gate expression stays readable).
static constexpr int POOL_MIN_COLS = 512;

// BitLinear input quantization: in-procedure RMSNorm + absmax int8.
// x_q and quant_scale are pure functions of x -- callers that share an input
// (Q/K/V) may compute this once and reuse it bit-for-bit.
static int8_t *quantize_input(const vector<float> &x, int in_dim,
                              vector<int8_t> &scratch, float *quant_scale) {
  // 0. Internal RMSNorm (Non-affine)
  // Per BitNet paper/impl: BitLinear(x) = Layernorm(x) ...
  // guirguispierre 2026-08-10 - double for the same reason as rms_norm
  double sum_sq_d = 0.0;
  for (float v : x)
    sum_sq_d += (double)v * (double)v;
  float rms = sqrt((float)(sum_sq_d / in_dim) + 1e-5f);

  // 1. Quantize Input
  // We effectively quantize (x / rms).
  // Scale s = 127 / max(|x/rms|)
  float max_val = 0.0f;
  for (float v : x)
    max_val = max(max_val, abs(v / rms)); // Optimize: abs(v)/rms

  if (max_val < 1e-5f)
    max_val = 1e-5f;

  *quant_scale = 127.0f / max_val;

  // guirguispierre 2026-08-10 - divide before scaling to match
  // activation_quant(); folding it into one multiply rounds differently
  // guirguispierre 2026-08-10 - caller's buffer avoids a malloc per call in the
  // hot loop on microcontrollers
  if ((int)scratch.size() < in_dim)
    scratch.resize(in_dim);
  int8_t *x_q = scratch.data();
  for (int i = 0; i < in_dim; ++i) {
    float val = round((x[i] / rms) * *quant_scale);
    if (val > 127)
      val = 127;
    if (val < -127)
      val = -127;
    x_q[i] = (int8_t)val;
  }
  return x_q;
}

// Serial ternary kernel + dequantize for one output tensor: the exact same
// arithmetic the pool's per-claim path applies to a column range.
static void bit_kernel_serial(const int8_t *x_q, const TensorI8 &w,
                              int32_t *acc, float *out, int in_dim, int out_dim,
                              float quant_scale) {
  memset(acc, 0, out_dim * sizeof(int32_t));

  // guirguispierre 2026-08-10 - i-outer streams each weight row contiguously
  // instead of striding by out_dim, worth ~14x; integer sums are associative so
  // the totals are unchanged
  for (int i = 0; i < in_dim; ++i) {
    int32_t xv = x_q[i];
    if (xv == 0)
      continue; // free sparsity: skip the whole row
    const int8_t *wrow = &w.data[(size_t)i * out_dim];
    // guirguispierre 2026-08-10 - w is {-1,0,1} so the multiply is the add/sub
    // and vectorizes; the branchy form stalls on mispredicts
    for (int j = 0; j < out_dim; ++j)
      acc[j] += wrow[j] * xv;
  }

  // out = (acc / x_scale) * w_scale
  for (int j = 0; j < out_dim; ++j)
    out[j] = ((float)acc[j] / quant_scale) * w.scale;
}

// 128B-aligned int32 scratch: with --threads, neighboring column claims never
// share a cache line, so workers can't ping-pong acc lines per row.
static int32_t *aligned_acc(vector<int32_t> &acc32, int cols) {
  if ((int)acc32.size() < cols + 32)
    acc32.resize(cols + 32);
  return (int32_t *)(((uintptr_t)acc32.data() + 127) & ~(uintptr_t)127);
}

// Global Matmul for BitLinear
// A: Float input [Dim]
// Weight: Int8 [In, Out]
// Out: Float output [Out]
// Scratch: [In] (Pre-allocated buffer to avoid malloc)
void bit_linear(const vector<float> &x, const TensorI8 &w, vector<float> &out,
                vector<int8_t> &scratch, vector<int32_t> &acc32) {
  int in_dim = w.rows;
  int out_dim = w.cols;

  float quant_scale;
  int8_t *x_q = quantize_input(x, in_dim, scratch, &quant_scale);

  if ((int)acc32.size() < out_dim + 32)
    acc32.resize(out_dim + 32);
  int32_t *acc = aligned_acc(acc32, out_dim);

  // Parallel path: split [0, out_dim) into CHUNK-column pieces claimed off an
  // atomic counter. Bit-identical to the serial kernel -- every acc[j] is the
  // same independent int32 sum over all i (see LinearPool).
  int min_cols = g_pool != nullptr ? g_pool->min_cols : POOL_MIN_COLS;
  if (g_pool != nullptr && out_dim >= min_cols) {
    LinearPool::Part p{w.data.data(), out.data(), w.scale, out_dim, 0, 0,
                       in_dim};
    g_pool->run(&p, 1, x_q, acc, in_dim, out_dim, quant_scale);
    return;
  }

  // Narrow output but fat input (e.g. fc2): split the INPUT dim instead.
  // Each part accumulates a partial int32 sum over its i-range into its own
  // acc region; the caller reduces after the join. Integer partial sums are
  // exact, so still bit-identical to the serial kernel.
  if (g_pool != nullptr && g_pool->in_split_ok && in_dim >= 512 &&
      (int64_t)in_dim * out_dim >= 131072 && out_dim >= 32) {
    int K = g_pool->nthreads;
    if (K > LinearPool::MAX_PARTS)
      K = LinearPool::MAX_PARTS;
    int inlen = in_dim / K;
    acc = aligned_acc(acc32, K * out_dim);
    LinearPool::Part ps[LinearPool::MAX_PARTS];
    for (int k = 0; k < K; ++k)
      ps[k] = {w.data.data(), nullptr, w.scale, out_dim, k * out_dim,
               k * inlen,
               k == K - 1 ? in_dim - k * inlen : inlen};
    g_pool->run(ps, K, x_q, acc, in_dim, K * out_dim, quant_scale, out.data(),
                out_dim);
    return;
  }

  bit_kernel_serial(x_q, w, acc, out.data(), in_dim, out_dim, quant_scale);
}

// Fused triple BitLinear for projections sharing one input (Q/K/V): one
// quantization, one parallel job of 3*out_dim columns instead of three small
// ones. Each part writes its own out; identical math to three bit_linear
// calls because x_q is a pure function of x.
void bit_linear3(const vector<float> &x, const TensorI8 &w1, const TensorI8 &w2,
                 const TensorI8 &w3, vector<float> &o1, vector<float> &o2,
                 vector<float> &o3, vector<int8_t> &scratch,
                 vector<int32_t> &acc32) {
  int in_dim = w1.rows;
  int cols = w1.cols + w2.cols + w3.cols;

  float quant_scale;
  int8_t *x_q = quantize_input(x, in_dim, scratch, &quant_scale);
  int32_t *acc = aligned_acc(acc32, cols);

  if (g_pool != nullptr && cols >= g_pool->min_cols) {
    LinearPool::Part ps[3] = {
        {w1.data.data(), o1.data(), w1.scale, w1.cols, 0, 0, in_dim},
        {w2.data.data(), o2.data(), w2.scale, w2.cols, w1.cols, 0, in_dim},
        {w3.data.data(), o3.data(), w3.scale, w3.cols, w1.cols + w2.cols, 0,
         in_dim}};
    g_pool->run(ps, 3, x_q, acc, in_dim, cols, quant_scale);
    return;
  }

  int col0 = 0;
  for (const TensorI8 *pw : {&w1, &w2, &w3}) {
    vector<float> *po = pw == &w1 ? &o1 : pw == &w2 ? &o2 : &o3;
    bit_kernel_serial(x_q, *pw, acc + col0, po->data(), in_dim, pw->cols,
                      quant_scale);
    col0 += pw->cols;
  }
}

// guirguispierre 2026-08-10 - full float32 precision so tools/parity_check.py
// can diff values exactly rather than by eye
void print_tensor(const char *name, const vector<float> &v, int count = 10) {
  int n = min((int)v.size(), count);
  cout << "[" << name << "] n=" << v.size() << " ";
  cout << std::setprecision(9);
  for (int i = 0; i < n; ++i)
    cout << v[i] << (i + 1 < n ? " " : "");
  cout << endl;
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
  if (version != 2 && version != 3) {
    cerr << "Unsupported version: " << version
         << " (expected 2 or 3; re-export with atomic_1bit/utils/export_to_cpp.py)"
         << endl;
    return false;
  }

  cerr << "Sizeof(Config): " << sizeof(Config) << " (Expected 24)" << endl;
  f.read((char *)&model.config, sizeof(Config));
  Config &c = model.config;

  cout << "Loaded Config: Dim=" << c.dim << " Layers=" << c.depth
       << " Heads=" << c.heads << " HasGist=" << c.has_gist << endl;
  cerr << "  File Pos after Config: " << f.tellg() << endl;

  // Read Gist if present
  if (c.has_gist) {
    model.gist_vector.resize(c.dim);
    f.read((char *)model.gist_vector.data(), c.dim * sizeof(float));
  }
  cerr << "  File Pos after Gist: " << f.tellg() << endl;

  // Helper
  auto read_f32 = [&](TensorF32 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    t.data.resize(r * c_dim);
    size_t bytes = t.data.size() * sizeof(float);
    f.read((char *)t.data.data(), bytes);
    cerr << "  Read F32 (" << r << "x" << c_dim << ") Bytes: " << bytes
         << " Pos: " << f.tellg() << endl;
  };

  // guirguispierre 2026-08-10 - weights arrive four per byte as w+1; unpack at
  // load so the hot loop still sees a plain int8 array
  auto read_i8 = [&](TensorI8 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    f.read((char *)&t.scale, sizeof(float));

    size_t n = (size_t)r * c_dim;
    size_t bytes = (n + 3) / 4;
    vector<uint8_t> packed(bytes);
    f.read((char *)packed.data(), bytes);

    t.data.resize(n);
    for (size_t i = 0; i < n; ++i)
      t.data[i] = (int8_t)((packed[i >> 2] >> (2 * (i & 3))) & 3) - 1;

    cerr << "    Read I8 (" << r << "x" << c_dim << ") Bytes: " << bytes
         << " Pos: " << f.tellg() << endl;
  };

  // guirguispierre 2026-11-08 - v3 stores each embedding row as int8 codes
  // plus one float32 absmax scale per row (all scales first, then all codes);
  // dequantize at load so the forward pass still sees a plain fp32 table
  auto read_emb = [&](TensorF32 &t, int r, int c_dim) {
    t.rows = r;
    t.cols = c_dim;
    t.data.resize((size_t)r * c_dim);
    if (version >= 3) {
      vector<float> scales(r);
      f.read((char *)scales.data(), r * sizeof(float));
      vector<int8_t> q(t.data.size());
      f.read((char *)q.data(), q.size());
      for (int i = 0; i < r; ++i) {
        float s = scales[i];
        for (int j = 0; j < c_dim; ++j)
          t.data[(size_t)i * c_dim + j] = (float)q[(size_t)i * c_dim + j] * s;
      }
    } else {
      f.read((char *)t.data.data(), t.data.size() * sizeof(float));
    }
    cerr << "  Read EMB (" << r << "x" << c_dim << ") v" << version
         << " Pos: " << f.tellg() << endl;
  };

  // Embeddings
  read_emb(model.token_emb, c.vocab_size, c.dim);
  read_emb(model.pos_emb, c.max_seq_len, c.dim);

  // Layers
  model.layers.resize(c.depth);
  for (int i = 0; i < c.depth; ++i) {
    cerr << "Loading Layer " << i << "..." << endl;
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
  vector<int32_t> acc32;

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

// --- KV Cache ---
// Per-layer key/value rows produced by earlier tokens, laid out
// [layer][pos][dim] so head slices stay contiguous. Positions are absolute:
// the model uses learned position embeddings added at input time, so a cached
// row is only valid while its position index is unchanged. The generation
// loop drops the whole cache (len = -1) whenever the context window slides
// and rebases positions, then refills it with a full forward pass.
struct KVCache {
  vector<float> K, V;
  int max_seq = 0;
  int dim = 0;
  int len = -1; // valid positions (includes the gist slot); -1 = empty/stale

  void init(int depth, int max_seq_len, int dim_) {
    max_seq = max_seq_len;
    dim = dim_;
    K.assign((size_t)depth * max_seq * dim, 0.0f);
    V.assign((size_t)depth * max_seq * dim, 0.0f);
    len = -1;
  }
  float *k_row(int layer, int pos) {
    return &K[((size_t)layer * max_seq + pos) * dim];
  }
  float *v_row(int layer, int pos) {
    return &V[((size_t)layer * max_seq + pos) * dim];
  }
};

// --- Forward Pass ---
void forward(AtomicModel &model, const vector<int> &tokens,
             vector<float> &logits, Workspace &ws, KVCache &cache,
             bool parity_mode = false) {
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
  if (parity_mode) {
    vector<float> tmp(&ws.X[(seq_len - 1) * dim], &ws.X[seq_len * dim]);
    print_tensor("Embedding_Sum", tmp, dim);
  }

  int l_idx = 0;
  for (auto &layer : model.layers) {
    // Norm
    for (int t = 0; t < seq_len; ++t) {
      memcpy(ws.xt.data(), &ws.X[t * dim], dim * sizeof(float));
      rms_norm(ws.xt, layer.ln1.data, ws.xn, dim);
      memcpy(&ws.X_norm[t * dim], ws.xn.data(), dim * sizeof(float));
    }

    if (parity_mode) {
      vector<float> tmp(&ws.X_norm[(seq_len - 1) * dim],
                        &ws.X_norm[seq_len * dim]);
      print_tensor(("Layer" + to_string(l_idx) + "_LN1").c_str(), tmp, dim);
    }

    // Attention
    for (int t = 0; t < seq_len; ++t) {
      memcpy(ws.xt.data(), &ws.X_norm[t * dim], dim * sizeof(float));

      bit_linear3(ws.xt, layer.q_w, layer.k_w, layer.v_w, ws.q, ws.k, ws.v,
                  ws.scratch_q, ws.acc32);

      memcpy(&ws.Q[t * dim], ws.q.data(), dim * sizeof(float));
      memcpy(&ws.K[t * dim], ws.k.data(), dim * sizeof(float));
      memcpy(&ws.V[t * dim], ws.v.data(), dim * sizeof(float));
      memcpy(cache.k_row(l_idx, t), ws.k.data(), dim * sizeof(float));
      memcpy(cache.v_row(l_idx, t), ws.v.data(), dim * sizeof(float));
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
      bit_linear(ws.xt, layer.o_w, ws.layer_out, ws.scratch_q, ws.acc32);

      // Residual
      for (int i = 0; i < dim; ++i)
        ws.X[t * dim + i] += ws.layer_out[i];
    }

    // guirguispierre 2026-08-10 - layer_out is pre-residual here, matching what
    // tools/parity_check.py prints on the python side
    if (parity_mode)
      print_tensor(("Layer" + to_string(l_idx) + "_Attn_Out").c_str(),
                   ws.layer_out, dim);

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
      bit_linear(ws.xn, layer.fc1_w, ws.fc_act, ws.scratch_q, ws.acc32);
      for (int i = 0; i < hidden; ++i)
        ws.fc_act[i] = gelu(ws.fc_act[i]);

      bit_linear(ws.fc_act, layer.fc2_w, ws.mlp_out, ws.scratch_q, ws.acc32);

      for (int i = 0; i < dim; ++i)
        ws.X[t * dim + i] += ws.mlp_out[i];
    }

    if (parity_mode)
      print_tensor(("Layer" + to_string(l_idx) + "_MLP_Out").c_str(),
                   ws.mlp_out, dim);

    l_idx++;
  }

  // Final Output
  int last_t = seq_len - 1;
  memcpy(ws.xt.data(), &ws.X[last_t * dim], dim * sizeof(float));
  rms_norm(ws.xt, model.ln_f.data, ws.xn, dim);

  if (parity_mode)
    print_tensor("Final_LN", ws.xn, dim);

  bit_linear(ws.xn, model.head_w, logits, ws.scratch_q, ws.acc32);

  if (parity_mode)
    print_tensor("Final_Logits", logits, (int)logits.size());

  cache.len = seq_len;
}

// Single-token decode step against the KV cache.
//
// Appends one K/V row per layer at `pos` and attends the new query over
// cache rows 0..pos. Every op is the same call in the same order as the
// last-token computation inside forward(), so the logits are identical to
// recomputing the full prefix -- the cache only avoids the wasted work.
void forward_one(AtomicModel &model, int token_id, int pos,
                 vector<float> &logits, Workspace &ws, KVCache &cache) {
  int dim = model.config.dim;
  int heads = model.config.heads;
  int head_dim = dim / heads;

  ws.ensure_size(model.config.max_seq_len, dim);

  if (pos >= model.config.max_seq_len)
    pos = model.config.max_seq_len - 1;
  if (token_id >= model.config.vocab_size)
    token_id = 0;

  // Residual stream for this token lives in row `pos` of ws.X, matching the
  // indexing forward() uses for the same position.
  float *x = &ws.X[(size_t)pos * dim];
  float *tok_ptr = &model.token_emb.data[(size_t)token_id * dim];
  float *pos_ptr = &model.pos_emb.data[(size_t)pos * dim];
  for (int d = 0; d < dim; ++d)
    x[d] = tok_ptr[d] + pos_ptr[d];

  int l_idx = 0;
  for (auto &layer : model.layers) {
    memcpy(ws.xt.data(), x, dim * sizeof(float));
    rms_norm(ws.xt, layer.ln1.data, ws.xn, dim);

    bit_linear3(ws.xn, layer.q_w, layer.k_w, layer.v_w, ws.q, ws.k, ws.v,
                ws.scratch_q, ws.acc32);

    memcpy(cache.k_row(l_idx, pos), ws.k.data(), dim * sizeof(float));
    memcpy(cache.v_row(l_idx, pos), ws.v.data(), dim * sizeof(float));

    float *attn = &ws.AttnOut[(size_t)pos * dim];
    fill(attn, attn + dim, 0.0f);

    for (int h = 0; h < heads; ++h) {
      float *q_ptr = &ws.q[h * head_dim];
      float *out_ptr = &attn[h * head_dim];
      vector<float> scores;
      scores.reserve(pos + 1);

      for (int src = 0; src <= pos; ++src) {
        float *k_ptr = cache.k_row(l_idx, src) + h * head_dim;
        float dp = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dp += q_ptr[d] * k_ptr[d];
        scores.push_back(dp / sqrt((float)head_dim));
      }
      softmax(scores);

      for (int src = 0; src <= pos; ++src) {
        float w = scores[src];
        float *v_ptr = cache.v_row(l_idx, src) + h * head_dim;
        for (int d = 0; d < head_dim; ++d)
          out_ptr[d] += w * v_ptr[d];
      }
    }

    memcpy(ws.xt.data(), attn, dim * sizeof(float));
    bit_linear(ws.xt, layer.o_w, ws.layer_out, ws.scratch_q, ws.acc32);
    for (int i = 0; i < dim; ++i)
      x[i] += ws.layer_out[i];

    memcpy(ws.xt.data(), x, dim * sizeof(float));
    rms_norm(ws.xt, layer.ln2.data, ws.xn, dim);

    int hidden = 4 * dim;
    bit_linear(ws.xn, layer.fc1_w, ws.fc_act, ws.scratch_q, ws.acc32);
    for (int i = 0; i < hidden; ++i)
      ws.fc_act[i] = gelu(ws.fc_act[i]);
    bit_linear(ws.fc_act, layer.fc2_w, ws.mlp_out, ws.scratch_q, ws.acc32);
    for (int i = 0; i < dim; ++i)
      x[i] += ws.mlp_out[i];

    l_idx++;
  }

  memcpy(ws.xt.data(), x, dim * sizeof(float));
  rms_norm(ws.xt, model.ln_f.data, ws.xn, dim);
  bit_linear(ws.xn, model.head_w, logits, ws.scratch_q, ws.acc32);
}

// Sampling: temperature + optional top-k + optional top-p (nucleus).
//
// Order of operations matches the conventional implementation:
//   1) temperature scales logits
//   2) softmax to probabilities
//   3) top-k truncation (if top_k > 0)
//   4) top-p truncation (if top_p in (0, 1))
//   5) renormalize and sample
//
// temp == 0 => argmax (greedy), ignoring top_k / top_p.
int sample_logits(const vector<float> &logits, float temp, int top_k,
                  float top_p, std::mt19937 &rng) {
  if (temp < 1e-5f) {
    int best_token = 0;
    float best_val = -1e9f;
    for (size_t i = 0; i < logits.size(); ++i) {
      if (logits[i] > best_val) {
        best_val = logits[i];
        best_token = (int)i;
      }
    }
    return best_token;
  }

  int V = (int)logits.size();
  vector<float> probs(V);
  float max_val = -1e9f;
  for (float v : logits)
    max_val = max(max_val, v);

  float sum_exp = 0.0f;
  for (int i = 0; i < V; ++i) {
    probs[i] = exp((logits[i] - max_val) / temp);
    sum_exp += probs[i];
  }
  if (sum_exp <= 0.0f)
    return 0;
  for (int i = 0; i < V; ++i)
    probs[i] /= sum_exp;

  // Build (prob, idx) for ranking only if we need to truncate.
  bool use_topk = top_k > 0 && top_k < V;
  bool use_topp = top_p > 0.0f && top_p < 1.0f;

  if (!use_topk && !use_topp) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < V; ++i) {
      acc += probs[i];
      if (r <= acc)
        return i;
    }
    return V - 1;
  }

  vector<pair<float, int>> ranked(V);
  for (int i = 0; i < V; ++i)
    ranked[i] = {probs[i], i};

  // Sort descending by probability.
  std::sort(ranked.begin(), ranked.end(),
            [](const pair<float, int> &a, const pair<float, int> &b) {
              return a.first > b.first;
            });

  int kept = use_topk ? top_k : V;

  if (use_topp) {
    float cum = 0.0f;
    int p_kept = 0;
    for (; p_kept < kept; ++p_kept) {
      cum += ranked[p_kept].first;
      if (cum >= top_p) {
        p_kept += 1;
        break;
      }
    }
    if (p_kept < 1)
      p_kept = 1; // never empty
    kept = p_kept;
  }

  float renorm = 0.0f;
  for (int i = 0; i < kept; ++i)
    renorm += ranked[i].first;
  if (renorm <= 0.0f)
    return ranked[0].second;

  std::uniform_real_distribution<float> dist(0.0f, renorm);
  float r = dist(rng);
  float acc = 0.0f;
  for (int i = 0; i < kept; ++i) {
    acc += ranked[i].first;
    if (r <= acc)
      return ranked[i].second;
  }
  return ranked[kept - 1].second;
}

// Debug Utils (Moved to top)

// Parse "1,2,3,4" -> [1, 2, 3, 4]. Empty string => empty vector.
// Optional detokenizer: a .vocab file holds one line per token id, each line
// the lowercase hex of that token's raw bytes (empty line = empty token).
// Produced by tools/make_vocab.py. Printing raw bytes keeps multi-byte utf-8
// sequences intact across token boundaries.
static bool load_vocab(const char *path, vector<string> &vocab) {
  ifstream f(path);
  if (!f)
    return false;
  string line;
  while (getline(f, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.size() % 2 != 0)
      return false;
    string bytes;
    bytes.reserve(line.size() / 2);
    for (size_t i = 0; i + 1 < line.size(); i += 2) {
      auto nyb = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      int hi = nyb(line[i]), lo = nyb(line[i + 1]);
      if (hi < 0 || lo < 0)
        return false;
      bytes.push_back((char)((hi << 4) | lo));
    }
    vocab.push_back(bytes);
  }
  return !vocab.empty();
}

static void print_token(int t, const vector<string> &vocab) {
  if (t >= 0 && t < (int)vocab.size()) {
    cout << vocab[t];
  } else {
    cout << t << " ";
  }
}

static vector<int> parse_token_list(const string &s) {
  vector<int> out;
  if (s.empty())
    return out;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t comma = s.find(',', pos);
    string tok =
        s.substr(pos, comma == string::npos ? string::npos : comma - pos);
    if (!tok.empty()) {
      try {
        out.push_back(std::stoi(tok));
      } catch (...) {
        cerr << "Warning: skipping non-integer token '" << tok << "'" << endl;
      }
    }
    if (comma == string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

static void print_help() {
  cout << "Atomic-1Bit Runner\n"
       << "  --model PATH         Model .bin path (default atomic_model.bin)\n"
       << "  --steps N            Number of tokens to generate (default 50)\n"
       << "  --temp F             Sampling temperature, 0 = greedy (default 0)\n"
       << "  --top_k N            Keep top N logits before sampling\n"
       << "  --top_p F            Nucleus sampling, keep smallest set with cum prob >= F\n"
       << "  --seed N             RNG seed (default 42)\n"
       << "  --start_token N      Single starting token id (default 42)\n"
       << "  --prompt 1,2,3       Comma-separated starting token ids (overrides --start_token)\n"
       << "  --vocab PATH         .vocab file (hex-per-line, tools/make_vocab.py) to print text instead of ids\n"
       << "  --threads N          Worker threads for the ternary matmuls\n"
       << "                       (default: min(8, hardware_concurrency); 1 = exact serial path).\n"
       << "                       Splits each projection's column dim across a persistent\n"
       << "                       pool; output is bit-identical for any N.\n"
       << "  --stream             Print tokens as they are generated (default on)\n"
       << "  --no-stream          Buffer all tokens and print at end\n"
       << "  --parity             Run a single-step parity check\n"
       << "  --help               Show this message\n";
}

int main(int argc, char **argv) {
  string model_path = "atomic_model.bin";
  string prompt_str;
  int gen_len = 50;
  float temp = 0.0f;
  int top_k = 0;
  float top_p = 0.0f;
  int seed = 42;
  int start_token = 42;
  int threads = 0; // 0 = default, resolved after parsing
  bool parity_mode = false;
  bool stream = true;
  string vocab_path;

  auto need_value = [&](int i, const string &arg) -> bool {
    if (i + 1 < argc)
      return true;
    cerr << "Error: " << arg << " requires a value (use --help)" << endl;
    return false;
  };

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_help();
      return 0;
    } else if (arg == "--model") {
      if (!need_value(i, arg)) return 2;
      model_path = argv[++i];
    } else if (arg == "--steps") {
      if (!need_value(i, arg)) return 2;
      gen_len = atoi(argv[++i]);
    } else if (arg == "--temp") {
      if (!need_value(i, arg)) return 2;
      temp = (float)atof(argv[++i]);
    } else if (arg == "--top_k") {
      if (!need_value(i, arg)) return 2;
      top_k = atoi(argv[++i]);
    } else if (arg == "--top_p") {
      if (!need_value(i, arg)) return 2;
      top_p = (float)atof(argv[++i]);
    } else if (arg == "--seed") {
      if (!need_value(i, arg)) return 2;
      seed = atoi(argv[++i]);
    } else if (arg == "--start_token") {
      if (!need_value(i, arg)) return 2;
      start_token = atoi(argv[++i]);
    } else if (arg == "--prompt") {
      if (!need_value(i, arg)) return 2;
      prompt_str = argv[++i];
    } else if (arg == "--vocab") {
      if (!need_value(i, arg)) return 2;
      vocab_path = argv[++i];
    } else if (arg == "--threads") {
      if (!need_value(i, arg)) return 2;
      threads = atoi(argv[++i]);
    } else if (arg == "--stream")
      stream = true;
    else if (arg == "--no-stream")
      stream = false;
    else if (arg == "--parity")
      parity_mode = true;
    else {
      cerr << "Unknown arg: " << arg << " (use --help)" << endl;
      return 2;
    }
  }

  {
    int hw = (int)thread::hardware_concurrency();
    if (hw < 1)
      hw = 1;
    if (threads == 0) {
      // Default: measured best around 6-8 on an M5 Pro (6P+12E) and ~neutral
      // on tiny models; output is bit-identical either way. --threads 1
      // forces the exact serial path.
      threads = min(8, hw);
    } else if (threads > hw) {
      cerr << "Warning: --threads " << threads
           << " exceeds hardware_concurrency " << hw << ", clamping" << endl;
      threads = hw;
    } else if (threads < 1) {
      cerr << "Warning: --threads " << threads << " invalid, using 1" << endl;
      threads = 1;
    }
    if (threads > 1) {
      g_linear_pool.start(threads);
      g_pool = &g_linear_pool;
    }
  }

  if (parity_mode) {
    cout << "--- C++ Parity Check ---" << endl;
    gen_len = 1;
  }

  cout << "Atomic-1Bit Engine | Model: " << model_path
       << " | Steps: " << gen_len << " | Temp: " << temp
       << " | Top-k: " << top_k << " | Top-p: " << top_p
       << " | Seed: " << seed << endl;

  std::mt19937 rng(seed);

  AtomicModel model;
  if (!load_model(model_path.c_str(), model)) {
    cerr << "Error: Could not load model file." << endl;
    return 1;
  }

  vector<int> context = parse_token_list(prompt_str);
  if (context.empty())
    context.push_back(start_token);

  vector<string> vocab;
  if (!vocab_path.empty() && !load_vocab(vocab_path.c_str(), vocab)) {
    cerr << "Error: could not load vocab file " << vocab_path << endl;
    return 1;
  }

  for (int t : context) {
    if (t < 0 || t >= model.config.vocab_size) {
      cerr << "Error: prompt token " << t << " out of vocab range [0, "
           << model.config.vocab_size << ")" << endl;
      return 1;
    }
  }

  if (stream) {
    if (vocab.empty())
      cout << "Generating: ";
    for (int t : context)
      print_token(t, vocab);
    if (vocab.empty())
      cout << " ";
    else
      cout << "\n";
    cout.flush();
  }
  if (parity_mode)
    cout << endl; // keep the first probe off the prompt-echo line

  Workspace ws;
  ws.ensure_size(model.config.max_seq_len, model.config.dim);

  KVCache cache;
  cache.init(model.config.depth, model.config.max_seq_len, model.config.dim);
  int t_offset = model.config.has_gist ? 1 : 0;

  vector<int> generated;
  auto start_time = std::chrono::high_resolution_clock::now();

  // A gist occupies one context row, so only this many token slots remain.
  int ctx_cap = model.config.max_seq_len - t_offset;

  for (int step = 0; step < gen_len; ++step) {
    if ((int)context.size() > ctx_cap) {
      // Keep ~3/4 of the window: the slide rebases every learned position so
      // the cache must be refilled anyway, and erasing a chunk leaves headroom
      // for several cached decode steps before the next rebuild.
      int keep = ctx_cap - ctx_cap / 4;
      if (keep < 1)
        keep = 1;
      int erase_count = (int)context.size() - keep;
      if (erase_count < 1)
        erase_count = 1;
      context.erase(context.begin(), context.begin() + erase_count);
      cache.len = -1;
    }

    vector<float> logits(model.config.vocab_size);
    if ((int)context.size() + t_offset <= model.config.max_seq_len) {
      if (cache.len == (int)context.size() - 1 + t_offset) {
        // Cache holds every position but the new last token: decode just it.
        forward_one(model, context.back(), cache.len, logits, ws, cache);
        cache.len++;
      } else {
        forward(model, context, logits, ws, cache, parity_mode);
      }
    } else {
      // Defensive: unreachable after the ctx_cap slide above, but keep the
      // guard so a future edit can't turn an overflow into silent zero logits.
      cache.len = -1;
    }
    int next_token = sample_logits(logits, temp, top_k, top_p, rng);
    generated.push_back(next_token);
    context.push_back(next_token);
    if (stream) {
      if (vocab.empty())
        cout << next_token << " " << flush;
      else
        print_token(next_token, vocab);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  double tps = gen_len / diff.count();

  if (!stream) {
    if (vocab.empty())
      cout << "Generated: ";
    for (int t : generated)
      print_token(t, vocab);
  }

  cout << endl << "Done." << endl;
  cout << "TPS: " << tps << endl;

  return 0;
}
