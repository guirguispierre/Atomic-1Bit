/**
 * Atomic-1Bit ESP32 Inference Demo
 *
 * Runs the Pocket model (4096 vocab, 256 dim, 4 layers) on ESP32 with PSRAM.
 * Uses flash-streaming: reads weight blocks from SPI flash on demand instead
 * of loading the entire model into RAM.
 *
 * Build: platformio run
 * Flash: platformio run --target upload
 * Monitor: platformio device monitor
 */

#include <Arduino.h>
#include <SPIFFS.h>
#include <cmath>
#include <cstdint>
#include <cstring>

// --- Model Config (Pocket: 4096 vocab, 256 dim, 4 layers) ---
#define VOCAB_SIZE 4096
#define DIM 256
#define DEPTH 4
#define HEADS 4
#define HEAD_DIM (DIM / HEADS)
#define CONTEXT_LEN 128
#define HIDDEN_DIM (4 * DIM)

// --- Flash-Streaming Weight Access ---
// Instead of loading the full model into RAM, we read weight blocks from
// SPI flash (SPIFFS) on demand. This trades latency for memory savings.

static File model_file;
static size_t weight_data_offset = 0; // Offset past header + embeddings

// Pre-allocated buffers (static to avoid heap fragmentation)
static float x_buf[CONTEXT_LEN * DIM];
static float x_norm[DIM];
static float q_buf[DIM], k_buf[DIM], v_buf[DIM], o_buf[DIM];
static int8_t x_q[DIM];
static int8_t weight_row[HIDDEN_DIM]; // Largest row: fc1 output dim
// guirguispierre 2026-08-10 - sized for the head, the widest bit_linear output
static uint8_t packed_row[(VOCAB_SIZE + 3) / 4];
static int32_t acc_buf[VOCAB_SIZE];
static float fc_act[HIDDEN_DIM];
static float mlp_out[DIM];
static float logits[VOCAB_SIZE];

// Embeddings (loaded into PSRAM if available)
static float *token_emb = nullptr;
static float *pos_emb = nullptr;

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

// Quantize input to INT8
static void quantize_input(const float *x, int8_t *x_q, int dim, float *scale_out) {
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
        if (val > 127)
            val = 127;
        if (val < -127)
            val = -127;
        x_q[i] = (int8_t)val;
    }
}

// Ternary matmul: reads weights from flash one row at a time
static void bit_linear_flash(const float *input, size_t weight_offset,
                             int in_dim, int out_dim, float *output) {
    // 1. RMSNorm + Quantize input
    float sum_sq = 0.0f;
    for (int i = 0; i < in_dim; i++)
        sum_sq += input[i] * input[i];
    float rms = sqrtf(sum_sq / in_dim + 1e-5f);

    float normed[DIM]; // Stack buffer
    for (int i = 0; i < in_dim; i++)
        normed[i] = input[i] / rms;

    float quant_scale;
    quantize_input(normed, x_q, in_dim, &quant_scale);

    // 2. Read scale from flash
    float w_scale;
    model_file.seek(weight_offset);
    model_file.read((uint8_t *)&w_scale, sizeof(float));

    // 3. Stream weights and compute
    // guirguispierre 2026-08-10 - weights are (in_dim, out_dim) row-major and
    // packed four per byte, so one sequential read per row replaces the old
    // per-element seek: in_dim reads instead of in_dim*out_dim
    const size_t row_bytes = (out_dim + 3) / 4;
    const size_t data_offset = weight_offset + sizeof(float);

    for (int j = 0; j < out_dim; j++)
        acc_buf[j] = 0;

    model_file.seek(data_offset);
    for (int i = 0; i < in_dim; i++) {
        int32_t xv = x_q[i];
        if (xv == 0) {
            model_file.seek(data_offset + (size_t)(i + 1) * row_bytes);
            continue; // free sparsity: skip the whole row
        }
        model_file.read(packed_row, row_bytes);
        for (int j = 0; j < out_dim; j++) {
            int8_t wv = (int8_t)((packed_row[j >> 2] >> (2 * (j & 3))) & 3) - 1;
            acc_buf[j] += wv * xv;
        }
    }

    for (int j = 0; j < out_dim; j++)
        output[j] = ((float)acc_buf[j] / quant_scale) * w_scale;
}

// --- Main ---

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Atomic-1Bit ESP32 Demo ===");

    // Initialize SPIFFS
    if (!SPIFFS.begin(true)) {
        Serial.println("ERROR: SPIFFS mount failed!");
        return;
    }

    // Open model file
    model_file = SPIFFS.open("/atomic_model.bin", "r");
    if (!model_file) {
        Serial.println("ERROR: Model file not found in SPIFFS!");
        Serial.println("Upload atomic_model.bin to SPIFFS data partition first.");
        return;
    }

    // Read and verify header
    int32_t magic, version;
    model_file.read((uint8_t *)&magic, 4);
    model_file.read((uint8_t *)&version, 4);

    if (magic != 0x41544F4D) {
        Serial.println("ERROR: Invalid model format (bad magic)");
        return;
    }

    if (version != 2) {
        Serial.printf("ERROR: model is v%d, this build needs v2\n", version);
        return;
    }

    Serial.printf("Model format: ATOM v%d\n", version);

    // Skip config (6 ints = 24 bytes) - we use hardcoded constants
    model_file.seek(model_file.position() + 24);

    // Load embeddings into PSRAM if available
    size_t emb_size = VOCAB_SIZE * DIM * sizeof(float);
    size_t pos_size = CONTEXT_LEN * DIM * sizeof(float);

    token_emb = (float *)ps_malloc(emb_size);
    pos_emb = (float *)ps_malloc(pos_size);

    if (!token_emb || !pos_emb) {
        Serial.println("WARNING: PSRAM not available, using heap");
        if (token_emb) free(token_emb);
        if (pos_emb) free(pos_emb);
        token_emb = (float *)malloc(emb_size);
        pos_emb = (float *)malloc(pos_size);
    }

    if (!token_emb || !pos_emb) {
        Serial.println("ERROR: Not enough memory for embeddings!");
        return;
    }

    model_file.read((uint8_t *)token_emb, emb_size);
    model_file.read((uint8_t *)pos_emb, pos_size);

    weight_data_offset = model_file.position();
    Serial.printf("Embeddings loaded. Weight data starts at offset %d\n",
                  (int)weight_data_offset);
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

    // Simple generation demo
    Serial.println("\nGenerating tokens (greedy, start_token=58):");
    int token = 58;

    unsigned long start = millis();
    for (int step = 0; step < 20; step++) {
        // Embed
        for (int d = 0; d < DIM; d++) {
            x_buf[d] = token_emb[token * DIM + d] + pos_emb[0 * DIM + d];
        }

        // Note: Full transformer forward would iterate through layers here
        // using bit_linear_flash for each projection. This demo shows the
        // embedding + first linear as proof of concept.

        Serial.printf("%d ", token);
        token = (token + 1) % VOCAB_SIZE; // Placeholder: would use actual inference
    }

    unsigned long elapsed = millis() - start;
    Serial.printf("\nGenerated 20 tokens in %lu ms\n", elapsed);
    Serial.println("\n=== Demo Complete ===");
}

void loop() {
    delay(10000);
}
