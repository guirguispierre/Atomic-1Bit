/*
 * Atomic-1Bit for M5Stack Cardputer
 * File: CardputerChat.ino
 *
 * INSTRUCTIONS:
 * 1. Board: "M5Stack-STAMPS3" (or "ESP32S3 Dev Module")
 * 2. PSRAM: "OPI PSRAM" (Enabled) - CRITICAL!
 * 3. USB Mode: "Hardware CDC and JTAG"
 * 4. Copy 'atomic_model.bin' to MicroSD card root.
 * 5. Ensure 'M5Cardputer' library is installed.
 */

#include <M5Cardputer.h>
#include <SPI.h>
#include <SD.h>
#include <vector>
#include <cmath>

// --- Cardputer SD Pins ---
#define SD_SPI_SCK  40
#define SD_SPI_MISO 39
#define SD_SPI_MOSI 14
#define SD_SPI_CS   12

using namespace std;

// --- CONFIG ---
struct AtomicConfig {
    int vocab_size;
    int dim;
    int depth;
    int heads;
    int max_seq_len;
    int has_gist;
};

// --- LAYERS ---
struct AtomicLayer {
    vector<float> ln1;
    vector<int8_t> q_w, k_w, v_w, o_w;
    vector<float> ln2;
    vector<int8_t> fc1_w, fc2_w;
};

struct AtomicModel {
    AtomicConfig config;
    vector<float> token_emb;
    vector<float> pos_emb;
    vector<float> ln_f;
    vector<int8_t> head_w;
    vector<AtomicLayer> layers;
    vector<float> gist_vector;
};

AtomicModel model;
bool model_loaded = false;

// --- MATH OPS ---
inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
}

void rms_norm(const vector<float> &x, const vector<float> &w, vector<float> &out, int dim) {
    float ss = 0.0f;
    for (float f : x) ss += f * f;
    ss /= dim;
    float rms = 1.0f / sqrt(ss + 1e-5f);
    for (int i = 0; i < dim; ++i) out[i] = x[i] * rms * w[i];
}

void softmax(vector<float> &x) {
    float max_val = -1e9f;
    for (float f : x) if (f > max_val) max_val = f;
    float sum = 0.0f;
    for (float &f : x) { f = exp(f - max_val); sum += f; }
    for (float &f : x) f /= sum;
}

// 1.58-bit Linear
void bit_linear(const vector<float> &x, const vector<int8_t> &w, vector<float> &out) {
    float max_val = 0.0f;
    for (float f : x) max_val = max(max_val, abs(f));
    float scale = 127.0f / max(max_val, 1e-5f);

    int dim = x.size();
    int out_dim = out.size();
    static vector<int8_t> x_q; // Reuse memory
    if(x_q.size() != dim) x_q.resize(dim);

    for (int i = 0; i < dim; ++i) {
        x_q[i] = (int8_t)max(-127.0f, min(127.0f, round(x[i] * scale)));
    }

    for (int o = 0; o < out_dim; ++o) {
        int32_t acc = 0;
        for (int i = 0; i < dim; ++i) {
            acc += (int32_t)x_q[i] * (int32_t)w[i * out_dim + o];
        }
        out[o] = (float)acc / scale;
    }
}

// --- LOAD MODEL (STABLE VERSION) ---
bool load_model_sd(const char* filename) {
    File f = SD.open(filename, FILE_READ);
    if (!f) return false;

    // Read Config
    f.read((uint8_t*)&model.config, sizeof(AtomicConfig));
    int dim = model.config.dim;

    // Read Gist
    if (model.config.has_gist) {
        model.gist_vector.resize(dim);
        f.read((uint8_t*)model.gist_vector.data(), dim * 4);
    }

    // Read Embeddings
    model.token_emb.resize(model.config.vocab_size * dim);
    f.read((uint8_t*)model.token_emb.data(), model.token_emb.size() * 4);
    
    model.pos_emb.resize(model.config.max_seq_len * dim);
    f.read((uint8_t*)model.pos_emb.data(), model.pos_emb.size() * 4);

    // Read Layers
    for (int i = 0; i < model.config.depth; ++i) {
        AtomicLayer layer;
        layer.ln1.resize(dim); f.read((uint8_t*)layer.ln1.data(), dim * 4);
        
        int sq = dim*dim;
        layer.q_w.resize(sq); f.read((uint8_t*)layer.q_w.data(), sq);
        layer.k_w.resize(sq); f.read((uint8_t*)layer.k_w.data(), sq);
        layer.v_w.resize(sq); f.read((uint8_t*)layer.v_w.data(), sq);
        layer.o_w.resize(sq); f.read((uint8_t*)layer.o_w.data(), sq);
        
        layer.ln2.resize(dim); f.read((uint8_t*)layer.ln2.data(), dim * 4);
        
        int fc1 = dim * (dim*4);
        int fc2 = (dim*4) * dim;
        layer.fc1_w.resize(fc1); f.read((uint8_t*)layer.fc1_w.data(), fc1);
        layer.fc2_w.resize(fc2); f.read((uint8_t*)layer.fc2_w.data(), fc2);
        
        model.layers.push_back(layer);
        
        // --- STABILITY FIX ---
        M5Cardputer.Display.print("."); // Show progress
        delay(10); // Prevent Watchdog Timeout!
    }

    // Read Final
    model.ln_f.resize(dim); f.read((uint8_t*)model.ln_f.data(), dim * 4);
    model.head_w.resize(dim * model.config.vocab_size); f.read((uint8_t*)model.head_w.data(), dim * model.config.vocab_size);

    f.close();
    return true;
}

// --- INFERENCE ---
void run_inference(vector<int> &tokens) {
    int dim = model.config.dim;
    int heads = model.config.heads;
    int head_dim = dim / heads;
    int seq_len = tokens.size() + (model.config.has_gist ? 1 : 0);
    
    if (seq_len > model.config.max_seq_len) return;

    // Inputs
    vector<float> X(seq_len * dim);
    int t_offset = 0;
    
    if (model.config.has_gist) {
        memcpy(X.data(), model.gist_vector.data(), dim * 4);
        t_offset = 1;
    }

    for (int i = 0; i < tokens.size(); ++i) {
        int id = tokens[i];
        for (int d = 0; d < dim; ++d)
             X[(t_offset+i)*dim + d] = model.token_emb[id*dim+d] + model.pos_emb[i*dim+d];
    }

    vector<float> X_norm(seq_len * dim);
    
    // Layers
    for (auto &layer : model.layers) {
        // 1. Attention
        for(int t=0; t<seq_len; ++t) {
            vector<float> xt(dim); memcpy(xt.data(), &X[t*dim], dim*4);
            vector<float> xn(dim); rms_norm(xt, layer.ln1, xn, dim);
            memcpy(&X_norm[t*dim], xn.data(), dim*4);
        }

        vector<float> Q(seq_len*dim), K(seq_len*dim), V(seq_len*dim);
        
        // QKV Proj
        for(int t=0; t<seq_len; ++t) {
            vector<float> xt(dim); memcpy(xt.data(), &X_norm[t*dim], dim*4);
            vector<float> q(dim), k(dim), v(dim);
            bit_linear(xt, layer.q_w, q);
            bit_linear(xt, layer.k_w, k);
            bit_linear(xt, layer.v_w, v);
            memcpy(&Q[t*dim], q.data(), dim*4);
            memcpy(&K[t*dim], k.data(), dim*4);
            memcpy(&V[t*dim], v.data(), dim*4);
        }
        
        // Attention Scores
        vector<float> AttnOut(seq_len * dim, 0.0f);
        for(int t=0; t<seq_len; ++t) {
            for(int h=0; h<heads; ++h) {
                float* q_ptr = &Q[t*dim + h*head_dim];
                vector<float> scores;
                for(int s=0; s<=t; ++s) { // Causal
                    float* k_ptr = &K[s*dim + h*head_dim];
                    float dp = 0.0f;
                    for(int d=0; d<head_dim; ++d) dp += q_ptr[d]*k_ptr[d];
                    scores.push_back(dp / sqrt((float)head_dim));
                }
                softmax(scores);
                
                float* out_ptr = &AttnOut[t*dim + h*head_dim];
                for(int s=0; s<=t; ++s) {
                    float w = scores[s];
                    float* v_ptr = &V[s*dim + h*head_dim];
                    for(int d=0; d<head_dim; ++d) out_ptr[d] += w * v_ptr[d];
                }
            }
        }
        
        // O Proj + Resid
        for(int t=0; t<seq_len; ++t) {
            vector<float> attn(dim); memcpy(attn.data(), &AttnOut[t*dim], dim*4);
            vector<float> o(dim); bit_linear(attn, layer.o_w, o);
            for(int i=0; i<dim; ++i) X[t*dim+i] += o[i];
        }

        // 2. MLP
        for(int t=0; t<seq_len; ++t) {
            vector<float> xt(dim); memcpy(xt.data(), &X[t*dim], dim*4);
            vector<float> xn(dim); rms_norm(xt, layer.ln2, xn, dim);
            
            int hidden = dim * 4;
            vector<float> h(hidden); bit_linear(xn, layer.fc1_w, h);
            for(int i=0; i<hidden; ++i) h[i] = gelu(h[i]);
            
            vector<float> out(dim); bit_linear(h, layer.fc2_w, out);
            for(int i=0; i<dim; ++i) X[t*dim+i] += out[i];
        }
    }

    // Head
    int last_t = seq_len - 1;
    vector<float> xt(dim); memcpy(xt.data(), &X[last_t*dim], dim*4);
    vector<float> xn(dim); rms_norm(xt, model.ln_f, xn, dim);
    
    vector<float> logits(model.config.vocab_size);
    bit_linear(xn, model.head_w, logits);

    // Greedy Sample
    int best_id = 0; float best_val = -1e9;
    for(int i=0; i<model.config.vocab_size; ++i) {
        if(logits[i] > best_val) { best_val = logits[i]; best_id = i; }
    }
    
    tokens.push_back(best_id);
    
    // Print char (assuming ASCII model output)
    M5Cardputer.Display.print((char)best_id); 
}

// --- SETUP ---
void setup() {
    auto cfg = M5.config();
    M5Cardputer.begin(cfg, true); // Enable Keyboard
    M5Cardputer.Display.setRotation(1);
    M5Cardputer.Display.setTextSize(2);
    
    M5Cardputer.Display.println("Atomic-1Bit Init...");

    // 1. Check PSRAM
    if (ESP.getPsramSize() == 0) {
        M5Cardputer.Display.setTextColor(RED);
        M5Cardputer.Display.println("ERR: NO PSRAM!");
        M5Cardputer.Display.setTextSize(1);
        M5Cardputer.Display.println("Enable 'OPI PSRAM' in Tools Menu");
        while(1);
    } else {
        M5Cardputer.Display.setTextColor(GREEN);
        M5Cardputer.Display.print("RAM: ");
        M5Cardputer.Display.print(ESP.getPsramSize() / 1024 / 1024);
        M5Cardputer.Display.println("MB OK");
        M5Cardputer.Display.setTextColor(WHITE);
    }

    // 2. Init SD
    SPI.begin(SD_SPI_SCK, SD_SPI_MISO, SD_SPI_MOSI, SD_SPI_CS);
    if(!SD.begin(SD_SPI_CS, SPI, 25000000)) {
        M5Cardputer.Display.setTextColor(RED);
        M5Cardputer.Display.println("SD Fail!");
        while(1);
    }

    // 3. Load Model
    M5Cardputer.Display.print("Loading");
    if(!load_model_sd("/atomic_model.bin")) {
        M5Cardputer.Display.setTextColor(RED);
        M5Cardputer.Display.println("\nLoad Fail!");
        M5Cardputer.Display.setTextSize(1);
        M5Cardputer.Display.println("Check atomic_model.bin");
        while(1);
    }
    
    model_loaded = true;
    M5Cardputer.Display.println("\nReady.");
    M5Cardputer.Display.print("> ");
}

String input_buffer = "";

// --- MAIN LOOP ---
void loop() {
    M5Cardputer.update();
    
    if(M5Cardputer.Keyboard.isChange()) {
        if(M5Cardputer.Keyboard.isPressed()) {
            Keyboard_Class::KeysState status = M5Cardputer.Keyboard.keysState();
            
            for(auto i : status.word) {
                input_buffer += i;
                M5Cardputer.Display.print(i);
            }
            if(status.del) {
                if(input_buffer.length() > 0) {
                    input_buffer.remove(input_buffer.length()-1);
                    // Simple backspace visual hack
                    int x = M5Cardputer.Display.getCursorX();
                    int y = M5Cardputer.Display.getCursorY();
                    M5Cardputer.Display.setCursor(x-12, y);
                    M5Cardputer.Display.print(" ");
                    M5Cardputer.Display.setCursor(x-12, y);
                }
            }
            if(status.enter) {
                M5Cardputer.Display.println();
                M5Cardputer.Display.setTextColor(GREEN);
                
                vector<int> tokens; 
                for(int i=0; i<input_buffer.length(); ++i) tokens.push_back((int)input_buffer[i]);
                
                // Generate response (20 tokens)
                for(int k=0; k<20; ++k) run_inference(tokens);
                
                M5Cardputer.Display.println();
                M5Cardputer.Display.setTextColor(WHITE);
                M5Cardputer.Display.print("> ");
                input_buffer = "";
            }
        }
    }
}