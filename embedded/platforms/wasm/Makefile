# Atomic-1Bit WebAssembly Build
#
# Requirements:
#   - Emscripten SDK (https://emscripten.org/docs/getting_started/)
#   - source emsdk/emsdk_env.sh
#
# Build:
#   make          # Produces atomic.js + atomic.wasm
#   make clean    # Remove build artifacts
#   make serve    # Start local HTTP server for testing

EMCC = emcc
CXXFLAGS = -O3 -std=c++17

# Emscripten-specific flags
EMFLAGS = \
	-s WASM=1 \
	-s EXPORTED_FUNCTIONS='["_init_model","_generate_token","_get_logits","_get_vocab_size","_reset_context","_free_model","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","HEAPF32"]' \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s INITIAL_MEMORY=67108864 \
	-s MAXIMUM_MEMORY=268435456 \
	-s NO_EXIT_RUNTIME=1

SRC = atomic_wasm.cpp
OUT = atomic.js

.PHONY: all clean serve

all: $(OUT)

$(OUT): $(SRC)
	$(EMCC) $(CXXFLAGS) $(EMFLAGS) $(SRC) -o $(OUT)
	@echo "Build complete: $(OUT) + atomic.wasm"
	@echo "Open index.html in a browser to run the demo."

clean:
	rm -f atomic.js atomic.wasm

serve:
	@echo "Starting HTTP server on port 8080..."
	@echo "Open http://localhost:8080/index.html"
	python3 -m http.server 8080
