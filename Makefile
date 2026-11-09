# Atomic-1Bit top-level build.
#
# make            -> build the ternary kernel and the C++ runner
# make verify     -> clone-to-proof: build everything, check kernel parity,
#                    smoke-test the runner, and run the quickstart demo
# make test       -> full pytest suite (needs pytest installed)
# make clean      -> remove built artifacts
#
# See docs/COMMANDS.md for the full command reference.

CXX ?= g++
PY  ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: all kernel runner verify test clean

all: kernel runner

kernel:
	$(MAKE) -C atomic_1bit/core

runner:
	cd embedded && $(CXX) -O3 -std=c++17 atomic_runner.cpp -o runner

verify: kernel runner
	"$(PY)" atomic_1bit/python/inference.py
	./embedded/runner --help > /dev/null && echo "C++ runner OK"
	"$(PY)" examples/quickstart.py
	@echo "verify complete: kernel, runner, and quickstart all ran"

test:
	"$(PY)" -m pytest tests/ -x -q

bench: kernel runner
	"$(PY)" benchmarks/run_suite.py
	"$(PY)" benchmarks/kernel_micro.py

clean:
	$(MAKE) -C atomic_1bit/core clean
	rm -f embedded/runner embedded/atomic_model.bin
