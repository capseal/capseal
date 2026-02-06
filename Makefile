# CapSeal Makefile
# Build targets for installation, testing, and native code compilation

.PHONY: install install-dev install-full build-rust build-enn test test-all lint clean help

# Default target
help:
	@echo "CapSeal Build Targets"
	@echo "====================="
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install core package (Python only)"
	@echo "  make install-dev   Install with dev dependencies (pytest, ruff)"
	@echo "  make install-full  Install all optional dependencies"
	@echo ""
	@echo "Native Code (optional):"
	@echo "  make build-rust    Build bef_rust.so (FRI/STC proof generation)"
	@echo "  make build-enn     Build C++ ENN trainer (committor_train)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run integration tests only"
	@echo "  make test-all      Run all tests (including internal)"
	@echo ""
	@echo "Development:"
	@echo "  make lint          Run ruff linter on new code"
	@echo "  make clean         Remove build artifacts and caches"
	@echo ""
	@echo "Note: Run 'make build-rust' if you need proof generation features."
	@echo "      The bef_rust.so binary enables FRI and STC verification."

# Installation targets
install:
	pip install -e .
	@echo ""
	@echo "Installed capseal (core). Run 'capseal --help' to verify."
	@echo "Note: For proof generation, also run 'make build-rust'"

install-dev:
	pip install -e '.[dev]'
	@echo ""
	@echo "Installed capseal with dev dependencies."

install-full:
	pip install -e '.[full]'
	@echo ""
	@echo "Installed capseal with all dependencies."

# Native code builds
build-rust:
	@echo "Building bef_rust.so via maturin..."
	@if command -v maturin >/dev/null 2>&1; then \
		cd BEF-main/bef_rust && maturin develop --release; \
		echo ""; \
		echo "bef_rust.so built successfully."; \
	else \
		echo "maturin not found. Install with: pip install maturin"; \
		echo "Then run: make build-rust"; \
		exit 1; \
	fi

build-enn:
	@echo "Building C++ ENN trainer..."
	@if [ -d "otherstuff/enn-cpp" ]; then \
		cd otherstuff/enn-cpp && \
		mkdir -p build && \
		cd build && \
		cmake .. && \
		make -j$$(nproc); \
		echo ""; \
		echo "committor_train binary built at otherstuff/enn-cpp/build/committor_train"; \
	else \
		echo "enn-cpp directory not found"; \
		exit 1; \
	fi

# Testing
test:
	python -m pytest tests/ -v

test-all:
	python -m pytest tests/ -v
	@echo ""
	@echo "Note: To run full internal tests, use:"
	@echo "  PYTHONPATH=BEF-main:otherstuff python -m pytest BEF-main/bef_zk/ otherstuff/tests/ -v"

# Development
lint:
	@echo "Linting capseal_cli and new integration code..."
	ruff check capseal_cli/ --fix
	ruff check BEF-main/bef_zk/shared/ --fix
	ruff check BEF-main/bef_zk/capsule/cli/eval_cmd.py --fix
	ruff check BEF-main/bef_zk/capsule/committor_gate.py --fix
	@echo "Lint complete."

lint-all:
	@echo "Linting entire codebase (may have many issues)..."
	ruff check BEF-main/bef_zk/ otherstuff/ capseal_cli/ --statistics

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts and caches."

# Verify installation works
verify:
	@echo "Verifying installation..."
	@python -c "from capseal_cli import __version__; print(f'capseal_cli version: {__version__}')"
	@python -c "from bef_zk.capsule.cli.shell import CapsealShell; print('shell import: OK')"
	@python -c "from bef_zk.shared.scoring import compute_acquisition_score; print('scoring import: OK')"
	@python -c "from bef_zk.shared.receipts import build_round_receipt; print('receipts import: OK')"
	@capseal --help >/dev/null && echo "capseal CLI: OK"
	@echo ""
	@echo "All verifications passed."
