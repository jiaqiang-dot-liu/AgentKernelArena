# Makefile for KernelBench HIP Kernel Development
# Dynamic ROCm Environment Setup

SHELL := /bin/bash
PYTHON_VERSION := 3.12
VENV_DIR := .venv
REQUIREMENTS := requirements.txt

# Detect ROCm version (supports 7.1, 7.0, 6.4, including patch releases)
ROCM_PATH_DETECTED := $(shell \
	for d in /opt/rocm-7.1.* /opt/rocm-7.1 /opt/rocm-7.0.* /opt/rocm-7.0 /opt/rocm-6.4.* /opt/rocm-6.4 /opt/rocm; do \
		if [ -d "$$d" ]; then readlink -f "$$d"; exit 0; fi; \
	done; \
	echo "/opt/rocm")
ROCM_VERSION := $(shell \
	rocm_path="$(ROCM_PATH_DETECTED)"; \
	if [ -f "$$rocm_path/.info/version" ]; then \
		version=$$(sed -n '1p' "$$rocm_path/.info/version"); \
	else \
		version=$$(basename "$$rocm_path" | sed -n 's/^rocm-\([0-9]\+\.[0-9]\+\).*/\1/p'); \
	fi; \
	if printf '%s\n' "$$version" | grep -q '^7\.1'; then echo "7.1"; \
	elif printf '%s\n' "$$version" | grep -q '^7\.0'; then echo "7.0"; \
	elif printf '%s\n' "$$version" | grep -q '^6\.4'; then echo "6.4"; \
	else echo "unknown"; fi)

# ROCm environment variables
export ROCM_PATH := $(ROCM_PATH_DETECTED)
export CMAKE_PREFIX_PATH := $(ROCM_PATH_DETECTED):$(ROCM_PATH_DETECTED)/hip:/usr/local:/usr
export MAX_JOBS := 8
export HIP_FORCE_DEV_KERNARG := 1
export HSA_NO_SCRATCH_RECLAIM := 1

.PHONY: help setup setup-venv setup-flydsl verify-flydsl clean cleanup-venv cleanup-works install-cursor-agent act vllm docker-shell docker-check-agents docker-run docker-smoke

help:
	@echo "AgentKernelArena Evaluation Framework - Makefile Commands"
	@echo "======================================================"
	@echo "Docker-first workflow:"
	@echo "make docker-shell        - Enter the benchmark Docker image with repo and agent auth mounted"
	@echo "make docker-check-agents - Verify Codex, Claude Code, and Cursor Agent login reuse in Docker"
	@echo "make docker-smoke        - Verify Docker Python, ROCm tools, imports, and GPU access"
	@echo "make docker-run CONFIG=config.yaml RUN_ARGS=\"--run-suffix test\" - Run benchmark in Docker"
	@echo "                         Images: gfx942->mi30x, gfx950->mi35x; override with AKA_DOCKER_IMAGE=..."
	@echo ""
	@echo "Legacy venv workflow:"
	@echo "make setup              - Complete environment setup (venv + deps, includes FlyDSL by default)"
	@echo "make setup WITH_FLYDSL=0 - Setup without FlyDSL"
	@echo "make setup-flydsl       - Install and verify FlyDSL dependency for flydsl2flydsl tasks"
	@echo "make verify-flydsl      - Verify FlyDSL import and ROCm PyTorch GPU availability"
	@echo "make clean              - Remove virtual environment"

WITH_FLYDSL ?= 1

setup: setup-venv
ifeq ($(WITH_FLYDSL),1)
setup: setup-flydsl
endif

setup-venv:
	@echo "Detected ROCm version: $(ROCM_VERSION) at $(ROCM_PATH_DETECTED)"
	@if [ "$(ROCM_VERSION)" = "unknown" ]; then \
		echo "ERROR: Could not detect ROCm installation"; \
		exit 1; \
	fi
	@echo "Creating virtual environment with uv..."
	@uv venv $(VENV_DIR) --python $(PYTHON_VERSION)
	@echo "✓ Virtual environment created"
	@echo "Installing PyTorch for ROCm $(ROCM_VERSION)..."
	@source $(VENV_DIR)/bin/activate && \
		uv pip install --upgrade pip setuptools wheel && \
		uv pip install setuptools==75.8.0 && \
		uv pip install setuptools_scm packaging && \
		if [ "$(ROCM_VERSION)" = "7.1" ]; then \
			uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1; \
		elif [ "$(ROCM_VERSION)" = "7.0" ]; then \
			uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0; \
		else \
			uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4; \
		fi
	@echo "✓ PyTorch installed"
	@echo "Installing Python dependencies..."
	@if [ ! -f $(REQUIREMENTS) ]; then \
		echo "Creating requirements.txt..."; \
		echo "# Build tools" > $(REQUIREMENTS); \
		echo "ninja" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# LLM service dependencies" >> $(REQUIREMENTS); \
		echo "pyyaml" >> $(REQUIREMENTS); \
		echo "httpx" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# Utilities" >> $(REQUIREMENTS); \
		echo "numpy" >> $(REQUIREMENTS); \
	fi
	@source $(VENV_DIR)/bin/activate && uv pip install -r $(REQUIREMENTS)
	@echo "✓ Setup complete! Activate with: source $(VENV_DIR)/bin/activate"

setup-flydsl: setup-venv
	@echo "Installing FlyDSL..."
	@source $(VENV_DIR)/bin/activate && \
		uv pip install flydsl
	@$(MAKE) verify-flydsl
	@echo "✓ FlyDSL installed"

verify-flydsl:
	@echo "Verifying FlyDSL and ROCm PyTorch GPU availability..."
	@source $(VENV_DIR)/bin/activate && \
		python3 -c 'import flydsl, torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is False; FlyDSL GPU tasks require ROCm PyTorch with GPU access"; print("✓ FlyDSL import OK:", getattr(flydsl, "__version__", "unknown")); print("✓ ROCm PyTorch GPU OK:", torch.cuda.get_device_name(0))'

cleanup-venv:
	@echo "Removing virtual environment and build caches..."
	@rm -rf $(VENV_DIR)
	@find . -type d -name "build_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

cleanup-works:
	@echo "Removing workspace directories and logs..."
	@rm -rf workspace_*
	@rm -rf logs
	@echo "✓ Workspace directories and logs removed"

install-cursor-agent:
	@echo "Installing Cursor agent..."
	@curl https://cursor.com/install -fsSL | bash


ACTIVATE_VENV_CMD = exec bash -c "source .venv/bin/activate && exec bash"
act:
	$(ACTIVATE_VENV_CMD) 

DOCKER_RUNNER := scripts/docker_benchmark.sh
CONFIG ?= config.yaml
RUN_ARGS ?=

docker-shell:
	@$(DOCKER_RUNNER) shell

docker-check-agents:
	@$(DOCKER_RUNNER) check-agents

docker-smoke:
	@$(DOCKER_RUNNER) smoke

docker-run:
	@$(DOCKER_RUNNER) run --config_name $(CONFIG) $(RUN_ARGS)


# Run vLLM server with latest ROCm 6.4.1 and vLLM 0.10.1
vllm:
	@if ss -ltn | grep ':30001 ' > /dev/null; then \
		echo "vLLM server is already running on port 30001."; \
	else \
		docker run -d \
			--ipc=host \
			--network=host \
			--privileged \
			--cap-add=SYS_ADMIN \
			--cap-add=SYS_PTRACE \
			--device=/dev/kfd \
			--device=/dev/dri \
			--device=/dev/mem \
			--group-add=render \
			--security-opt=seccomp=unconfined \
			rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909 \
			vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
			--served-model-name llamas_team_local_llm \
			--api-key dummy \
			--host 0.0.0.0 \
			--port 30001 \
			--enable-auto-tool-choice \
			--tool-call-parser hermes \
			--trust-remote-code; \
		echo "Don't forget to set local_llm_enabled: true in configs/config.yml"; \
		echo "vLLM server will be running on port 30001, please wait 3 minutes for it to start..."; \
		echo "You can use docker logs -f container_id to check the server status"; \
	fi
