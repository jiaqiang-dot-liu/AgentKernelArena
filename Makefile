# Makefile for AgentKernelArena — Docker-first workflow
#
# All benchmarking runs inside the pinned ROCm/SGLang container. Docker is the only
# supported path; the legacy host venv / `python main.py` workflow has been removed.
# See scripts/docker_benchmark.sh and docs/install/install.md.

SHELL := /bin/bash

.PHONY: help docker-shell docker-check-agents docker-smoke docker-run docker-setup-flydsl \
        sync-perf-helpers check-perf-helpers cleanup-works install-cursor-agent vllm

help:
	@echo "AgentKernelArena Evaluation Framework - Makefile Commands"
	@echo "======================================================"
	@echo "Docker-first workflow (the only supported path):"
	@echo "make docker-shell        - Enter the benchmark Docker image with repo and agent auth mounted"
	@echo "make docker-check-agents - Verify Codex, Claude Code, and Cursor Agent login reuse in Docker"
	@echo "make docker-smoke        - Verify Docker Python, ROCm tools, imports, and GPU access"
	@echo "make docker-run CONFIG=config.yaml RUN_ARGS=\"--run-suffix test\" - Run benchmark in Docker"
	@echo "                         Images: gfx942->mi30x, gfx950->mi35x; override with AKA_DOCKER_IMAGE=..."
	@echo "make docker-setup-flydsl - Install FlyDSL into the container (needed for flydsl2flydsl tasks)"
	@echo ""
	@echo "Maintenance:"
	@echo "make sync-perf-helpers   - Propagate canonical perf helpers (tools/perf/) into task copies"
	@echo "make check-perf-helpers  - Verify perf-helper copies are in sync"
	@echo "make cleanup-works       - Remove workspace_* directories and logs"
	@echo "make install-cursor-agent- Install the Cursor Agent CLI on the host"

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

# Install FlyDSL into the container's persistent pip user-base (the base image does
# not ship it). Run once per machine/image; needed only for flydsl2flydsl tasks.
docker-setup-flydsl:
	@$(DOCKER_RUNNER) setup-flydsl

# Propagate the canonical perf-benchmark helpers (tools/perf/) into every task copy.
# Edit tools/perf/*, then run this.
sync-perf-helpers:
	@python3 tools/sync_perf_helpers.py

# Verify all task perf-helper copies match the canonical source (CI-friendly).
check-perf-helpers:
	@python3 tools/sync_perf_helpers.py --check

cleanup-works:
	@echo "Removing workspace directories and logs..."
	@rm -rf workspace_*
	@rm -rf logs
	@echo "✓ Workspace directories and logs removed"

install-cursor-agent:
	@echo "Installing Cursor agent..."
	@curl https://cursor.com/install -fsSL | bash

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
