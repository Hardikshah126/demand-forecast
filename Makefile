# Developer commands (Unix-like shells). On Windows PowerShell use ./dev.ps1 with the same target names.
PY ?= .venv/bin/python
DFS ?= .venv/bin/dfs

.PHONY: setup db-up db-init download ingest quality train serve test lint frontend-install frontend-dev frontend-build all-data

setup:
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dev]"

db-up:
	docker compose up -d --wait

db-init:
	$(DFS) db-init

download:
	$(DFS) download

ingest:
	$(DFS) ingest

quality:
	$(DFS) quality

train:
	$(DFS) train

serve:
	$(DFS) serve --reload

# One-shot data + model pipeline (assumes `make setup` and a populated .env)
all-data: db-up db-init download ingest quality train

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check dfs tests
	cd frontend && npm run lint

frontend-install:
	cd frontend && npm ci

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build
