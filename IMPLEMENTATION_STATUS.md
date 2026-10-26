# Implementation status

Last updated 2026-10-26. All core components are implemented and verified locally.

## Environment
Windows 11, Python 3.11.9, Node 24, Docker 29 (daemon had to be started), no `psql`/`make`. Repo was empty and is **not yet a git repository**.
A native PostgreSQL service already occupies host port 5432, so the project's Docker database uses **5433** (see `.env.example`).
M5 data: Kaggle needs a login, so files come from the Hugging Face mirror `kashif/M5` (sizes identical to the originals).

## Checklist
- [x] Data download, ingestion (wide -> long, pre-launch trimming), star schema, SQL features, 13 SQL quality checks
- [x] Leakage-safe features + tests; SQL-vs-Python feature equivalence test
- [x] Naive, seasonal naive, 28-day MA, global XGBoost (recursive); 3 rolling-origin folds; WAPE/MAE/RMSE/bias; empirical 80% intervals
- [x] Error analysis + one controlled feature experiment (`xgboost_v2`)
- [x] Monte Carlo simulator (10,000 paths, (s,S) lost-sales policy) + held-out backtest
- [x] FastAPI (service/repository layers, read-only role, CORS, validation) - 7 endpoints
- [x] Next.js dashboard, 4 views, real API only
- [x] Tests, ruff, frontend lint + production build, CI workflow, Docker Compose, Makefile/dev.ps1, Dockerfile
- [x] README with real results and resume bullets
- [ ] Deployment (needs hosting credentials) - see README "Deployment"
- [ ] `git init` / first commit (not requested)

## Commands run and observed results
- `dfs db-init`, `dfs ingest` (538,515 sales rows, 79,545 price rows), `dfs quality` -> 13/13 pass
- `dfs train` (run 3): 168,000 forecasts, 1,630 metric rows, 126,000 residuals, 9,600 backtest rows; ~6 min
- `pytest` -> 109 passed (incl. 11 PostgreSQL integration tests); `ruff check dfs tests` -> clean
- `npm run lint` and `npm run build` (frontend) -> pass
- `./dev.ps1 quality` and `./dev.ps1 lint` -> pass
- `docker build` of the API image succeeds; the container served `/health` and `/simulate` against the local database
- Browser verification: the Chrome extension could not reach localhost, so pages were rendered with headless Chrome and inspected as screenshots (`docs/screenshots/`)

## Problems met and how they were resolved
- Port 5432 taken by a native PostgreSQL -> Docker DB moved to 5433.
- `CREATE OR REPLACE VIEW` cannot change column types -> views are dropped and recreated in `sql/features.sql`.
- FastAPI could not resolve a locally defined annotation under `from __future__ import annotations` (422s) -> removed that import from `main.py`.
- First simulator (7-day block bootstrap of own residuals) was badly over-confident (59-71% of realized 28-day totals outside the simulated 10-90% band) -> replaced by pooled whole-path resampling; documented in README. Calibration numbers are mildly optimistic because the diagnosis used the same held-out folds.
- A Python patch script wrote an en dash in cp1252, breaking `simulator/page.tsx` as invalid UTF-8 -> repaired; whole repo scanned for invalid UTF-8.
- Initial training runs 1-2 (older simulator) were deleted from the database; run 3 is the only run.

## Not verified
- GitHub Actions has never run (no remote).
- No deployment to Vercel / an API host.

## Next steps
1. `git init`, commit, push, and confirm CI is green.
2. Deploy per README "Deployment".
3. Optional: conformal intervals (current coverage 69% vs 80%), Tweedie objective for intermittent items.
