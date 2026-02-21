# OSRS Flipping AI

Python + MongoDB backend for OSRS Grand Exchange flip scoring, personalization, alerts, and dashboard APIs.

## Local setup

1. Create virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill values.
3. Start API service:
   `RUN_MODE=api uvicorn backend.app:app --host 0.0.0.0 --port 8001`
4. Start worker service in a second terminal:
   `RUN_MODE=worker python -m backend.worker`

## Core endpoints

- `GET /flips/top`
- `GET /flips/filtered`
- `GET /flips/top5`
- `POST /portfolio/optimize`
- `POST /api/user/risk_profile`
- `POST /api/user/flips/import`
- `GET /api/user/insights`
- `PATCH /api/user/alerts/settings`
- `GET /api/user/alerts/test`
- `GET /health`

## Load test

Run:
`k6 run loadtests/k6_flips.js -e BASE_URL=http://localhost:8001`

## QA checklist

See `docs/QA.md`.

## Railway deployment (2 services)

Deploy two Railway services from the same repo:

1. Web/API service:
   - Start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - Env: `RUN_MODE=api`
2. Worker service:
   - Start command: `python -m backend.worker`
   - Env: `RUN_MODE=worker`

Both services should share the same `MONGODB_URL` and `DATABASE_NAME`.

