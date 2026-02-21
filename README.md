# OSRS Flipping AI

Python + MongoDB backend for OSRS Grand Exchange flip scoring, personalization, alerts, and dashboard APIs.

## Local setup

1. Create virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill values.
3. Start backend:
   `uvicorn backend.app:app --host 0.0.0.0 --port 8001`

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

