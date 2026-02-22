# QA Checklist

## Pre-deploy

1. Run compile check: `python -m compileall backend tests`
2. Run tests: `python -m pytest -q`
3. Run load test:
   `k6 run loadtests/k6_flips.js -e BASE_URL=http://localhost:8001`
4. Run 1-hour soak: keep backend + poller running and monitor logs for exceptions.

## Post-deploy (Railway/Vercel)

1. Verify health endpoint: `GET /health`
2. Verify scoring endpoints:
   - `GET /flips/top?limit=20&profile=balanced`
   - `GET /flips/filtered?min_confidence=40&max_risk=7`
   - `GET /flips/top5?profile=balanced`
3. Confirm `/flips/top5` p95 under 200ms.
4. Confirm dashboard columns render:
   - margin_after_tax
   - liquidity_score
   - fill_probability
   - confidence_pct
   - risk_level
   - final_score
5. Confirm user profile endpoints:
   - `POST /api/user/risk_profile`
   - `POST /api/user/flips/import`
   - `GET /api/user/insights`
   - `PATCH /api/user/alerts/settings`
   - `GET /api/user/alerts/test`

## Manual bug hunt

1. Poller writes snapshots and no crash loops.
2. Negative margin items are capped to very low score.
3. Conservative profile surfaces fewer low-fill items than aggressive.
4. Imported flip outcomes update calibration multipliers.
5. Portfolio optimizer respects capital/slot constraints.
6. Alert test ping succeeds and cooldown behavior is observed in logs.

