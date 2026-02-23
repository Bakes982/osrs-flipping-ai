# OSRS Flipping AI — Production Reference

This file covers every environment variable the application reads, expected
log patterns to verify things are working, and a deploy checklist for
Railway.

---

## 1. Required Environment Variables

### Core infrastructure (must be set)

| Variable | Example | Notes |
|---|---|---|
| `MONGODB_URL` | `mongodb+srv://user:pass@cluster.mongodb.net/` | Atlas connection string |
| `DATABASE_NAME` | `osrs_flipping_ai` | MongoDB database name |
| `REDIS_URL` | `redis://default:pass@host:6379` | Upstash / Railway Redis |
| `AUTH_SECRET` | `<64-char hex>` | JWT signing key — keep secret |
| `DISCORD_CLIENT_ID` | `123456789` | OAuth app client ID |
| `DISCORD_CLIENT_SECRET` | `…` | OAuth app client secret |
| `ALLOWED_DISCORD_IDS` | `123,456` | Comma-separated Discord user IDs allowed to log in |
| `DISCORD_REDIRECT_URI` | `https://backend.railway.app/api/auth/callback` | Must match Discord developer portal |

### Service URLs

| Variable | Example | Notes |
|---|---|---|
| `FRONTEND_URL` | `https://osrs-flipping-ai.vercel.app` | Used for CORS + OAuth redirect |
| `BACKEND_URL` | `https://osrs-flipping-ai-backend.railway.app` | Used for self-referencing links |
| `PORT` | `8001` | Railway sets this automatically |
| `RUN_MODE` | `api` or `worker` | `api` = FastAPI server, `worker` = background tasks only |

---

## 2. Optional / Tuning Variables

### Opportunities cache

| Variable | Default | Notes |
|---|---|---|
| `FLIPS_CACHE_TTL_SECONDS` | `600` | How long a warm flip list stays valid in Redis |
| `FLIPS_CACHE_WARM_INTERVAL_SECONDS` | `30` | Worker re-warms caches this often (measured from start of warm) |
| `FLIPS_FRESH_MAX_PER_MINUTE` | `6` | Rate-limit for fresh (non-cached) opportunity calls |
| `SCORE_STALE_MAX_MINUTES` | `45` | Items with data older than this are excluded from recommendations |

### Worker reliability

| Variable | Default | Notes |
|---|---|---|
| `WORKER_RETRY_INITIAL_SECONDS` | `2` | Backoff on session startup failure |
| `WORKER_RETRY_MAX_SECONDS` | `60` | Maximum backoff cap |
| `WORKER_CIRCUIT_FAILURE_THRESHOLD` | `5` | Failures before circuit opens |
| `WORKER_CIRCUIT_OPEN_SECONDS` | `300` | How long circuit stays open |
| `WORKER_OK_MAX_AGE_SECONDS` | `180` | `/health` `worker_ok` field max age |

### Dump detector v2 quality filters

| Variable | Default | Notes |
|---|---|---|
| `DUMP_V2_MIN_PRICE_GP` | `500000` | Skip items below this insta-sell price |
| `DUMP_V2_MIN_DROP_PCT` | `4.0` | Minimum % drop from 4h average |
| `DUMP_V2_MIN_VOLUME_TRADES` | `25` | Minimum 5m volume (sold + bought) |
| `DUMP_V2_MIN_SELL_RATIO` | `0.80` | Panic selling threshold (sold / total) |
| `DUMP_V2_MIN_PROFIT_PER_ITEM` | `2000` | Minimum net profit per item (GP) |
| `DUMP_V2_MIN_TOTAL_PROFIT` | `150000` | Minimum estimated total profit (GP) |
| `DUMP_V2_COOLDOWN_MINUTES` | `60` | Per-item cooldown between alerts |
| `DUMP_V2_POSITION_CAP_PCT` | `0.10` | Max fraction of capital per dump trade |

### Dump detector v2 speed + alert quality (Phase 4)

| Variable | Default | Notes |
|---|---|---|
| `DUMP_MAX_TICK_SECONDS` | `10.0` | Scan stops early after this many seconds |
| `DUMP_SUPPRESS_LOW` | `1` (true) | Set to `0` to re-enable ⭐☆☆ LOW alerts |
| `SHOW_FULL_DUMP_CHART` | `0` (false) | Set to `1` to attach price chart PNG (slow) |

### Discord webhooks per alert type (Phase 5)

| Variable | Alert type | Notes |
|---|---|---|
| `DISCORD_WEBHOOK_DUMPS` | Dump detector alerts | Highest priority for dump alerts |
| `DISCORD_WEBHOOK_OPPORTUNITIES` | Top-5 opportunity digests | Highest priority for opp alerts |
| `DISCORD_WEBHOOK_POSITIONS` | Position monitor / sell alerts | Highest priority for sell alerts |

Leave any of these empty to fall back to the `discord_webhook` DB setting
configured via the Settings page.

### Capital + position sizing

| Variable | Default | Notes |
|---|---|---|
| `DEFAULT_CAPITAL_GP` | `50000000` | Bankroll used for trade-plan sizing (50 M GP) |

---

## 3. Expected Log Patterns

Use these to verify a deployment is healthy.

### API server startup

```
INFO     uvicorn.error: Application startup complete.
```

### Worker startup

```
INFO     backend.worker: Starting worker in continuous mode.
INFO     backend.worker: Worker session started.
```

### Cache warm cycle (every ~30 s)

```
INFO     backend.flips_cache: OPP_CACHE_WRITE flips:top5:balanced ttl=600s items=5
INFO     backend.flips_cache: OPP_CACHE_WRITE flips:top100:balanced ttl=600s items=97
INFO     backend.worker: Cache warm complete in 82.3s: {'balanced': 97, 'aggressive': 89, ...}
```

If you see `Cache warm complete in Xs` where X ≈ interval, the loop is healthy.
If `items=0` on every cycle, the price collector may not be writing to Redis.

### Dump scan tick

```
INFO     dump_detector: DUMP_TICK start: scan_for_dumps_v2  max_tick=10.0s  suppress_low=True  charts=False
INFO     dump_detector: scan_for_dumps_v2: 12 candidates after pre-filters (of 4312 items)  elapsed=0.34s
INFO     dump_detector: DUMP_TICK end: scan_for_dumps_v2 complete  total=8.12s  alerts=2
```

If you see `DUMP_TICK deadline:` the scan is being cut short — increase
`DUMP_MAX_TICK_SECONDS` if you want more candidates scanned.

### Dump alert sent

```
INFO     dump_detector: DUMP DETECTED (v2): Dragon claws  -8.4%  conf=HIGH  net/item=312000 GP  total=1560000 GP  tick=3.21s
INFO     backend.tasks: NOTIFIER=dump WEBHOOK=DISCORD_WEBHOOK_DUMPS
```

### Opportunity alert sent

```
INFO     backend.tasks: NOTIFIER=opportunities WEBHOOK=DISCORD_WEBHOOK_OPPORTUNITIES
```

### Position / sell alert sent

```
INFO     backend.tasks: NOTIFIER=positions WEBHOOK=DISCORD_WEBHOOK_POSITIONS
```

---

## 4. Deploy Checklist

Run through this after every production deploy.

### Pre-deploy

- [ ] All required env vars are set in Railway (see Section 1)
- [ ] `DISCORD_WEBHOOK_DUMPS` points to the correct Discord channel
- [ ] `MONGODB_URL` is the Atlas connection string (not `localhost`)
- [ ] `REDIS_URL` is set (Upstash or Railway Redis plugin)
- [ ] `RUN_MODE=api` on the API service, `RUN_MODE=worker` on the worker service

### Post-deploy (API service)

- [ ] `GET /api/health` returns `{"status": "ok", "worker_ok": true, ...}`
- [ ] `GET /api/opportunities` returns `{"count": N, "items": [...]}` with N > 0
  - If `count=0`, check worker logs for `OPP_CACHE_WRITE` — may be still warming
- [ ] `GET /api/model/status` returns `{"status": "ready"}` or similar

### Post-deploy (Worker service)

- [ ] Worker logs show `Cache warm complete in Xs` within 2 minutes of startup
- [ ] Worker logs show `OPP_CACHE_WRITE flips:top100:balanced` with `items > 0`
- [ ] No `Cache warm failed` or `Worker session crashed` errors

### Alert verification

- [ ] Test dump webhook: set `DUMP_SMOKE_TEST=1` on the worker and redeploy once
  — should log `DUMP_SMOKE_TEST: Discord message sent OK`
- [ ] Test opportunity webhook: POST `/api/alerts/test-webhook` from the Settings page
- [ ] Check Railway logs for `NOTIFIER=dump WEBHOOK=DISCORD_WEBHOOK_DUMPS`

---

## 5. Troubleshooting

### `GET /api/opportunities` returns `count: 0`

1. Check worker is running: look for `OPP_CACHE_WRITE` in worker logs.
2. If no `OPP_CACHE_WRITE`, the warm loop may have crashed — look for `Cache warm failed`.
3. If `OPP_CACHE_WRITE items=0`, the price collector is running but no items pass
   scoring — check `SCORE_STALE_MAX_MINUTES` and whether the price DB has recent data.
4. If `OPP_CACHE_WRITE` is present with `items > 0` but the API still returns 0, check
   `REDIS_URL` is the same on both API and worker services.

### No dump alerts firing

1. Check `DUMP_TICK end: ... alerts=0` in worker logs — the scan is running but nothing qualifies.
2. Lower `DUMP_V2_MIN_DROP_PCT` (default 4.0) or `DUMP_V2_MIN_SELL_RATIO` (default 0.80).
3. Check `DUMP_SUPPRESS_LOW` — if set to `0`, LOW-confidence alerts are enabled too.
4. Confirm `DISCORD_WEBHOOK_DUMPS` is set and correct.

### MongoDB connection refused

1. Confirm `MONGODB_URL` is the Atlas SRV string, not `localhost`.
2. Atlas → Network Access → ensure Railway IP (or `0.0.0.0/0`) is allowed.
3. `python-dotenv` must be installed for `.env` to be loaded locally.

### Redis TTL causing empty cache

The flip cache TTL defaults to 600 s and the warm interval to 30 s, so there
is a comfortable 10× buffer.  If you see the cache expiring between writes:

- Check `FLIPS_CACHE_TTL_SECONDS` ≥ `FLIPS_CACHE_WARM_INTERVAL_SECONDS` × 3.
- Check `DUMP_MAX_TICK_SECONDS` — a very long scan delays the warm loop.
- Worker logs show `Cache warm complete in Xs`; ensure X < TTL.
