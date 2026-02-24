import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RefreshCw, Search, TrendingUp, TrendingDown, Minus, Filter,
  ArrowUpRight, Info, Zap, Shield, BarChart3, Target, AlertTriangle, Check,
} from 'lucide-react';
import { api, API_BASE } from '../api/client';
import { useApi } from '../hooks/useApi';

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

/* ── Score pill ──────────────────────────────────────────────────────────── */

function scoreColor(score) {
  if (score >= 70) return { bg: 'rgba(34,197,94,0.15)',  color: '#22c55e', border: 'rgba(34,197,94,0.3)' };
  if (score >= 55) return { bg: 'rgba(6,182,212,0.15)',  color: '#06b6d4', border: 'rgba(6,182,212,0.3)' };
  if (score >= 40) return { bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: 'rgba(245,158,11,0.3)' };
  return              { bg: 'rgba(239,68,68,0.15)',   color: '#ef4444', border: 'rgba(239,68,68,0.3)' };
}

function ScorePill({ score }) {
  const { bg, color, border } = scoreColor(score ?? 0);
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      minWidth: 38, padding: '2px 7px', borderRadius: 99,
      background: bg, color, border: `1px solid ${border}`,
      fontWeight: 700, fontSize: 13,
    }}>
      {score != null ? score.toFixed(0) : '—'}
    </span>
  );
}

/* ── Trend badge ─────────────────────────────────────────────────────────── */

function trendBadge(trend) {
  if (!trend) return { icon: '─',  color: '#06b6d4', label: 'Neutral' };
  if (trend === 'STRONG_UP')   return { icon: '▲▲', color: '#22c55e', label: 'Strong Up' };
  if (trend === 'UP')          return { icon: '▲',  color: '#22c55e', label: 'Up' };
  if (trend === 'DOWN')        return { icon: '▼',  color: '#ef4444', label: 'Down' };
  if (trend === 'STRONG_DOWN') return { icon: '▼▼', color: '#ef4444', label: 'Strong Down' };
  return { icon: '─', color: '#06b6d4', label: 'Neutral' };
}

/* ── Small inline chip ───────────────────────────────────────────────────── */

function Chip({ label, color }) {
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, letterSpacing: 0.3, textTransform: 'uppercase',
      padding: '1px 5px', borderRadius: 4,
      background: `${color}18`, color, border: `1px solid ${color}40`,
    }}>
      {label}
    </span>
  );
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

const FILTERS = [
  { key: 'All', icon: null, desc: 'All items' },
  { key: 'High Volume', icon: BarChart3, desc: 'Volume ≥ 500' },
  { key: 'High Value 1M+', icon: TrendingUp, desc: 'Buy price ≥ 1M' },
  { key: 'High Value 10M+', icon: TrendingUp, desc: 'Buy price ≥ 10M' },
  { key: 'Low Risk', icon: Shield, desc: 'Stable / calm setups' },
  { key: 'Best EV', icon: Zap, desc: 'Best expected value (profit × volume)' },
];

/* ── Skeleton loader row ─────────────────────────────────────────────────── */


function timeAgo(ts) {
  if (!ts) return 'Never';
  let epoch = Number(ts);
  if (!Number.isFinite(epoch)) {
    const parsed = Date.parse(String(ts));
    if (!Number.isFinite(parsed)) return 'Unknown';
    epoch = parsed / 1000;
  }
  const delta = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (delta < 10) return 'just now';
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  return `${Math.floor(delta / 3600)}h ago`;
}

function LoadingSkeletonRows({ rows = 8 }) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th style={{ width: 30 }}>#</th><th>Item</th><th>RUNE SCORE</th><th>Buy</th><th>Sell</th><th>Margin</th><th>Profit</th><th>ROI</th><th>Vol</th><th>Trend</th><th>AI</th><th style={{ width: 30 }}></th>
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: rows }).map((_, i) => (
          <tr key={`skeleton-${i}`}>
            {Array.from({ length: 12 }).map((__, j) => (
              <td key={`${i}-${j}`}>
                <div style={{ height: 10, borderRadius: 6, background: 'var(--bg-secondary)', opacity: 0.8, width: j === 1 ? '90%' : '70%' }} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}


function DumpSparkline({ dumpPrice, refAvg }) {
  const low = Math.min(dumpPrice || 0, refAvg || 0) || 1;
  const high = Math.max(dumpPrice || 0, refAvg || 0) || 1;
  const norm = (v) => 28 - ((v - low) / Math.max(1, high - low)) * 20;
  const y1 = norm(refAvg || low);
  const y2 = norm(dumpPrice || low);
  return (
    <svg width="90" height="30" viewBox="0 0 90 30" aria-hidden>
      <polyline points={`2,${y1} 45,${(y1 + y2) / 2} 88,${y2}`} fill="none" stroke="var(--cyan)" strokeWidth="2" />
      <circle cx="88" cy={y2} r="2.5" fill="var(--red)" />
    </svg>
  );
}


function riskBadge(opp) {
  const safety = Number(opp.stability_score || 0);
  const score = Number(opp.flip_score || 0);
  if (safety >= 70 && score >= 60) return { label: 'CALM', cls: 'badge-cyan' };
  if (score >= 65) return { label: 'HOT', cls: 'badge-green' };
  return { label: 'SPIKY', cls: 'badge-yellow' };
}

function scoreChips(opp) {
  return [
    `VOLUME ${Math.round(Number(opp.volume_score || 0))}`,
    `MARGIN ${Math.round(Number(opp.spread_score || 0))}`,
    `SAFETY ${Math.round(Number(opp.stability_score || 0))}`,
    `SPEED ${Math.round(Number(opp.freshness_score || 0))}`,
  ];
}

function MiniSparkline({ buyPrice, sellPrice }) {
  const a = Number(buyPrice || 0);
  const b = Number(sellPrice || 0);
  const low = Math.min(a || 1, b || 1);
  const high = Math.max(a || 1, b || 1);
  const n = (v) => 22 - ((v - low) / Math.max(1, high - low)) * 14;
  const y1 = n(a || low);
  const y2 = n((a + b) / 2 || low);
  const y3 = n(b || low);
  return (
    <svg width="68" height="24" viewBox="0 0 68 24" aria-hidden>
      <polyline points={`2,${y1} 34,${y2} 66,${y3}`} fill="none" stroke="var(--cyan)" strokeWidth="2" />
      <circle cx="66" cy={y3} r="2" fill="var(--green)" />
    </svg>
  );
}

/* ── Score bar mini-component ────────────────────────────────────────────── */

function ScoreBar({ score, max = 100 }) {
  const pct = Math.min(100, (score / max) * 100);
  const color = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--cyan)' : 'var(--red)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 80 }}>
      <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'var(--bg-secondary)' }}>
        <div style={{ height: '100%', borderRadius: 2, width: `${pct}%`, background: color, transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontSize: 10, color: 'var(--text-secondary)', width: 24, textAlign: 'right' }}>{pct.toFixed(0)}</span>
    </div>
  );
}

/* ── Expanded Row Detail ─────────────────────────────────────────────────── */

function ExpandedDetail({ opp }) {
  return (
    <tr>
      <td colSpan={13} style={{ padding: 0, background: 'rgba(6,182,212,0.03)' }}>
        <div style={{ padding: '16px 20px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20, fontSize: 12 }}>

          {/* RUNE SCORE Breakdown */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--cyan)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              RUNE SCORE Breakdown
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {[
                ['Spread',    opp.spread_score],
                ['Volume',    opp.volume_score],
                ['Freshness', opp.freshness_score],
                ['Trend',     opp.trend_score],
                ['History',   opp.history_score],
                ['Stability', opp.stability_score],
              ].map(([label, val]) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="text-muted">{label}</span>
                  <ScoreBar score={val || 0} />
                </div>
              ))}
              {opp.ml_direction && (
                <div style={{ marginTop: 6, padding: '6px 10px', background: 'rgba(34,197,94,0.06)', borderRadius: 6 }}>
                  AI predicts{' '}
                  <strong style={{ color: opp.ml_direction === 'up' ? '#22c55e' : opp.ml_direction === 'down' ? '#ef4444' : '#f59e0b' }}>
                    {opp.ml_direction === 'up' ? '▲ Up' : opp.ml_direction === 'down' ? '▼ Down' : '— Flat'}
                  </strong>
                  {opp.ml_prediction_confidence != null && ` (${(opp.ml_prediction_confidence * 100).toFixed(0)}% conf)`}
                  {opp.ml_method && <span className="text-muted"> · {opp.ml_method === 'ml' ? 'ML Model' : 'Statistical'}</span>}
                </div>
              )}
            </div>
          </div>

          {/* Pricing Detail */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--cyan)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Pricing Detail
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div><span className="text-muted">Buy @ </span><span className="text-red">{formatGP(opp.buy_price || opp.recommended_buy)}</span></div>
              <div><span className="text-muted">Sell @ </span><span className="text-green">{formatGP(opp.sell_price || opp.recommended_sell)}</span></div>
              <div><span className="text-muted">Gross Spread: </span>{formatGP(opp.margin || opp.gross_profit)}</div>
              <div><span className="text-muted">GE Tax (2%): </span><span className="text-red">−{formatGP(opp.tax)}</span></div>
              <div><span className="text-muted">Net Profit: </span><span className="text-green">+{formatGP(opp.potential_profit)}</span></div>
              <div><span className="text-muted">ROI: </span><span style={{ color: (opp.roi_pct || 0) > 0 ? 'var(--green)' : 'var(--red)' }}>{opp.roi_pct?.toFixed(2)}%</span></div>
              <div><span className="text-muted">GP/hr est: </span>{formatGP(opp.gp_per_hour)}</div>
              <div><span className="text-muted">Est. fill: </span>{opp.est_fill_time_minutes?.toFixed(0) ?? '—'} min</div>
            </div>
          </div>

          {/* Position Sizing */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--cyan)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Position Sizing
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {opp.trade_plan ? (
                <>
                  <div><span className="text-muted">Qty: </span>{opp.trade_plan.qty_to_buy?.toLocaleString()}</div>
                  <div><span className="text-muted">Max Invest: </span>{formatGP(opp.trade_plan.max_invest_gp)}</div>
                  <div><span className="text-muted">Est Profit: </span><span className="text-green">+{formatGP(opp.trade_plan.total_profit)}</span></div>
                </>
              ) : opp.qty_suggested > 0 ? (
                <>
                  <div><span className="text-muted">Qty Suggested: </span>{opp.qty_suggested?.toLocaleString()}</div>
                  <div><span className="text-muted">Est Profit: </span><span className="text-green">+{formatGP(opp.expected_profit)}</span></div>
                </>
              ) : <div className="text-muted">No sizing data</div>}
              {opp.win_rate != null && <div><span className="text-muted">Win Rate: </span>{(opp.win_rate * 100)?.toFixed(0)}%</div>}
              {opp.total_flips > 0   && <div><span className="text-muted">Your Flips: </span>{opp.total_flips}</div>}
              {opp.avg_profit != null && <div><span className="text-muted">Avg Profit: </span><span className="text-green">{formatGP(opp.avg_profit)}</span></div>}
            </div>
            {opp.reason && (
              <div className="text-muted" style={{ marginTop: 10, fontSize: 11, lineHeight: 1.5, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                {opp.reason}
              </div>
            )}
          </div>
        </div>
      </td>
    </tr>
  );
}

/* ── Filter definitions ──────────────────────────────────────────────────── */

/* ── Main Component ──────────────────────────────────────────────────────── */

export default function Opportunities() {
  const nav = useNavigate();
  const [filter, setFilter] = useState('All');
  const [sortCol, setSortCol] = useState('flip_score');
  const [sortDir, setSortDir] = useState('desc');
  const [search, setSearch] = useState('');
  const [minPrice, setMinPrice] = useState(0);
  const [profile, setProfile] = useState('balanced');
  const [expandedId, setExpandedId] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [replaceForItem, setReplaceForItem] = useState(null);
  const [acceptingId, setAcceptingId] = useState(null);
  const [viewMode, setViewMode] = useState('opportunities');
  const debugEnabled = import.meta.env.DEV || new URLSearchParams(window.location.search).get('debug') === '1';
  const opportunitiesParams = useMemo(
    () => ({ limit: 200, min_price: minPrice, profile }),
    [minPrice, profile],
  );
  const opportunitiesRequestUrl = useMemo(() => {
    const qs = new URLSearchParams(opportunitiesParams).toString();
    return `${API_BASE}/opportunities${qs ? `?${qs}` : ''}`;
  }, [opportunitiesParams]);
  const { data: raw, loading, error, reload } = useApi(
    ({ signal }) => api.getOpportunities(
      opportunitiesParams,
      { signal, timeoutMs: 15000 },
    ),
    [opportunitiesParams],
    autoRefresh ? 60_000 : null,  // 60 s auto-refresh, cancellable
  );
  const { data: tradeData, reload: reloadTrades } = useApi(
    () => api.getActiveTrades(),
    [], 10000,
  );
  const { data: dumpsRaw, loading: dumpsLoading, error: dumpsError, reload: reloadDumps } = useApi(
    () => api.getDumps(),
    [], 120000,
  );

  const firstOpportunityExample = useMemo(() => {
    const first = raw?.items?.[0];
    if (!first) return null;
    return {
      name: first.name,
      buy_price: first.buy_price,
      sell_price: first.sell_price,
      volume_5m: first.volume_5m,
      flip_score: first.flip_score,
    };
  }, [raw]);

  const opps = useMemo(() => raw?.items || [], [raw]);
  const dumps = useMemo(() => dumpsRaw?.items || [], [dumpsRaw]);
  const activeMode = raw?.profile || 'balanced';
  const apiCount = Number(raw?.count || 0);
  const lastUpdated = timeAgo(raw?.generated_at);
  const activeTrades = tradeData?.items || [];
  const slotsUsed = tradeData?.slots_used || 0;
  const slotsTotal = tradeData?.slots_total || 8;
  const freeSlots = Math.max(0, tradeData?.free_slots ?? (slotsTotal - slotsUsed));


  const acceptOpportunity = async (opp, replaceTradeId = null, overrides = {}) => {
    try {
      setAcceptingId(opp.item_id);
      await api.acceptTrade({
        item_id: opp.item_id,
        name: opp.name,
        buy_target: opp.buy_price || opp.instant_buy || opp.dump_price || 0,
        sell_target: opp.sell_price || opp.instant_sell || opp.ref_avg || 0,
        qty_target: Math.max(1, opp.position_sizing?.quantity || 1),
        max_invest_gp: Math.max(0, opp.position_sizing?.max_investment || (opp.buy_price || opp.dump_price || 0)),
        type: (opp.dump_signal || '').toLowerCase() === 'high' ? 'dump' : 'normal',
        volume_5m: opp.volume_5m || opp.volume,
        replace_trade_id: replaceTradeId,
        ...overrides,
      });
      setReplaceForItem(null);
      reloadTrades();
      if (viewMode === "dumps") reloadDumps();
    } catch (e) {
      window.alert(e.message || 'Failed to accept trade');
    } finally {
      setAcceptingId(null);
    }
  };

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filtered = useMemo(() => {
    let items = [...opps];

    if (search) {
      const q = search.toLowerCase();
      items = items.filter(o =>
        (o.name || o.item_name)?.toLowerCase().includes(q) || String(o.item_id).includes(q)
      );
    }

    if (filter === 'High Volume') items = items.filter(o => (o.volume_5m || 0) >= 500);
    else if (filter === 'High Value 1M+') items = items.filter(o => (o.buy_price || 0) >= 1_000_000);
    else if (filter === 'High Value 10M+') items = items.filter(o => (o.buy_price || 0) >= 10_000_000);
    else if (filter === 'Best EV') items.sort((a, b) => (b.potential_profit * (b.volume_5m || 1)) - (a.potential_profit * (a.volume_5m || 1)));
    else if (filter === 'Low Risk') items = items.filter(o => (o.stability_score || 0) >= 70);

    items.sort((a, b) => {
      const av = a[sortCol] ?? 0;
      const bv = b[sortCol] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });

    return items;
  }, [opps, filter, sortCol, sortDir, search]);

  const summaryStats = useMemo(() => {
    if (!filtered.length) return null;
    const avgMargin  = filtered.reduce((s, o) => s + (o.margin_pct ?? 0), 0) / filtered.length;
    const avgScore   = filtered.reduce((s, o) => s + (o.flip_score ?? 0), 0) / filtered.length;
    const totalProfit= filtered.reduce((s, o) => s + (o.potential_profit ?? 0), 0);
    const totalVol   = filtered.reduce((s, o) => s + (o.volume_5m ?? 0), 0);
    const best       = filtered.reduce(
      (b, o) => (o.flip_score ?? 0) > (b.flip_score ?? 0) ? o : b,
      filtered[0],
    );
    return { avgMargin, avgScore, totalProfit, totalVol, best };
  }, [filtered]);

  const th = (label, col) => (
    <th
      className={sortCol === col ? 'sorted' : ''}
      onClick={() => toggleSort(col)}
      style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
    >
      {label}{sortCol === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''}
    </th>
  );

  return (
    <div>
      {debugEnabled && (
        <div className="card" style={{ marginBottom: 12, border: '1px solid #f59e0b', background: 'rgba(245,158,11,0.08)' }}>
          <div style={{ padding: 12, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12, lineHeight: 1.5 }}>
            <div style={{ fontWeight: 700, marginBottom: 6, color: '#f59e0b' }}>Debug banner (temporary)</div>
            <div><strong>apiBaseUrl:</strong> {API_BASE}</div>
            <div><strong>opportunitiesRequestUrl:</strong> {opportunitiesRequestUrl}</div>
            <div>
              <strong>response:</strong>{' '}
              generated_at={raw?.generated_at ?? '—'}, count={raw?.count ?? '—'}, profile={raw?.profile ?? '—'}
            </div>
            <div>
              <strong>firstItem:</strong>{' '}
              {firstOpportunityExample ? JSON.stringify(firstOpportunityExample) : '—'}
            </div>
          </div>
        </div>
      )}

      {/* ── Header ── */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Opportunities</h2>
          <p className="page-subtitle">
            {apiCount} items · ranked by{' '}
            {sortCol === 'flip_score' ? 'flip score' : sortCol === 'potential_profit' ? 'profit' : sortCol} · last updated {lastUpdated} · slots {slotsUsed}/{slotsTotal}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select
            value={profile}
            onChange={e => setProfile(e.target.value)}
            style={{ padding: '7px 12px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12 }}
          >
            <option value="conservative">Conservative</option>
            <option value="balanced">Balanced</option>
            <option value="aggressive">Aggressive</option>
          </select>
          <button
            className={`pill ${autoRefresh ? 'active' : ''}`}
            onClick={() => setAutoRefresh(v => !v)}
            title={autoRefresh ? 'Auto-refresh ON (60s) – click to pause' : 'Auto-refresh OFF – click to enable'}
            style={{ fontSize: 11 }}
          >
            {autoRefresh ? '⟳ Live' : '⟳ Paused'}
          </button>
          <button className="btn" onClick={() => (viewMode === 'opportunities' ? reload() : reloadDumps())} disabled={loading}>
            <RefreshCw size={14} style={loading ? { animation: 'spin 1s linear infinite' } : {}} /> Refresh
          </button>
        </div>
      </div>


      <div className="filter-bar" style={{ marginBottom: 12 }}>
        <button className={`pill ${viewMode === 'opportunities' ? 'active' : ''}`} onClick={() => setViewMode('opportunities')}>Opportunities</button>
        <button className={`pill ${viewMode === 'dumps' ? 'active' : ''}`} onClick={() => setViewMode('dumps')}>Dumps</button>
      </div>

      <div className="filter-bar" style={{ marginBottom: 12 }}>
        <span className={`pill active`} style={{ textTransform: 'capitalize' }}>Mode: {activeMode}</span>
      </div>

      {/* Summary Stats */}
      {summaryStats && (
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', marginBottom: 20 }}>
          {[
            { title: 'Items Shown',       value: loading ? '…' : filtered.length,                             color: null },
            { title: 'Avg Score',         value: loading ? '…' : (summaryStats?.avgScore.toFixed(0) + '/100'),color: null },
            { title: 'Avg Margin',        value: loading ? '…' : (summaryStats?.avgMargin.toFixed(1) + '%'),  color: (summaryStats?.avgMargin ?? 0) > 0 ? 'var(--green)' : null },
            { title: 'Total Liq Score',   value: loading ? '…' : summaryStats?.totalVol.toFixed(0),           color: null },
            { title: 'Best Opportunity',  value: loading ? '…' : (summaryStats?.best?.name || summaryStats?.best?.item_name), sub: summaryStats?.best ? `score ${summaryStats.best.flip_score?.toFixed(0)}` : null, color: 'var(--green)' },
          ].map(({ title, value, sub, color }) => (
            <div key={title} className="card" style={{ padding: '14px 16px' }}>
              <div className="card-title">{title}</div>
              <div style={{ fontSize: sub ? 13 : 20, fontWeight: 700, color: color || undefined, marginTop: 4 }}>{value}</div>
              {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
            </div>
          ))}
        </div>
      )}

      {/* ── Filter Bar ── */}
      <div className="filter-bar" style={{ alignItems: 'center', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {FILTERS.map(f => (
          <button
            key={f.key}
            className={`pill ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
            title={f.desc}
          >
            {f.icon && <f.icon size={11} style={{ marginRight: 3, verticalAlign: 'middle' }} />}
            {f.key}
          </button>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search items..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ padding: '7px 14px 7px 30px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12, width: 190 }}
            />
          </div>
          <select
            value={minPrice}
            onChange={e => setMinPrice(Number(e.target.value))}
            style={{ padding: '7px 12px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12 }}
          >
            <option value={0}>Any price</option>
            <option value={100000}>100K+</option>
            <option value={1000000}>1M+</option>
            <option value={10000000}>10M+</option>
            <option value={50000000}>50M+</option>
          </select>
        </div>
      </div>

      {viewMode === 'dumps' ? (
        <div>
          {dumpsLoading ? (
            <div className="card" style={{ padding: 16 }}>Loading dump candidates…</div>
          ) : dumpsError ? (
            <div className="empty" style={{ color: '#ef4444' }}>{dumpsError.message || 'Failed to load dumps'}</div>
          ) : dumps.length === 0 ? (
            <div className="empty">No dump candidates right now.</div>
          ) : (
            <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
              {dumps.map((d) => (
                <div key={d.item_id} className="card" style={{ padding: 14 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontWeight: 700 }}>{d.name}</div>
                    <span className={`badge ${d.stars >= 3 ? 'badge-red' : d.stars === 2 ? 'badge-yellow' : 'badge-cyan'}`}>{'★'.repeat(d.stars || 1)}</span>
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <DumpSparkline dumpPrice={d.dump_price} refAvg={d.ref_avg} />
                    <div style={{ fontSize: 12 }}>
                      <div>Drop: <strong>{d.drop_pct?.toFixed?.(1) ?? d.drop_pct}%</strong></div>
                      <div>Vol: <strong>{(d.volume_5m || 0).toLocaleString()}</strong></div>
                      <div>Est profit: <strong>+{formatGP(d.est_profit)}</strong></div>
                    </div>
                  </div>
                  <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                    {freeSlots > 0 ? (
                      <button className="btn" onClick={() => acceptOpportunity(d, null, { type: 'dump' })}>
                        <Check size={12} /> Accept dump
                      </button>
                    ) : (
                      <select
                        defaultValue=""
                        onChange={(e) => e.target.value && acceptOpportunity(d, e.target.value, { type: 'dump' })}
                        style={{ width: '100%' }}
                      >
                        <option value="" disabled>Replace slot to accept</option>
                        {activeTrades.map(t => (
                          <option key={t.trade_id} value={t.trade_id}>Slot {t.slot_index}: {t.name}</option>
                        ))}
                      </select>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
      /* ── Table ── */
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        {loading ? (
          <div>
            <div className="text-muted" style={{ padding: '10px 14px' }}>Refreshing opportunities…</div>
            <LoadingSkeletonRows rows={8} />
          </div>
        ) : error ? (
          <div className="empty" style={{ color: '#ef4444' }}>
            <AlertTriangle size={24} style={{ marginBottom: 8 }} /><br />
            <strong>Failed to load opportunities</strong><br />
            <small className="text-muted">{error.message || 'Connection error'} — auto-retrying</small>
          </div>
        ) : apiCount === 0 ? (
          <div className="empty">
            <Filter size={24} style={{ marginBottom: 8, opacity: 0.5 }} /><br />
            No opportunities in cache yet.
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty">
            <Filter size={24} style={{ marginBottom: 8, opacity: 0.5 }} /><br />
            No items match your filters. Try adjusting criteria.
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 30 }}>#</th>
                <th>Item</th>
                {th('RUNE SCORE', 'flip_score')}
                {th('Buy', 'buy_price')}
                {th('Sell', 'sell_price')}
                {th('Margin', 'margin_gp')}
                {th('Profit', 'potential_profit')}
                {th('ROI',    'roi_pct')}
                {th('Vol',    'volume_5m')}
                <th>Trend</th>
                <th>AI</th>
                <th style={{ width: 30 }}></th>
                <th style={{ width: 180 }}>Trade</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((opp, i) => {
                const t = trendBadge(opp.trend);
                const isExpanded = expandedId === opp.item_id;
                return [
                  <tr key={`row-${i}`}
                    onClick={() => setExpandedId(isExpanded ? null : opp.item_id)}
                    style={isExpanded ? { background: 'rgba(6,182,212,0.05)' } : {}}
                  >
                    <td className="text-muted">{i + 1}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <img src={IMG(opp.item_id)} alt="" width={28} height={28}
                          style={{ imageRendering: 'pixelated', flexShrink: 0 }}
                          onError={e => { e.target.style.display = 'none'; }} />
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                            {opp.name}
                            <span className={`badge ${riskBadge(opp).cls}`} style={{ fontSize: 10 }}>{riskBadge(opp).label}</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <MiniSparkline buyPrice={opp.buy_price} sellPrice={opp.sell_price} />
                            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                              {scoreChips(opp).map((chip) => (
                                <span key={chip} className="badge badge-cyan" style={{ fontSize: 9 }}>{chip}</span>
                              ))}
                            </div>
                          </div>
                          {opp.win_rate != null && (
                            <div className="text-muted" style={{ fontSize: 10 }}>
                              {opp.total_flips} flips · {opp.win_rate?.toFixed(0)}% WR
                            </div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${scoreColor(opp.flip_score)}`}>
                        {opp.flip_score?.toFixed(0)}
                      </span>
                    </td>
                    <td className="gp text-green">{formatGP(opp.buy_price)}</td>
                    <td className="gp text-cyan">{formatGP(opp.sell_price)}</td>
                    <td className="gp">{formatGP(opp.margin_gp)}</td>
                    <td className="gp text-green">+{formatGP(opp.potential_profit)}</td>
                    <td className="gp">{opp.roi_pct?.toFixed(1) || '—'}%</td>
                    <td className="gp">{opp.volume_5m || 0}</td>
                    <td><span className={`badge ${t.cls}`} title={t.label}>{t.icon}</span></td>
                    <td>
                      {(opp.confidence ?? opp.ml_confidence) != null ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <div style={{ width: 36, height: 5, borderRadius: 3, background: 'var(--bg-secondary)' }}>
                            <div style={{
                              height: '100%', borderRadius: 3,
                              width: `${Math.min(100, ((opp.confidence ?? opp.ml_confidence) || 0) * 100)}%`,
                              background: ((opp.confidence ?? opp.ml_confidence) || 0) > 0.7 ? 'var(--green)' : ((opp.confidence ?? opp.ml_confidence) || 0) > 0.5 ? 'var(--yellow)' : 'var(--red)',
                            }} />
                          </div>
                          <span className="text-muted" style={{ fontSize: 10 }}>{((((opp.confidence ?? opp.ml_confidence) || 0) * 100)).toFixed(0)}%</span>
                        </div>
                      ) : <span className="text-muted">—</span>}
                    </td>
                    <td>
                      <button
                        className="btn"
                        style={{ padding: '4px 8px', fontSize: 11 }}
                        onClick={e => { e.stopPropagation(); nav(`/item/${opp.item_id}`); }}
                        title="View full analysis"
                      >
                        <ArrowUpRight size={12} />
                      </button>
                    </td>

                        <td>
                          {freeSlots > 0 ? (
                            <button
                              className="btn"
                              disabled={acceptingId === opp.item_id}
                              onClick={e => { e.stopPropagation(); acceptOpportunity(opp); }}
                              style={{ fontSize: 11 }}
                            >
                              <Check size={12} /> Accept
                            </button>
                          ) : (
                            <div>
                              <button
                                className="btn"
                                disabled={acceptingId === opp.item_id}
                                onClick={e => { e.stopPropagation(); setReplaceForItem(replaceForItem === opp.item_id ? null : opp.item_id); }}
                                style={{ fontSize: 11 }}
                              >
                                Replace slot
                              </button>
                              {replaceForItem === opp.item_id && (
                                <select
                                  onClick={e => e.stopPropagation()}
                                  onChange={(e) => e.target.value && acceptOpportunity(opp, e.target.value)}
                                  defaultValue=""
                                  style={{ marginTop: 6, width: '100%', fontSize: 11 }}
                                >
                                  <option value="" disabled>Select trade to replace</option>
                                  {activeTrades.map(t => (
                                    <option key={t.trade_id} value={t.trade_id}>Slot {t.slot_index}: {t.name}</option>
                                  ))}
                                </select>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>,
                  isExpanded && <ExpandedDetail key={`detail-${opp.item_id ?? i}`} opp={opp} />,
                ];
              })}
            </tbody>
          </table>
        )}
      </div>
      )}
    </div>
  );
}
