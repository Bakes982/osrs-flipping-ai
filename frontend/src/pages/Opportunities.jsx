import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RefreshCw, Search, TrendingUp, Filter,
  ArrowUpRight, Zap, Shield, BarChart3, Target, AlertTriangle,
} from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function formatVol(n) {
  if (n == null || n === 0) return '—';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}

function relativeTime(isoStr) {
  if (!isoStr) return null;
  try {
    const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
    if (diff < 5)    return 'just now';
    if (diff < 60)   return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  } catch { return null; }
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

/* ── Confidence badge ────────────────────────────────────────────────────── */

function confBadge(conf, score) {
  const c = (conf ?? 0) > 1 ? (conf ?? 0) / 100 : (conf ?? 0);
  const isHigh = c > 0.70 || score >= 70;
  const isMed  = c > 0.50 || score >= 55;
  if (isHigh) return { label: 'HIGH', color: '#22c55e', bg: 'rgba(34,197,94,0.12)' };
  if (isMed)  return { label: 'MED',  color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' };
  return              { label: 'LOW',  color: '#ef4444', bg: 'rgba(239,68,68,0.12)' };
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

/* ── Skeleton loader row ─────────────────────────────────────────────────── */

function SkeletonRow() {
  return (
    <tr style={{ opacity: 0.5 }}>
      {[30, 220, 54, 80, 80, 60, 80, 60, 50, 50, 60, 32].map((w, i) => (
        <td key={i}>
          <div style={{
            height: 14, borderRadius: 4, background: 'var(--bg-secondary)',
            width: w, animation: 'pulse 1.4s ease-in-out infinite',
          }} />
        </td>
      ))}
    </tr>
  );
}

/* ── Score bar ───────────────────────────────────────────────────────────── */

function ScoreBar({ score }) {
  const pct = Math.min(100, score ?? 0);
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
      <td colSpan={12} style={{ padding: 0, background: 'rgba(6,182,212,0.03)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ padding: '16px 20px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20, fontSize: 12 }}>

          {/* Score Breakdown */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--cyan)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Score Breakdown
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

const FILTERS = [
  { key: 'All',            icon: null,       desc: 'All items' },
  { key: 'High Score',     icon: Target,     desc: 'Score ≥ 60' },
  { key: 'High Margin',    icon: TrendingUp, desc: 'Margin > 2%' },
  { key: 'High Liquidity', icon: BarChart3,  desc: 'Vol score > 50' },
  { key: 'Best EV',        icon: Zap,        desc: 'Profit × liquidity' },
  { key: 'Low Risk',       icon: Shield,     desc: 'Risk = LOW' },
];

/* ── Main Component ──────────────────────────────────────────────────────── */

export default function Opportunities() {
  const nav = useNavigate();
  const [filter, setFilter]       = useState('All');
  const [sortCol, setSortCol]     = useState('flip_score');
  const [sortDir, setSortDir]     = useState('desc');
  const [search, setSearch]       = useState('');
  const [minPrice, setMinPrice]   = useState(0);
  const [profile, setProfile]     = useState('balanced');
  const [expandedId, setExpandedId]     = useState(null);
  const [autoRefresh, setAutoRefresh]   = useState(true);

  const { data: raw, loading, error, reload } = useApi(
    () => api.getOpportunities({ limit: 200, min_price: minPrice, profile }),
    [minPrice, profile],
    autoRefresh ? 60_000 : null,  // 60 s auto-refresh, cancellable
  );

  const opps        = raw?.items || raw || [];
  const lastUpdated = raw?.generated_at ? relativeTime(raw.generated_at) : null;

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

    if      (filter === 'High Score')     items = items.filter(o => (o.flip_score ?? 0) >= 60);
    else if (filter === 'High Margin')    items = items.filter(o => (o.margin_pct ?? 0) > 2);
    else if (filter === 'High Liquidity') items = items.filter(o => (o.volume_score ?? o.volume ?? 0) > 50);
    else if (filter === 'Best EV')        items.sort((a, b) =>
      (b.potential_profit ?? 0) * Math.max(1, b.volume_score ?? b.volume ?? 1) -
      (a.potential_profit ?? 0) * Math.max(1, a.volume_score ?? a.volume ?? 1)
    );
    else if (filter === 'Low Risk')       items = items.filter(o => (o.risk_level ?? '').toUpperCase() === 'LOW');

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
    const totalVol   = filtered.reduce((s, o) => s + (o.volume_score ?? o.volume ?? 0), 0);
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
      {/* ── Header ── */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Opportunities</h2>
          <p className="page-subtitle">
            {loading ? 'Loading…' : `${filtered.length} items`}
            {!loading && ` · ranked by ${sortCol === 'flip_score' ? 'score' : sortCol}`}
            {lastUpdated && <span className="text-muted"> · updated {lastUpdated}</span>}
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
          <button className="btn" onClick={reload} disabled={loading}>
            <RefreshCw size={14} style={loading ? { animation: 'spin 1s linear infinite' } : {}} /> Refresh
          </button>
        </div>
      </div>

      {/* ── Summary Cards ── */}
      {(summaryStats || loading) && (
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

      {/* ── Table ── */}
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        {error ? (
          <div className="empty" style={{ color: '#ef4444', padding: '40px 20px' }}>
            <AlertTriangle size={28} style={{ marginBottom: 12 }} />
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Failed to load opportunities</div>
            <div className="text-muted" style={{ fontSize: 12, marginBottom: 16 }}>
              {error.message || 'Connection error'}
            </div>
            <button className="btn" onClick={reload}><RefreshCw size={13} /> Retry</button>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 30 }}>#</th>
                <th style={{ minWidth: 200 }}>Item</th>
                {th('Score',  'flip_score')}
                {th('Buy',    'buy_price')}
                {th('Sell',   'sell_price')}
                {th('Margin', 'margin_pct')}
                {th('Profit', 'potential_profit')}
                {th('ROI',    'roi_pct')}
                {th('Vol',    'volume_score')}
                <th>Trend</th>
                <th>Conf</th>
                <th style={{ width: 32 }}></th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 8 }, (_, i) => <SkeletonRow key={i} />)
                : filtered.length === 0
                ? (
                  <tr>
                    <td colSpan={12}>
                      <div className="empty" style={{ padding: '40px 20px' }}>
                        <Filter size={24} style={{ marginBottom: 12, opacity: 0.4 }} />
                        <div style={{ fontWeight: 600, marginBottom: 6 }}>No opportunities found</div>
                        <div className="text-muted" style={{ fontSize: 12, marginBottom: 16 }}>
                          Try lowering your filters or switching to a different profile
                        </div>
                        <button className="btn" onClick={() => { setFilter('All'); setSearch(''); setMinPrice(0); }}>
                          Clear Filters
                        </button>
                      </div>
                    </td>
                  </tr>
                )
                : filtered.map((opp, i) => {
                    const t    = trendBadge(opp.trend);
                    const cb   = confBadge(opp.ml_confidence, opp.flip_score);
                    const isX  = expandedId === opp.item_id;
                    const profit = opp.potential_profit ?? 0;
                    const margin = opp.margin_pct ?? 0;
                    return [
                      <tr
                        key={`row-${opp.item_id ?? i}`}
                        onClick={() => setExpandedId(isX ? null : opp.item_id)}
                        style={isX ? { background: 'rgba(6,182,212,0.04)', cursor: 'pointer' } : { cursor: 'pointer' }}
                      >
                        <td className="text-muted" style={{ fontSize: 11 }}>{i + 1}</td>

                        {/* Item chip */}
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <img
                              src={IMG(opp.item_id)} alt="" width={26} height={26}
                              style={{ imageRendering: 'pixelated', flexShrink: 0, borderRadius: 4 }}
                              onError={e => { e.target.style.display = 'none'; }}
                            />
                            <div>
                              <div style={{ fontWeight: 600, fontSize: 13, lineHeight: 1.2 }}>
                                {opp.name || opp.item_name}
                              </div>
                              <div style={{ display: 'flex', gap: 3, marginTop: 3, flexWrap: 'wrap' }}>
                                {opp.risk_level && (
                                  <Chip
                                    label={opp.risk_level}
                                    color={opp.risk_level === 'LOW' ? '#22c55e' : opp.risk_level === 'MEDIUM' ? '#f59e0b' : '#ef4444'}
                                  />
                                )}
                                {(opp.volume_score ?? 0) >= 65 && <Chip label="Liquid" color="#06b6d4" />}
                                {(opp.flip_score ?? 0) >= 70   && <Chip label="Top"    color="#22c55e" />}
                              </div>
                            </div>
                          </div>
                        </td>

                        <td><ScorePill score={opp.flip_score} /></td>

                        <td className="gp" style={{ color: 'var(--green)' }}>{formatGP(opp.buy_price)}</td>
                        <td className="gp" style={{ color: 'var(--cyan)' }}>{formatGP(opp.sell_price)}</td>

                        <td className="gp" style={{ color: margin > 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                          {margin > 0 ? '+' : ''}{margin.toFixed(1)}%
                        </td>

                        <td className="gp" style={{ color: profit > 0 ? 'var(--green)' : 'var(--text-secondary)', fontWeight: 600 }}>
                          {profit > 0 ? '+' : ''}{formatGP(profit)}
                        </td>

                        <td className="gp" style={{ color: (opp.roi_pct ?? 0) > 0 ? 'var(--green)' : 'var(--text-secondary)' }}>
                          {opp.roi_pct != null ? `${opp.roi_pct.toFixed(1)}%` : '—'}
                        </td>

                        <td className="gp text-muted">{formatVol(opp.volume_score ?? opp.volume)}</td>

                        <td>
                          <span style={{ color: t.color, fontWeight: 700, fontSize: 12 }} title={t.label}>
                            {t.icon}
                          </span>
                        </td>

                        <td>
                          <span style={{
                            display: 'inline-block', padding: '2px 6px', borderRadius: 4,
                            background: cb.bg, color: cb.color,
                            fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
                          }}>
                            {cb.label}
                          </span>
                        </td>

                        <td>
                          <button
                            className="btn"
                            style={{ padding: '3px 7px', fontSize: 11 }}
                            onClick={e => { e.stopPropagation(); nav(`/item/${opp.item_id}`); }}
                            title="View full analysis"
                          >
                            <ArrowUpRight size={12} />
                          </button>
                        </td>
                      </tr>,
                      isX && <ExpandedDetail key={`detail-${opp.item_id ?? i}`} opp={opp} />,
                    ];
                  })
              }
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
