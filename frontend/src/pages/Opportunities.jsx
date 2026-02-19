import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RefreshCw, Search, TrendingUp, TrendingDown, Minus, Filter,
  ArrowUpRight, Info, Zap, Shield, BarChart3, Target, AlertTriangle,
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

function scoreColor(score) {
  if (score >= 70) return 'badge-green';
  if (score >= 55) return 'badge-cyan';
  if (score >= 45) return 'badge-yellow';
  return 'badge-red';
}

function trendBadge(trend) {
  if (!trend) return { icon: '►', cls: 'badge-cyan', label: 'Neutral' };
  if (trend === 'STRONG_UP') return { icon: '▲▲', cls: 'badge-green', label: 'Strong Up' };
  if (trend === 'UP') return { icon: '▲', cls: 'badge-green', label: 'Up' };
  if (trend === 'DOWN') return { icon: '▼', cls: 'badge-red', label: 'Down' };
  if (trend === 'STRONG_DOWN') return { icon: '▼▼', cls: 'badge-red', label: 'Strong Down' };
  return { icon: '►', cls: 'badge-cyan', label: 'Neutral' };
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

const FILTERS = [
  { key: 'All',            icon: null,        desc: 'All items' },
  { key: 'High Score',     icon: Target,      desc: 'Score ≥ 60' },
  { key: 'High Margin',    icon: TrendingUp,  desc: 'Margin > 2%' },
  { key: 'High Liquidity', icon: BarChart3,   desc: 'Volume > 10' },
  { key: 'Best EV',        icon: Zap,         desc: 'Profit × Volume' },
  { key: 'Low Risk',       icon: Shield,      desc: 'Stability ≥ 60' },
];

/* ── Score bar mini-component ────────────────────────────────────────────── */

function ScoreBar({ score, label, max = 100 }) {
  const pct = Math.min(100, (score / max) * 100);
  const color = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--cyan)' : 'var(--red)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 80 }}>
      <div style={{ flex: 1, height: 5, borderRadius: 3, background: 'var(--bg-secondary)' }}>
        <div style={{ height: '100%', borderRadius: 3, width: `${pct}%`, background: color, transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontSize: 10, color: 'var(--text-secondary)', width: 24, textAlign: 'right' }}>{score?.toFixed(0)}</span>
    </div>
  );
}

/* ── Expanded Row Detail ─────────────────────────────────────────────────── */

function ExpandedDetail({ opp }) {
  return (
    <tr>
      <td colSpan={12} style={{ padding: 0, background: 'rgba(6,182,212,0.03)' }}>
        <div style={{ padding: '16px 20px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20, fontSize: 12 }}>
          {/* Score Breakdown */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--cyan)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Score Breakdown
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">Spread</span>
                <ScoreBar score={opp.spread_score || 0} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">Volume</span>
                <ScoreBar score={opp.volume_score || 0} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">Freshness</span>
                <ScoreBar score={opp.freshness_score || 0} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">Trend</span>
                <ScoreBar score={opp.trend_score || 0} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">History</span>
                <ScoreBar score={opp.history_score || 0} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">Stability</span>
                <ScoreBar score={opp.stability_score || 0} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">🧠 AI Signal</span>
                <ScoreBar score={opp.ml_signal_score || 0} />
              </div>
              {opp.ml_direction && (
                <div style={{ marginTop: 6, padding: '6px 10px', background: 'rgba(34,197,94,0.06)', borderRadius: 6, fontSize: 12 }}>
                  AI predicts <strong style={{ color: opp.ml_direction === 'up' ? '#22c55e' : opp.ml_direction === 'down' ? '#ef4444' : '#f59e0b' }}>
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
              <div><span className="text-muted">Instant Buy: </span><span className="text-red">{formatGP(opp.instant_buy)}</span></div>
              <div><span className="text-muted">Instant Sell: </span><span className="text-green">{formatGP(opp.instant_sell)}</span></div>
              <div><span className="text-muted">Gross Margin: </span>{formatGP(opp.margin)}</div>
              <div><span className="text-muted">Tax (2%): </span><span className="text-red">{formatGP(opp.tax)}</span></div>
              <div><span className="text-muted">Net Profit: </span><span className="text-green">{formatGP(opp.potential_profit)}</span></div>
              <div><span className="text-muted">ROI: </span>{opp.roi_pct?.toFixed(2)}%</div>
            </div>
          </div>

          {/* Position Sizing */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--cyan)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Position Sizing
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div><span className="text-muted">Kelly: </span>{(opp.position_sizing?.kelly * 100)?.toFixed(1) || 0}%</div>
              <div><span className="text-muted">Max Investment: </span>{formatGP(opp.position_sizing?.max_investment)}</div>
              <div><span className="text-muted">Recommended Qty: </span>{opp.position_sizing?.quantity || 0}</div>
              <div><span className="text-muted">Stop Loss: </span><span className="text-red">{opp.position_sizing?.stop_loss_pct?.toFixed(1)}%</span></div>
              {opp.win_rate != null && <div><span className="text-muted">Historical WR: </span>{opp.win_rate?.toFixed(0)}%</div>}
              {opp.total_flips > 0 && <div><span className="text-muted">Your Flips: </span>{opp.total_flips}</div>}
              {opp.avg_profit != null && <div><span className="text-muted">Avg Profit: </span><span className="text-green">{formatGP(opp.avg_profit)}</span></div>}
            </div>
            <div style={{ marginTop: 10 }}>
              <div className="text-muted" style={{ fontSize: 11, lineHeight: 1.5 }}>{opp.reason}</div>
            </div>
          </div>
        </div>
      </td>
    </tr>
  );
}

/* ── Main Component ──────────────────────────────────────────────────────── */

export default function Opportunities() {
  const nav = useNavigate();
  const [filter, setFilter] = useState('All');
  const [sortCol, setSortCol] = useState('flip_score');
  const [sortDir, setSortDir] = useState('desc');
  const [search, setSearch] = useState('');
  const [minPrice, setMinPrice] = useState(0);
  const [expandedId, setExpandedId] = useState(null);

  const { data: raw, loading, error, reload } = useApi(
    () => api.getOpportunities({ limit: 200, min_profit: minPrice }),
    [minPrice], 120000,
  );

  const opps = raw?.items || raw || [];

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filtered = useMemo(() => {
    let items = [...opps];

    if (search) {
      const q = search.toLowerCase();
      items = items.filter(o => o.name?.toLowerCase().includes(q) || String(o.item_id).includes(q));
    }

    if (filter === 'High Score') items = items.filter(o => o.flip_score >= 60);
    else if (filter === 'High Margin') items = items.filter(o => o.margin_pct > 2);
    else if (filter === 'High Liquidity') items = items.filter(o => o.volume > 10);
    else if (filter === 'Best EV') items.sort((a, b) => (b.potential_profit * (b.volume || 1)) - (a.potential_profit * (a.volume || 1)));
    else if (filter === 'Low Risk') items = items.filter(o => (o.stability_score || 0) >= 60);

    items.sort((a, b) => {
      const av = a[sortCol] ?? 0;
      const bv = b[sortCol] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });

    return items;
  }, [opps, filter, sortCol, sortDir, search]);

  // Market summary stats
  const summaryStats = useMemo(() => {
    if (!filtered.length) return null;
    const avgMargin = filtered.reduce((s, o) => s + (o.margin_pct || 0), 0) / filtered.length;
    const avgScore = filtered.reduce((s, o) => s + (o.flip_score || 0), 0) / filtered.length;
    const totalProfit = filtered.reduce((s, o) => s + (o.potential_profit || 0), 0);
    const totalVol = filtered.reduce((s, o) => s + (o.volume || 0), 0);
    const best = filtered.reduce((best, o) => (o.potential_profit || 0) > (best.potential_profit || 0) ? o : best, filtered[0]);
    return { avgMargin, avgScore, totalProfit, totalVol, best };
  }, [filtered]);

  const th = (label, col, width) => (
    <th className={sortCol === col ? 'sorted' : ''} onClick={() => toggleSort(col)} style={width ? { width } : {}}>
      {label} {sortCol === col && (sortDir === 'asc' ? '▲' : '▼')}
    </th>
  );

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Opportunities</h2>
          <p className="page-subtitle">
            {filtered.length} items · ranked by{' '}
            {sortCol === 'flip_score' ? 'flip score' : sortCol === 'potential_profit' ? 'profit' : sortCol}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn" onClick={reload}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      {summaryStats && (
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', marginBottom: 20 }}>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Items Shown</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{filtered.length}</div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Avg Score</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{summaryStats.avgScore.toFixed(0)}<span className="text-muted" style={{ fontSize: 12 }}>/100</span></div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Avg Margin</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{summaryStats.avgMargin.toFixed(1)}%</div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Total Volume</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{summaryStats.totalVol.toLocaleString()}</div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Best Opportunity</div>
            <div style={{ fontSize: 14, fontWeight: 700 }} className="text-green">
              {summaryStats.best?.name} (+{formatGP(summaryStats.best?.potential_profit)})
            </div>
          </div>
        </div>
      )}

      {/* Filters Row */}
      <div className="filter-bar" style={{ alignItems: 'center' }}>
        {FILTERS.map(f => (
          <button key={f.key}
            className={`pill ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
            title={f.desc}
          >
            {f.icon && <f.icon size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />}
            {f.key}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search items..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                padding: '7px 14px 7px 30px', borderRadius: 20, border: '1px solid var(--border)',
                background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12, width: 200,
              }}
            />
          </div>
          <select
            value={minPrice}
            onChange={e => setMinPrice(Number(e.target.value))}
            style={{
              padding: '7px 12px', borderRadius: 20, border: '1px solid var(--border)',
              background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12,
            }}
          >
            <option value={0}>Any price</option>
            <option value={100000}>100K+</option>
            <option value={1000000}>1M+</option>
            <option value={10000000}>10M+</option>
            <option value={50000000}>50M+</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        {loading ? (
          <div className="loading">Scanning market for opportunities...</div>
        ) : error ? (
          <div className="empty" style={{ color: '#ef4444' }}>
            <AlertTriangle size={24} style={{ marginBottom: 8 }} /><br />
            <strong>Failed to load opportunities</strong><br />
            <small className="text-muted">{error.message || 'Connection error'} — auto-retrying</small>
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
                {th('Score', 'flip_score')}
                {th('Buy', 'buy_price')}
                {th('Sell', 'sell_price')}
                {th('Margin', 'margin_pct')}
                {th('Profit', 'potential_profit')}
                {th('ROI', 'roi_pct')}
                {th('Vol', 'volume')}
                <th>Trend</th>
                <th>AI</th>
                <th style={{ width: 30 }}></th>
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
                          <div style={{ fontWeight: 600, fontSize: 13 }}>{opp.name}</div>
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
                    <td className="gp">{opp.margin_pct?.toFixed(1)}%</td>
                    <td className="gp text-green">+{formatGP(opp.potential_profit)}</td>
                    <td className="gp">{opp.roi_pct?.toFixed(1) || '—'}%</td>
                    <td className="gp">{opp.volume || 0}</td>
                    <td><span className={`badge ${t.cls}`} title={t.label}>{t.icon}</span></td>
                    <td>
                      {opp.ml_confidence != null ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <div style={{ width: 36, height: 5, borderRadius: 3, background: 'var(--bg-secondary)' }}>
                            <div style={{
                              height: '100%', borderRadius: 3,
                              width: `${Math.min(100, (opp.ml_confidence || 0) * 100)}%`,
                              background: opp.ml_confidence > 0.7 ? 'var(--green)' : opp.ml_confidence > 0.5 ? 'var(--yellow)' : 'var(--red)',
                            }} />
                          </div>
                          <span className="text-muted" style={{ fontSize: 10 }}>{((opp.ml_confidence || 0) * 100).toFixed(0)}%</span>
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
                  </tr>,
                  isExpanded && <ExpandedDetail key={`detail-${i}`} opp={opp} />,
                ];
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
