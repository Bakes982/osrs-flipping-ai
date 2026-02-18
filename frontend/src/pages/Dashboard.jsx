import { useNavigate } from 'react-router-dom';
import { useState, useMemo } from 'react';
import {
  RefreshCw, TrendingUp, TrendingDown, Minus, DollarSign, Trophy,
  BarChart3, Target, Zap, Clock, ArrowUpRight, ArrowDownRight, Percent,
  Activity, Award, AlertTriangle, CheckCircle, Flame,
} from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, PieChart, Pie,
} from 'recharts';
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

function trendIcon(trend) {
  if (!trend) return { icon: '►', cls: 'badge-cyan', label: 'Neutral' };
  if (trend === 'STRONG_UP') return { icon: '▲▲', cls: 'badge-green', label: 'Strong Up' };
  if (trend === 'UP') return { icon: '▲', cls: 'badge-green', label: 'Up' };
  if (trend === 'DOWN') return { icon: '▼', cls: 'badge-red', label: 'Down' };
  if (trend === 'STRONG_DOWN') return { icon: '▼▼', cls: 'badge-red', label: 'Strong Down' };
  return { icon: '►', cls: 'badge-cyan', label: 'Neutral' };
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

/* ── Dashboard ───────────────────────────────────────────────────────────── */

export default function Dashboard() {
  const nav = useNavigate();

  const { data: raw, loading: oppsLoading, error: oppsError, reload: reloadOpps } = useApi(
    () => api.getOpportunities({ limit: 20, sort_by: 'total_score' }),
    [], 120000,
  );
  const { data: perf, loading: perfLoading } = useApi(() => api.getPerformance(), [], 120000);
  const { data: portfolio } = useApi(() => api.getPortfolio(), [], 120000);

  const opps = raw?.items || raw || [];

  // Derived stats
  const totalInvested = useMemo(() => {
    return (portfolio?.holdings || []).reduce((s, h) => s + (h.total_cost || 0), 0);
  }, [portfolio]);

  const holdingsCount = portfolio?.holdings?.length || 0;

  // Profit history for sparkline
  const profitHistory = useMemo(() => {
    const ph = perf?.profit_history || [];
    if (ph.length === 0) return [];
    // Group by day for a clean chart
    const byDay = {};
    ph.forEach(p => {
      const day = (p.sell_time || p.date || '').slice(0, 10);
      if (!day) return;
      if (!byDay[day]) byDay[day] = { day, profit: 0, count: 0 };
      byDay[day].profit += p.profit || p.net_profit || 0;
      byDay[day].count += 1;
    });
    return Object.values(byDay).sort((a, b) => a.day.localeCompare(b.day));
  }, [perf]);

  // Top performing items
  const topItems = useMemo(() => {
    return (perf?.item_performance || [])
      .filter(i => i.flip_count >= 2)
      .sort((a, b) => b.total_profit - a.total_profit)
      .slice(0, 5);
  }, [perf]);

  // Worst performing items
  const worstItems = useMemo(() => {
    return (perf?.item_performance || [])
      .filter(i => i.total_profit < 0)
      .sort((a, b) => a.total_profit - b.total_profit)
      .slice(0, 3);
  }, [perf]);

  // Market overview stats from opportunities
  const marketStats = useMemo(() => {
    if (!opps.length) return null;
    const avgMargin = opps.reduce((s, o) => s + (o.margin_pct || 0), 0) / opps.length;
    const avgScore = opps.reduce((s, o) => s + (o.flip_score || 0), 0) / opps.length;
    const totalVolume = opps.reduce((s, o) => s + (o.volume || 0), 0);
    const profitable = opps.filter(o => (o.potential_profit || 0) > 0).length;
    const upTrend = opps.filter(o => o.trend?.includes('UP')).length;
    const downTrend = opps.filter(o => o.trend?.includes('DOWN')).length;
    return { avgMargin, avgScore, totalVolume, profitable, upTrend, downTrend };
  }, [opps]);

  // Cumulative profit for chart
  const cumulativeProfit = useMemo(() => {
    let running = 0;
    return profitHistory.map(d => {
      running += d.profit;
      return { ...d, cumulative: running };
    });
  }, [profitHistory]);

  const bestFlip = perf?.best_flip;
  const worstFlip = perf?.worst_flip;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Dashboard</h2>
          <p className="page-subtitle">Real-time market overview & performance analytics</p>
        </div>
        <button className="btn" onClick={reloadOpps}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* ─── Primary KPI Row ─────────────────────────────────────────── */}
      <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}>
        <div className="card kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="card-title">Total Profit</div>
              <div className={`card-value ${(perf?.total_profit || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                {formatGP(perf?.total_profit || 0)}
              </div>
            </div>
            <div className="kpi-icon kpi-icon-green"><DollarSign size={18} /></div>
          </div>
          <div className="kpi-sub text-muted">
            Tax paid: {formatGP(perf?.total_tax_paid || 0)}
          </div>
        </div>

        <div className="card kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="card-title">Completed Flips</div>
              <div className="card-value">{perf?.total_flips || 0}</div>
            </div>
            <div className="kpi-icon kpi-icon-cyan"><BarChart3 size={18} /></div>
          </div>
          <div className="kpi-sub text-muted">
            Avg: {formatGP(perf?.avg_profit_per_flip || 0)}/flip
          </div>
        </div>

        <div className="card kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="card-title">Win Rate</div>
              <div className="card-value" style={{ color: (perf?.win_rate || 0) >= 70 ? 'var(--green)' : (perf?.win_rate || 0) >= 50 ? 'var(--yellow)' : 'var(--red)' }}>
                {perf?.win_rate?.toFixed(1) || 0}%
              </div>
            </div>
            <div className="kpi-icon kpi-icon-green"><Target size={18} /></div>
          </div>
          <div className="kpi-sub">
            <span className="text-green">{Math.round((perf?.total_flips || 0) * (perf?.win_rate || 0) / 100)}W</span>
            {' / '}
            <span className="text-red">{(perf?.total_flips || 0) - Math.round((perf?.total_flips || 0) * (perf?.win_rate || 0) / 100)}L</span>
          </div>
        </div>

        <div className="card kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="card-title">GP / Hour</div>
              <div className="card-value text-cyan">{formatGP(perf?.gp_per_hour || 0)}</div>
            </div>
            <div className="kpi-icon kpi-icon-cyan"><Clock size={18} /></div>
          </div>
          <div className="kpi-sub text-muted">Effective rate</div>
        </div>

        <div className="card kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="card-title">Active Holdings</div>
              <div className="card-value">{holdingsCount}</div>
            </div>
            <div className="kpi-icon kpi-icon-purple"><Activity size={18} /></div>
          </div>
          <div className="kpi-sub text-muted">
            Invested: {formatGP(totalInvested)}
          </div>
        </div>

        <div className="card kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="card-title">Items Tracked</div>
              <div className="card-value">{perf?.item_performance?.length || 0}</div>
            </div>
            <div className="kpi-icon kpi-icon-yellow"><Zap size={18} /></div>
          </div>
          <div className="kpi-sub text-muted">Unique items flipped</div>
        </div>
      </div>

      {/* ─── Profit Chart + Best/Worst Cards ─────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Cumulative Profit Chart */}
        <div className="card">
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <TrendingUp size={16} style={{ color: 'var(--green)' }} />
            Cumulative Profit
          </h3>
          {cumulativeProfit.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={cumulativeProfit} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                <defs>
                  <linearGradient id="profitGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.12)" />
                <XAxis dataKey="day" stroke="#6b7280" fontSize={10} interval="preserveStartEnd"
                  tickFormatter={v => { const d = new Date(v); return `${d.getDate()}/${d.getMonth()+1}`; }}
                />
                <YAxis stroke="#6b7280" fontSize={10} tickFormatter={formatGP} />
                <Tooltip
                  contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8, fontSize: 12 }}
                  formatter={(v, name) => [formatGP(v) + ' GP', name === 'cumulative' ? 'Total Profit' : 'Daily']}
                  labelFormatter={v => new Date(v).toLocaleDateString()}
                />
                <Area type="monotone" dataKey="cumulative" stroke="#10b981" fill="url(#profitGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty" style={{ padding: 40 }}>Import trade data to see profit chart</div>
          )}
        </div>

        {/* Best & Worst Flips */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card" style={{ flex: 1, borderLeft: '3px solid var(--green)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <Trophy size={16} style={{ color: 'var(--green)' }} />
              <h4 style={{ fontSize: 13, fontWeight: 600 }}>Best Flip</h4>
            </div>
            {bestFlip ? (
              <div onClick={() => nav(`/item/${bestFlip.item_id}`)} style={{ cursor: 'pointer' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <img src={IMG(bestFlip.item_id)} alt="" width={28} height={28}
                    style={{ imageRendering: 'pixelated' }}
                    onError={e => { e.target.style.display = 'none'; }} />
                  <span style={{ fontWeight: 600 }}>{bestFlip.item_name}</span>
                </div>
                <div className="card-value text-green" style={{ fontSize: 20, marginBottom: 4 }}>
                  +{formatGP(bestFlip.net_profit)}
                </div>
                <div className="text-muted" style={{ fontSize: 11 }}>
                  {bestFlip.margin_pct?.toFixed(1)}% margin · qty {bestFlip.quantity}
                </div>
              </div>
            ) : <div className="text-muted" style={{ fontSize: 12 }}>No data yet</div>}
          </div>

          <div className="card" style={{ flex: 1, borderLeft: '3px solid var(--red)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <AlertTriangle size={16} style={{ color: 'var(--red)' }} />
              <h4 style={{ fontSize: 13, fontWeight: 600 }}>Worst Flip</h4>
            </div>
            {worstFlip ? (
              <div onClick={() => nav(`/item/${worstFlip.item_id}`)} style={{ cursor: 'pointer' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <img src={IMG(worstFlip.item_id)} alt="" width={28} height={28}
                    style={{ imageRendering: 'pixelated' }}
                    onError={e => { e.target.style.display = 'none'; }} />
                  <span style={{ fontWeight: 600 }}>{worstFlip.item_name}</span>
                </div>
                <div className="card-value text-red" style={{ fontSize: 20, marginBottom: 4 }}>
                  {formatGP(worstFlip.net_profit)}
                </div>
                <div className="text-muted" style={{ fontSize: 11 }}>
                  {worstFlip.margin_pct?.toFixed(1)}% margin · qty {worstFlip.quantity}
                </div>
              </div>
            ) : <div className="text-muted" style={{ fontSize: 12 }}>No data yet</div>}
          </div>
        </div>
      </div>

      {/* ─── Market Overview Row ─────────────────────────────────────── */}
      {marketStats && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Activity size={16} style={{ color: 'var(--cyan)' }} />
            Live Market Overview
          </h3>
          <div className="stats-grid" style={{ marginBottom: 0, gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))' }}>
            <div>
              <div className="card-title">Avg Margin</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{marketStats.avgMargin.toFixed(1)}%</div>
            </div>
            <div>
              <div className="card-title">Avg Score</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{marketStats.avgScore.toFixed(0)}<span className="text-muted">/100</span></div>
            </div>
            <div>
              <div className="card-title">Total Volume</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{marketStats.totalVolume.toLocaleString()}</div>
            </div>
            <div>
              <div className="card-title">Profitable</div>
              <div className="text-green" style={{ fontSize: 18, fontWeight: 700 }}>{marketStats.profitable}/{opps.length}</div>
            </div>
            <div>
              <div className="card-title">Trending Up</div>
              <div className="text-green" style={{ fontSize: 18, fontWeight: 700 }}>
                <TrendingUp size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} />{marketStats.upTrend}
              </div>
            </div>
            <div>
              <div className="card-title">Trending Down</div>
              <div className="text-red" style={{ fontSize: 18, fontWeight: 700 }}>
                <TrendingDown size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} />{marketStats.downTrend}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── Top Performers + Daily Breakdown ────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Top Earners */}
        <div className="card">
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Flame size={16} style={{ color: 'var(--orange)' }} />
            Top Earners (All Time)
          </h3>
          {topItems.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {topItems.map((item, i) => (
                <div key={i} onClick={() => nav(`/item/${item.item_id}`)}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', borderRadius: 8, cursor: 'pointer', transition: 'background 0.15s' }}
                  className="hover-row"
                >
                  <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-muted)', width: 20 }}>#{i + 1}</span>
                  <img src={IMG(item.item_id)} alt="" width={28} height={28}
                    style={{ imageRendering: 'pixelated', flexShrink: 0 }}
                    onError={e => { e.target.style.display = 'none'; }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.item_name}</div>
                    <div className="text-muted" style={{ fontSize: 11 }}>{item.flip_count} flips · {item.win_rate?.toFixed(0)}% WR</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div className="text-green" style={{ fontWeight: 700, fontSize: 14 }}>+{formatGP(item.total_profit)}</div>
                    <div className="text-muted" style={{ fontSize: 11 }}>{formatGP(item.avg_profit)}/flip</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty" style={{ padding: 30 }}>No data yet — import trades</div>
          )}
        </div>

        {/* Daily Volume Chart */}
        <div className="card">
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart3 size={16} style={{ color: 'var(--purple)' }} />
            Daily Flip Volume
          </h3>
          {profitHistory.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={profitHistory.slice(-30)} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.12)" />
                <XAxis dataKey="day" stroke="#6b7280" fontSize={10} interval="preserveStartEnd"
                  tickFormatter={v => { const d = new Date(v); return `${d.getDate()}/${d.getMonth()+1}`; }}
                />
                <YAxis stroke="#6b7280" fontSize={10} />
                <Tooltip
                  contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => [v, 'Flips']}
                  labelFormatter={v => new Date(v).toLocaleDateString()}
                />
                <Bar dataKey="count" fill="rgba(139, 92, 246, 0.6)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty" style={{ padding: 30 }}>No data yet</div>
          )}
        </div>
      </div>

      {/* ─── Top Opportunities Table ─────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Zap size={16} style={{ color: 'var(--yellow)' }} />
            Top Opportunities
          </h3>
          <button className="btn" onClick={() => nav('/opportunities')} style={{ fontSize: 12 }}>
            View All <ArrowUpRight size={12} />
          </button>
        </div>

        {oppsLoading ? (
          <div className="loading">Scanning market...</div>
        ) : oppsError ? (
          <div className="empty" style={{ color: '#ef4444' }}>
            <AlertTriangle size={20} style={{ marginBottom: 8 }} /><br />
            <strong>Backend connection failed</strong><br />
            <small className="text-muted">Auto-retrying every 30s</small>
          </div>
        ) : !opps?.length ? (
          <div className="empty">No opportunities found. Backend may be scanning...</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 30 }}>#</th>
                  <th>Item</th>
                  <th>Score</th>
                  <th>Buy</th>
                  <th>Sell</th>
                  <th>Margin</th>
                  <th>Profit</th>
                  <th>ROI</th>
                  <th>Vol</th>
                  <th>Trend</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {opps.slice(0, 10).map((opp, i) => {
                  const t = trendIcon(opp.trend);
                  return (
                    <tr key={i} onClick={() => nav(`/item/${opp.item_id}`)}>
                      <td className="text-muted">{i + 1}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <img src={IMG(opp.item_id)} alt="" width={28} height={28}
                            style={{ imageRendering: 'pixelated', flexShrink: 0 }}
                            onError={e => { e.target.style.display = 'none'; }} />
                          <div>
                            <div style={{ fontWeight: 600, fontSize: 13 }}>{opp.name}</div>
                            <div className="text-muted" style={{ fontSize: 10, maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {opp.reason?.split('|')[0]?.trim()}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td><span className={`badge ${scoreColor(opp.flip_score)}`}>{opp.flip_score?.toFixed(0)}</span></td>
                      <td className="gp text-green">{formatGP(opp.buy_price)}</td>
                      <td className="gp text-cyan">{formatGP(opp.sell_price)}</td>
                      <td className="gp">{opp.margin_pct?.toFixed(1)}%</td>
                      <td className="gp text-green">+{formatGP(opp.potential_profit)}</td>
                      <td className="gp">{opp.roi_pct?.toFixed(1) || '—'}%</td>
                      <td className="gp">{opp.volume || 0}</td>
                      <td><span className={`badge ${t.cls}`}>{t.icon}</span></td>
                      <td>
                        {opp.ml_confidence != null ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <div style={{ width: 40, height: 5, borderRadius: 3, background: 'var(--bg-secondary)' }}>
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
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ─── Biggest Losers ──────────────────────────────────────────── */}
      {worstItems.length > 0 && (
        <div className="card" style={{ borderLeft: '3px solid var(--red)' }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <ArrowDownRight size={16} style={{ color: 'var(--red)' }} />
            Biggest Loss Items
          </h3>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {worstItems.map((item, i) => (
              <div key={i} onClick={() => nav(`/item/${item.item_id}`)}
                style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px', borderRadius: 8, cursor: 'pointer', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.15)' }}
              >
                <img src={IMG(item.item_id)} alt="" width={24} height={24}
                  style={{ imageRendering: 'pixelated' }}
                  onError={e => { e.target.style.display = 'none'; }} />
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12 }}>{item.item_name}</div>
                  <div className="text-red" style={{ fontWeight: 700 }}>{formatGP(item.total_profit)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
