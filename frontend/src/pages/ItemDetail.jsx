import { useParams, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { ArrowLeft, TrendingUp, TrendingDown, Minus, BarChart3 } from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Legend, ComposedChart,
} from 'recharts';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function formatTime(ts, timestep) {
  const d = new Date(ts * 1000);
  if (timestep === '5m') return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (timestep === '1h') return d.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

const HORIZONS = ['1m', '5m', '30m', '2h', '8h', '24h'];
const GE_TIMESTEPS = [
  { key: '5m',  label: '30h',  desc: '5-min intervals' },
  { key: '1h',  label: '15d',  desc: '1-hour intervals' },
  { key: '6h',  label: '3mo',  desc: '6-hour intervals' },
  { key: '24h', label: '1yr',  desc: 'Daily intervals' },
];

function TrendIcon({ trend }) {
  if (!trend) return <Minus size={14} />;
  if (trend.includes('UP')) return <TrendingUp size={14} style={{ color: 'var(--green)' }} />;
  if (trend.includes('DOWN')) return <TrendingDown size={14} style={{ color: 'var(--red)' }} />;
  return <Minus size={14} style={{ color: 'var(--cyan)' }} />;
}

/* ── Custom Tooltip ──────────────────────────────────────────────────────── */

function PriceTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  return (
    <div style={{
      background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8,
      padding: '10px 14px', fontSize: 12, lineHeight: 1.6,
    }}>
      <div style={{ color: '#9ca3af', marginBottom: 4 }}>
        {new Date(d.timestamp * 1000).toLocaleString()}
      </div>
      {d.high != null && <div><span style={{ color: '#ef4444' }}>Buy (High): </span><span style={{ color: '#f9fafb' }}>{formatGP(d.high)} GP</span></div>}
      {d.low != null && <div><span style={{ color: '#10b981' }}>Sell (Low): </span><span style={{ color: '#f9fafb' }}>{formatGP(d.low)} GP</span></div>}
      {d.high != null && d.low != null && (
        <div><span style={{ color: '#06b6d4' }}>Spread: </span><span style={{ color: '#f9fafb' }}>{formatGP(d.high - d.low)} GP ({((d.high - d.low) / d.low * 100).toFixed(2)}%)</span></div>
      )}
      {(d.highVol > 0 || d.lowVol > 0) && (
        <div style={{ marginTop: 4, borderTop: '1px solid #374151', paddingTop: 4 }}>
          <div><span style={{ color: '#9ca3af' }}>Buy vol: </span>{d.highVol ?? 0}</div>
          <div><span style={{ color: '#9ca3af' }}>Sell vol: </span>{d.lowVol ?? 0}</div>
        </div>
      )}
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────────────────────── */

export default function ItemDetail() {
  const { itemId } = useParams();
  const nav = useNavigate();
  const [horizon, setHorizon] = useState('5m');
  const [geTimestep, setGeTimestep] = useState('1h');

  // Fetch item detail, predictions, and GE price history
  const { data: detail, loading } = useApi(
    () => api.getOpportunityDetail(itemId),
    [itemId],
    60000,
  );
  const { data: predictions } = useApi(
    () => api.getPredictions(itemId),
    [itemId],
    60000,
  );
  const { data: priceHistory, loading: histLoading } = useApi(
    () => api.getPriceHistory(itemId, geTimestep),
    [itemId, geTimestep],
    120000,
  );

  if (loading) return <div className="loading">Loading item analysis...</div>;
  if (!detail) return <div className="empty">Item not found</div>;

  // Map backend fields
  const itemName = detail.item_name || `Item ${itemId}`;
  const rec = detail.recommendation || {};
  const fs = detail.flip_score || {};
  const hist = detail.history || {};
  const vwap = detail.vwap || {};
  const boll = detail.bollinger || {};
  const suggested = predictions?.suggested_action;
  const posSize = detail.position_sizing;

  // Format GE history chart data
  const chartData = (priceHistory?.data || []).map(pt => ({
    ...pt,
    spread: pt.high != null && pt.low != null ? pt.high - pt.low : null,
    totalVol: (pt.highVol || 0) + (pt.lowVol || 0),
    label: formatTime(pt.timestamp, geTimestep),
  }));

  // Compute stats from chart data
  const lastPrice = chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const firstPrice = chartData.length > 0 ? chartData[0] : null;
  const priceChange = (lastPrice?.high && firstPrice?.high)
    ? lastPrice.high - firstPrice.high : null;
  const pctChange = (priceChange != null && firstPrice?.high)
    ? (priceChange / firstPrice.high * 100) : null;
  const avgSpread = chartData.length > 0
    ? chartData.filter(d => d.spread != null).reduce((s, d) => s + d.spread, 0) / (chartData.filter(d => d.spread != null).length || 1)
    : null;
  const totalVol = chartData.reduce((s, d) => s + d.totalVol, 0);
  const maxHigh = chartData.reduce((m, d) => d.high != null ? Math.max(m, d.high) : m, 0);
  const minLow = chartData.reduce((m, d) => d.low != null ? Math.min(m, d.low) : m, Infinity);

  // Build flip history chart
  const flipChart = (detail.recent_flips || []).map((f, i) => ({
    idx: i + 1,
    profit: f.net_profit,
    margin: f.margin_pct,
  }));

  return (
    <div>
      {/* Header with image */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="btn" onClick={() => nav(-1)}><ArrowLeft size={16} /></button>
          <img
            src={`https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${itemId}`}
            alt={itemName}
            width={36}
            height={36}
            style={{ imageRendering: 'pixelated' }}
            onError={e => { e.target.style.display = 'none'; }}
          />
          <div>
            <h2 className="page-title">{itemName}</h2>
            <p className="page-subtitle">
              ID: {itemId}
              {rec.trend && <> — <TrendIcon trend={rec.trend} /> {rec.trend}</>}
              {pctChange != null && (
                <span style={{ marginLeft: 8, color: pctChange >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {pctChange >= 0 ? '▲' : '▼'} {Math.abs(pctChange).toFixed(2)}%
                </span>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Price & Margin Cards */}
      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Buy At</div>
          <div className="card-value text-green">{formatGP(rec.recommended_buy || detail.current_sell)}</div>
        </div>
        <div className="card">
          <div className="card-title">Sell At</div>
          <div className="card-value text-cyan">{formatGP(rec.recommended_sell || detail.current_buy)}</div>
        </div>
        <div className="card">
          <div className="card-title">Spread</div>
          <div className="card-value">{formatGP(detail.current_buy && detail.current_sell ? detail.current_buy - detail.current_sell : null)}</div>
          <div className="text-muted" style={{ fontSize: 11 }}>Tax: {formatGP(rec.tax)}</div>
        </div>
        <div className="card">
          <div className="card-title">Profit</div>
          <div className="card-value" style={{ color: rec.expected_profit >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {rec.expected_profit >= 0 ? '+' : ''}{formatGP(rec.expected_profit)}
          </div>
          <div className="text-muted" style={{ fontSize: 11 }}>{rec.expected_profit_pct != null ? rec.expected_profit_pct.toFixed(2) + '% ROI' : ''}</div>
        </div>
        <div className="card">
          <div className="card-title">Volume (5m)</div>
          <div className="card-value">{detail.volume_5m || 0}</div>
        </div>
        <div className="card">
          <div className="card-title">Score</div>
          <div className="card-value">{fs.total ? `${fs.total}/100` : '—'}</div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          GRAND EXCHANGE PRICE HISTORY
          ═══════════════════════════════════════════════════════════════════ */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart3 size={16} style={{ color: 'var(--cyan)' }} />
            <h3 style={{ fontSize: 14, margin: 0 }}>GE Price History</h3>
          </div>
          <div className="horizon-tabs">
            {GE_TIMESTEPS.map(ts => (
              <button
                key={ts.key}
                className={`horizon-tab ${geTimestep === ts.key ? 'active' : ''}`}
                onClick={() => setGeTimestep(ts.key)}
                title={ts.desc}
              >
                {ts.label}
              </button>
            ))}
          </div>
        </div>

        {/* Stats bar */}
        {chartData.length > 0 && (
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 16, fontSize: 12 }}>
            <div><span className="text-muted">Period High: </span><span className="text-red">{formatGP(maxHigh)}</span></div>
            <div><span className="text-muted">Period Low: </span><span className="text-green">{formatGP(minLow === Infinity ? null : minLow)}</span></div>
            <div><span className="text-muted">Avg Spread: </span><span className="text-cyan">{formatGP(avgSpread)}</span></div>
            <div><span className="text-muted">Total Volume: </span><span>{totalVol.toLocaleString()}</span></div>
            {priceChange != null && (
              <div>
                <span className="text-muted">Change: </span>
                <span style={{ color: priceChange >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {priceChange >= 0 ? '+' : ''}{formatGP(priceChange)}
                </span>
              </div>
            )}
          </div>
        )}

        {histLoading ? (
          <div className="loading" style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            Loading GE history...
          </div>
        ) : chartData.length > 0 ? (
          <>
            {/* Price chart */}
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                <defs>
                  <linearGradient id="spreadFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.15)" />
                <XAxis
                  dataKey="label"
                  stroke="#6b7280"
                  fontSize={10}
                  interval="preserveStartEnd"
                  tick={{ fill: '#6b7280' }}
                />
                <YAxis
                  stroke="#6b7280"
                  fontSize={10}
                  tickFormatter={formatGP}
                  domain={['auto', 'auto']}
                  tick={{ fill: '#6b7280' }}
                />
                <Tooltip content={<PriceTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                  iconType="line"
                />

                {/* Spread area between high and low */}
                <Area
                  type="monotone"
                  dataKey="high"
                  stroke="none"
                  fill="url(#spreadFill)"
                  name="Spread Zone"
                  connectNulls
                  legendType="none"
                />

                {/* Buy price line (instant-buy / high) */}
                <Line
                  type="monotone"
                  dataKey="high"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  name="Buy (High)"
                  connectNulls
                />

                {/* Sell price line (instant-sell / low) */}
                <Line
                  type="monotone"
                  dataKey="low"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  name="Sell (Low)"
                  connectNulls
                />

                {/* Bollinger bands if available */}
                {boll.upper && <ReferenceLine y={boll.upper} stroke="rgba(239,68,68,0.3)" strokeDasharray="5 5" />}
                {boll.lower && <ReferenceLine y={boll.lower} stroke="rgba(16,185,129,0.3)" strokeDasharray="5 5" />}
              </ComposedChart>
            </ResponsiveContainer>

            {/* Volume chart */}
            <div style={{ marginTop: 8 }}>
              <div className="text-muted" style={{ fontSize: 11, marginBottom: 4 }}>Volume</div>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={chartData} margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.1)" />
                  <XAxis dataKey="label" hide />
                  <YAxis stroke="#6b7280" fontSize={10} tick={{ fill: '#6b7280' }} />
                  <Tooltip
                    contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8, fontSize: 12 }}
                    formatter={(v, name) => [v.toLocaleString(), name === 'highVol' ? 'Buy Vol' : 'Sell Vol']}
                    labelFormatter={(_, payload) => payload?.[0]?.payload ? new Date(payload[0].payload.timestamp * 1000).toLocaleString() : ''}
                  />
                  <Bar dataKey="highVol" fill="rgba(239,68,68,0.5)" stackId="vol" name="Buy Vol" />
                  <Bar dataKey="lowVol" fill="rgba(16,185,129,0.5)" stackId="vol" name="Sell Vol" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <div className="empty" style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            No GE price history available for this timestep.
          </div>
        )}
      </div>

      {/* Score Breakdown */}
      {fs.total > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Score Breakdown</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
            {[
              { label: 'Spread', val: fs.spread, max: 100 },
              { label: 'Volume', val: fs.volume, max: 100 },
              { label: 'Freshness', val: fs.freshness, max: 100 },
              { label: 'Trend', val: fs.trend, max: 100 },
              { label: 'History', val: fs.history, max: 100 },
              { label: 'Stability', val: fs.stability, max: 100 },
            ].map(s => (
              <div key={s.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span className="text-muted">{s.label}</span>
                  <span>{s.val?.toFixed(0) ?? '—'}</span>
                </div>
                <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-secondary)' }}>
                  <div style={{
                    height: '100%', borderRadius: 3, width: `${Math.min(100, s.val || 0)}%`,
                    background: (s.val || 0) >= 70 ? 'var(--green)' : (s.val || 0) >= 40 ? 'var(--cyan)' : 'var(--red)',
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VWAP & Bollinger */}
      {(vwap['1m'] || vwap['5m'] || vwap['30m'] || vwap['2h'] || boll.upper) && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Technical Indicators</h3>
          <div className="stats-grid" style={{ marginBottom: boll.upper ? 16 : 0 }}>
            {vwap['1m'] && <div><div className="card-title">VWAP 1m</div><div style={{ fontSize: 16 }}>{formatGP(vwap['1m'])}</div></div>}
            {vwap['5m'] && <div><div className="card-title">VWAP 5m</div><div style={{ fontSize: 16 }}>{formatGP(vwap['5m'])}</div></div>}
            {vwap['30m'] && <div><div className="card-title">VWAP 30m</div><div style={{ fontSize: 16 }}>{formatGP(vwap['30m'])}</div></div>}
            {vwap['2h'] && <div><div className="card-title">VWAP 2h</div><div style={{ fontSize: 16 }}>{formatGP(vwap['2h'])}</div></div>}
          </div>
          {boll.upper && (
            <div className="stats-grid" style={{ marginBottom: 0 }}>
              <div><div className="card-title">BB Upper</div><div style={{ fontSize: 16 }}>{formatGP(boll.upper)}</div></div>
              <div><div className="card-title">BB Middle</div><div style={{ fontSize: 16 }}>{formatGP(boll.middle)}</div></div>
              <div><div className="card-title">BB Lower</div><div style={{ fontSize: 16 }}>{formatGP(boll.lower)}</div></div>
              <div>
                <div className="card-title">BB Position</div>
                <div style={{ fontSize: 16 }}>
                  {boll.position != null ? (
                    <span className={boll.position > 0.8 ? 'text-red' : boll.position < 0.2 ? 'text-green' : ''}>
                      {(boll.position * 100).toFixed(0)}%
                    </span>
                  ) : '—'}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Flip History Chart */}
      {flipChart.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Your Flip Performance</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={flipChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.2)" />
              <XAxis dataKey="idx" stroke="#6b7280" label={{ value: 'Flip #', fill: '#6b7280', fontSize: 10, position: 'insideBottom', offset: -5 }} />
              <YAxis stroke="#6b7280" tickFormatter={formatGP} />
              <Tooltip
                contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8 }}
                formatter={(v, name) => [name === 'profit' ? formatGP(v) + ' GP' : v.toFixed(1) + '%', name === 'profit' ? 'Profit' : 'Margin']}
              />
              <Legend />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
              <Line type="monotone" dataKey="profit" stroke="#10b981" dot strokeWidth={2} name="Profit" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Multi-Horizon Predictions */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 14 }}>Price Predictions</h3>
          <div className="horizon-tabs">
            {HORIZONS.map(h => (
              <button
                key={h}
                className={`horizon-tab ${horizon === h ? 'active' : ''}`}
                onClick={() => setHorizon(h)}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        {predictions?.predictions ? (
          <table className="data-table">
            <thead>
              <tr>
                <th>Horizon</th>
                <th>Predicted Buy</th>
                <th>Predicted Sell</th>
                <th>Direction</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {HORIZONS.map(h => {
                const p = predictions.predictions[h];
                if (!p) return null;
                return (
                  <tr key={h} style={h === horizon ? { background: 'rgba(6,182,212,0.05)' } : {}}>
                    <td style={{ fontWeight: h === horizon ? 600 : 400 }}>{h}</td>
                    <td className="gp text-green">{formatGP(p.buy)}</td>
                    <td className="gp text-cyan">{formatGP(p.sell)}</td>
                    <td>
                      <span className={`badge ${p.direction === 'up' ? 'badge-green' : p.direction === 'down' ? 'badge-red' : 'badge-yellow'}`}>
                        {p.direction === 'up' ? '\u25B2 Up' : p.direction === 'down' ? '\u25BC Down' : '— Flat'}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${p.confidence > 0.7 ? 'badge-green' : p.confidence > 0.5 ? 'badge-yellow' : 'badge-red'}`}>
                        {(p.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <div className="empty">Predictions loading... ML models need 7+ days of data.</div>
        )}
      </div>

      {/* Position Sizing */}
      {posSize && !posSize.error && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Position Sizing (Kelly Criterion)</h3>
          <div className="stats-grid" style={{ marginBottom: 0 }}>
            <div><div className="card-title">Recommended Qty</div><div className="card-value">{posSize.quantity?.toLocaleString() || 0}</div></div>
            <div><div className="card-title">Max Investment</div><div className="card-value text-cyan">{formatGP(posSize.max_investment)}</div></div>
            <div><div className="card-title">Stop Loss</div><div className="card-value text-red">{posSize.stop_loss_pct?.toFixed(1)}%</div></div>
            <div><div className="card-title">Kelly Fraction</div><div className="card-value">{(posSize.kelly_fraction * 100)?.toFixed(1)}%</div></div>
          </div>
          {posSize.warnings?.length > 0 && (
            <div style={{ marginTop: 12, fontSize: 12, color: 'var(--yellow)' }}>
              {posSize.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
            </div>
          )}
        </div>
      )}

      {/* Suggested Action */}
      {suggested && (
        <div className="card" style={{
          marginBottom: 24,
          borderColor: 'rgba(6,182,212,0.3)',
          background: 'linear-gradient(135deg, rgba(6,182,212,0.05), rgba(139,92,246,0.05))',
        }}>
          <h3 style={{ fontSize: 14, marginBottom: 12, color: 'var(--cyan)' }}>AI Suggested Action</h3>
          <div className="stats-grid" style={{ marginBottom: 0 }}>
            <div>
              <div className="card-title">Buy At (undercut)</div>
              <div className="card-value text-green">{formatGP(suggested.buy_at)}</div>
            </div>
            <div>
              <div className="card-title">Sell At (overcut)</div>
              <div className="card-value text-cyan">{formatGP(suggested.sell_at)}</div>
            </div>
            <div>
              <div className="card-title">Expected Profit</div>
              <div className="card-value text-green">+{formatGP(suggested.expected_profit)}</div>
            </div>
            <div>
              <div className="card-title">Trend</div>
              <div className="card-value"><TrendIcon trend={suggested.trend} /> {suggested.trend}</div>
            </div>
            <div>
              <div className="card-title">Confidence</div>
              <div className="card-value">{((suggested.confidence || 0) * 100).toFixed(0)}%</div>
            </div>
          </div>
          {suggested.reason && (
            <div className="text-muted" style={{ marginTop: 12, fontSize: 12 }}>{suggested.reason}</div>
          )}
        </div>
      )}

      {/* Historical Performance */}
      {hist.total_flips > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Historical Performance</h3>
          <div className="stats-grid" style={{ marginBottom: 0 }}>
            <div><div className="card-title">Total Flips</div><div className="card-value">{hist.total_flips}</div></div>
            <div><div className="card-title">Win Rate</div><div className="card-value">{hist.win_rate?.toFixed(0)}%</div></div>
            <div><div className="card-title">Avg Profit</div><div className="card-value text-green">{formatGP(hist.avg_profit)}</div></div>
          </div>
        </div>
      )}

      {/* Recent Flips Table */}
      {detail.recent_flips?.length > 0 && (
        <div className="card" style={{ padding: 0, overflow: 'auto' }}>
          <div style={{ padding: '16px 16px 0' }}>
            <h3 style={{ fontSize: 14 }}>Your Recent Flips</h3>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Buy</th>
                <th>Sell</th>
                <th>Qty</th>
                <th>Profit</th>
                <th>Margin</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {detail.recent_flips.map((f, i) => (
                <tr key={i}>
                  <td className="gp text-green">{formatGP(f.buy_price)}</td>
                  <td className="gp text-cyan">{formatGP(f.sell_price)}</td>
                  <td>{f.quantity}</td>
                  <td className={`gp ${f.net_profit >= 0 ? 'text-green' : 'text-red'}`}>{formatGP(f.net_profit)}</td>
                  <td>{f.margin_pct?.toFixed(1)}%</td>
                  <td className="text-muted">{f.sell_time ? new Date(f.sell_time).toLocaleDateString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
