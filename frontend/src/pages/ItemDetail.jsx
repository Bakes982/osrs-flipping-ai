import { useParams, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { ArrowLeft, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid, Legend } from 'recharts';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

const HORIZONS = ['1m', '5m', '30m', '2h', '8h', '24h'];

function TrendIcon({ trend }) {
  if (!trend) return <Minus size={14} />;
  if (trend.includes('UP')) return <TrendingUp size={14} style={{ color: 'var(--green)' }} />;
  if (trend.includes('DOWN')) return <TrendingDown size={14} style={{ color: 'var(--red)' }} />;
  return <Minus size={14} style={{ color: 'var(--cyan)' }} />;
}

export default function ItemDetail({ prices }) {
  const { itemId } = useParams();
  const nav = useNavigate();
  const [horizon, setHorizon] = useState('5m');

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

  if (loading) return <div className="loading">Loading item analysis...</div>;
  if (!detail) return <div className="empty">Item not found</div>;

  // Map backend fields
  const itemName = detail.item_name || `Item ${itemId}`;
  const rec = detail.recommendation || {};
  const fs = detail.flip_score || {};
  const hist = detail.history || {};
  const vwap = detail.vwap || {};
  const boll = detail.bollinger || {};
  const pred = predictions?.predictions?.[horizon];
  const suggested = predictions?.suggested_action;
  const posSize = detail.position_sizing;

  // Build VWAP chart data from available VWAP values
  const vwapData = [
    { period: '1m', vwap: vwap['1m'] },
    { period: '5m', vwap: vwap['5m'] },
    { period: '30m', vwap: vwap['30m'] },
    { period: '2h', vwap: vwap['2h'] },
  ].filter(d => d.vwap != null);

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
          <div className="card-title">Profit</div>
          <div className="card-value text-green">+{formatGP(rec.expected_profit)}</div>
          <div className="text-muted" style={{ fontSize: 11 }}>Tax: {formatGP(rec.tax)}</div>
        </div>
        <div className="card">
          <div className="card-title">ROI</div>
          <div className="card-value">{rec.expected_profit_pct != null ? rec.expected_profit_pct.toFixed(2) + '%' : '—'}</div>
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
      {(vwapData.length > 0 || boll.upper) && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Technical Indicators</h3>
          <div className="stats-grid" style={{ marginBottom: 16 }}>
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

      {/* VWAP Chart */}
      {vwapData.length > 1 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>VWAP Trend</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={vwapData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.2)" />
              <XAxis dataKey="period" stroke="#6b7280" />
              <YAxis stroke="#6b7280" tickFormatter={formatGP} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8 }}
                formatter={(v) => [formatGP(v) + ' GP', 'VWAP']}
              />
              <Line type="monotone" dataKey="vwap" stroke="#06b6d4" dot strokeWidth={2} />
              {boll.upper && <ReferenceLine y={boll.upper} stroke="rgba(239,68,68,0.5)" strokeDasharray="5 5" label={{ value: 'BB Upper', fill: '#ef4444', fontSize: 10 }} />}
              {boll.lower && <ReferenceLine y={boll.lower} stroke="rgba(16,185,129,0.5)" strokeDasharray="5 5" label={{ value: 'BB Lower', fill: '#10b981', fontSize: 10 }} />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Flip History Chart */}
      {flipChart.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Recent Flip Performance</h3>
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
        <div className="card" style={{ marginTop: 24 }}>
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
        <div className="card" style={{ marginTop: 24, padding: 0, overflow: 'auto' }}>
          <div style={{ padding: '16px 16px 0' }}>
            <h3 style={{ fontSize: 14 }}>Recent Flips</h3>
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
