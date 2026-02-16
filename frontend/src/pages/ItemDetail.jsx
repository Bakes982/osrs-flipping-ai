import { useParams, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
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

export default function ItemDetail({ prices }) {
  const { itemId } = useParams();
  const nav = useNavigate();
  const [horizon, setHorizon] = useState('5m');

  const { data: detail, loading } = useApi(
    () => api.getOpportunityDetail(itemId),
    [itemId],
    10000,
  );
  const { data: predictions } = useApi(
    () => api.getPredictions(itemId),
    [itemId],
    10000,
  );

  if (loading) return <div className="loading">Loading item analysis...</div>;
  if (!detail) return <div className="empty">Item not found</div>;

  const pred = predictions?.predictions?.[horizon];
  const suggested = predictions?.suggested_action;

  // Mock price history chart data (will be replaced with real data from API)
  const chartData = detail.price_history?.map((p, i) => ({
    time: i,
    buy: p.instant_buy,
    sell: p.instant_sell,
  })) || [];

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="btn" onClick={() => nav(-1)}><ArrowLeft size={16} /></button>
          <div>
            <h2 className="page-title">{detail.name || `Item ${itemId}`}</h2>
            <p className="page-subtitle">ID: {itemId} — {detail.category || 'Unknown category'}</p>
          </div>
        </div>
      </div>

      {/* Price & Margin Cards */}
      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Buy At</div>
          <div className="card-value text-green">{formatGP(detail.buy_at || detail.instant_sell)}</div>
        </div>
        <div className="card">
          <div className="card-title">Sell At</div>
          <div className="card-value text-cyan">{formatGP(detail.sell_at || detail.instant_buy)}</div>
        </div>
        <div className="card">
          <div className="card-title">Profit</div>
          <div className="card-value text-green">+{formatGP(detail.expected_profit)}</div>
          <div className="text-muted" style={{ fontSize: 11 }}>Tax: {formatGP(detail.tax)}</div>
        </div>
        <div className="card">
          <div className="card-title">ROI</div>
          <div className="card-value">{detail.margin_pct?.toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="card-title">Volume (5m)</div>
          <div className="card-value">{detail.volume_5m || 0}</div>
        </div>
        <div className="card">
          <div className="card-title">Buy Limit</div>
          <div className="card-value">{(detail.buy_limit || 0).toLocaleString()}</div>
        </div>
      </div>

      {/* Price Chart */}
      {chartData.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Price History</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <XAxis dataKey="time" stroke="#6b7280" />
              <YAxis stroke="#6b7280" tickFormatter={formatGP} />
              <Tooltip
                contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8 }}
                formatter={(v) => [formatGP(v) + ' GP']}
              />
              <Line type="monotone" dataKey="buy" stroke="#10b981" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="sell" stroke="#06b6d4" dot={false} strokeWidth={2} />
              {pred?.buy && <ReferenceLine y={pred.buy} stroke="#10b981" strokeDasharray="5 5" label="Pred Buy" />}
              {pred?.sell && <ReferenceLine y={pred.sell} stroke="#06b6d4" strokeDasharray="5 5" label="Pred Sell" />}
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
              <div className="card-title">Best Horizon</div>
              <div className="card-value">{suggested.horizon}</div>
            </div>
            <div>
              <div className="card-title">Confidence</div>
              <div className="card-value">{((suggested.confidence || 0) * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
