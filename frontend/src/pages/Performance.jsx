import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid } from 'recharts';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { useAccount } from '../hooks/useAccount';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

export default function Performance() {
  const nav = useNavigate();
  const { activeAccount } = useAccount();
  const { data: perf, loading, error, reload } = useApi(
    () => api.getPerformance(activeAccount), [activeAccount], 120000,
  );

  if (loading) return <div className="loading">Loading performance data...</div>;
  if (error) return (
    <div className="empty" style={{ color: '#ef4444' }}>
      <AlertTriangle size={24} style={{ marginBottom: 8 }} /><br />
      <strong>Failed to load performance data</strong><br />
      <small className="text-muted">{error.message || 'Connection error'}</small><br /><br />
      <button className="btn" onClick={reload}><RefreshCw size={14} /> Retry</button>
    </div>
  );
  if (!perf || perf.total_flips === 0) return (
    <div className="empty">
      No performance data available yet.
      <br /><br />
      <button className="btn" onClick={() => nav('/import')} style={{ background: 'var(--cyan)', color: '#000', padding: '10px 20px' }}>
        Import Your Trade History
      </button>
    </div>
  );

  // Map backend profit_history (time, profit) to chart format
  const profitHistory = (perf.profit_history || []).map((p, i) => ({
    idx: i + 1,
    date: p.time ? new Date(p.time).toLocaleDateString() : `#${i + 1}`,
    profit: p.profit,
    item: p.item,
    flip_profit: p.flip_profit,
  }));
  const itemPerf = perf.item_performance || [];

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Performance</h2>
          <p className="page-subtitle">
            {activeAccount
              ? `${activeAccount} — ${perf.total_flips} completed flips`
              : `Your flipping performance from ${perf.total_flips} completed flips`}
          </p>
        </div>
      </div>

      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Total Profit</div>
          <div className={`card-value ${(perf.total_profit || 0) >= 0 ? 'text-green' : 'text-red'}`}>
            {formatGP(perf.total_profit || 0)}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Completed Flips</div>
          <div className="card-value">{perf.total_flips || 0}</div>
        </div>
        <div className="card">
          <div className="card-title">Win Rate</div>
          <div className="card-value">{(perf.win_rate || 0).toFixed(0)}%</div>
        </div>
        <div className="card">
          <div className="card-title">GP/Hour</div>
          <div className="card-value text-cyan">{formatGP(perf.gp_per_hour || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Avg Profit/Flip</div>
          <div className="card-value">{formatGP(perf.avg_profit_per_flip || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Tax Paid</div>
          <div className="card-value text-red">{formatGP(perf.total_tax_paid || 0)}</div>
        </div>
      </div>

      {/* Best / Worst Flip */}
      {(perf.best_flip || perf.worst_flip) && (
        <div className="stats-grid" style={{ marginBottom: 24 }}>
          {perf.best_flip && (
            <div className="card" style={{ borderColor: 'rgba(16,185,129,0.3)' }}>
              <div className="card-title" style={{ color: 'var(--green)' }}>Best Flip</div>
              <div style={{ fontWeight: 500, fontSize: 14 }}>{perf.best_flip.item_name}</div>
              <div className="text-green" style={{ fontSize: 18, fontWeight: 600 }}>+{formatGP(perf.best_flip.net_profit)}</div>
              <div className="text-muted" style={{ fontSize: 11 }}>x{perf.best_flip.quantity} @ {perf.best_flip.margin_pct}% margin</div>
            </div>
          )}
          {perf.worst_flip && (
            <div className="card" style={{ borderColor: 'rgba(239,68,68,0.3)' }}>
              <div className="card-title" style={{ color: 'var(--red)' }}>Worst Flip</div>
              <div style={{ fontWeight: 500, fontSize: 14 }}>{perf.worst_flip.item_name}</div>
              <div className="text-red" style={{ fontSize: 18, fontWeight: 600 }}>{formatGP(perf.worst_flip.net_profit)}</div>
              <div className="text-muted" style={{ fontSize: 11 }}>x{perf.worst_flip.quantity} @ {perf.worst_flip.margin_pct}% margin</div>
            </div>
          )}
        </div>
      )}

      {/* Profit Over Time */}
      {profitHistory.length > 2 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Cumulative Profit</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={profitHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.2)" />
              <XAxis dataKey="idx" stroke="#6b7280" label={{ value: 'Flip #', fill: '#6b7280', fontSize: 10, position: 'insideBottom', offset: -5 }} />
              <YAxis stroke="#6b7280" tickFormatter={formatGP} />
              <Tooltip
                contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8 }}
                formatter={(v, name) => [formatGP(v) + ' GP', name === 'profit' ? 'Cumulative' : name]}
                labelFormatter={(idx) => {
                  const p = profitHistory[idx - 1];
                  return p ? `${p.item} (${p.date}) — ${formatGP(p.flip_profit)} GP` : '';
                }}
              />
              <Line type="monotone" dataKey="profit" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Item Performance */}
      {itemPerf.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Performance by Item</h3>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Flips</th>
                  <th>Win Rate</th>
                  <th>Total Profit</th>
                  <th>Avg Profit</th>
                  <th>Avg Duration</th>
                </tr>
              </thead>
              <tbody>
                {itemPerf.map((item, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 8 }}>
                      {item.item_id > 0 && (
                        <img
                          src={`https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${item.item_id}`}
                          alt="" width={24} height={24}
                          style={{ imageRendering: 'pixelated' }}
                          onError={e => { e.target.style.display = 'none'; }}
                        />
                      )}
                      {item.item_name}
                    </td>
                    <td>{item.flip_count}</td>
                    <td>
                      <span className={`badge ${item.win_rate >= 60 ? 'badge-green' : item.win_rate >= 40 ? 'badge-yellow' : 'badge-red'}`}>
                        {(item.win_rate || 0).toFixed(0)}%
                      </span>
                    </td>
                    <td className={item.total_profit >= 0 ? 'text-green' : 'text-red'}>
                      {formatGP(item.total_profit)}
                    </td>
                    <td className="gp">{formatGP(item.avg_profit)}</td>
                    <td className="text-muted">{item.avg_duration_min?.toFixed(0) || '—'}min</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
