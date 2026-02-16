import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

export default function Performance() {
  const { data: perf, loading } = useApi(() => api.getPerformance(), [], 60000);

  if (loading) return <div className="loading">Loading performance data...</div>;
  if (!perf) return <div className="empty">No performance data available yet. Start flipping!</div>;

  const profitHistory = perf.profit_history || [];
  const itemPerf = perf.item_performance || [];

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Performance</h2>
          <p className="page-subtitle">Track your flipping performance over time</p>
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
          <div className="card-value">{formatGP(perf.avg_profit || 0)}</div>
        </div>
      </div>

      {/* Profit Over Time */}
      {profitHistory.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Cumulative Profit</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={profitHistory}>
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" tickFormatter={formatGP} />
              <Tooltip
                contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8 }}
                formatter={(v) => [formatGP(v) + ' GP']}
              />
              <Line type="monotone" dataKey="cumulative_profit" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Item Performance */}
      {itemPerf.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 14, marginBottom: 16 }}>Item Performance</h3>
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
                  <td style={{ fontWeight: 500 }}>{item.item_name}</td>
                  <td>{item.flip_count}</td>
                  <td>{(item.win_rate || 0).toFixed(0)}%</td>
                  <td className={item.total_profit >= 0 ? 'text-green' : 'text-red'}>
                    {formatGP(item.total_profit)}
                  </td>
                  <td className="gp">{formatGP(item.avg_profit)}</td>
                  <td className="text-muted">{item.avg_duration_min?.toFixed(0) || '?'}min</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
