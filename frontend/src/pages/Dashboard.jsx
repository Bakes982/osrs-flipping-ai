import { useNavigate } from 'react-router-dom';
import { RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

export default function Dashboard({ prices }) {
  const nav = useNavigate();
  const { data: opps, loading: oppsLoading, reload: reloadOpps } = useApi(
    () => api.getOpportunities({ limit: 10, sort_by: 'profit' }),
    [],
    30000,
  );
  const { data: perf } = useApi(() => api.getPerformance(), [], 60000);
  const { data: portfolio } = useApi(() => api.getPortfolio(), [], 60000);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Dashboard</h2>
          <p className="page-subtitle">Real-time market overview</p>
        </div>
        <button className="btn" onClick={reloadOpps}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* KPI Cards */}
      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Cash</div>
          <div className="card-value text-green">{formatGP(portfolio?.cash || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Total Profit (24h)</div>
          <div className={`card-value ${(perf?.total_profit || 0) >= 0 ? 'text-green' : 'text-red'}`}>
            {formatGP(perf?.total_profit || 0)}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Flips Today</div>
          <div className="card-value">{perf?.total_flips || 0}</div>
        </div>
        <div className="card">
          <div className="card-title">Win Rate</div>
          <div className="card-value">{perf?.win_rate?.toFixed(0) || 0}%</div>
        </div>
        <div className="card">
          <div className="card-title">GP/Hour</div>
          <div className="card-value text-cyan">{formatGP(perf?.gp_per_hour || 0)}</div>
        </div>
      </div>

      {/* Top Opportunities */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 16, fontWeight: 600 }}>Top Opportunities</h3>
          <button className="btn" onClick={() => nav('/opportunities')}>View All</button>
        </div>

        {oppsLoading ? (
          <div className="loading">Loading opportunities...</div>
        ) : !opps?.length ? (
          <div className="empty">No opportunities found. Backend may be starting up.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Item</th>
                <th>Buy Price</th>
                <th>Sell Price</th>
                <th>Margin</th>
                <th>Profit</th>
                <th>Risk</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {opps.slice(0, 10).map((opp, i) => (
                <tr key={i} onClick={() => nav(`/item/${opp.item_id}`)}>
                  <td style={{ fontWeight: 500 }}>{opp.name}</td>
                  <td className="gp text-green">{formatGP(opp.buy_at)}</td>
                  <td className="gp text-cyan">{formatGP(opp.sell_at)}</td>
                  <td className="gp">{opp.margin_pct?.toFixed(1)}%</td>
                  <td className="gp text-green">+{formatGP(opp.expected_profit)}</td>
                  <td>
                    <span className={`badge ${opp.risk_score <= 4 ? 'badge-green' : opp.risk_score <= 6 ? 'badge-yellow' : 'badge-red'}`}>
                      {opp.risk_score}/10
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${opp.confidence === 'HIGH' ? 'badge-green' : opp.confidence === 'MEDIUM' ? 'badge-yellow' : 'badge-red'}`}>
                      {opp.confidence}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
