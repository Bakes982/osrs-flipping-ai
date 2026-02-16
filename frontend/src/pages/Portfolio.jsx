import { RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

export default function Portfolio({ prices }) {
  const { data: portfolio, loading, reload } = useApi(() => api.getPortfolio(), [], 30000);
  const { data: trades } = useApi(() => api.getTrades({ limit: 50 }), [], 30000);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Portfolio</h2>
          <p className="page-subtitle">Track holdings and trade history</p>
        </div>
        <button className="btn" onClick={reload}><RefreshCw size={14} /> Refresh</button>
      </div>

      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Total Invested</div>
          <div className="card-value">{formatGP(portfolio?.total_invested || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Current Value</div>
          <div className="card-value text-cyan">{formatGP(portfolio?.current_value || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Unrealized P/L</div>
          <div className={`card-value ${(portfolio?.unrealized_pl || 0) >= 0 ? 'text-green' : 'text-red'}`}>
            {formatGP(portfolio?.unrealized_pl || 0)}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Realized P/L</div>
          <div className={`card-value ${(portfolio?.realized_pl || 0) >= 0 ? 'text-green' : 'text-red'}`}>
            {formatGP(portfolio?.realized_pl || 0)}
          </div>
        </div>
      </div>

      {/* Holdings */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Holdings</h3>
        {loading ? <div className="loading">Loading...</div> : (
          !portfolio?.investments?.length ? (
            <div className="empty">No active holdings</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Qty</th>
                  <th>Buy Price</th>
                  <th>Current</th>
                  <th>P/L</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.investments.map((inv, i) => {
                  const pl = ((inv.current_price || inv.buy_price) - inv.buy_price) * inv.quantity;
                  return (
                    <tr key={i}>
                      <td style={{ fontWeight: 500 }}>{inv.item_name}</td>
                      <td>{inv.quantity?.toLocaleString()}</td>
                      <td className="gp">{formatGP(inv.buy_price)}</td>
                      <td className="gp">{formatGP(inv.current_price || inv.buy_price)}</td>
                      <td className={pl >= 0 ? 'text-green' : 'text-red'}>{formatGP(pl)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )
        )}
      </div>

      {/* Recent Trades */}
      <div className="card">
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Recent Trades</h3>
        {!trades?.length ? <div className="empty">No trades recorded yet</div> : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Item</th>
                <th>Type</th>
                <th>Qty</th>
                <th>Price</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 30).map((t, i) => (
                <tr key={i}>
                  <td className="text-muted">{new Date(t.timestamp).toLocaleTimeString()}</td>
                  <td style={{ fontWeight: 500 }}>{t.item_name}</td>
                  <td>
                    <span className={`badge ${t.trade_type === 'BUY' ? 'badge-green' : 'badge-red'}`}>
                      {t.trade_type}
                    </span>
                  </td>
                  <td>{t.quantity?.toLocaleString()}</td>
                  <td className="gp">{formatGP(t.price)}</td>
                  <td className="gp">{formatGP(t.total_value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
