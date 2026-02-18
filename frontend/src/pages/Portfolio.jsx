import { RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

export default function Portfolio({ prices }) {
  const nav = useNavigate();
  const { data: portfolio, loading, reload } = useApi(() => api.getPortfolio(), [], 120000);
  const { data: trades } = useApi(() => api.getTrades({ limit: 50 }), [], 120000);

  // Holdings come from backend as "holdings" array
  const holdings = portfolio?.holdings || portfolio?.investments || [];
  const totalInvested = holdings.reduce((s, h) => s + (h.total_cost || (h.buy_price * h.quantity) || 0), 0);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Portfolio</h2>
          <p className="page-subtitle">Track holdings and trade history</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn" onClick={() => nav('/import')}>Import CSV</button>
          <button className="btn" onClick={reload}><RefreshCw size={14} /> Refresh</button>
        </div>
      </div>

      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Active Holdings</div>
          <div className="card-value">{holdings.length}</div>
        </div>
        <div className="card">
          <div className="card-title">Total Invested</div>
          <div className="card-value text-cyan">{formatGP(totalInvested)}</div>
        </div>
        <div className="card">
          <div className="card-title">Cash</div>
          <div className="card-value text-green">{formatGP(portfolio?.cash || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Trades Recorded</div>
          <div className="card-value">{trades?.length || 0}</div>
        </div>
      </div>

      {/* Holdings */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Active Holdings</h3>
        {loading ? <div className="loading">Loading...</div> : (
          !holdings.length ? (
            <div className="empty">No active holdings — import your trades CSV to populate</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Qty</th>
                  <th>Buy Price</th>
                  <th>Total Cost</th>
                  <th>Account</th>
                  <th>Bought At</th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((h, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 8 }}>
                      {h.item_id > 0 && (
                        <img
                          src={`https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${h.item_id}`}
                          alt="" width={24} height={24}
                          style={{ imageRendering: 'pixelated' }}
                          onError={e => { e.target.style.display = 'none'; }}
                        />
                      )}
                      {h.item_name}
                    </td>
                    <td>{h.quantity?.toLocaleString()}</td>
                    <td className="gp">{formatGP(h.buy_price)}</td>
                    <td className="gp">{formatGP(h.total_cost || (h.buy_price * h.quantity))}</td>
                    <td className="text-muted">{h.player || '—'}</td>
                    <td className="text-muted">
                      {h.bought_at ? new Date(h.bought_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        )}
      </div>

      {/* Recent Trades */}
      <div className="card">
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Recent Trades</h3>
        {!trades?.length ? (
          <div className="empty">No trades recorded — <a href="/import" style={{ color: 'var(--cyan)' }}>import your CSV</a></div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Item</th>
                <th>Type</th>
                <th>Qty</th>
                <th>Price</th>
                <th>Total</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 30).map((t, i) => (
                <tr key={i}>
                  <td className="text-muted">{t.timestamp ? new Date(t.timestamp).toLocaleString() : '—'}</td>
                  <td style={{ fontWeight: 500 }}>{t.item_name}</td>
                  <td>
                    <span className={`badge ${t.trade_type === 'BUY' ? 'badge-green' : 'badge-red'}`}>
                      {t.trade_type}
                    </span>
                  </td>
                  <td>{t.quantity?.toLocaleString()}</td>
                  <td className="gp">{formatGP(t.price)}</td>
                  <td className="gp">{formatGP(t.total_value)}</td>
                  <td className="text-muted" style={{ fontSize: 11 }}>{t.source || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
