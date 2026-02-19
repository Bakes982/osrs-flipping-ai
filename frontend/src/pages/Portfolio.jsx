import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, TrendingDown, TrendingUp, AlertTriangle, Eye, X, Trash2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { api, createPriceSocket } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

export default function Portfolio({ prices }) {
  const nav = useNavigate();
  const [sourceFilter, setSourceFilter] = useState('dink'); // 'dink' | 'all'
  const { data: portfolio, loading, reload } = useApi(() => api.getPortfolio(), [], 120000);
  const { data: trades } = useApi(() => api.getTrades({ limit: 50 }), [], 120000);
  const { data: posData, loading: posLoading, reload: reloadPos } = useApi(
    () => api.getActivePositions(sourceFilter === 'all' ? undefined : sourceFilter),
    [sourceFilter],
    60000
  );

  // Live position updates from WebSocket
  const [livePositions, setLivePositions] = useState(null);

  useEffect(() => {
    const socket = createPriceSocket((msg) => {
      if (msg.type === 'position_update' && msg.positions) {
        setLivePositions(msg.positions);
      }
      if (msg.type === 'position_opened' || msg.type === 'position_closed') {
        reloadPos();
      }
    });
    return () => socket.close();
  }, [reloadPos]);

  // Use live data if available, otherwise fall back to API snapshot
  const positions = livePositions || posData?.positions || [];

  const totalInvested = positions.reduce(
    (s, p) => s + (p.buy_price || 0) * (p.quantity || 0), 0
  );
  const totalCurrentValue = positions.reduce(
    (s, p) => s + (p.current_price || p.buy_price || 0) * (p.quantity || 0), 0
  );
  const totalPnL = positions.reduce((s, p) => s + (p.recommended_profit || 0), 0);

  const [actionError, setActionError] = useState(null);

  const handleDismiss = async (e, tradeId) => {
    e.stopPropagation();
    setActionError(null);
    try {
      await api.dismissPosition(tradeId);
      reloadPos();
      setLivePositions(null);
    } catch (err) {
      setActionError(`Failed to dismiss position: ${err.message}`);
    }
  };

  const handleClearCsv = async () => {
    setActionError(null);
    try {
      await api.clearCsvPositions();
      reloadPos();
      setLivePositions(null);
    } catch (err) {
      setActionError(`Failed to clear CSV positions: ${err.message}`);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Portfolio</h2>
          <p className="page-subtitle">Active positions &amp; trade history</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn" onClick={() => nav('/import')}>Import CSV</button>
          <button className="btn" onClick={() => { reload(); reloadPos(); }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Open Positions</div>
          <div className="card-value">{positions.length}</div>
        </div>
        <div className="card">
          <div className="card-title">Total Invested</div>
          <div className="card-value text-cyan">{formatGP(totalInvested)}</div>
        </div>
        <div className="card">
          <div className="card-title">Current Value</div>
          <div className="card-value" style={{ color: totalCurrentValue >= totalInvested ? 'var(--green)' : 'var(--red)' }}>
            {positions.length ? formatGP(totalCurrentValue) : '—'}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Est. Profit (at rec. sell)</div>
          <div className="card-value" style={{ color: totalPnL >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {positions.length ? `${totalPnL >= 0 ? '+' : ''}${formatGP(totalPnL)}` : '—'}
          </div>
        </div>
      </div>

      {/* Error banner */}
      {actionError && (
        <div style={{
          padding: '10px 16px', marginBottom: 16, borderRadius: 8,
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          color: 'var(--red)', fontSize: 13, display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span><AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />{actionError}</span>
          <button onClick={() => setActionError(null)} style={{ background: 'none', border: 'none', color: 'var(--red)', cursor: 'pointer' }}>&times;</button>
        </div>
      )}

      {/* Active Positions (with live pricing) */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Eye size={16} /> Active Positions
            {livePositions && <span className="badge badge-green" style={{ fontSize: 10 }}>LIVE</span>}
          </h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select
              value={sourceFilter}
              onChange={e => { setSourceFilter(e.target.value); setLivePositions(null); }}
              className="input"
              style={{ padding: '4px 8px', fontSize: 12, width: 'auto' }}
            >
              <option value="dink">DINK Only</option>
              <option value="all">All Sources</option>
            </select>
            {sourceFilter === 'all' && positions.some(p => p.source === 'csv_import') && (
              <button className="btn" style={{ fontSize: 11, padding: '4px 10px' }} onClick={handleClearCsv}>
                <Trash2 size={12} /> Clear CSV
              </button>
            )}
          </div>
        </div>
        {posLoading && !positions.length ? <div className="loading">Loading positions...</div> : (
          !positions.length ? (
            <div className="empty">
              No active positions — when DINK sends a BUY confirmation, positions appear here
              with live price tracking and sell recommendations.
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Qty</th>
                  <th>Buy Price</th>
                  <th>Current Price</th>
                  <th>Rec. Sell</th>
                  <th>Change</th>
                  <th>Est. Profit</th>
                  <th style={{ width: 36 }}></th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p, i) => {
                  const pnlPct = p.pnl_pct || 0;
                  const recProfit = p.recommended_profit;
                  const recProfitPct = p.recommended_profit_pct || 0;
                  const isDown = pnlPct < 0;
                  return (
                    <tr key={p.trade_id || i} style={{ cursor: 'pointer' }}
                        onClick={() => p.item_id && nav(`/item/${p.item_id}`)}>
                      <td style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 8 }}>
                        {p.item_id > 0 && (
                          <img src={IMG(p.item_id)} alt="" width={24} height={24}
                               style={{ imageRendering: 'pixelated' }}
                               onError={e => { e.target.style.display = 'none'; }} />
                        )}
                        {p.item_name}
                        {Math.abs(pnlPct) >= 5 && (
                          <AlertTriangle size={14} color="var(--red)" style={{ marginLeft: 4 }} />
                        )}
                      </td>
                      <td>{(p.quantity || 0).toLocaleString()}</td>
                      <td className="gp">{formatGP(p.buy_price)}</td>
                      <td className="gp">
                        {p.current_price ? formatGP(p.current_price) : '—'}
                      </td>
                      <td className="gp" style={{ color: 'var(--cyan)', fontWeight: 600 }}>
                        {p.recommended_sell ? formatGP(p.recommended_sell) : '—'}
                      </td>
                      <td>
                        {p.current_price ? (
                          <span className={`badge ${isDown ? 'badge-red' : 'badge-green'}`}
                                style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                            {isDown ? <TrendingDown size={12} /> : <TrendingUp size={12} />}
                            {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
                          </span>
                        ) : '—'}
                      </td>
                      <td style={{ color: recProfit >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                        {recProfit != null
                          ? `${recProfit >= 0 ? '+' : ''}${formatGP(recProfit)} (${recProfitPct >= 0 ? '+' : ''}${recProfitPct.toFixed(1)}%)`
                          : '—'}
                      </td>
                      <td>
                        <button
                          onClick={(e) => handleDismiss(e, p.trade_id)}
                          title="Dismiss position"
                          style={{
                            background: 'none', border: 'none', cursor: 'pointer',
                            color: 'var(--text-muted)', padding: 4,
                          }}
                        >
                          <X size={14} />
                        </button>
                      </td>
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
        {!trades?.length ? (
          <div className="empty">No trades recorded — <span onClick={() => nav('/import')} style={{ color: 'var(--cyan)', cursor: 'pointer' }}>import your CSV</span></div>
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
