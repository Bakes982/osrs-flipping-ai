import { useState, useEffect, useCallback, useRef } from 'react';
import { RefreshCw, TrendingDown, TrendingUp, AlertTriangle, Eye, X, Trash2, Clock, Filter } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { api, createPriceSocket } from '../api/client';
import { useApi } from '../hooks/useApi';
import { useAccount } from '../hooks/useAccount';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function timeAgo(dateStr) {
  if (!dateStr) return '';
  const diff = (Date.now() - new Date(dateStr).getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

export default function Portfolio({ prices }) {
  const nav = useNavigate();
  const { activeAccount } = useAccount();
  const [sourceFilter, setSourceFilter] = useState('dink');
  const [showPending, setShowPending] = useState(false);          // toggle for BUYING/SELLING
  const [lastRefresh, setLastRefresh] = useState(Date.now());     // for "updated X ago"
  const refreshTimer = useRef(null);

  // ---- Active positions (only BOUGHT, unmatched) ----
  const { data: posData, loading: posLoading, reload: reloadPos } = useApi(
    () => api.getActivePositions(sourceFilter === 'all' ? undefined : sourceFilter, activeAccount),
    [sourceFilter, activeAccount],
    30000                                                        // auto-refresh every 30s
  );

  // ---- Trade history (only filled by default) ----
  const { data: trades, reload: reloadTrades } = useApi(
    () => api.getTrades({
      limit: 50,
      ...(activeAccount ? { player: activeAccount } : {}),
      completed_only: showPending ? 'false' : 'true',
    }),
    [activeAccount, showPending], 120000,
  );

  // WebSocket for live position updates
  const [livePositions, setLivePositions] = useState(null);

  useEffect(() => {
    const socket = createPriceSocket((msg) => {
      if (msg.type === 'position_update' && msg.positions) {
        setLivePositions(msg.positions);
        setLastRefresh(Date.now());
      }
      if (msg.type === 'position_opened' || msg.type === 'position_closed') {
        reloadPos();
        reloadTrades();
        setLastRefresh(Date.now());
      }
    });
    return () => socket.close();
  }, [reloadPos, reloadTrades]);

  // Tick the "updated X ago" label every 10s
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 10000);
    return () => clearInterval(id);
  }, []);

  // Use live data if available, otherwise fall back to API snapshot
  const positions = livePositions || posData?.positions || [];

  // ---- Summary stats (only from real filled positions) ----
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
      setActionError(`Failed to dismiss: ${err.message}`);
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

  const handleRefreshAll = () => {
    reloadPos();
    reloadTrades();
    setLivePositions(null);
    setLastRefresh(Date.now());
  };

  return (
    <div>
      {/* ---- Header ---- */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Portfolio</h2>
          <p className="page-subtitle">
            {activeAccount
              ? `${activeAccount} — filled positions & trade history`
              : 'All accounts — filled positions & trade history'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className="text-muted" style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Clock size={12} /> {timeAgo(new Date(lastRefresh).toISOString())}
          </span>
          <button className="btn" onClick={() => nav('/import')}>Import CSV</button>
          <button className="btn" onClick={handleRefreshAll}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* ---- Summary cards ---- */}
      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Filled Positions</div>
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

      {/* ---- Error ---- */}
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

      {/* ================================================================== */}
      {/*  Active Positions — only items actually BOUGHT (filled)             */}
      {/* ================================================================== */}
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
              No filled positions yet.
              <br />
              <span className="text-muted" style={{ fontSize: 12 }}>
                Only items that were actually bought appear here — pending/cancelled offers are excluded.
              </span>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Item</th>
                  {!activeAccount && <th>Account</th>}
                  <th>Qty</th>
                  <th>Buy Price</th>
                  <th>Current</th>
                  <th>Rec. Sell</th>
                  <th>Change</th>
                  <th>Est. Profit</th>
                  <th style={{ width: 80 }}>Bought</th>
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
                      {!activeAccount && (
                        <td className="text-muted" style={{ fontSize: 11 }}>{p.player || '—'}</td>
                      )}
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
                      <td className="text-muted" style={{ fontSize: 11 }}>
                        {timeAgo(p.bought_at)}
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

      {/* ================================================================== */}
      {/*  Trade History — only filled by default, toggle to see pending      */}
      {/* ================================================================== */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, margin: 0 }}>Trade History</h3>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)', cursor: 'pointer' }}>
            <Filter size={12} />
            <input
              type="checkbox"
              checked={showPending}
              onChange={e => setShowPending(e.target.checked)}
              style={{ accentColor: 'var(--cyan)' }}
            />
            Include pending (BUYING / SELLING)
          </label>
        </div>
        {!trades?.length ? (
          <div className="empty">No completed trades — <span onClick={() => nav('/import')} style={{ color: 'var(--cyan)', cursor: 'pointer' }}>import your CSV</span> or wait for DINK webhooks.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Item</th>
                {!activeAccount && <th>Account</th>}
                <th>Type</th>
                <th>Status</th>
                <th>Qty</th>
                <th>Price</th>
                <th>Total</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 30).map((t, i) => {
                const isFilled = t.status === 'BOUGHT' || t.status === 'SOLD';
                return (
                  <tr key={i} style={{ opacity: isFilled ? 1 : 0.5 }}>
                    <td className="text-muted" style={{ fontSize: 12 }}>
                      {t.timestamp ? timeAgo(t.timestamp) : '—'}
                    </td>
                    <td style={{ fontWeight: 500 }}>{t.item_name}</td>
                    {!activeAccount && (
                      <td className="text-muted" style={{ fontSize: 11 }}>{t.player || '—'}</td>
                    )}
                    <td>
                      <span className={`badge ${t.trade_type === 'BUY' ? 'badge-green' : 'badge-red'}`}>
                        {t.trade_type}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${isFilled ? 'badge-green' : 'badge-yellow'}`} style={{ fontSize: 10 }}>
                        {t.status}
                      </span>
                    </td>
                    <td>{t.quantity?.toLocaleString()}</td>
                    <td className="gp">{formatGP(t.price)}</td>
                    <td className="gp">{formatGP(t.total_value)}</td>
                    <td className="text-muted" style={{ fontSize: 11 }}>{t.source || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
