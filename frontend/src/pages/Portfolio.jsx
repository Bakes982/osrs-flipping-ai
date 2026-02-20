import { useState, useMemo, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RefreshCw, TrendingUp, ArrowUpRight,
  ShoppingCart, Tag, Clock, AlertTriangle, X,
} from 'lucide-react';
import { api, createPriceSocket } from '../api/client';
import { useApi } from '../hooks/useApi';

/* ── Helpers ───────────────────────────────────────────────────────────────── */

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function elapsed(isoStr) {
  if (!isoStr) return '—';
  const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  const h = Math.floor(diff / 3600);
  const m = Math.floor((diff % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function pnlColor(pct) {
  if (pct == null) return 'var(--text-secondary)';
  if (pct >= 1) return 'var(--green)';
  if (pct >= 0) return 'var(--yellow)';
  return 'var(--red)';
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

/* ── Sub-components ────────────────────────────────────────────────────────── */

function SectionHeader({ icon: Icon, title, count, color = 'var(--cyan)' }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14,
      paddingBottom: 10, borderBottom: '1px solid var(--border)',
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: 8,
        background: `linear-gradient(135deg, ${color}22, ${color}44)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: `1px solid ${color}44`,
      }}>
        <Icon size={16} color={color} />
      </div>
      <span style={{ fontWeight: 700, fontSize: 15 }}>{title}</span>
      {count != null && (
        <span style={{
          background: 'var(--bg-secondary)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '2px 8px', fontSize: 11, color: 'var(--text-secondary)',
        }}>{count}</span>
      )}
    </div>
  );
}

function HoldingCard({ pos, onDismiss, flashId, nav }) {
  const isFlashing = flashId === pos.item_id;
  const pnl = pos.pnl_pct;
  const profit = pos.recommended_profit;

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: `1px solid ${isFlashing ? 'var(--red)' : pnl != null && pnl < -2 ? 'rgba(239,68,68,0.4)' : 'var(--border)'}`,
      borderRadius: 12, padding: '14px 16px',
      display: 'grid',
      gridTemplateColumns: '36px 1fr auto auto auto auto auto',
      alignItems: 'center', gap: 12,
      transition: 'border-color 0.4s',
      boxShadow: isFlashing ? '0 0 12px rgba(239,68,68,0.3)' : 'none',
    }}>
      <img src={IMG(pos.item_id)} alt="" width={32} height={32}
        style={{ imageRendering: 'pixelated' }}
        onError={e => { e.target.style.display = 'none'; }} />

      <div>
        <div style={{ fontWeight: 600, fontSize: 13 }}>{pos.item_name}</div>
        <div className="text-muted" style={{ fontSize: 10 }}>
          {pos.player} · {pos.quantity?.toLocaleString()}×
        </div>
      </div>

      <div style={{ textAlign: 'right' }}>
        <div className="text-muted" style={{ fontSize: 10 }}>Bought</div>
        <div style={{ fontSize: 12, fontWeight: 600 }}>{formatGP(pos.buy_price)}</div>
      </div>

      <div style={{ textAlign: 'right' }}>
        <div className="text-muted" style={{ fontSize: 10 }}>Now</div>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--cyan)' }}>{formatGP(pos.current_price)}</div>
      </div>

      <div style={{ textAlign: 'right' }}>
        <div className="text-muted" style={{ fontSize: 10 }}>Sell at</div>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--green)' }}>{formatGP(pos.recommended_sell)}</div>
      </div>

      <div style={{ textAlign: 'right', minWidth: 72 }}>
        <div className="text-muted" style={{ fontSize: 10 }}>Est. Profit</div>
        <div style={{ fontSize: 12, fontWeight: 700, color: pnlColor(pos.recommended_profit_pct) }}>
          {profit != null ? (profit >= 0 ? '+' : '') + formatGP(profit) : '—'}
          {pos.recommended_profit_pct != null && (
            <span style={{ fontSize: 10, marginLeft: 4 }}>
              ({pos.recommended_profit_pct >= 0 ? '+' : ''}{pos.recommended_profit_pct?.toFixed(1)}%)
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
          <Clock size={10} /> {elapsed(pos.bought_at)}
        </span>
        <button onClick={() => nav(`/item/${pos.item_id}`)} className="btn"
          style={{ padding: '4px 7px', fontSize: 10 }} title="View chart">
          <ArrowUpRight size={11} />
        </button>
        <button onClick={() => onDismiss(pos.trade_id)} className="btn"
          style={{ padding: '4px 7px', fontSize: 10, color: 'var(--red)' }} title="Dismiss">
          <X size={11} />
        </button>
      </div>
    </div>
  );
}

function SellOfferCard({ sell, alertMap, nav }) {
  const alert = alertMap[sell.item_id];
  const isAlerting = !!alert;
  const dropPct = alert?.drop_pct;

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: `1px solid ${isAlerting ? 'rgba(239,68,68,0.6)' : 'rgba(245,158,11,0.3)'}`,
      borderRadius: 12, padding: '14px 16px',
      display: 'grid',
      gridTemplateColumns: '36px 1fr auto auto auto auto auto',
      alignItems: 'center', gap: 12,
      transition: 'border-color 0.4s',
      boxShadow: isAlerting ? '0 0 14px rgba(239,68,68,0.25)' : 'none',
    }}>
      <img src={IMG(sell.item_id)} alt="" width={32} height={32}
        style={{ imageRendering: 'pixelated' }}
        onError={e => { e.target.style.display = 'none'; }} />

      <div>
        <div style={{ fontWeight: 600, fontSize: 13 }}>{sell.item_name}</div>
        <div className="text-muted" style={{ fontSize: 10 }}>
          {sell.player} · {sell.quantity?.toLocaleString()}×
        </div>
      </div>

      <div style={{ textAlign: 'right' }}>
        <div className="text-muted" style={{ fontSize: 10 }}>Listed at</div>
        <div style={{ fontSize: 12, fontWeight: 600 }}>{formatGP(sell.listed_sell_price)}</div>
      </div>

      <div style={{ textAlign: 'right' }}>
        <div className="text-muted" style={{ fontSize: 10 }}>Market now</div>
        <div style={{ fontSize: 12, fontWeight: 600, color: isAlerting ? 'var(--red)' : 'var(--cyan)' }}>
          {alert ? formatGP(alert.current_market_price) : '—'}
        </div>
      </div>

      <div style={{ textAlign: 'right' }}>
        {isAlerting ? (
          <span className="badge badge-red" style={{ fontSize: 11 }}>▼ {dropPct?.toFixed(1)}% below</span>
        ) : (
          <span className="badge badge-yellow" style={{ fontSize: 11 }}>Monitoring</span>
        )}
      </div>

      <div style={{ textAlign: 'right', minWidth: 80 }}>
        {alert?.suggested_relist ? (
          <>
            <div className="text-muted" style={{ fontSize: 10 }}>Re-list at</div>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--yellow)' }}>
              {formatGP(alert.suggested_relist)}
            </div>
          </>
        ) : <span />}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
          <Clock size={10} /> {elapsed(sell.listed_at)}
        </span>
        <button onClick={() => nav(`/item/${sell.item_id}`)} className="btn"
          style={{ padding: '4px 7px', fontSize: 10 }} title="View chart">
          <ArrowUpRight size={11} />
        </button>
      </div>
    </div>
  );
}

/* ── Main Component ────────────────────────────────────────────────────────── */

export default function Portfolio() {
  const nav = useNavigate();
  const [sourceFilter, setSourceFilter] = useState('');
  const [activeAccount, setActiveAccount] = useState('');
  const [flashId, setFlashId] = useState(null);
  const [sellAlertMap, setSellAlertMap] = useState({});
  const wsRef = useRef(null);

  const { data: posData, loading: posLoading, reload: reloadPos } = useApi(
    () => api.getActivePositions(sourceFilter || undefined, activeAccount || undefined),
    [sourceFilter, activeAccount],
    5000,
  );

  const { data: sellData, reload: reloadSells } = useApi(
    () => api.getSellOffers(activeAccount || undefined),
    [activeAccount],
    10000,
  );

  const { data: tradeData, loading: tradeLoading } = useApi(
    () => api.getTrades({ limit: 50, completed_only: 'true' }),
    [],
    120000,
  );

  const positions = posData?.positions || posData || [];
  const sells = sellData?.sells || [];
  const trades = tradeData?.trades || tradeData || [];

  useEffect(() => {
    const { socket, close } = createPriceSocket((msg) => {
      if (msg.type === 'position_update') {
        reloadPos();
      } else if (msg.type === 'position_opened' || msg.type === 'position_closed') {
        reloadPos();
        reloadSells();
      } else if (msg.type === 'selling_price_alert') {
        const map = {};
        for (const a of (msg.alerts || [])) {
          map[a.item_id] = a;
          setFlashId(a.item_id);
          setTimeout(() => setFlashId(null), 4000);
        }
        setSellAlertMap(prev => ({ ...prev, ...map }));
      }
    });
    wsRef.current = socket;
    return () => close();
  }, []);

  const handleDismiss = async (tradeId) => {
    try {
      await api.dismissPosition(tradeId);
      reloadPos();
    } catch (e) {
      console.error('Dismiss failed', e);
    }
  };

  const stats = useMemo(() => {
    if (!positions.length) return null;
    const totalCost = positions.reduce((s, p) => s + (p.buy_price * p.quantity || 0), 0);
    const totalProfit = positions.reduce((s, p) => s + (p.recommended_profit || 0), 0);
    const avgPnl = positions.reduce((s, p) => s + (p.pnl_pct || 0), 0) / positions.length;
    return { totalCost, totalProfit, avgPnl, count: positions.length };
  }, [positions]);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Active Flips</h2>
          <p className="page-subtitle">
            Live positions · updates every 5s
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginLeft: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--green)', display: 'inline-block', animation: 'pulse 2s infinite' }} />
              Live
            </span>
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select
            value={sourceFilter}
            onChange={e => setSourceFilter(e.target.value)}
            style={{ padding: '7px 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12 }}
          >
            <option value="">All sources</option>
            <option value="dink">DINK only</option>
            <option value="csv_import">CSV import</option>
          </select>
          <button className="btn" onClick={() => { reloadPos(); reloadSells(); }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {stats && (
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', marginBottom: 20 }}>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Holdings</div>
            <div style={{ fontSize: 22, fontWeight: 700 }}>{stats.count}</div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Total Invested</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{formatGP(stats.totalCost)}</div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Est. Total Profit</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: pnlColor(stats.totalProfit) }}>
              {stats.totalProfit >= 0 ? '+' : ''}{formatGP(stats.totalProfit)}
            </div>
          </div>
          <div className="card" style={{ padding: '14px 16px' }}>
            <div className="card-title">Avg P&L</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: pnlColor(stats.avgPnl) }}>
              {stats.avgPnl >= 0 ? '+' : ''}{stats.avgPnl.toFixed(1)}%
            </div>
          </div>
          {sells.length > 0 && (
            <div className="card" style={{ padding: '14px 16px', border: '1px solid rgba(245,158,11,0.3)' }}>
              <div className="card-title">Sell Offers</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--yellow)' }}>{sells.length}</div>
            </div>
          )}
        </div>
      )}

      {/* Sell Offers */}
      {sells.length > 0 && (
        <div className="card" style={{ marginBottom: 20, padding: '18px 20px' }}>
          <SectionHeader icon={Tag} title="Sell Offers" count={sells.length} color="var(--yellow)" />
          {Object.keys(sellAlertMap).length > 0 && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12,
              padding: '8px 12px', background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, fontSize: 12,
            }}>
              <AlertTriangle size={14} color="var(--red)" />
              <span style={{ color: 'var(--red)', fontWeight: 600 }}>
                Market price has dropped below your listed price on {Object.keys(sellAlertMap).length} item(s) — consider re-listing.
              </span>
            </div>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {sells.map(sell => (
              <SellOfferCard key={sell.trade_id} sell={sell} alertMap={sellAlertMap} nav={nav} />
            ))}
          </div>
        </div>
      )}

      {/* Active Holdings */}
      <div className="card" style={{ marginBottom: 20, padding: '18px 20px' }}>
        <SectionHeader icon={ShoppingCart} title="Active Holdings" count={positions.length} color="var(--cyan)" />
        {posLoading && positions.length === 0 ? (
          <div className="loading">Loading positions...</div>
        ) : positions.length === 0 ? (
          <div className="empty" style={{ padding: '30px 0' }}>
            <ShoppingCart size={22} style={{ opacity: 0.4, marginBottom: 8 }} />
            <div>No active holdings — buy something via the GE and DINK will pick it up.</div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {positions.map(pos => (
              <HoldingCard
                key={pos.trade_id}
                pos={pos}
                onDismiss={handleDismiss}
                flashId={flashId}
                nav={nav}
              />
            ))}
          </div>
        )}
      </div>

      {/* Recent Completed Trades */}
      <div className="card" style={{ padding: '18px 20px' }}>
        <SectionHeader icon={TrendingUp} title="Recent Trades" count={trades.length} color="var(--green)" />
        {tradeLoading && !trades.length ? (
          <div className="loading">Loading trades...</div>
        ) : trades.length === 0 ? (
          <div className="empty" style={{ padding: '24px 0' }}>No completed trades yet.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Type</th>
                  <th>Qty</th>
                  <th>Price</th>
                  <th>Status</th>
                  <th>Player</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => (
                  <tr key={i}>
                    <td style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                      <img src={IMG(t.item_id)} alt="" width={22} height={22}
                        style={{ imageRendering: 'pixelated' }}
                        onError={e => { e.target.style.display = 'none'; }} />
                      {t.item_name}
                    </td>
                    <td>
                      <span className={`badge ${t.trade_type === 'BUY' ? 'badge-cyan' : 'badge-green'}`}>
                        {t.trade_type === 'BUY' ? '⬇ BUY' : '⬆ SELL'}
                      </span>
                    </td>
                    <td>{t.quantity?.toLocaleString()}</td>
                    <td className="gp">{formatGP(t.price)}</td>
                    <td>
                      <span className={`badge ${
                        t.status === 'BOUGHT' ? 'badge-green' :
                        t.status === 'SOLD' ? 'badge-cyan' :
                        t.status === 'BUYING' || t.status === 'SELLING' ? 'badge-yellow' : 'badge-red'
                      }`}>{t.status}</span>
                    </td>
                    <td className="text-muted">{t.player}</td>
                    <td className="text-muted" style={{ fontSize: 11 }}>
                      <span title={t.timestamp}>{elapsed(t.timestamp)} ago</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
