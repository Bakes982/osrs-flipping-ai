import { RefreshCw, XCircle, ArrowRightCircle } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

const NEXT_EVENT_BY_STATE = {
  BUY_PENDING: 'BUY_PLACED',
  BUYING: 'BUY_FILLED',
  HOLDING: 'SELL_PLACED',
  SELLING: 'SOLD',
};

const NEXT_LABEL = {
  BUY_PENDING: 'Mark buy placed',
  BUYING: 'Mark buy filled',
  HOLDING: 'Place sell',
  SELLING: 'Mark sold',
};

export default function ActiveTrades() {
  const { data, loading, error, reload } = useApi(() => api.getActiveTrades(), [], 10000);
  const trades = data?.items || [];
  const slotsUsed = data?.slots_used || 0;
  const slotsTotal = data?.slots_total || 8;

  const doEvent = async (tradeId, event) => {
    try {
      await api.tradeEvent(tradeId, event);
      reload();
    } catch (e) {
      window.alert(e.message || 'Failed to update trade');
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Active Trades</h2>
          <p className="page-subtitle">Slots used {slotsUsed}/{slotsTotal}</p>
        </div>
        <button className="btn" onClick={reload}><RefreshCw size={14} /> Refresh</button>
      </div>

      {loading && trades.length === 0 ? (
        <div className="loading">Loading active trades...</div>
      ) : error ? (
        <div className="empty" style={{ color: '#ef4444' }}>{error.message || 'Failed to load trades'}</div>
      ) : trades.length === 0 ? (
        <div className="empty">No active trades.</div>
      ) : (
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
          {trades.map((t) => (
            <div key={t.trade_id} className="card" style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 700 }}>{t.name}</div>
                  <div className="text-muted" style={{ fontSize: 12 }}>Slot {t.slot_index} · {t.state}</div>
                </div>
                <span className={`badge ${t.type === 'dump' ? 'badge-red' : 'badge-cyan'}`}>{t.type}</span>
              </div>

              <div style={{ marginTop: 10, fontSize: 12, lineHeight: 1.6 }}>
                <div>Buy target: {Number(t.buy_target || 0).toLocaleString()} gp</div>
                <div>Sell target: {Number(t.sell_target || 0).toLocaleString()} gp</div>
                <div>Qty target: {t.qty_target || 0}</div>
                <div>Last action: {t.last_action || '—'}</div>
              </div>

              <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {NEXT_EVENT_BY_STATE[t.state] && (
                  <button className="btn" onClick={() => doEvent(t.trade_id, NEXT_EVENT_BY_STATE[t.state])}>
                    <ArrowRightCircle size={13} /> {NEXT_LABEL[t.state]}
                  </button>
                )}
                <button className="btn" onClick={() => doEvent(t.trade_id, 'CANCEL')}>
                  <XCircle size={13} /> Cancel/Close
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
