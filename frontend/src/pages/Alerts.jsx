import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bell, Check, Trash2, Plus, AlertTriangle, TrendingUp, TrendingDown, Target, Eye } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function timeAgo(ts) {
  if (!ts) return '';
  const diff = (Date.now() - new Date(ts).getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const ALERT_ICONS = {
  price_target: Target,
  dump: AlertTriangle,
  opportunity: TrendingUp,
  ml_signal: TrendingUp,
  position_change: Eye,
};

const ALERT_COLORS = {
  price_target: 'badge-cyan',
  dump: 'badge-red',
  opportunity: 'badge-green',
  ml_signal: 'badge-yellow',
  position_change: 'badge-cyan',
};

export default function Alerts() {
  const nav = useNavigate();
  const [filter, setFilter] = useState('all');
  const [showForm, setShowForm] = useState(false);
  const [newTarget, setNewTarget] = useState({ item_id: '', item_name: '', target_price: '', direction: 'below' });

  const { data: alertData, loading, reload } = useApi(
    () => api.getAlerts({ limit: 100, hours: 48 }),
    [],
    15000,
  );

  const { data: targetData, reload: reloadTargets } = useApi(
    () => api.getPriceTargets(),
    [],
  );

  const alerts = alertData?.alerts || [];
  const targets = targetData?.targets || [];
  const unackCount = alertData?.unacknowledged || 0;

  const filtered = filter === 'all'
    ? alerts
    : alerts.filter(a => a.alert_type === filter);

  const handleAckAll = async () => {
    await api.acknowledgeAlerts({ acknowledge_all: true });
    reload();
  };

  const handleAck = async (id) => {
    await api.acknowledgeAlerts({ alert_ids: [id] });
    reload();
  };

  const handleCreateTarget = async (e) => {
    e.preventDefault();
    if (!newTarget.item_id || !newTarget.target_price) return;
    await api.createPriceTarget({
      item_id: parseInt(newTarget.item_id),
      item_name: newTarget.item_name,
      target_price: parseInt(newTarget.target_price),
      direction: newTarget.direction,
    });
    setNewTarget({ item_id: '', item_name: '', target_price: '', direction: 'below' });
    setShowForm(false);
    reloadTargets();
  };

  const handleDeleteTarget = async (itemId, direction) => {
    await api.deletePriceTarget(itemId, direction);
    reloadTargets();
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Alerts</h2>
          <p className="page-subtitle">
            {unackCount > 0 ? `${unackCount} new alerts` : 'No new alerts'} — {alerts.length} total in 48h
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {unackCount > 0 && (
            <button className="btn btn-primary" onClick={handleAckAll}>
              <Check size={14} /> Acknowledge All
            </button>
          )}
          <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
            <Plus size={14} /> Price Alert
          </button>
        </div>
      </div>

      {/* Price target form */}
      {showForm && (
        <div className="card" style={{ marginBottom: 16, padding: 16 }}>
          <h3 style={{ marginBottom: 12, fontSize: 14 }}>Create Price Target Alert</h3>
          <form onSubmit={handleCreateTarget} style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'end' }}>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Item ID</label>
              <input type="number" value={newTarget.item_id} onChange={e => setNewTarget(t => ({ ...t, item_id: e.target.value }))}
                placeholder="e.g. 13652" style={inputStyle} required />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Item Name</label>
              <input type="text" value={newTarget.item_name} onChange={e => setNewTarget(t => ({ ...t, item_name: e.target.value }))}
                placeholder="e.g. Dragon claws" style={inputStyle} />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Target Price</label>
              <input type="number" value={newTarget.target_price} onChange={e => setNewTarget(t => ({ ...t, target_price: e.target.value }))}
                placeholder="e.g. 50000000" style={inputStyle} required />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Direction</label>
              <select value={newTarget.direction} onChange={e => setNewTarget(t => ({ ...t, direction: e.target.value }))} style={inputStyle}>
                <option value="below">Drops below</option>
                <option value="above">Rises above</option>
              </select>
            </div>
            <button type="submit" className="btn btn-primary" style={{ height: 34 }}>Create</button>
          </form>
        </div>
      )}

      {/* Active price targets */}
      {targets.length > 0 && (
        <div className="card" style={{ marginBottom: 16, padding: 16 }}>
          <h3 style={{ marginBottom: 12, fontSize: 14 }}>Active Price Targets</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {targets.map((t, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px',
                borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                fontSize: 12,
              }}>
                <Target size={14} style={{ color: 'var(--cyan)' }} />
                <span>{t.item_name || `Item ${t.item_id}`}</span>
                <span className={`badge ${t.direction === 'below' ? 'badge-red' : 'badge-green'}`}>
                  {t.direction === 'below' ? '\u25BC' : '\u25B2'} {formatGP(t.target_price)}
                </span>
                <button onClick={() => handleDeleteTarget(t.item_id, t.direction)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 2 }}>
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filter pills */}
      <div className="filter-bar">
        {['all', 'opportunity', 'dump', 'price_target', 'ml_signal'].map(f => (
          <button key={f} className={`pill ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
            {f === 'all' ? 'All' : f === 'ml_signal' ? 'ML Signal' : f.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
          </button>
        ))}
      </div>

      {/* Alert list */}
      <div className="card" style={{ padding: 0 }}>
        {loading ? (
          <div className="loading">Loading alerts...</div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
            No alerts found. The system will automatically generate alerts for price dumps, high-score opportunities, and your price targets.
          </div>
        ) : (
          <div>
            {filtered.map(a => {
              const Icon = ALERT_ICONS[a.alert_type] || Bell;
              const color = ALERT_COLORS[a.alert_type] || 'badge-cyan';
              return (
                <div key={a.id} style={{
                  display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px',
                  borderBottom: '1px solid var(--border)',
                  opacity: a.acknowledged ? 0.5 : 1,
                  cursor: 'pointer',
                }} onClick={() => a.item_id && nav(`/item/${a.item_id}`)}>
                  <Icon size={18} style={{ color: `var(--${a.alert_type === 'dump' ? 'red' : a.alert_type === 'opportunity' ? 'green' : 'cyan'})`, flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: a.acknowledged ? 400 : 600, marginBottom: 2 }}>
                      {a.message}
                    </div>
                    <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-muted)' }}>
                      <span className={`badge ${color}`} style={{ fontSize: 10 }}>{a.alert_type.replace('_', ' ')}</span>
                      <span>{timeAgo(a.timestamp)}</span>
                    </div>
                  </div>
                  {!a.acknowledged && (
                    <button onClick={(e) => { e.stopPropagation(); handleAck(a.id); }}
                      style={{ background: 'none', border: 'none', color: 'var(--cyan)', cursor: 'pointer', padding: 4 }}
                      title="Acknowledge">
                      <Check size={16} />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

const inputStyle = {
  padding: '6px 10px',
  borderRadius: 6,
  border: '1px solid var(--border)',
  background: 'var(--bg-card)',
  color: 'var(--text-primary)',
  fontSize: 12,
  width: 150,
};
