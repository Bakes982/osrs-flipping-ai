import { useState } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

export default function SettingsPage() {
  const { data: settings, loading, reload } = useApi(() => api.getSettings(), []);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');

  const handleSave = async (key, value) => {
    setSaving(true);
    try {
      await api.updateSettings({ [key]: value });
      setMsg('Saved!');
      setTimeout(() => setMsg(''), 2000);
      reload();
    } catch (e) {
      setMsg('Error: ' + e.message);
    }
    setSaving(false);
  };

  if (loading) return <div className="loading">Loading settings...</div>;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Settings</h2>
          <p className="page-subtitle">Configure your flipping AI</p>
        </div>
        {msg && <span className="badge badge-green">{msg}</span>}
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Discord Notifications</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Webhook URL
            <input
              type="password"
              defaultValue={settings?.discord_webhook?.url || ''}
              onBlur={e => handleSave('discord_webhook', { ...settings?.discord_webhook, url: e.target.value })}
              style={{
                display: 'block', width: '100%', marginTop: 4, padding: '8px 12px',
                background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                borderRadius: 8, color: 'var(--text-primary)', fontSize: 13,
              }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <input
              type="checkbox"
              defaultChecked={settings?.discord_webhook?.enabled}
              onChange={e => handleSave('discord_webhook', { ...settings?.discord_webhook, enabled: e.target.checked })}
            />
            Enable Discord notifications
          </label>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Risk Settings</h3>
        <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          Risk Tolerance
          <select
            defaultValue={settings?.risk_tolerance || 'MEDIUM'}
            onChange={e => handleSave('risk_tolerance', e.target.value)}
            style={{
              display: 'block', marginTop: 4, padding: '8px 12px',
              background: 'var(--bg-secondary)', border: '1px solid var(--border)',
              borderRadius: 8, color: 'var(--text-primary)', fontSize: 13,
            }}
          >
            <option value="LOW">Low (conservative)</option>
            <option value="MEDIUM">Medium (balanced)</option>
            <option value="HIGH">High (aggressive)</option>
          </select>
        </label>
      </div>

      <div className="card">
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>API Settings</h3>
        <p className="text-muted" style={{ fontSize: 12 }}>
          Price polling: every 10 seconds | Feature updates: every 60 seconds | ML scoring: every 60 seconds
        </p>
        <p className="text-muted" style={{ fontSize: 12, marginTop: 8 }}>
          Data retention: 7 days raw (10s), 30 days aggregated (5m), 1 year hourly
        </p>
      </div>
    </div>
  );
}
