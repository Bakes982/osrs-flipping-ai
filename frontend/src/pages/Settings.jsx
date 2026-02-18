import { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

export default function SettingsPage() {
  const { data: settings, loading, reload } = useApi(() => api.getSettings(), []);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');

  // Local state for discord webhook (controlled inputs)
  const [webhookUrl, setWebhookUrl] = useState('');
  const [webhookEnabled, setWebhookEnabled] = useState(false);
  const [notifyInterval, setNotifyInterval] = useState(30);
  const [riskTolerance, setRiskTolerance] = useState('MEDIUM');
  const [sending, setSending] = useState(false);
  const [testing, setTesting] = useState(false);

  // Sync local state when settings load
  useEffect(() => {
    if (settings) {
      setWebhookUrl(settings.discord_webhook?.url || settings.discord_webhook_url || '');
      setWebhookEnabled(settings.discord_webhook?.enabled ?? settings.discord_alerts_enabled ?? false);
      setNotifyInterval(settings.discord_top5_interval_minutes ?? 30);
      setRiskTolerance(settings.risk_tolerance || 'MEDIUM');
    }
  }, [settings]);

  const showMsg = (text, duration = 2500) => {
    setMsg(text);
    setTimeout(() => setMsg(''), duration);
  };

  const saveDiscord = async () => {
    setSaving(true);
    try {
      await api.updateSettings({
        discord_webhook: { url: webhookUrl, enabled: webhookEnabled },
        discord_top5_interval_minutes: notifyInterval,
      });
      showMsg('Discord settings saved!');
      reload();
    } catch (e) {
      showMsg('Error: ' + e.message, 4000);
    }
    setSaving(false);
  };

  const testWebhook = async () => {
    setTesting(true);
    try {
      const res = await api.testWebhook();
      showMsg(res.ok ? '✅ Test message sent!' : '❌ Webhook test failed', 3000);
    } catch (e) {
      showMsg('Error: ' + e.message, 4000);
    }
    setTesting(false);
  };

  const sendTop5 = async () => {
    setSending(true);
    try {
      const res = await api.sendTop5Now();
      showMsg(`✅ Sent ${res.items_sent} opportunities to Discord!`, 4000);
    } catch (e) {
      const detail = e.message || 'Unknown error';
      showMsg('Error: ' + detail, 4000);
    }
    setSending(false);
  };

  const saveRisk = async (value) => {
    setRiskTolerance(value);
    setSaving(true);
    try {
      await api.updateSettings({ risk_tolerance: value });
      showMsg('Risk setting saved!');
      reload();
    } catch (e) {
      showMsg('Error: ' + e.message, 4000);
    }
    setSaving(false);
  };

  if (loading) return <div className="loading">Loading settings...</div>;

  const inputStyle = {
    display: 'block', width: '100%', marginTop: 4, padding: '8px 12px',
    background: 'var(--bg-secondary)', border: '1px solid var(--border)',
    borderRadius: 8, color: 'var(--text-primary)', fontSize: 13,
  };

  const btnStyle = (color = '#4fc3f7') => ({
    padding: '8px 16px', fontSize: 13, fontWeight: 600,
    background: color, color: '#000', border: 'none', borderRadius: 8,
    cursor: 'pointer', opacity: saving ? 0.6 : 1,
  });

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Settings</h2>
          <p className="page-subtitle">Configure your flipping AI</p>
        </div>
        {msg && <span className={`badge ${msg.startsWith('Error') || msg.startsWith('❌') ? 'badge-red' : 'badge-green'}`}>{msg}</span>}
      </div>

      {/* Discord Notifications */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>🔔 Discord Notifications</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Webhook URL
            <input
              type="password"
              value={webhookUrl}
              onChange={e => setWebhookUrl(e.target.value)}
              placeholder="https://discord.com/api/webhooks/..."
              style={inputStyle}
            />
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <input
              type="checkbox"
              checked={webhookEnabled}
              onChange={e => setWebhookEnabled(e.target.checked)}
            />
            Enable automatic Top-5 opportunity alerts
          </label>

          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Notification Interval
            <select
              value={notifyInterval}
              onChange={e => setNotifyInterval(Number(e.target.value))}
              style={{ ...inputStyle, width: 'auto' }}
            >
              <option value={15}>Every 15 minutes</option>
              <option value={30}>Every 30 minutes</option>
              <option value={60}>Every 1 hour</option>
              <option value={120}>Every 2 hours</option>
              <option value={360}>Every 6 hours</option>
            </select>
          </label>

          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <button onClick={saveDiscord} disabled={saving} style={btnStyle('#4fc3f7')}>
              {saving ? 'Saving…' : '💾 Save'}
            </button>
            <button onClick={testWebhook} disabled={testing || !webhookUrl} style={btnStyle('#ff9800')}>
              {testing ? 'Sending…' : '🧪 Test Webhook'}
            </button>
            <button onClick={sendTop5} disabled={sending || !webhookUrl || !webhookEnabled} style={btnStyle('#00ff88')}>
              {sending ? 'Scanning…' : '🚀 Send Top 5 Now'}
            </button>
          </div>

          <p className="text-muted" style={{ fontSize: 11, marginTop: 4 }}>
            When enabled, the backend will automatically scan for the best flip opportunities
            and send them to your Discord channel with price charts attached.
          </p>
        </div>
      </div>

      {/* Risk Settings */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>⚖️ Risk Settings</h3>
        <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          Risk Tolerance
          <select
            value={riskTolerance}
            onChange={e => saveRisk(e.target.value)}
            style={{ ...inputStyle, width: 'auto' }}
          >
            <option value="LOW">Low (conservative)</option>
            <option value="MEDIUM">Medium (balanced)</option>
            <option value="HIGH">High (aggressive)</option>
          </select>
        </label>
      </div>

      {/* API Settings */}
      <div className="card">
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>⚙️ API Settings</h3>
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
