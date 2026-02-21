import { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

export default function SettingsPage() {
  const { data: settings, loading, reload } = useApi(() => api.getSettings(), []);
  const { data: me } = useApi(() => api.getMe(), []);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');

  // Local state for discord webhook (controlled inputs)
  const [webhookUrl, setWebhookUrl] = useState('');
  const [webhookEnabled, setWebhookEnabled] = useState(false);
  const [sellAlertWebhook, setSellAlertWebhook] = useState('');
  const [notifyInterval, setNotifyInterval] = useState(30);
  const [riskTolerance, setRiskTolerance] = useState('MEDIUM');
  const [sending, setSending] = useState(false);
  const [testing, setTesting] = useState(false);
  const [autoArchiveDays, setAutoArchiveDays] = useState(7);

  // Security state
  const [allowedIds, setAllowedIds] = useState([]);
  const [newDiscordId, setNewDiscordId] = useState('');

  // Sync local state when settings load
  useEffect(() => {
    if (settings) {
      setWebhookUrl(settings.discord_webhook?.url || settings.discord_webhook_url || '');
      setWebhookEnabled(settings.discord_webhook?.enabled ?? settings.discord_alerts_enabled ?? false);
      setSellAlertWebhook(settings.sell_alert_webhook_url || '');
      setNotifyInterval(settings.discord_top5_interval_minutes ?? 30);
      setRiskTolerance(settings.risk_tolerance || 'MEDIUM');
      setAllowedIds(settings.allowed_discord_ids || []);
      setAutoArchiveDays(settings.position_auto_archive_days ?? 7);
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
        sell_alert_webhook_url: sellAlertWebhook,
      });
      showMsg('Discord settings saved!');
      reload();
    } catch (e) {
      showMsg('Error: ' + e.message, 4000);
    }
    setSaving(false);
  };

  const addDiscordId = async () => {
    const trimmed = newDiscordId.trim();
    if (!trimmed || allowedIds.includes(trimmed)) return;
    const updated = [...allowedIds, trimmed];
    setAllowedIds(updated);
    setNewDiscordId('');
    try {
      await api.updateSettings({ allowed_discord_ids: updated });
      showMsg('Allowed user added!');
    } catch (e) {
      showMsg('Error: ' + e.message, 4000);
    }
  };

  const removeDiscordId = async (id) => {
    const updated = allowedIds.filter(x => x !== id);
    setAllowedIds(updated);
    try {
      await api.updateSettings({ allowed_discord_ids: updated });
      showMsg('User removed.');
    } catch (e) {
      showMsg('Error: ' + e.message, 4000);
    }
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
      await api.sendTop5Now();
      showMsg('⏳ Scanning & generating charts… this takes ~1-2 min', 6000);
      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const s = await api.getSendTop5Status();
          if (s.status === 'sending') {
            showMsg('📤 Sending to Discord…', 3000);
          } else if (s.status === 'done') {
            clearInterval(poll);
            setSending(false);
            showMsg(`✅ Sent ${s.items_sent} opportunities to Discord!`, 5000);
          } else if (s.status === 'error') {
            clearInterval(poll);
            setSending(false);
            showMsg('❌ ' + (s.error || 'Send failed'), 5000);
          }
        } catch { /* keep polling */ }
      }, 3000);
      // Safety timeout after 5 minutes
      setTimeout(() => { clearInterval(poll); setSending(false); }, 300000);
    } catch (e) {
      setSending(false);
      showMsg('Error: ' + (e.message || 'Unknown error'), 4000);
    }
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

          <label style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 8 }}>
            Sell-Alert Webhook URL
            <input
              type="password"
              value={sellAlertWebhook}
              onChange={e => setSellAlertWebhook(e.target.value)}
              placeholder="https://discord.com/api/webhooks/... (for sell price drop alerts)"
              style={inputStyle}
            />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Receives alerts when a sell offer price drops while you're listing. Leave blank to use the main webhook above.
            </span>
          </label>

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

      {/* Portfolio auto-archive */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 8 }}>🗂️ Position Auto-Archive</h3>
        <p className="text-muted" style={{ fontSize: 12, marginBottom: 12 }}>
          Positions (BUY trades) that haven't been matched to a sell after this many days are
          automatically dismissed. This clears ghost positions caused by the Dink plugin missing
          a sell event, or CSV imports of already-completed trades.
          Set to <strong>0</strong> to disable.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <select
            value={autoArchiveDays}
            onChange={async e => {
              const val = Number(e.target.value);
              setAutoArchiveDays(val);
              try {
                await api.updateSettings({ position_auto_archive_days: val });
                showMsg(val === 0 ? 'Auto-archive disabled.' : `Auto-archive set to ${val} days.`);
              } catch (err) {
                showMsg('Error: ' + err.message, 4000);
              }
            }}
            style={{ ...inputStyle, width: 'auto', marginTop: 0 }}
          >
            <option value={0}>Disabled</option>
            <option value={3}>3 days</option>
            <option value={7}>7 days (default)</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
          </select>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {autoArchiveDays === 0
              ? 'Positions will never auto-expire — dismiss them manually.'
              : `Positions older than ${autoArchiveDays} days will be auto-dismissed every 6 hours.`}
          </span>
        </div>
      </div>

      {/* Security */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>🔒 Security — Allowed Users</h3>

        {/* Current user info */}
        {me && me.id !== 'anonymous' && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16,
            padding: '10px 14px', background: 'rgba(6,182,212,0.07)', borderRadius: 8,
            border: '1px solid rgba(6,182,212,0.2)',
          }}>
            {me.avatar && (
              <img
                src={`https://cdn.discordapp.com/avatars/${me.id}/${me.avatar}.png?size=40`}
                alt="" width={36} height={36} style={{ borderRadius: '50%' }}
              />
            )}
            <div>
              <div style={{ fontWeight: 600, fontSize: 13 }}>{me.username}</div>
              <div className="text-muted" style={{ fontSize: 11 }}>Discord ID: {me.id}</div>
            </div>
            <span className="badge badge-green" style={{ marginLeft: 'auto' }}>Logged in</span>
          </div>
        )}

        <p className="text-muted" style={{ fontSize: 12, marginBottom: 12 }}>
          Only Discord users in this list can log in. The first user to authenticate is automatically added.
          Remove yourself only if another owner is already listed.
        </p>

        {/* List of allowed IDs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
          {allowedIds.length === 0 && (
            <p className="text-muted" style={{ fontSize: 12 }}>
              No users configured — anyone who authenticates with Discord can log in.
            </p>
          )}
          {allowedIds.map(id => (
            <div key={id} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '6px 12px', background: 'var(--bg-secondary)', borderRadius: 6,
              border: '1px solid var(--border)', fontSize: 12,
            }}>
              <span style={{ fontFamily: 'monospace' }}>{id}</span>
              {id === me?.id && <span className="badge badge-cyan" style={{ margin: '0 8px' }}>You</span>}
              <button
                onClick={() => removeDiscordId(id)}
                style={{ background: 'none', border: 'none', color: 'var(--red)', cursor: 'pointer', fontSize: 16, padding: '0 4px' }}
                title="Remove"
              >×</button>
            </div>
          ))}
        </div>

        {/* Add new ID */}
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="text"
            value={newDiscordId}
            onChange={e => setNewDiscordId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addDiscordId()}
            placeholder="Discord user ID (18-digit number)"
            style={{ ...inputStyle, flex: 1, marginTop: 0 }}
          />
          <button onClick={addDiscordId} style={btnStyle('#4fc3f7')}>Add</button>
        </div>
        <p className="text-muted" style={{ fontSize: 11, marginTop: 8 }}>
          To find a Discord ID: enable Developer Mode in Discord → right-click any user → Copy User ID.
        </p>
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
