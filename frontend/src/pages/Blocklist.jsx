import { useState, useEffect } from 'react';
import { Search, Shield, Brain, List, Save, Loader, X, Plus } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

const SUGGESTION_STYLE = {
  block:   { cls: 'badge-red',    label: 'Block',   icon: '🚫' },
  monitor: { cls: 'badge-yellow', label: 'Monitor', icon: '⚠️' },
  keep:    { cls: 'badge-green',  label: 'Keep',    icon: '✅' },
};

export default function Blocklist() {
  const [tab, setTab] = useState('suggestions');
  const [msg, setMsg] = useState('');
  const [saving, setSaving] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // Current blocklist
  const { data: blocklistData, reload: reloadBlocklist } = useApi(() => api.getBlocklist(), []);
  const [blockedIds, setBlockedIds] = useState(new Set());

  useEffect(() => {
    if (blocklistData?.items) {
      setBlockedIds(new Set(blocklistData.items.map(i => i.item_id)));
    }
  }, [blocklistData]);

  // AI suggestions
  const [suggestions, setSuggestions] = useState(null);
  const [suggestSearch, setSuggestSearch] = useState('');
  const [suggestFilter, setSuggestFilter] = useState('all');

  // Manual tab
  const [manualSearch, setManualSearch] = useState('');

  const showMsg = (text, ms = 2500) => {
    setMsg(text);
    setTimeout(() => setMsg(''), ms);
  };

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      const data = await api.analyzeBlocklist();
      setSuggestions(data.items || []);
      showMsg(`Analysed ${data.total} items from your trade history`);
    } catch (e) {
      showMsg('Analysis failed: ' + e.message, 4000);
    }
    setAnalyzing(false);
  };

  const saveBlocklist = async (ids) => {
    setSaving(true);
    try {
      await api.setBlocklist([...ids]);
      setBlockedIds(new Set(ids));
      reloadBlocklist();
      showMsg(`Blocklist saved — ${ids.size} items blocked`);
    } catch (e) {
      showMsg('Save failed: ' + e.message, 4000);
    }
    setSaving(false);
  };

  const toggleBlock = (itemId) => {
    setBlockedIds(prev => {
      const next = new Set(prev);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  };

  const applyAllSuggested = () => {
    if (!suggestions) return;
    const toBlock = suggestions.filter(s => s.suggestion === 'block').map(s => s.item_id);
    const next = new Set(blockedIds);
    toBlock.forEach(id => next.add(id));
    saveBlocklist(next);
  };

  // Styles
  const tabStyle = (t) => ({
    padding: '8px 18px', borderRadius: 20, fontSize: 13, fontWeight: 500,
    border: '1px solid var(--border)', cursor: 'pointer',
    background: tab === t ? 'linear-gradient(135deg, var(--cyan), var(--purple))' : 'var(--bg-card)',
    color: tab === t ? '#fff' : 'var(--text-primary)',
    transition: 'all 0.2s',
  });

  // Filtered suggestions
  const filteredSuggestions = (suggestions || []).filter(s => {
    if (suggestFilter !== 'all' && s.suggestion !== suggestFilter) return false;
    if (suggestSearch) return s.name.toLowerCase().includes(suggestSearch.toLowerCase());
    return true;
  });

  // Blocklist items for manual tab
  const blockedItems = blocklistData?.items || [];
  const filteredBlocked = manualSearch
    ? blockedItems.filter(i => i.name.toLowerCase().includes(manualSearch.toLowerCase()))
    : blockedItems;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Blocklist</h2>
          <p className="page-subtitle">
            {blockedIds.size} items blocked · excluded from opportunity scan
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {msg && (
            <span className={`badge ${msg.includes('failed') || msg.includes('Failed') ? 'badge-red' : 'badge-green'}`}>
              {msg}
            </span>
          )}
          <button
            className="btn btn-primary"
            onClick={() => saveBlocklist(blockedIds)}
            disabled={saving}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {saving ? <Loader size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <Save size={14} />}
            {saving ? 'Saving…' : 'Save Blocklist'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button style={tabStyle('suggestions')} onClick={() => setTab('suggestions')}>
          <Brain size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />
          AI Suggestions
        </button>
        <button style={tabStyle('manual')} onClick={() => setTab('manual')}>
          <Shield size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />
          Current Blocklist
        </button>
      </div>

      {/* ── AI Suggestions Tab ─────────────────────────────────────────────── */}
      {tab === 'suggestions' && (
        <div>
          <div className="card" style={{ marginBottom: 16, padding: '18px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
              <div>
                <h3 style={{ fontSize: 14, marginBottom: 6 }}>AI Analysis</h3>
                <p className="text-muted" style={{ fontSize: 12, maxWidth: 500 }}>
                  Analyses your complete flip history to identify items you consistently lose money on,
                  have a low win rate, or underperform. Suggested blocks are pre-selected for you.
                </p>
              </div>
              <button
                className="btn btn-primary"
                onClick={runAnalysis}
                disabled={analyzing}
                style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: 6 }}
              >
                {analyzing
                  ? <><Loader size={14} style={{ animation: 'spin 1s linear infinite' }} /> Analysing…</>
                  : <><Brain size={14} /> Get AI Analysis</>}
              </button>
            </div>

            {suggestions !== null && (
              <div style={{ marginTop: 14, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</span>
                {['all', 'block', 'monitor', 'keep'].map(f => (
                  <button key={f} onClick={() => setSuggestFilter(f)}
                    style={{
                      padding: '4px 12px', borderRadius: 12, fontSize: 11, border: '1px solid var(--border)',
                      background: suggestFilter === f ? 'var(--bg-secondary)' : 'transparent',
                      color: suggestFilter === f ? 'var(--text-primary)' : 'var(--text-muted)', cursor: 'pointer',
                    }}>
                    {f === 'all' ? 'All' : SUGGESTION_STYLE[f]?.label}
                    <span style={{ marginLeft: 4, opacity: 0.7 }}>
                      ({f === 'all' ? suggestions.length : suggestions.filter(s => s.suggestion === f).length})
                    </span>
                  </button>
                ))}
                <div style={{ position: 'relative', marginLeft: 'auto' }}>
                  <Search size={12} style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                  <input
                    type="text" placeholder="Search…" value={suggestSearch}
                    onChange={e => setSuggestSearch(e.target.value)}
                    style={{ padding: '5px 10px 5px 26px', borderRadius: 16, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12, width: 150 }}
                  />
                </div>
                <button
                  className="btn" style={{ marginLeft: 8, fontSize: 12 }}
                  onClick={applyAllSuggested}
                  title="Block all items marked 'Block'"
                >
                  Apply All Suggested Blocks
                </button>
              </div>
            )}
          </div>

          {suggestions === null ? (
            <div className="card">
              <div className="empty" style={{ padding: '40px 0' }}>
                <Brain size={28} style={{ opacity: 0.3, marginBottom: 12 }} />
                <div>Click "Get AI Analysis" to analyse your trade history</div>
                <div className="text-muted" style={{ fontSize: 12, marginTop: 6 }}>
                  Needs at least a few completed flips in your import history
                </div>
              </div>
            </div>
          ) : filteredSuggestions.length === 0 ? (
            <div className="card">
              <div className="empty" style={{ padding: '30px 0' }}>No items match the filter.</div>
            </div>
          ) : (
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: 40 }}>Block</th>
                    <th>Item</th>
                    <th>Flips</th>
                    <th>Win Rate</th>
                    <th>Total P&L</th>
                    <th>Avg Margin</th>
                    <th>Suggestion</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredSuggestions.map(s => {
                    const style = SUGGESTION_STYLE[s.suggestion] || SUGGESTION_STYLE.keep;
                    const isBlocked = blockedIds.has(s.item_id);
                    return (
                      <tr key={s.item_id} style={{ opacity: s.suggestion === 'keep' ? 0.8 : 1 }}>
                        <td style={{ textAlign: 'center' }}>
                          <input
                            type="checkbox"
                            checked={isBlocked}
                            onChange={() => toggleBlock(s.item_id)}
                            style={{ width: 16, height: 16, cursor: 'pointer', accentColor: 'var(--red)' }}
                          />
                        </td>
                        <td style={{ fontWeight: 600 }}>{s.name}</td>
                        <td>{s.total_flips}</td>
                        <td>
                          <span style={{ color: s.win_rate < 50 ? 'var(--red)' : s.win_rate < 60 ? 'var(--yellow)' : 'var(--green)', fontWeight: 600 }}>
                            {s.win_rate}%
                          </span>
                        </td>
                        <td style={{ color: s.total_pnl < 0 ? 'var(--red)' : 'var(--green)', fontWeight: 600 }}>
                          {s.total_pnl >= 0 ? '+' : ''}{formatGP(s.total_pnl)}
                        </td>
                        <td>{s.avg_margin_pct?.toFixed(1)}%</td>
                        <td>
                          <span className={`badge ${style.cls}`}>{style.icon} {style.label}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Current Blocklist Tab ─────────────────────────────────────────── */}
      {tab === 'manual' && (
        <div>
          <div className="card" style={{ marginBottom: 16, padding: '18px 20px' }}>
            <h3 style={{ fontSize: 14, marginBottom: 12 }}>Blocked Items</h3>
            <p className="text-muted" style={{ fontSize: 12, marginBottom: 14 }}>
              Items in this list are excluded from all opportunity scans. You can also use the AI Suggestions tab to auto-populate this list.
            </p>

            {/* Search */}
            <div style={{ position: 'relative', marginBottom: 12 }}>
              <Search size={12} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text" placeholder="Search blocked items…" value={manualSearch}
                onChange={e => setManualSearch(e.target.value)}
                style={{ padding: '7px 12px 7px 28px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12, width: '100%' }}
              />
            </div>

            {filteredBlocked.length === 0 ? (
              <div className="empty" style={{ padding: '24px 0' }}>
                <Shield size={20} style={{ opacity: 0.3, marginBottom: 8 }} />
                <div>{manualSearch ? 'No matching blocked items.' : 'No items are blocked yet. Use AI Suggestions to get started.'}</div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {filteredBlocked.map(item => (
                  <div key={item.item_id} style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '8px 14px', background: 'var(--bg-secondary)',
                    borderRadius: 8, border: '1px solid var(--border)',
                  }}>
                    <div>
                      <span style={{ fontWeight: 600, fontSize: 13 }}>{item.name}</span>
                      <span className="text-muted" style={{ fontSize: 11, marginLeft: 8 }}>#{item.item_id}</span>
                    </div>
                    <button
                      onClick={() => {
                        const next = new Set(blockedIds);
                        next.delete(item.item_id);
                        saveBlocklist(next);
                      }}
                      style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 6, padding: '4px 10px', color: 'var(--red)', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                    >
                      <X size={12} /> Unblock
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
