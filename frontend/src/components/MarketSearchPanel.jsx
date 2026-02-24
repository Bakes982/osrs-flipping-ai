import { useEffect, useMemo, useState } from 'react';
import { Search, X } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { api } from '../api/client';

const RANGES = ['1h', '6h', '24h', '7d'];

function formatGP(n) {
  if (n == null) return '-';
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return Number(n).toLocaleString();
}

function formatTs(ts) {
  const date = new Date(Number(ts) * 1000);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleString();
}

export default function MarketSearchPanel() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [selected, setSelected] = useState(null);
  const [range, setRange] = useState('24h');
  const [graph, setGraph] = useState(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [graphError, setGraphError] = useState(null);

  useEffect(() => {
    const term = query.trim();
    if (!term) {
      setResults([]);
      setLoadingSearch(false);
      return undefined;
    }

    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setLoadingSearch(true);
      try {
        const payload = await api.searchItems(term, 20, { signal: ctrl.signal, timeoutMs: 10_000 });
        setResults(Array.isArray(payload) ? payload : (payload?.items || []));
      } catch (err) {
        if (err?.name !== 'AbortError') setResults([]);
      } finally {
        setLoadingSearch(false);
      }
    }, 250);

    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [query]);

  useEffect(() => {
    if (!selected?.item_id) return;
    let cancelled = false;

    (async () => {
      setLoadingGraph(true);
      setGraphError(null);
      try {
        const payload = await api.getItemGraph(selected.item_id, range);
        if (!cancelled) setGraph(payload || null);
      } catch (err) {
        if (!cancelled) {
          setGraph(null);
          setGraphError(err?.message || 'Failed to load graph');
        }
      } finally {
        if (!cancelled) setLoadingGraph(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selected, range]);

  const chartData = useMemo(
    () => (graph?.points || []).map((p) => ({ ...p, label: new Date(p.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })),
    [graph],
  );

  const latest = graph?.latest || {};

  return (
    <div className="card" style={{ marginBottom: 16, overflow: 'visible' }}>
      <div className="card-title">Market Search</div>
      <div style={{ position: 'relative', marginTop: 6 }}>
        <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search any item (name or ID)..."
          style={{ width: '100%', padding: '8px 10px 8px 30px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
        />
        {(loadingSearch || results.length > 0 || query.trim()) && (
          <div style={{ position: 'absolute', left: 0, right: 0, top: 40, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, maxHeight: 220, overflowY: 'auto', zIndex: 30, touchAction: 'pan-y' }}>
            {loadingSearch ? (
              <div style={{ padding: 10, color: 'var(--text-muted)', fontSize: 12 }}>Searching...</div>
            ) : results.length === 0 ? (
              <div style={{ padding: 10, color: 'var(--text-muted)', fontSize: 12 }}>No items found</div>
            ) : (
              results.map((item) => (
                <button
                  key={item.item_id}
                  type="button"
                  onClick={() => {
                    setSelected(item);
                    setRange('24h');
                    setResults([]);
                    setQuery(item.name);
                  }}
                  style={{ width: '100%', border: 'none', borderBottom: '1px solid rgba(45,55,72,0.4)', background: 'transparent', color: 'var(--text-primary)', textAlign: 'left', padding: '8px 10px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', gap: 8 }}
                >
                  <span>{item.name}</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>#{item.item_id}</span>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {selected && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(10,14,26,0.75)', zIndex: 260, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 12 }}>
          <div className="card" style={{ width: 'min(960px, calc(100vw - 24px))', maxHeight: '90vh', overflowY: 'auto', padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 18 }}>{selected.name}</h3>
                <div className="text-muted" style={{ fontSize: 12 }}>Item #{selected.item_id}</div>
              </div>
              <button type="button" className="btn" onClick={() => { setSelected(null); setGraph(null); }}>
                <X size={14} />
              </button>
            </div>

            <div className="filter-bar" style={{ marginBottom: 10 }}>
              {RANGES.map((r) => (
                <button key={r} className={`pill ${range === r ? 'active' : ''}`} onClick={() => setRange(r)}>
                  {r}
                </button>
              ))}
            </div>

            {loadingGraph ? (
              <div className="loading" style={{ padding: 30 }}>Loading graph...</div>
            ) : graphError ? (
              <div className="empty" style={{ color: 'var(--red)' }}>{graphError}</div>
            ) : (
              <>
                <div className="stats-grid" style={{ marginBottom: 12 }}>
                  <div className="card"><div className="card-title">Buy</div><div className="card-value text-red">{formatGP(latest.buy_price)}</div></div>
                  <div className="card"><div className="card-title">Sell</div><div className="card-value text-green">{formatGP(latest.sell_price)}</div></div>
                  <div className="card"><div className="card-title">Margin</div><div className="card-value">{formatGP(latest.margin_gp)}</div></div>
                  <div className="card"><div className="card-title">Volume (5m)</div><div className="card-value">{Number(latest.volume_5m || 0).toLocaleString()}</div></div>
                  <div className="card"><div className="card-title">Trend</div><div className="card-value" style={{ textTransform: 'uppercase' }}>{latest.trend || '-'}</div></div>
                  <div className="card"><div className="card-title">Updated</div><div className="card-value" style={{ fontSize: 14 }}>{formatTs(latest.updated_at)}</div></div>
                </div>

                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.18)" />
                      <XAxis dataKey="label" stroke="#6b7280" fontSize={10} />
                      <YAxis stroke="#6b7280" fontSize={10} tickFormatter={formatGP} />
                      <Tooltip
                        contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8, fontSize: 12 }}
                        formatter={(value, name) => [`${formatGP(value)} GP`, name === 'buy' ? 'Buy' : 'Sell']}
                        labelFormatter={(_, payload) => {
                          const row = payload?.[0]?.payload;
                          return row ? formatTs(row.ts) : '';
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="buy" stroke="#ef4444" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="sell" stroke="#10b981" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty">No history points for this range yet.</div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

