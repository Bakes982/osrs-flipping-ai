import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw, ArrowUpDown } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

const FILTERS = ['All', 'High Margin', 'High Liquidity', 'Best EV', 'Low Risk'];

export default function Opportunities({ prices }) {
  const nav = useNavigate();
  const [filter, setFilter] = useState('All');
  const [sortCol, setSortCol] = useState('expected_profit');
  const [sortDir, setSortDir] = useState('desc');
  const [search, setSearch] = useState('');
  const [minPrice, setMinPrice] = useState(100000);

  const { data: opps, loading, reload } = useApi(
    () => api.getOpportunities({ limit: 200, min_profit: minPrice }),
    [minPrice],
    30000,
  );

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filtered = useMemo(() => {
    if (!opps) return [];
    let items = [...opps];

    // Text search
    if (search) {
      const q = search.toLowerCase();
      items = items.filter(o => o.name?.toLowerCase().includes(q));
    }

    // Filter pills
    if (filter === 'High Margin') items = items.filter(o => o.margin_pct > 2);
    else if (filter === 'High Liquidity') items = items.filter(o => o.volume_5m > 10);
    else if (filter === 'Best EV') items.sort((a, b) => (b.expected_profit * (b.volume_5m || 1)) - (a.expected_profit * (a.volume_5m || 1)));
    else if (filter === 'Low Risk') items = items.filter(o => o.risk_score <= 4);

    // Sort
    items.sort((a, b) => {
      const av = a[sortCol] ?? 0;
      const bv = b[sortCol] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });

    return items;
  }, [opps, filter, sortCol, sortDir, search]);

  const th = (label, col) => (
    <th className={sortCol === col ? 'sorted' : ''} onClick={() => toggleSort(col)}>
      {label} {sortCol === col && (sortDir === 'asc' ? '\u25B2' : '\u25BC')}
    </th>
  );

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Opportunities</h2>
          <p className="page-subtitle">{filtered.length} items found — prices update every 10s</p>
        </div>
        <button className="btn-primary btn" onClick={reload}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="filter-bar">
        {FILTERS.map(f => (
          <button key={f} className={`pill ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
            {f}
          </button>
        ))}
        <input
          type="text"
          placeholder="Search items..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            padding: '6px 14px', borderRadius: 20, border: '1px solid var(--border)',
            background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12,
            marginLeft: 'auto', width: 200,
          }}
        />
        <select
          value={minPrice}
          onChange={e => setMinPrice(Number(e.target.value))}
          style={{
            padding: '6px 10px', borderRadius: 20, border: '1px solid var(--border)',
            background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12,
          }}
        >
          <option value={0}>Any price</option>
          <option value={100000}>100K+</option>
          <option value={1000000}>1M+</option>
          <option value={10000000}>10M+</option>
          <option value={50000000}>50M+</option>
        </select>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        {loading ? (
          <div className="loading">Loading opportunities...</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Item</th>
                {th('Buy Price', 'buy_at')}
                {th('Sell Price', 'sell_at')}
                {th('Margin', 'margin_pct')}
                {th('Profit', 'expected_profit')}
                {th('Volume', 'volume_5m')}
                {th('Risk', 'risk_score')}
                <th>Confidence</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((opp, i) => (
                <tr key={i} onClick={() => nav(`/item/${opp.item_id}`)}>
                  <td style={{ fontWeight: 500 }}>{opp.name}</td>
                  <td className="gp text-green">{formatGP(opp.buy_at)}</td>
                  <td className="gp text-cyan">{formatGP(opp.sell_at)}</td>
                  <td className="gp">{opp.margin_pct?.toFixed(1)}%</td>
                  <td className="gp text-green">+{formatGP(opp.expected_profit)}</td>
                  <td className="gp">{opp.volume_5m || 0}</td>
                  <td>
                    <span className={`badge ${opp.risk_score <= 4 ? 'badge-green' : opp.risk_score <= 6 ? 'badge-yellow' : 'badge-red'}`}>
                      {opp.risk_score}/10
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${opp.confidence === 'HIGH' ? 'badge-green' : opp.confidence === 'MEDIUM' ? 'badge-yellow' : 'badge-red'}`}>
                      {opp.confidence}
                    </span>
                  </td>
                  <td className="text-muted">{opp.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
