import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function scoreColor(score) {
  if (score >= 70) return 'badge-green';
  if (score >= 55) return 'badge-cyan';
  if (score >= 45) return 'badge-yellow';
  return 'badge-red';
}

const FILTERS = ['All', 'High Score', 'High Margin', 'High Liquidity', 'Best EV'];

export default function Opportunities({ prices }) {
  const nav = useNavigate();
  const [filter, setFilter] = useState('All');
  const [sortCol, setSortCol] = useState('flip_score');
  const [sortDir, setSortDir] = useState('desc');
  const [search, setSearch] = useState('');
  const [minPrice, setMinPrice] = useState(0);

  const { data: raw, loading, reload } = useApi(
    () => api.getOpportunities({ limit: 200, min_profit: minPrice }),
    [minPrice],
    120000,
  );

  // API now returns { items: [...], total: N }
  const opps = raw?.items || raw || [];

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filtered = useMemo(() => {
    let items = [...opps];

    // Text search
    if (search) {
      const q = search.toLowerCase();
      items = items.filter(o => o.name?.toLowerCase().includes(q));
    }

    // Filter pills
    if (filter === 'High Score') items = items.filter(o => o.flip_score >= 60);
    else if (filter === 'High Margin') items = items.filter(o => o.margin_pct > 2);
    else if (filter === 'High Liquidity') items = items.filter(o => o.volume > 10);
    else if (filter === 'Best EV') items.sort((a, b) => (b.potential_profit * (b.volume || 1)) - (a.potential_profit * (a.volume || 1)));

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
          <p className="page-subtitle">{filtered.length} scored items — ranked by flip score</p>
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
                {th('Score', 'flip_score')}
                {th('Buy', 'buy_price')}
                {th('Sell', 'sell_price')}
                {th('Margin%', 'margin_pct')}
                {th('Profit', 'potential_profit')}
                {th('Volume', 'volume')}
                {th('Win Rate', 'win_rate')}
                <th>Trend</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((opp, i) => (
                <tr key={i} onClick={() => nav(`/item/${opp.item_id}`)}>
                  <td style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <img
                      src={`https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${opp.item_id}`}
                      alt=""
                      width={24}
                      height={24}
                      style={{ imageRendering: 'pixelated', flexShrink: 0 }}
                      onError={e => { e.target.style.display = 'none'; }}
                    />
                    {opp.name}</td>
                  <td>
                    <span className={`badge ${scoreColor(opp.flip_score)}`}>
                      {opp.flip_score?.toFixed(0)}/100
                    </span>
                  </td>
                  <td className="gp text-green">{formatGP(opp.buy_price)}</td>
                  <td className="gp text-cyan">{formatGP(opp.sell_price)}</td>
                  <td className="gp">{opp.margin_pct?.toFixed(1)}%</td>
                  <td className="gp text-green">+{formatGP(opp.potential_profit)}</td>
                  <td className="gp">{opp.volume || 0}</td>
                  <td>
                    {opp.win_rate != null ? (
                      <span className={`badge ${opp.win_rate >= 80 ? 'badge-green' : opp.win_rate >= 60 ? 'badge-yellow' : 'badge-red'}`}>
                        {opp.win_rate?.toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    <span className={`badge ${
                      opp.trend === 'NEUTRAL' ? 'badge-cyan' :
                      opp.trend?.includes('UP') ? 'badge-green' : 'badge-red'
                    }`}>
                      {opp.trend === 'STRONG_UP' ? '\u25B2\u25B2' :
                       opp.trend === 'UP' ? '\u25B2' :
                       opp.trend === 'NEUTRAL' ? '\u25BA' :
                       opp.trend === 'DOWN' ? '\u25BC' : '\u25BC\u25BC'}
                    </span>
                  </td>
                  <td className="text-muted" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {opp.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
