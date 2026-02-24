import { useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, BarChart3 } from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
  Bar,
} from 'recharts';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

const RANGES = [
  { key: '30h', label: '30h' },
  { key: '15d', label: '15d' },
  { key: '3m', label: '3m' },
  { key: '1y', label: '1y' },
];

function formatGP(n) {
  if (n == null) return '-';
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return Number(n).toLocaleString();
}

function formatTime(ts, rangeKey) {
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return '-';
  if (rangeKey === '30h') return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (rangeKey === '15d') return d.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function Item() {
  const { itemId } = useParams();
  const nav = useNavigate();
  const [range, setRange] = useState('15d');

  const { data: detail, loading, error } = useApi(
    () => api.getItem(itemId, range),
    [itemId, range],
    60_000,
  );

  const latest = detail?.latest || {};
  const meta = detail?.meta || {};
  const chartData = useMemo(() => {
    const ts = detail?.series?.ts || [];
    const buy = detail?.series?.buy || [];
    const sell = detail?.series?.sell || [];
    const volume = detail?.series?.volume || [];
    const points = [];
    const len = Math.min(ts.length, buy.length, sell.length);
    for (let i = 0; i < len; i += 1) {
      points.push({
        ts: ts[i],
        buy: buy[i],
        sell: sell[i],
        volume: volume[i] ?? 0,
        label: formatTime(ts[i], range),
      });
    }
    return points;
  }, [detail, range]);

  if (loading) return <div className="loading">Loading item...</div>;
  if (error) {
    return (
      <div className="empty" style={{ color: 'var(--red)' }}>
        <strong>Failed to load item</strong>
        <br />
        <small>{error.message || String(error)}</small>
      </div>
    );
  }
  if (!detail) return <div className="empty">Item not found.</div>;

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button className="btn" onClick={() => nav(-1)}>
            <ArrowLeft size={15} />
          </button>
          <img
            src={`https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${itemId}`}
            alt=""
            width={36}
            height={36}
            style={{ imageRendering: 'pixelated' }}
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          />
          <div>
            <h2 className="page-title">{detail.name || `Item ${itemId}`}</h2>
            <p className="page-subtitle">ID: {itemId}</p>
          </div>
        </div>
      </div>

      <div className="stats-grid">
        <div className="card">
          <div className="card-title">Buy</div>
          <div className="card-value text-red">{formatGP(latest.buy)}</div>
        </div>
        <div className="card">
          <div className="card-title">Sell</div>
          <div className="card-value text-green">{formatGP(latest.sell)}</div>
        </div>
        <div className="card">
          <div className="card-title">Spread</div>
          <div className="card-value">{formatGP(latest.spread_gp)}</div>
        </div>
        <div className="card">
          <div className="card-title">Profit</div>
          <div className="card-value text-cyan">{formatGP(latest.spread_gp)}</div>
          <div className="text-muted" style={{ fontSize: 11 }}>{Number(latest.roi_pct || 0).toFixed(2)}% ROI</div>
        </div>
        <div className="card">
          <div className="card-title">Volume (5m)</div>
          <div className="card-value">{Number(latest.volume_5m || 0).toLocaleString()}</div>
        </div>
        <div className="card">
          <div className="card-title">Score</div>
          <div className="card-value">-</div>
        </div>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 10, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart3 size={16} style={{ color: 'var(--cyan)' }} />
            <h3 style={{ margin: 0, fontSize: 14 }}>GE Price History</h3>
          </div>
          <div className="horizon-tabs">
            {RANGES.map((entry) => (
              <button
                key={entry.key}
                className={`horizon-tab ${range === entry.key ? 'active' : ''}`}
                onClick={() => setRange(entry.key)}
              >
                {entry.label}
              </button>
            ))}
          </div>
        </div>

        <div className="text-muted" style={{ marginBottom: 10, fontSize: 12 }}>
          {meta.points || 0} points · source {meta.source || 'unknown'}
        </div>

        {chartData.length > 0 ? (
          <>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.2)" />
                <XAxis dataKey="label" stroke="#6b7280" fontSize={10} />
                <YAxis
                  yAxisId="price"
                  stroke="#6b7280"
                  fontSize={10}
                  tickFormatter={formatGP}
                  width={65}
                />
                <YAxis yAxisId="volume" orientation="right" stroke="#6b7280" fontSize={10} width={50} />
                <Tooltip
                  contentStyle={{ background: '#1a1f35', border: '1px solid #2d3748', borderRadius: 8, fontSize: 12 }}
                  formatter={(value, name) => {
                    if (name === 'volume') return [Number(value || 0).toLocaleString(), 'Volume'];
                    return [`${formatGP(value)} GP`, name === 'buy' ? 'Buy' : 'Sell'];
                  }}
                  labelFormatter={(_, payload) => {
                    const point = payload?.[0]?.payload;
                    return point ? new Date(point.ts * 1000).toLocaleString() : '';
                  }}
                />
                <Legend />
                <Bar yAxisId="volume" dataKey="volume" fill="rgba(6,182,212,0.25)" name="volume" />
                <Line yAxisId="price" type="monotone" dataKey="buy" stroke="#ef4444" strokeWidth={2} dot={false} name="buy" />
                <Line yAxisId="price" type="monotone" dataKey="sell" stroke="#10b981" strokeWidth={2} dot={false} name="sell" />
              </ComposedChart>
            </ResponsiveContainer>
          </>
        ) : (
          <div className="empty" style={{ padding: 24 }}>
            No chart data yet
            <div style={{ marginTop: 8, fontSize: 12 }}>
              Buy: {formatGP(latest.buy)} · Sell: {formatGP(latest.sell)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

