import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

const HORIZONS = ['1m', '5m', '30m', '2h', '8h', '24h'];
const HORIZON_LABELS = { '1m': '1 Min', '5m': '5 Min', '30m': '30 Min', '2h': '2 Hour', '8h': '8 Hour', '24h': '24 Hour' };

export default function ModelDashboard() {
  const { data: status, loading } = useApi(() => api.getModelStatus(), [], 30000);
  const { data: metrics } = useApi(() => api.getModelMetrics(), [], 60000);

  if (loading) return <div className="loading">Loading ML pipeline status...</div>;

  const mlActive = status?.ml_active;
  const liveAcc = status?.live_accuracy;
  const dataColl = status?.data_collection;
  const pipeline = status?.pipeline_info;
  const learningCurve = status?.learning_curve || [];
  const horizonData = status?.horizons || {};

  // Build horizon accuracy chart data
  const horizonChartData = HORIZONS.map(h => ({
    horizon: HORIZON_LABELS[h] || h,
    accuracy: horizonData[h]?.direction_accuracy || 0,
    samples: horizonData[h]?.sample_count || 0,
    trained: horizonData[h]?.trained,
  }));

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">🧠 AI Learning Dashboard</h2>
          <p className="page-subtitle">
            {mlActive
              ? 'ML models are actively learning from live price data'
              : 'Collecting data — statistical models active while ML trains'}
          </p>
        </div>
        <span className={`badge ${mlActive ? 'badge-green' : 'badge-yellow'}`} style={{ fontSize: '1rem', padding: '8px 16px' }}>
          {mlActive ? '🟢 ML Active' : '🟡 Training...'}
        </span>
      </div>

      {/* KPI Summary row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 24 }}>
        <div className="card" style={{ textAlign: 'center', padding: 20 }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: mlActive ? 'var(--color-success)' : 'var(--color-warning)' }}>
            {status?.prediction_method === 'ml' ? 'ML' : 'Statistical'}
          </div>
          <div className="text-muted" style={{ marginTop: 4 }}>Active Method</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 20 }}>
          <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--color-primary)' }}>
            {liveAcc?.direction_accuracy_pct != null ? `${liveAcc.direction_accuracy_pct}%` : '—'}
          </div>
          <div className="text-muted" style={{ marginTop: 4 }}>Live Direction Accuracy</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 20 }}>
          <div style={{ fontSize: 28, fontWeight: 700 }}>
            {liveAcc?.total_predictions?.toLocaleString() || 0}
          </div>
          <div className="text-muted" style={{ marginTop: 4 }}>Total Predictions</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 20 }}>
          <div style={{ fontSize: 28, fontWeight: 700 }}>
            {dataColl?.total_snapshots?.toLocaleString() || 0}
          </div>
          <div className="text-muted" style={{ marginTop: 4 }}>Price Snapshots</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 20 }}>
          <div style={{ fontSize: 28, fontWeight: 700 }}>
            {dataColl?.tracked_items || 0}
          </div>
          <div className="text-muted" style={{ marginTop: 4 }}>Items Tracked</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 20 }}>
          <div style={{ fontSize: 28, fontWeight: 700 }}>
            {dataColl?.data_span_hours ? `${(dataColl.data_span_hours / 24).toFixed(1)}d` : '—'}
          </div>
          <div className="text-muted" style={{ marginTop: 4 }}>Data Span</div>
        </div>
      </div>

      {/* Learning Curve Chart */}
      {learningCurve.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ marginBottom: 16 }}>📈 Learning Curve — Accuracy Over Time</h3>
          <p className="text-muted" style={{ marginBottom: 12 }}>
            Shows how prediction accuracy changes as the AI learns from more data.
            An upward trend means the models are getting smarter.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={learningCurve}>
              <defs>
                <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fill: '#999', fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#999', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                formatter={(val, name) => [
                  name === 'accuracy' ? `${val}%` : val,
                  name === 'accuracy' ? 'Direction Accuracy' : 'Predictions',
                ]}
              />
              <Area type="monotone" dataKey="accuracy" stroke="#22c55e" fill="url(#accGrad)" strokeWidth={2} dot={{ r: 3 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Horizon Accuracy Bars */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ marginBottom: 16 }}>🎯 Accuracy by Prediction Horizon</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={horizonChartData} barSize={40}>
            <XAxis dataKey="horizon" tick={{ fill: '#999', fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fill: '#999', fontSize: 11 }} unit="%" />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
              formatter={(val, name) => [
                name === 'accuracy' ? `${val.toFixed(1)}%` : val.toLocaleString(),
                name === 'accuracy' ? 'Accuracy' : 'Training Samples',
              ]}
            />
            <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
              {horizonChartData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={!entry.trained ? '#555' : entry.accuracy >= 60 ? '#22c55e' : entry.accuracy >= 45 ? '#f59e0b' : '#ef4444'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Horizon Table */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ marginBottom: 16 }}>📊 Model Details per Horizon</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Horizon</th>
                <th>Status</th>
                <th>Direction Accuracy</th>
                <th>Price MAE</th>
                <th>Price MAPE</th>
                <th>Profit Accuracy</th>
                <th>Samples</th>
                <th>Last Trained</th>
              </tr>
            </thead>
            <tbody>
              {HORIZONS.map(h => {
                const hd = horizonData[h] || {};
                const md = metrics?.horizons?.[h] || {};
                return (
                  <tr key={h}>
                    <td style={{ fontWeight: 600 }}>{HORIZON_LABELS[h]}</td>
                    <td>
                      <span className={`badge ${hd.trained ? 'badge-green' : 'badge-yellow'}`}>
                        {hd.trained ? '✓ Trained' : '⏳ Pending'}
                      </span>
                    </td>
                    <td>
                      {hd.direction_accuracy != null ? (
                        <span style={{ fontWeight: 600, color: hd.direction_accuracy >= 60 ? '#22c55e' : hd.direction_accuracy >= 45 ? '#f59e0b' : '#ef4444' }}>
                          {hd.direction_accuracy}%
                        </span>
                      ) : '—'}
                    </td>
                    <td className="gp">{hd.price_mae != null ? `${hd.price_mae.toLocaleString()} GP` : '—'}</td>
                    <td>{hd.price_mape != null ? `${hd.price_mape}%` : '—'}</td>
                    <td>
                      {hd.profit_accuracy != null ? (
                        <span style={{ fontWeight: 600, color: hd.profit_accuracy >= 60 ? '#22c55e' : '#f59e0b' }}>
                          {hd.profit_accuracy}%
                        </span>
                      ) : '—'}
                    </td>
                    <td>{hd.sample_count?.toLocaleString() || 0}</td>
                    <td className="text-muted" style={{ fontSize: '0.85em' }}>
                      {hd.last_trained ? new Date(hd.last_trained).toLocaleString() : 'Never'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pipeline Info */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ marginBottom: 16 }}>⚙️ How the AI Learns</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
          <div>
            <h4 style={{ color: 'var(--color-primary)', marginBottom: 8 }}>📡 Data Collection</h4>
            <ul style={{ listStyle: 'none', padding: 0, lineHeight: 2 }}>
              <li>🔄 Live prices collected every <strong>{pipeline?.price_collection_interval || '10s'}</strong></li>
              <li>🧮 Feature vectors recomputed every <strong>{pipeline?.feature_computation_interval || '60s'}</strong></li>
              <li>📊 <strong>{dataColl?.total_snapshots?.toLocaleString() || 0}</strong> price snapshots stored</li>
              <li>📦 Raw data kept for <strong>{pipeline?.data_retention_raw || '7 days'}</strong></li>
              <li>📈 Aggregated candles kept <strong>{pipeline?.data_retention_aggregated || '30+ days'}</strong></li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--color-primary)', marginBottom: 8 }}>🧠 Model Training</h4>
            <ul style={{ listStyle: 'none', padding: 0, lineHeight: 2 }}>
              <li>🔄 Models retrain every <strong>{pipeline?.retrain_interval || '6h'}</strong></li>
              <li>🎯 Predictions run every <strong>{pipeline?.prediction_interval || '60s'}</strong></li>
              <li>📐 <strong>34 features</strong> per item (price, volume, technical, temporal, history)</li>
              <li>🏗️ <strong>18 models</strong> total (6 horizons × 3 targets)</li>
              <li>✅ Outcomes tracked automatically for accuracy measurement</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--color-primary)', marginBottom: 8 }}>🔬 ML Architecture</h4>
            <ul style={{ listStyle: 'none', padding: 0, lineHeight: 2 }}>
              <li>🌳 {status?.predictor?.forecaster?.backend === 'lightgbm' ? 'LightGBM' : 'Random Forest'} models</li>
              <li>📊 Direction classifier (up/down/flat)</li>
              <li>💰 Price regressor (predicted buy/sell)</li>
              <li>🎯 Confidence estimator (prediction reliability)</li>
              <li>📉 Time-series train/val split (no data leakage)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Live Accuracy Details */}
      {liveAcc && liveAcc.total_predictions > 0 && (
        <div className="card">
          <h3 style={{ marginBottom: 16 }}>📋 Prediction Outcome Tracking</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
            <div style={{ textAlign: 'center', padding: 16, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
              <div style={{ fontSize: 24, fontWeight: 700 }}>{liveAcc.total_predictions.toLocaleString()}</div>
              <div className="text-muted">Predictions Made</div>
            </div>
            <div style={{ textAlign: 'center', padding: 16, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
              <div style={{ fontSize: 24, fontWeight: 700 }}>{liveAcc.graded_predictions.toLocaleString()}</div>
              <div className="text-muted">Outcomes Verified</div>
            </div>
            <div style={{ textAlign: 'center', padding: 16, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#22c55e' }}>{liveAcc.correct_predictions.toLocaleString()}</div>
              <div className="text-muted">Correct Predictions</div>
            </div>
            <div style={{ textAlign: 'center', padding: 16, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: liveAcc.direction_accuracy_pct >= 55 ? '#22c55e' : '#f59e0b' }}>
                {liveAcc.direction_accuracy_pct != null ? `${liveAcc.direction_accuracy_pct}%` : '—'}
              </div>
              <div className="text-muted">Overall Accuracy</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
