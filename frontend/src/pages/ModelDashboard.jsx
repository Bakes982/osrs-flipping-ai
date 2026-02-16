import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

const HORIZONS = ['1m', '5m', '30m', '2h', '8h', '24h'];

export default function ModelDashboard() {
  const { data: metrics, loading } = useApi(() => api.getModelMetrics(), [], 60000);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">ML Models</h2>
          <p className="page-subtitle">Model accuracy per prediction horizon</p>
        </div>
      </div>

      {loading ? <div className="loading">Loading model metrics...</div> : (
        !metrics?.length ? (
          <div className="card">
            <div className="empty">
              <h3 style={{ marginBottom: 8 }}>No model metrics yet</h3>
              <p>Models need at least 7 days of price data to train. The system is collecting data now.</p>
              <p style={{ marginTop: 8 }}>During this period, statistical methods (VWAP + trend clamping) are used for predictions.</p>
            </div>
          </div>
        ) : (
          <div className="card">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Horizon</th>
                  <th>Direction Accuracy</th>
                  <th>Price MAE</th>
                  <th>Price MAPE</th>
                  <th>Profit Accuracy</th>
                  <th>Samples</th>
                  <th>Model Version</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 500 }}>{m.horizon}</td>
                    <td>
                      <span className={`badge ${m.direction_accuracy > 0.6 ? 'badge-green' : m.direction_accuracy > 0.45 ? 'badge-yellow' : 'badge-red'}`}>
                        {(m.direction_accuracy * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="gp">{m.price_mae?.toLocaleString() || '—'}</td>
                    <td>{m.price_mape?.toFixed(2) || '—'}%</td>
                    <td>
                      <span className={`badge ${m.profit_accuracy > 0.6 ? 'badge-green' : 'badge-yellow'}`}>
                        {(m.profit_accuracy * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>{m.sample_count?.toLocaleString()}</td>
                    <td className="text-muted">{m.model_version}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}
    </div>
  );
}
