const API_BASE = 'http://localhost:8000/api';
const WS_BASE = 'ws://localhost:8000/ws';

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

export const api = {
  // Opportunities
  getOpportunities(params = {}) {
    const qs = new URLSearchParams(params).toString();
    return fetchJSON(`/opportunities${qs ? '?' + qs : ''}`);
  },
  getOpportunityDetail(itemId) {
    return fetchJSON(`/opportunities/${itemId}`);
  },

  // Predictions
  getPredictions(itemId, horizon) {
    const qs = horizon ? `?horizon=${horizon}` : '';
    return fetchJSON(`/predict/${itemId}${qs}`);
  },

  // Portfolio
  getPortfolio() {
    return fetchJSON('/portfolio');
  },
  getTrades(params = {}) {
    const qs = new URLSearchParams(params).toString();
    return fetchJSON(`/trades${qs ? '?' + qs : ''}`);
  },
  getPerformance() {
    return fetchJSON('/performance');
  },

  // Model
  getModelMetrics() {
    return fetchJSON('/model/metrics');
  },

  // Settings
  getSettings() {
    return fetchJSON('/settings');
  },
  updateSettings(data) {
    return fetchJSON('/settings', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};

export function createPriceSocket(onMessage) {
  let ws = null;
  let reconnectTimer = null;

  function connect() {
    ws = new WebSocket(`${WS_BASE}/prices`);
    ws.onopen = () => console.log('WebSocket connected');
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        onMessage(data);
      } catch { /* ignore parse errors */ }
    };
    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting in 5s...');
      reconnectTimer = setTimeout(connect, 5000);
    };
    ws.onerror = () => ws.close();
  }

  connect();

  return {
    close() {
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
    },
  };
}
