// API base URL: use VITE_API_URL env var for cross-domain deployment,
// fall back to relative /api for same-origin or localhost for dev server.
const API_BASE = import.meta.env.VITE_API_URL
  || (window.location.port === '5173' ? 'http://localhost:8001/api' : '/api');

// WebSocket base: derive from API_BASE
const _apiUrl = new URL(API_BASE, window.location.origin);
const WS_BASE = import.meta.env.VITE_WS_URL
  || `${_apiUrl.protocol === 'https:' ? 'wss:' : 'ws:'}//${_apiUrl.host}/ws`;

// ---------------------------------------------------------------------------
// Token management (stored in localStorage for cross-domain auth)
// ---------------------------------------------------------------------------

const TOKEN_KEY = 'flipping_ai_token';

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

// On page load, check for token in URL query params (OAuth callback redirect)
(function captureTokenFromURL() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token');
  if (token) {
    setToken(token);
    // Clean the URL
    window.history.replaceState({}, '', window.location.pathname);
  }
})();

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

function authHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: authHeaders(),
    credentials: 'include', // send cookies for same-origin fallback
    ...options,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// API methods
// ---------------------------------------------------------------------------

export const api = {
  // Auth
  getMe() {
    return fetchJSON('/auth/me');
  },
  logout() {
    return fetchJSON('/auth/logout', { method: 'POST' });
  },

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

  // Alerts
  getAlerts(params = {}) {
    const qs = new URLSearchParams(params).toString();
    return fetchJSON(`/alerts${qs ? '?' + qs : ''}`);
  },
  acknowledgeAlerts(data) {
    return fetchJSON('/alerts/acknowledge', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
  createPriceTarget(data) {
    return fetchJSON('/alerts/price-target', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
  getPriceTargets() {
    return fetchJSON('/alerts/price-targets');
  },
  deletePriceTarget(itemId, direction) {
    const qs = direction ? `?direction=${direction}` : '';
    return fetchJSON(`/alerts/price-target/${itemId}${qs}`, { method: 'DELETE' });
  },

  // Arbitrage
  getArbitrage() {
    return fetchJSON('/opportunities/arbitrage');
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

// ---------------------------------------------------------------------------
// WebSocket with auth
// ---------------------------------------------------------------------------

export function createPriceSocket(onMessage) {
  let ws = null;
  let reconnectTimer = null;

  function connect() {
    // Append token as query param for WS auth
    const token = getToken();
    const url = token ? `${WS_BASE}/prices?token=${token}` : `${WS_BASE}/prices`;
    ws = new WebSocket(url);
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

// Export the base URL for use elsewhere
export { API_BASE };
