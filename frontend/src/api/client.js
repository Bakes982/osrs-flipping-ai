// API base URL: VITE_API_URL must be set for production (Vercel env vars).
// Dev server (port 5173) falls back to localhost backend.
const API_BASE = import.meta.env.VITE_API_URL
  || (window.location.port === '5173'
    ? 'http://localhost:8001/api'
    : 'https://osrs-flipping-ai-backend-production.up.railway.app/api');

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

function authHeaders(hasBody = false) {
  const headers = {};
  if (hasBody) headers['Content-Type'] = 'application/json';
  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

async function fetchJSON(path, options = {}, retries = 3) {
  const url = `${API_BASE}${path}`;
  const hasBody = !!options.body;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        headers: authHeaders(hasBody),
        ...options,
      });

      // If the backend returns HTML instead of JSON, something is wrong.
      const contentType = res.headers.get('content-type') || '';
      const isHTML = contentType.includes('text/html');

      if ((res.status === 503 || isHTML) && attempt < retries) {
        const delay = 3000 * (attempt + 1);
        console.log(`Backend not ready (attempt ${attempt + 1}/${retries + 1}), retrying in ${delay / 1000}s...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }

      if (isHTML) {
        throw new Error('Backend returned unexpected HTML. It may be restarting.');
      }
      if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
      if (res.status === 204) return null;
      const text = await res.text();
      return text ? JSON.parse(text) : null;
    } catch (err) {
      if (attempt < retries && err instanceof TypeError) {
        // Network error (CORS block, DNS, offline) – retry
        const delay = 5000 * (attempt + 1);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw err;
    }
  }
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

  // Price History (GE timeseries from Wiki API)
  getPriceHistory(itemId, timestep = '1h') {
    return fetchJSON(`/prices/${itemId}/history?timestep=${timestep}`);
  },

  // Portfolio
  getPortfolio(player) {
    const qs = player ? `?player=${encodeURIComponent(player)}` : '';
    return fetchJSON(`/portfolio${qs}`);
  },
  getActivePositions(source, player) {
    const params = new URLSearchParams();
    if (source) params.set('source', source);
    if (player) params.set('player', player);
    const qs = params.toString();
    return fetchJSON(`/positions${qs ? '?' + qs : ''}`);
  },
  getSellOffers(player) {
    const qs = player ? `?player=${encodeURIComponent(player)}` : '';
    return fetchJSON(`/positions/selling${qs}`);
  },
  dismissPosition(tradeId) {
    return fetchJSON(`/positions/dismiss?trade_id=${tradeId}`, { method: 'POST' });
  },
  clearCsvPositions() {
    return fetchJSON('/positions/clear-csv', { method: 'POST' });
  },
  getTrades(params = {}) {
    const qs = new URLSearchParams(params).toString();
    return fetchJSON(`/trades${qs ? '?' + qs : ''}`);
  },
  getPerformance(player) {
    const qs = player ? `?player=${encodeURIComponent(player)}` : '';
    return fetchJSON(`/performance${qs}`);
  },

  // Trade Import
  async uploadTradesCSV(file) {
    const url = `${API_BASE}/trades/import`;
    const formData = new FormData();
    formData.append('file', file);
    const headers = {};
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(url, { method: 'POST', headers, body: formData });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },
  clearTradeHistory() {
    return fetchJSON('/trades/clear', { method: 'POST' });
  },

  // Model
  getModelMetrics() {
    return fetchJSON('/model/metrics');
  },
  getModelStatus() {
    return fetchJSON('/model/status');
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
  sendTop5Now() {
    return fetchJSON('/alerts/send-top5', { method: 'POST' });
  },
  getSendTop5Status() {
    return fetchJSON('/alerts/send-top5/status');
  },
  testWebhook() {
    return fetchJSON('/alerts/test-webhook', { method: 'POST' });
  },

  // Arbitrage
  getArbitrage() {
    return fetchJSON('/opportunities/arbitrage');
  },

  // Accounts
  getAccounts() {
    return fetchJSON('/accounts');
  },

  // Settings
  // Blocklist
  getBlocklist() {
    return fetchJSON('/blocklist');
  },
  setBlocklist(itemIds) {
    return fetchJSON('/blocklist', {
      method: 'POST',
      body: JSON.stringify({ item_ids: itemIds }),
    });
  },
  analyzeBlocklist() {
    return fetchJSON('/blocklist/analyze', { method: 'POST' });
  },

  getSettings() {
    return fetchJSON('/settings');
  },
  updateSettings(data) {
    return fetchJSON('/settings', {
      method: 'POST',
      body: JSON.stringify({ settings: data }),
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
