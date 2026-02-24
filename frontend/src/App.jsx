import { BrowserRouter, Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import {
  LayoutDashboard, TrendingUp, Briefcase, BarChart3,
  Brain, Settings, LogOut, Bell, Upload, Users, ShieldOff, Search,
} from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Opportunities from './pages/Opportunities';
import Item from './pages/Item';
import Portfolio from './pages/Portfolio';
import ActiveTrades from './pages/ActiveTrades';
import Performance from './pages/Performance';
import ModelDashboard from './pages/ModelDashboard';
import Alerts from './pages/Alerts';
import SettingsPage from './pages/Settings';
import Import from './pages/Import';
import Blocklist from './pages/Blocklist';
import Login from './pages/Login';
import { createPriceSocket, api, clearToken } from './api/client';
import { AccountProvider, useAccount } from './hooks/useAccount';
import { ErrorProvider } from './components/ErrorPanel';
import './App.css';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/opportunities', label: 'Opportunities', icon: TrendingUp },
  { path: '/alerts', label: 'Alerts', icon: Bell },
  { path: '/portfolio', label: 'Portfolio', icon: Briefcase },
  { path: '/trades', label: 'Active Trades', icon: Briefcase },
  { path: '/performance', label: 'Performance', icon: BarChart3 },
  { path: '/import', label: 'Import', icon: Upload },
  { path: '/models', label: 'ML Models', icon: Brain },
  { path: '/blocklist', label: 'Blocklist', icon: ShieldOff },
  { path: '/settings', label: 'Settings', icon: Settings },
];

function AccountSelector() {
  const { accounts, activeAccount, setActiveAccount } = useAccount();
  if (!accounts.length) return null;
  return (
    <div className="account-selector">
      <Users size={14} />
      <select
        value={activeAccount || ''}
        onChange={(e) => setActiveAccount(e.target.value || null)}
      >
        <option value="">All Accounts</option>
        {accounts.map((a) => (
          <option key={a} value={a}>{a}</option>
        ))}
      </select>
    </div>
  );
}

function SidebarItemSearch({ onNavigate }) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const term = query.trim();
    if (!term) {
      setResults([]);
      setOpen(false);
      setLoading(false);
      return undefined;
    }

    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const payload = await api.searchItems(term, 20, {
          signal: ctrl.signal,
          timeoutMs: 10_000,
        });
        setResults(Array.isArray(payload) ? payload : (payload?.items || []));
        setOpen(true);
      } catch (err) {
        if (err?.name !== 'AbortError') {
          setResults([]);
          setOpen(true);
        }
      } finally {
        setLoading(false);
      }
    }, 250);

    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [query]);

  const selectItem = (itemId) => {
    setQuery('');
    setResults([]);
    setOpen(false);
    navigate(`/item/${itemId}`);
    if (onNavigate) onNavigate();
  };

  return (
    <div className="sidebar-item-search" onBlur={() => setTimeout(() => setOpen(false), 120)}>
      <Search size={13} className="sidebar-item-search-icon" />
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => query.trim() && setOpen(true)}
        placeholder="Search item or ID..."
      />
      {open && (
        <div className="sidebar-item-search-results">
          {loading ? (
            <div className="sidebar-item-search-empty">Searching...</div>
          ) : results.length === 0 ? (
            <div className="sidebar-item-search-empty">No items found</div>
          ) : (
            results.map((item) => (
              <button
                key={item.item_id}
                className="sidebar-item-search-result"
                onClick={() => selectItem(item.item_id)}
                type="button"
              >
                <span>{item.name}</span>
                <small>#{item.item_id}</small>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [livePrices, setLivePrices] = useState({});
  const [wsConnected, setWsConnected] = useState(false);
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const [alertCount, setAlertCount] = useState(0);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Check if user is logged in (uses Bearer token or cookie via api client)
  useEffect(() => {
    api.getMe()
      .then((data) => {
        if (data) setUser(data);
        setAuthChecked(true);
      })
      .catch((err) => {
        if (err.message && err.message.includes('401')) {
          setAuthRequired(true);
        }
        setAuthChecked(true);
      });
  }, []);

  // Only connect WebSocket if authenticated (or auth not required)
  useEffect(() => {
    if (authRequired && !user) return;
    const socket = createPriceSocket(
      (data) => setLivePrices(data),
      (connected) => setWsConnected(connected),
    );
    return () => socket.close();
  }, [authRequired, user]);

  // Poll unacknowledged alert count
  useEffect(() => {
    if (authRequired && !user) return;
    const poll = () => {
      api.getAlerts({ unacknowledged_only: true, limit: 1 })
        .then(d => setAlertCount(d.unacknowledged || 0))
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 30000);
    return () => clearInterval(id);
  }, [authRequired, user]);

  const handleLogout = async () => {
    try { await api.logout(); } catch { /* ignore */ }
    clearToken();
    setUser(null);
    setAuthRequired(true);
  };

  // Still checking auth
  if (!authChecked) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: '100vh', background: '#0f1923', color: '#8899aa',
      }}>
        Loading...
      </div>
    );
  }

  // Auth required but not logged in
  if (authRequired && !user) {
    return <Login />;
  }

  return (
    <ErrorProvider>
    <AccountProvider>
    <BrowserRouter>
      <div className="app">
        {/* Mobile menu overlay */}
        <div
          className={`mobile-overlay ${mobileMenuOpen ? 'visible' : ''}`}
          onClick={() => setMobileMenuOpen(false)}
        />
        <nav className={`sidebar ${mobileMenuOpen ? 'open' : ''}`}>
          <div className="sidebar-header">
            <h1 className="logo">OSRS<span>Flipper</span></h1>
            <div className="status-badge">
              <span className={`dot ${wsConnected ? 'online' : 'offline'}`} />
              {wsConnected ? 'Live' : 'Connecting...'}
            </div>
            <SidebarItemSearch onNavigate={() => setMobileMenuOpen(false)} />
            <AccountSelector />
          </div>
          <div className="nav-links">
            {NAV_ITEMS.map(({ path, label, icon }) => {
              const NavIcon = icon;
              return (
                <NavLink
                  key={path}
                  to={path}
                  end={path === '/'}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <NavIcon size={18} />
                  <span>{label}</span>
                  {path === '/alerts' && alertCount > 0 && (
                    <span className="alert-badge">{alertCount}</span>
                  )}
                </NavLink>
              );
            })}
          </div>
          <div className="sidebar-footer">
            {user && (
              <button onClick={handleLogout} className="nav-link" style={{
                background: 'none', border: 'none', cursor: 'pointer',
                width: '100%', textAlign: 'left',
              }}>
                <LogOut size={18} />
                <span>{user.username}</span>
              </button>
            )}
            <div className="version">v2.0 — AI Powered</div>
          </div>
        </nav>
        <main className="content">
          <button
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(o => !o)}
            style={{ marginBottom: 12 }}
          >
            ☰
          </button>
          <Routes>
            <Route path="/" element={<Dashboard prices={livePrices} />} />
            <Route path="/opportunities" element={<Opportunities prices={livePrices} />} />
            <Route path="/item/:itemId" element={<Item prices={livePrices} />} />
            <Route path="/portfolio" element={<Portfolio prices={livePrices} />} />
            <Route path="/trades" element={<ActiveTrades />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/import" element={<Import />} />
            <Route path="/models" element={<ModelDashboard />} />
            <Route path="/blocklist" element={<Blocklist />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
    </AccountProvider>
    </ErrorProvider>
  );
}
