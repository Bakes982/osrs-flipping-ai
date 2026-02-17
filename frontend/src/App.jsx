import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { useState, useEffect } from 'react';
import {
  LayoutDashboard, TrendingUp, Briefcase, BarChart3,
  Brain, Settings, LogOut, Bell,
} from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Opportunities from './pages/Opportunities';
import ItemDetail from './pages/ItemDetail';
import Portfolio from './pages/Portfolio';
import Performance from './pages/Performance';
import ModelDashboard from './pages/ModelDashboard';
import Alerts from './pages/Alerts';
import SettingsPage from './pages/Settings';
import Login from './pages/Login';
import { createPriceSocket, api } from './api/client';
import './App.css';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/opportunities', label: 'Opportunities', icon: TrendingUp },
  { path: '/alerts', label: 'Alerts', icon: Bell },
  { path: '/portfolio', label: 'Portfolio', icon: Briefcase },
  { path: '/performance', label: 'Performance', icon: BarChart3 },
  { path: '/models', label: 'ML Models', icon: Brain },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export default function App() {
  const [livePrices, setLivePrices] = useState({});
  const [wsConnected, setWsConnected] = useState(false);
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const [alertCount, setAlertCount] = useState(0);

  // Check if user is logged in
  useEffect(() => {
    fetch('/api/auth/me', { credentials: 'same-origin' })
      .then((r) => {
        if (r.ok) return r.json();
        if (r.status === 401) {
          setAuthRequired(true);
          return null;
        }
        // Auth not configured or other error - allow open access
        return null;
      })
      .then((data) => {
        if (data) setUser(data);
        setAuthChecked(true);
      })
      .catch(() => setAuthChecked(true));
  }, []);

  // Only connect WebSocket if authenticated (or auth not required)
  useEffect(() => {
    if (authRequired && !user) return;
    const socket = createPriceSocket((data) => {
      setLivePrices(data);
      setWsConnected(true);
    });
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
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' });
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
    <BrowserRouter>
      <div className="app">
        <nav className="sidebar">
          <div className="sidebar-header">
            <h1 className="logo">OSRS<span>Flipper</span></h1>
            <div className="status-badge">
              <span className={`dot ${wsConnected ? 'online' : 'offline'}`} />
              {wsConnected ? 'Live' : 'Connecting...'}
            </div>
          </div>
          <div className="nav-links">
            {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                end={path === '/'}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                <span>{label}</span>
                {path === '/alerts' && alertCount > 0 && (
                  <span className="alert-badge">{alertCount}</span>
                )}
              </NavLink>
            ))}
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
          <Routes>
            <Route path="/" element={<Dashboard prices={livePrices} />} />
            <Route path="/opportunities" element={<Opportunities prices={livePrices} />} />
            <Route path="/item/:itemId" element={<ItemDetail prices={livePrices} />} />
            <Route path="/portfolio" element={<Portfolio prices={livePrices} />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/models" element={<ModelDashboard />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
