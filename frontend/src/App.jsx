import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { useState, useEffect } from 'react';
import {
  LayoutDashboard, TrendingUp, Briefcase, BarChart3,
  Brain, Settings,
} from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Opportunities from './pages/Opportunities';
import ItemDetail from './pages/ItemDetail';
import Portfolio from './pages/Portfolio';
import Performance from './pages/Performance';
import ModelDashboard from './pages/ModelDashboard';
import SettingsPage from './pages/Settings';
import { createPriceSocket } from './api/client';
import './App.css';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/opportunities', label: 'Opportunities', icon: TrendingUp },
  { path: '/portfolio', label: 'Portfolio', icon: Briefcase },
  { path: '/performance', label: 'Performance', icon: BarChart3 },
  { path: '/models', label: 'ML Models', icon: Brain },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export default function App() {
  const [livePrices, setLivePrices] = useState({});
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    const socket = createPriceSocket((data) => {
      setLivePrices(data);
      setWsConnected(true);
    });
    return () => socket.close();
  }, []);

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
              </NavLink>
            ))}
          </div>
          <div className="sidebar-footer">
            <div className="version">v2.0 — AI Powered</div>
          </div>
        </nav>
        <main className="content">
          <Routes>
            <Route path="/" element={<Dashboard prices={livePrices} />} />
            <Route path="/opportunities" element={<Opportunities prices={livePrices} />} />
            <Route path="/item/:itemId" element={<ItemDetail prices={livePrices} />} />
            <Route path="/portfolio" element={<Portfolio prices={livePrices} />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/models" element={<ModelDashboard />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
