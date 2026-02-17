import { LogIn } from 'lucide-react';
import { API_BASE } from '../api/client';

export default function Login() {
  const handleLogin = () => {
    // Redirect to the backend's Discord OAuth login endpoint
    // API_BASE ends with /api, so strip it and append the auth path
    const backendBase = API_BASE.replace(/\/api$/, '');
    window.location.href = `${backendBase}/api/auth/login`;
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      background: '#0f1923',
      color: '#e0e0e0',
    }}>
      <div style={{
        textAlign: 'center',
        padding: '3rem',
        background: '#1a2332',
        borderRadius: '12px',
        border: '1px solid #2a3a4a',
        maxWidth: '400px',
        width: '100%',
      }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
          OSRS<span style={{ color: '#ffd700' }}>Flipper</span>
        </h1>
        <p style={{ color: '#8899aa', marginBottom: '2rem' }}>
          AI-powered Grand Exchange flipping assistant
        </p>
        <button
          onClick={handleLogin}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.85rem 2rem',
            fontSize: '1rem',
            fontWeight: '600',
            color: '#fff',
            background: '#5865F2',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'background 0.2s',
          }}
          onMouseOver={(e) => e.target.style.background = '#4752c4'}
          onMouseOut={(e) => e.target.style.background = '#5865F2'}
        >
          <LogIn size={20} />
          Sign in with Discord
        </button>
        <p style={{ color: '#556677', fontSize: '0.8rem', marginTop: '1.5rem' }}>
          Only authorised users can access the dashboard.
        </p>
      </div>
    </div>
  );
}
