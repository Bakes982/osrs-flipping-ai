/**
 * Global error panel — surfaces API/runtime errors with full diagnostic info.
 * Wrap your app with <ErrorProvider> and errors reported via useApi() or
 * addError() will appear in a dismissable panel at the bottom-right.
 */
import { createContext, useContext, useState, useCallback } from 'react';

export const ErrorContext = createContext(null);

// Maximum errors shown simultaneously
const MAX_ERRORS = 5;

export function ErrorProvider({ children }) {
  const [errors, setErrors] = useState([]);

  const addError = useCallback((err, context = '') => {
    const entry = {
      id: `${Date.now()}-${Math.random()}`,
      message: err?.message || String(err),
      status: err?.status,
      detail: err?.detail,
      url: err?.url,
      context,
      time: new Date().toISOString(),
    };
    setErrors(prev => [entry, ...prev].slice(0, MAX_ERRORS));
  }, []);

  const dismiss = useCallback((id) => {
    setErrors(prev => prev.filter(e => e.id !== id));
  }, []);

  const dismissAll = useCallback(() => setErrors([]), []);

  return (
    <ErrorContext.Provider value={{ addError, dismiss, dismissAll }}>
      {children}
      <ErrorPanel errors={errors} onDismiss={dismiss} onDismissAll={dismissAll} />
    </ErrorContext.Provider>
  );
}

export function useErrorPanel() {
  return useContext(ErrorContext);
}

// ---------------------------------------------------------------------------
// Panel UI
// ---------------------------------------------------------------------------

function copyDiagnostics(errors) {
  const text = errors.map(e => [
    `[${e.time}] ${e.context ? `(${e.context}) ` : ''}${e.message}`,
    e.status  ? `  HTTP Status : ${e.status}` : null,
    e.url     ? `  URL         : ${e.url}` : null,
    e.detail  ? `  Detail      : ${e.detail}` : null,
  ].filter(Boolean).join('\n')).join('\n\n');

  navigator.clipboard.writeText(text).catch(() => {});
}

function ErrorPanel({ errors, onDismiss, onDismissAll }) {
  if (!errors.length) return null;

  return (
    <div style={{
      position: 'fixed',
      bottom: 16,
      right: 16,
      zIndex: 9999,
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      maxWidth: 420,
      width: '100%',
      pointerEvents: 'none',
    }}>
      {errors.length > 1 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, pointerEvents: 'auto' }}>
          <button
            onClick={() => copyDiagnostics(errors)}
            style={btnStyle('#1e3a5f', '#60a5fa')}
            title="Copy all diagnostics to clipboard"
          >
            Copy all
          </button>
          <button
            onClick={onDismissAll}
            style={btnStyle('#3a1e1e', '#f87171')}
          >
            Dismiss all
          </button>
        </div>
      )}
      {errors.map(err => (
        <ErrorCard key={err.id} err={err} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function ErrorCard({ err, onDismiss }) {
  const [copied, setCopied] = useState(false);

  function copyThis() {
    copyDiagnostics([err]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div style={{
      background: '#1a1a2e',
      border: '1px solid #7f1d1d',
      borderLeft: '4px solid #ef4444',
      borderRadius: 8,
      padding: '10px 12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
      pointerEvents: 'auto',
      fontFamily: 'monospace',
      fontSize: 12,
      color: '#e5e7eb',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 6 }}>
        <span style={{ color: '#ef4444', fontWeight: 700, flexShrink: 0 }}>
          {err.status ? `HTTP ${err.status}` : 'Error'}
        </span>
        {err.context && (
          <span style={{ color: '#9ca3af', fontSize: 11 }}>[{err.context}]</span>
        )}
        <span style={{ marginLeft: 'auto', color: '#6b7280', fontSize: 10, flexShrink: 0 }}>
          {new Date(err.time).toLocaleTimeString()}
        </span>
        <button
          onClick={() => onDismiss(err.id)}
          style={{ background: 'none', border: 'none', color: '#6b7280', cursor: 'pointer', padding: 0, fontSize: 14, lineHeight: 1 }}
          title="Dismiss"
        >
          ×
        </button>
      </div>

      {/* Message */}
      <div style={{ color: '#fca5a5', marginBottom: err.url || err.detail ? 6 : 0, wordBreak: 'break-word' }}>
        {err.message}
      </div>

      {/* Extra diagnostic fields */}
      {err.url && (
        <div style={{ color: '#9ca3af', fontSize: 11, marginTop: 2 }}>
          <span style={{ color: '#6b7280' }}>URL: </span>{err.url}
        </div>
      )}
      {err.detail && err.detail !== err.message && (
        <div style={{ color: '#9ca3af', fontSize: 11, marginTop: 2, wordBreak: 'break-word' }}>
          <span style={{ color: '#6b7280' }}>Detail: </span>{err.detail}
        </div>
      )}

      {/* Actions */}
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button
          onClick={copyThis}
          style={btnStyle('#1e3a5f', '#60a5fa')}
          title="Copy diagnostic info"
        >
          {copied ? 'Copied!' : 'Copy diagnostic'}
        </button>
      </div>
    </div>
  );
}

function btnStyle(bg, color) {
  return {
    background: bg,
    border: `1px solid ${color}`,
    color,
    borderRadius: 4,
    padding: '3px 10px',
    fontSize: 11,
    cursor: 'pointer',
    fontFamily: 'monospace',
  };
}
