import { useState, useRef } from 'react';
import { Upload, FileText, Trash2, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

export default function Import() {
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [clearing, setClearing] = useState(false);
  const [clearResult, setClearResult] = useState(null);

  // Load existing stats
  const { data: perf, reload: reloadPerf } = useApi(() => api.getPerformance(), [], 0);
  const { data: trades, reload: reloadTrades } = useApi(() => api.getTrades({ limit: 10 }), [], 0);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f && f.name.endsWith('.csv')) {
      setFile(f);
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.uploadTradesCSV(file);
      setResult(res);
      reloadPerf();
      reloadTrades();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm('This will delete ALL trade history and flip records. Are you sure?')) return;
    setClearing(true);
    setClearResult(null);
    try {
      const res = await api.clearTradeHistory();
      setClearResult(res);
      setResult(null);
      reloadPerf();
      reloadTrades();
    } catch (err) {
      setError(err.message);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Import Trades</h2>
          <p className="page-subtitle">Upload your flip history CSV for AI analysis</p>
        </div>
      </div>

      {/* Current Data Summary */}
      <div className="stats-grid" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="card-title">Flips in Database</div>
          <div className="card-value">{perf?.total_flips || 0}</div>
        </div>
        <div className="card">
          <div className="card-title">Total Profit</div>
          <div className="card-value text-green">{formatGP(perf?.total_profit || 0)}</div>
        </div>
        <div className="card">
          <div className="card-title">Win Rate</div>
          <div className="card-value">{perf?.win_rate || 0}%</div>
        </div>
        <div className="card">
          <div className="card-title">Recent Trades</div>
          <div className="card-value">{trades?.length || 0}</div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>
          <Upload size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
          Upload CSV File
        </h3>
        <p className="text-muted" style={{ fontSize: 12, marginBottom: 16, lineHeight: 1.6 }}>
          Upload a CSV file exported from RuneLite flipping plugins (e.g. Flipping Utilities).
          <br />
          Expected columns: <code>First buy time, Last sell time, Account, Item, Status, Bought, Sold,
          Avg. buy price, Avg. sell price, Tax, Profit, Profit ea.</code>
          <br />
          Status values: <strong>FINISHED</strong> (completed flips), <strong>BUYING</strong> / <strong>SELLING</strong> (active offers)
        </p>

        {/* Drop zone */}
        <div
          onClick={() => fileRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{
            border: '2px dashed rgba(6,182,212,0.3)',
            borderRadius: 12,
            padding: '40px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            background: file ? 'rgba(16,185,129,0.05)' : 'rgba(6,182,212,0.02)',
            transition: 'all 0.2s',
            marginBottom: 16,
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          {file ? (
            <div>
              <FileText size={32} style={{ color: 'var(--green)', marginBottom: 8 }} />
              <div style={{ fontSize: 14, fontWeight: 500 }}>{file.name}</div>
              <div className="text-muted" style={{ fontSize: 12 }}>
                {(file.size / 1024).toFixed(1)} KB — Click to change
              </div>
            </div>
          ) : (
            <div>
              <Upload size={32} style={{ color: 'var(--cyan)', marginBottom: 8 }} />
              <div style={{ fontSize: 14 }}>Drop CSV file here or click to browse</div>
              <div className="text-muted" style={{ fontSize: 12 }}>Supports .csv files</div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 12 }}>
          <button
            className="btn"
            onClick={handleUpload}
            disabled={!file || uploading}
            style={{
              background: 'var(--green)',
              color: '#000',
              opacity: !file || uploading ? 0.5 : 1,
              padding: '10px 24px',
              fontWeight: 600,
            }}
          >
            {uploading ? (
              <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Importing...</>
            ) : (
              <><Upload size={14} /> Import Trades</>
            )}
          </button>
          <button
            className="btn"
            onClick={handleClear}
            disabled={clearing}
            style={{
              background: 'transparent',
              border: '1px solid rgba(239,68,68,0.4)',
              color: 'var(--red)',
              padding: '10px 24px',
            }}
          >
            <Trash2 size={14} /> {clearing ? 'Clearing...' : 'Clear All Data'}
          </button>
        </div>
      </div>

      {/* Import Result */}
      {result && (
        <div className="card" style={{
          marginBottom: 24,
          borderColor: result.errors?.length > 0 ? 'rgba(245,158,11,0.3)' : 'rgba(16,185,129,0.3)',
          background: result.errors?.length > 0
            ? 'rgba(245,158,11,0.03)'
            : 'rgba(16,185,129,0.03)',
        }}>
          <h3 style={{ fontSize: 14, marginBottom: 16, color: 'var(--green)' }}>
            <CheckCircle size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
            Import Complete
          </h3>
          <div className="stats-grid" style={{ marginBottom: 16 }}>
            <div>
              <div className="card-title">Total Rows</div>
              <div style={{ fontSize: 20, fontWeight: 600 }}>{result.total_rows}</div>
            </div>
            <div>
              <div className="card-title">Flips Imported</div>
              <div style={{ fontSize: 20, fontWeight: 600, color: 'var(--green)' }}>{result.flips_imported}</div>
            </div>
            <div>
              <div className="card-title">Active Trades</div>
              <div style={{ fontSize: 20, fontWeight: 600, color: 'var(--cyan)' }}>{result.active_trades_imported}</div>
            </div>
            <div>
              <div className="card-title">Skipped</div>
              <div style={{ fontSize: 20, fontWeight: 600, color: 'var(--yellow)' }}>{result.skipped}</div>
            </div>
          </div>

          {result.items_not_found?.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: 'var(--yellow)', marginBottom: 4 }}>
                <AlertTriangle size={12} style={{ verticalAlign: 'middle' }} /> Items not found in Wiki
                (imported with ID 0):
              </div>
              <div className="text-muted" style={{ fontSize: 11 }}>
                {result.items_not_found.join(', ')}
              </div>
            </div>
          )}

          {result.errors?.length > 0 && (
            <div>
              <div style={{ fontSize: 12, color: 'var(--red)', marginBottom: 4 }}>Errors:</div>
              <div className="text-muted" style={{ fontSize: 11, maxHeight: 100, overflow: 'auto' }}>
                {result.errors.map((e, i) => <div key={i}>{e}</div>)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Clear Result */}
      {clearResult && (
        <div className="card" style={{ marginBottom: 24, borderColor: 'rgba(239,68,68,0.3)' }}>
          <h3 style={{ fontSize: 14, color: 'var(--red)' }}>Data Cleared</h3>
          <div className="text-muted" style={{ fontSize: 12 }}>
            Deleted {clearResult.trades_deleted} trades and {clearResult.flips_deleted} flip records.
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card" style={{ marginBottom: 24, borderColor: 'rgba(239,68,68,0.3)' }}>
          <div style={{ color: 'var(--red)', fontSize: 13 }}>
            <AlertTriangle size={14} style={{ verticalAlign: 'middle' }} /> {error}
          </div>
        </div>
      )}

      {/* How it works */}
      <div className="card">
        <h3 style={{ fontSize: 14, marginBottom: 12 }}>How It Works</h3>
        <div className="text-muted" style={{ fontSize: 12, lineHeight: 1.8 }}>
          <strong>1. Export from RuneLite</strong> — Use "Flipping Utilities" plugin → Export → CSV
          <br />
          <strong>2. Upload here</strong> — The system parses your trades and resolves item IDs via OSRS Wiki
          <br />
          <strong>3. Completed flips</strong> (FINISHED) are stored as profit/loss records with buy/sell prices, tax, and timing
          <br />
          <strong>4. Active offers</strong> (BUYING/SELLING) become portfolio holdings
          <br />
          <strong>5. AI Analysis</strong> — Dashboard, Performance, and Portfolio pages now show your real data
          <br />
          <strong>6. Re-import</strong> — Use "Clear All Data" first, then upload again to avoid duplicates
        </div>
      </div>
    </div>
  );
}
