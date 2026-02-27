import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RefreshCw, Search, TrendingUp, TrendingDown, Minus, Shield,
  BarChart3, Check, AlertTriangle, Sword, FlaskConical,
  Pickaxe, ShoppingBag, Trophy, Filter, Wallet, Edit2, X,
} from 'lucide-react';
import { api, API_BASE } from '../api/client';
import { useApi } from '../hooks/useApi';
import MarketSearchPanel from '../components/MarketSearchPanel';

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function formatGP(n) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function formatQty(n) {
  const v = Number(n || 0);
  if (!Number.isFinite(v)) return '0';
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}m`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toLocaleString();
}

function timeAgo(ts) {
  if (!ts) return 'Never';
  let epoch = Number(ts);
  if (!Number.isFinite(epoch)) {
    const parsed = Date.parse(String(ts));
    if (!Number.isFinite(parsed)) return 'Unknown';
    epoch = parsed / 1000;
  }
  const delta = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (delta < 10) return 'just now';
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  return `${Math.floor(delta / 3600)}h ago`;
}

const IMG = (id) => `https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=${id}`;

function parseGP(str) {
  if (!str) return 0;
  const s = String(str).replace(/,/g, '').trim().toLowerCase();
  const n = parseFloat(s);
  if (!Number.isFinite(n)) return 0;
  if (s.endsWith('b')) return Math.round(n * 1e9);
  if (s.endsWith('m')) return Math.round(n * 1e6);
  if (s.endsWith('k')) return Math.round(n * 1e3);
  return Math.round(n);
}

/* ── Item category classifier ────────────────────────────────────────────── */

function classifyItem(opp) {
  const name = (opp.name || opp.item_name || '').toLowerCase();

  // ── 1. Construction flatpacks & materials (catch before broad keywords) ───
  if (
    name.includes('flatpack') || name.includes('wardrobe') ||
    name.includes('bookcase') || name.includes('bolts of cloth') ||
    name.includes('gold leaf') || name.includes('marble block') ||
    name.includes('carved teak') || name.includes('carved mahogany')
  ) return 'skilling';

  // ── 2. PvP / Wilderness ──────────────────────────────────────────────────
  const PVP = [
    "vesta's", "statius'", "morrigan's", "zuriel's", 'corrupt dragon',
    "craw's", "viggora's", "thammaron's", 'ursine chainmace', 'webweaver bow',
    'accursed sceptre', 'volatile orb', 'ancient godsword', 'forgotten brew',
    "ancient d'hide", 'ancient cloak', 'ancient coif',
    'blighted super restore', 'blighted vengeance', 'blighted ice barrage',
    'blighted ancient', 'blighted meleeing', 'revenant ether',
  ];
  if (PVP.some(k => name.includes(k))) return 'pvp';

  // ── 3. PvM gear ───────────────────────────────────────────────────────────
  const PVM = [
    // Melee armour
    'torva', 'bandos', 'inquisitor',
    // Ranged armour
    'armadyl', 'masori', 'justiciar',
    // Magic armour
    'ancestral',
    // Barrows sets
    'dharok', 'guthan', 'karil', 'torag', 'verac', 'ahrim', 'barrows',
    // Scythe / high-tier melee weapons
    'scythe of vitur', 'ghrazi rapier', 'dragon hunter lance',
    'dragon hunter crossbow', 'dragon warhammer', 'elder maul', 'voidwaker',
    'dragon claws', 'osmumten\'s fang', 'partisan',
    // Ranged weapons
    'twisted bow', 'toxic blowpipe', 'blowpipe', 'zaryte',
    'crystal bow', 'bowfa',
    // Magic weapons
    'tumeken\'s shadow', 'sanguinesti', 'kodai', 'toxic staff of the dead',
    // Rings (DT2 / BIS)
    'bellator ring', 'ultor ring', 'magus ring', 'venator ring',
    'bellator vestige', 'ultor vestige', 'magus vestige', 'venator vestige',
    'berserker ring', 'archers ring', 'seers ring', 'tyrannical ring',
    'treasonous ring', 'lightbearer', 'brimstone ring', 'ring of endurance',
    'ring of shadows',
    // Amulets
    'amulet of torture', 'necklace of anguish', 'tormented bracelet',
    'ring of suffering', 'amulet of fury', 'occult necklace',
    'amulet of blood fury', 'blood fury',
    'berserker necklace',
    // Boots
    'primordial boots', 'pegasian boots', 'eternal boots',
    // Helms / Capes
    'slayer helmet', 'black mask', 'serpentine helm',
    'fire cape', 'infernal cape', 'fighter torso', 'void knight',
    // Shields / sigils
    'avernic defender', 'dragonfire shield', "elidinis' ward",
    'spirit shield', 'arcane sigil', 'elysian sigil', 'spectral sigil',
    // Crystal armour
    'crystal helm', 'crystal body', 'crystal legs', 'crystal shield',
    // Other BIS / notable
    'imbued heart', 'rune pouch', "ava's assembler", "ava's accumulator",
    'pegasian crystal', 'eternal crystal', 'primordial crystal',
    // 2026 additions
    'tonalztics of ralos', 'echo crystal', 'sunfire fanatic',
  ];
  if (PVM.some(k => name.includes(k))) return 'pvm';

  // ── 4. Supplies: Food, Potions, Runes, Ammunition ────────────────────────
  const SUPPLY = [
    // Food
    'shark', 'anglerfish', 'manta ray', 'karambwan', 'dark crab',
    'tuna potato', 'monkfish', 'swordfish', 'lobster', 'bass', 'trout',
    'cooked', 'wine of',
    // Potions (match any dose: (4), (3), etc.)
    'potion', 'brew', 'restore', 'overload', 'stamina',
    'super combat', 'super attack', 'super strength', 'super defence',
    'ranging potion', 'magic potion', 'super energy', 'agility potion',
    'bastion potion', 'battlemage potion', 'prayer potion',
    'divine super', 'divine ranging', 'divine bastion', 'divine battlemage',
    'anti-', 'antivenin', 'antipoison', 'antidote', 'relicym',
    // Runes (' rune' with leading space avoids matching 'runite')
    ' rune', 'soul rune', 'death rune', 'blood rune', 'chaos rune',
    'nature rune', 'law rune', 'wrath rune', 'cosmic rune', 'astral rune',
    'mind rune', 'body rune', 'aether rune', 'sunfire rune',
    // Ammunition
    'cannonball', 'dart tip', 'arrowhead', 'arrow shaft',
    ' arrow', 'broad bolt', 'opal bolt', 'sapphire bolt', 'emerald bolt',
    'ruby bolt', 'diamond bolt', 'dragon bolt', 'runite bolt',
    'amethyst dart', 'dragon dart', 'rune dart', 'adamant dart', 'mithril dart',
    ' javelin', 'chinchompa', 'throwing axe', 'atlatl dart',
    // Teleport tabs
    'teleport to house', 'teletab',
  ];
  if (SUPPLY.some(k => name.includes(k))) return 'supplies';

  // ── 5. Skilling materials & gathering ────────────────────────────────────
  const SKILLING = [
    // Mining & Smithing
    ' ore', ' bar', 'coal', 'runite', 'adamantite', 'mithril',
    // Woodcutting / FM
    ' log', ' logs', 'magic log',
    // Farming / Herblore
    ' seed', ' herb', 'grimy ', 'clean ', 'torstol', 'ranarr weed',
    'snapdragon', 'lantadyme', 'dwarf weed', 'cadantine', 'kwuarm',
    'avantoe', 'irit', 'harralander', 'toadflax', 'marrentill', 'guam',
    'snape grass', 'crushed nest', 'demon tears',
    // Fishing
    'raw ', 'feather', 'fish',
    // Crafting
    'leather', 'dragon hide', 'd-hide', 'uncut ', 'sapphire', 'emerald',
    'ruby', 'diamond', 'dragonstone', 'amethyst', 'onyx gem',
    'bowstring', 'wool', 'flax', 'vial', 'chisel', 'needle', 'thread',
    'tiara', 'gold ore', 'gold bar',
    // Runecrafting
    'essence',
    // Construction
    'mahogany plank', 'teak plank', 'limestone brick', 'marble',
    // Misc gathering
    'bone', 'ashes', 'dust', 'powder', 'sand', 'bucket of',
    'tooth half', 'loop half',
    // Sailing (2025/26)
    'ironwood plank', 'rosewood', 'camphor log', 'narwhal horn', 'squid beak',
  ];
  if (SKILLING.some(k => name.includes(k))) return 'skilling';

  return 'merch';
}

/* ── Score colour ────────────────────────────────────────────────────────── */

function scoreColor(score) {
  if (score >= 70) return '#22c55e';
  if (score >= 55) return '#06b6d4';
  if (score >= 40) return '#f59e0b';
  return '#ef4444';
}

function scoreRing(score) {
  const pct  = Math.min(100, Math.max(0, score ?? 0));
  const c    = scoreColor(pct);
  const r    = 18;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return { pct, c, r, circ, dash };
}

function ScoreRing({ score }) {
  const { pct, c, r, circ, dash } = scoreRing(score);
  return (
    <div style={{ position: 'relative', width: 44, height: 44, flexShrink: 0 }}>
      <svg width="44" height="44" viewBox="0 0 44 44">
        <circle cx="22" cy="22" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="3.5" />
        <circle cx="22" cy="22" r={r} fill="none" stroke={c} strokeWidth="3.5"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 22 22)" />
      </svg>
      <span style={{
        position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700, color: c,
      }}>
        {pct.toFixed(0)}
      </span>
    </div>
  );
}

/* ── Mini score bars ─────────────────────────────────────────────────────── */

function MiniBar({ value, max = 100, color }) {
  const pct = Math.min(100, Math.max(0, ((value || 0) / max) * 100));
  return (
    <div style={{ flex: 1, height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
      <div style={{ height: '100%', width: `${pct}%`, borderRadius: 2, background: color || scoreColor(pct), transition: 'width 0.4s' }} />
    </div>
  );
}

/* ── Trend indicator ─────────────────────────────────────────────────────── */

function TrendIcon({ trend }) {
  if (trend === 'STRONG_UP')   return <TrendingUp size={13} color="#22c55e" />;
  if (trend === 'UP')          return <TrendingUp size={13} color="#86efac" />;
  if (trend === 'DOWN')        return <TrendingDown size={13} color="#ef4444" />;
  if (trend === 'STRONG_DOWN') return <TrendingDown size={13} color="#fca5a5" />;
  return <Minus size={13} color="#6b7280" />;
}

/* ── Dump sparkline ──────────────────────────────────────────────────────── */

function DumpSparkline({ dumpPrice, refAvg }) {
  const low  = Math.min(dumpPrice || 0, refAvg || 0) || 1;
  const high = Math.max(dumpPrice || 0, refAvg || 0) || 1;
  const norm = (v) => 26 - ((v - low) / Math.max(1, high - low)) * 18;
  const y1   = norm(refAvg || low);
  const y2   = norm(dumpPrice || low);
  return (
    <svg width="80" height="30" viewBox="0 0 80 30" aria-hidden>
      <polyline points={`2,${y1} 40,${(y1 + y2) / 2} 78,${y2}`}
        fill="none" stroke="var(--cyan)" strokeWidth="1.8" />
      <circle cx="78" cy={y2} r="2.5" fill="var(--red)" />
    </svg>
  );
}

/* ── Category definitions ────────────────────────────────────────────────── */

const CATEGORIES = [
  { key: 'all',      label: 'All',          Icon: BarChart3,    color: '#06b6d4' },
  { key: 'pvm',      label: 'PvM Gear',     Icon: Sword,        color: '#22c55e' },
  { key: 'pvp',      label: 'PvP Gear',     Icon: Shield,       color: '#ef4444' },
  { key: 'supplies', label: 'Supplies',     Icon: FlaskConical, color: '#f59e0b' },
  { key: 'skilling', label: 'Skilling',     Icon: Pickaxe,      color: '#a78bfa' },
  { key: 'merch',    label: 'Merch',        Icon: ShoppingBag,  color: '#fb923c' },
  { key: 'dumps',    label: 'Dumps',        Icon: Trophy,       color: '#f43f5e' },
];

/* ── Opportunity Card ────────────────────────────────────────────────────── */

function OpportunityCard({ opp, rank, activeScoreMode, freeSlots, activeTrades,
  acceptingId, replaceForItem, setReplaceForItem, onAccept, nav }) {
  const [expanded, setExpanded] = useState(false);

  const score     = activeScoreMode === 'margin_hunter'
    ? (opp.margin_hunter_score ?? opp.flip_score ?? 0)
    : (opp.flip_score ?? 0);
  const sc        = scoreColor(score);
  const margin    = opp.margin_gp ?? opp.margin ?? 0;
  const profit    = opp.potential_profit ?? opp.expected_profit ?? 0;
  const roi       = opp.roi_pct ?? 0;
  const vol       = opp.volume_5m ?? opp.volume ?? 0;
  const conf      = opp.confidence ?? opp.ml_confidence;

  const geLimit4h    = Number(opp.ge_limit_4h || 0);
  const qtySuggested = Number(opp.qty_suggested || 0);
  const qtyRaw       = Number(opp.qty_raw ?? opp.qty_suggested ?? 0);
  const geCapped     = geLimit4h > 0 && qtyRaw > qtySuggested;

  return (
    <div style={{
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      borderRadius: 12,
      overflow: 'hidden',
      transition: 'box-shadow 0.2s, border-color 0.2s',
      display: 'flex',
      flexDirection: 'column',
    }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = sc; e.currentTarget.style.boxShadow = `0 0 0 1px ${sc}22, 0 4px 20px ${sc}15`; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.boxShadow = 'none'; }}
    >

      {/* ── Card header ── */}
      <div style={{ padding: '12px 14px 10px', display: 'flex', alignItems: 'center', gap: 10 }}>
        {/* Rank badge */}
        <span style={{
          fontSize: 10, fontWeight: 700, color: 'var(--text-muted)',
          minWidth: 18, textAlign: 'center',
        }}>#{rank}</span>

        {/* Item image */}
        <img src={IMG(opp.item_id)} alt="" width={36} height={36}
          style={{ imageRendering: 'pixelated', flexShrink: 0, borderRadius: 6, background: 'rgba(0,0,0,0.2)' }}
          onError={e => { e.target.style.display = 'none'; }} />

        {/* Name + badges */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontWeight: 700, fontSize: 13, whiteSpace: 'nowrap',
            overflow: 'hidden', textOverflow: 'ellipsis', color: 'var(--text-primary)',
          }}>
            {opp.name || opp.item_name}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 3, flexWrap: 'wrap' }}>
            <TrendIcon trend={opp.trend} />
            {conf != null && (
              <span style={{ fontSize: 10, color: conf > 0.7 ? '#22c55e' : conf > 0.5 ? '#f59e0b' : '#6b7280' }}>
                AI {(conf * 100).toFixed(0)}%
              </span>
            )}
            {qtySuggested > 0 && (
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                ×{formatQty(qtySuggested)}{geCapped ? ' (GE cap)' : ''}
              </span>
            )}
          </div>
        </div>

        {/* Score ring */}
        <ScoreRing score={score} />
      </div>

      {/* ── Price strip ── */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        background: 'rgba(0,0,0,0.18)', borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ padding: '7px 12px', borderRight: '1px solid var(--border)' }}>
          <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 2 }}>Buy</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#4ade80' }}>{formatGP(opp.buy_price)}</div>
        </div>
        <div style={{ padding: '7px 12px' }}>
          <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 2 }}>Sell</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8' }}>{formatGP(opp.sell_price)}</div>
        </div>
      </div>

      {/* ── Stats grid ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0 }}>
        {[
          { label: 'Margin',  value: formatGP(margin),           color: '#f59e0b' },
          { label: 'Profit',  value: `+${formatGP(profit)}`,     color: '#22c55e' },
          { label: 'ROI',     value: `${roi.toFixed(1)}%`,       color: roi > 2 ? '#22c55e' : roi > 0 ? '#06b6d4' : '#ef4444' },
          { label: 'Volume',  value: formatQty(vol),             color: 'var(--text-primary)' },
        ].map(({ label, value, color }, idx) => (
          <div key={label} style={{
            padding: '8px 10px',
            borderRight: idx < 3 ? '1px solid var(--border)' : 'none',
            borderBottom: '1px solid var(--border)',
          }}>
            <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 3 }}>{label}</div>
            <div style={{ fontSize: 12, fontWeight: 700, color }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Score component bars ── */}
      <div style={{ padding: '8px 14px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        {[
          { label: 'VOL',    val: opp.volume_score,    color: '#06b6d4' },
          { label: 'MARGIN', val: opp.spread_score,    color: '#f59e0b' },
          { label: 'SAFETY', val: opp.stability_score, color: '#22c55e' },
          { label: 'SPEED',  val: opp.freshness_score, color: '#a78bfa' },
        ].map(({ label, val, color }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9, fontWeight: 700, color: 'var(--text-muted)', width: 40, letterSpacing: 0.4 }}>{label}</span>
            <MiniBar value={val} color={color} />
            <span style={{ fontSize: 9, color: 'var(--text-muted)', width: 22, textAlign: 'right' }}>
              {(val ?? 0).toFixed(0)}
            </span>
          </div>
        ))}
      </div>

      {/* ── Expanded detail ── */}
      {expanded && (
        <div style={{
          padding: '10px 14px',
          borderTop: '1px solid var(--border)',
          background: 'rgba(6,182,212,0.03)',
          fontSize: 11,
          display: 'flex', flexDirection: 'column', gap: 5,
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px' }}>
            {[
              ['GE Tax',      opp.tax != null ? `-${formatGP(opp.tax)}` : '—'],
              ['GP/hr est',   formatGP(opp.gp_per_hour)],
              ['Est fill',    opp.est_fill_time_minutes != null ? `${opp.est_fill_time_minutes.toFixed(0)} min` : '—'],
              ['GE Limit',    geLimit4h > 0 ? `${formatQty(geLimit4h)} / 4h` : '—'],
              ['Win Rate',    opp.win_rate != null ? `${(opp.win_rate * 100).toFixed(0)}%` : '—'],
              ['Your Flips',  opp.total_flips > 0 ? opp.total_flips : '—'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                <span style={{ fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>
          {opp.reason && (
            <div style={{ marginTop: 4, color: 'var(--text-muted)', lineHeight: 1.5, borderTop: '1px solid var(--border)', paddingTop: 6 }}>
              {opp.reason}
            </div>
          )}
          {opp.ml_direction && (
            <div style={{ padding: '5px 8px', borderRadius: 6, background: 'rgba(34,197,94,0.07)', marginTop: 2 }}>
              AI: <strong style={{ color: opp.ml_direction === 'up' ? '#22c55e' : opp.ml_direction === 'down' ? '#ef4444' : '#f59e0b' }}>
                {opp.ml_direction === 'up' ? '▲ Up' : opp.ml_direction === 'down' ? '▼ Down' : '— Flat'}
              </strong>
              {opp.ml_prediction_confidence != null && ` (${(opp.ml_prediction_confidence * 100).toFixed(0)}%)`}
            </div>
          )}
        </div>
      )}

      {/* ── Action footer ── */}
      <div style={{ padding: '10px 14px', marginTop: 'auto', display: 'flex', gap: 6, alignItems: 'center', borderTop: '1px solid var(--border)' }}>
        <button
          style={{
            flex: 1, padding: '6px 0', borderRadius: 6, border: '1px solid var(--border)',
            background: 'transparent', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer',
          }}
          onClick={() => setExpanded(v => !v)}
        >
          {expanded ? 'Less ▲' : 'More ▼'}
        </button>
        <button
          style={{
            padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'transparent', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer',
          }}
          onClick={() => nav(`/item/${opp.item_id}`)}
          title="View full analysis"
        >↗</button>

        {freeSlots > 0 ? (
          <button
            style={{
              flex: 2, padding: '6px 0', borderRadius: 6, fontSize: 11, fontWeight: 700, cursor: 'pointer',
              background: acceptingId === opp.item_id ? 'rgba(34,197,94,0.3)' : 'rgba(34,197,94,0.15)',
              border: '1px solid rgba(34,197,94,0.4)', color: '#22c55e',
            }}
            disabled={acceptingId === opp.item_id}
            onClick={() => onAccept(opp)}
          >
            <Check size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />
            Accept
          </button>
        ) : (
          <div style={{ flex: 2 }}>
            <button
              style={{
                width: '100%', padding: '6px 0', borderRadius: 6, fontSize: 11, fontWeight: 700, cursor: 'pointer',
                background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.35)', color: '#f59e0b',
              }}
              disabled={acceptingId === opp.item_id}
              onClick={() => setReplaceForItem(replaceForItem === opp.item_id ? null : opp.item_id)}
            >
              Replace slot
            </button>
            {replaceForItem === opp.item_id && (
              <select
                onChange={e => e.target.value && onAccept(opp, e.target.value)}
                defaultValue=""
                style={{ marginTop: 4, width: '100%', fontSize: 11, borderRadius: 6 }}
              >
                <option value="" disabled>Select trade to replace</option>
                {activeTrades.map(t => (
                  <option key={t.trade_id} value={t.trade_id}>Slot {t.slot_index}: {t.name}</option>
                ))}
              </select>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Dump Card ───────────────────────────────────────────────────────────── */

function DumpCard({ d, freeSlots, activeTrades, acceptingId, onAccept }) {
  const starColor = d.stars >= 3 ? '#ef4444' : d.stars === 2 ? '#f59e0b' : '#06b6d4';
  return (
    <div style={{
      background: 'var(--bg-secondary)', border: `1px solid rgba(239,68,68,0.25)`,
      borderRadius: 12, overflow: 'hidden', display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 10 }}>
        <img src={IMG(d.item_id)} alt="" width={36} height={36}
          style={{ imageRendering: 'pixelated', flexShrink: 0, borderRadius: 6, background: 'rgba(0,0,0,0.2)' }}
          onError={e => { e.target.style.display = 'none'; }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 13, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{d.name}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>Dump signal</div>
        </div>
        <span style={{ fontSize: 16, color: starColor, fontWeight: 900, letterSpacing: -1 }}>{'★'.repeat(d.stars || 1)}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', background: 'rgba(0,0,0,0.18)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ padding: '7px 12px', borderRight: '1px solid var(--border)' }}>
          <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 2 }}>Dump price</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#f87171' }}>{formatGP(d.dump_price)}</div>
        </div>
        <div style={{ padding: '7px 12px' }}>
          <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 2 }}>Avg price</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8' }}>{formatGP(d.ref_avg)}</div>
        </div>
      </div>

      <div style={{ padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <DumpSparkline dumpPrice={d.dump_price} refAvg={d.ref_avg} />
        <div style={{ fontSize: 11, textAlign: 'right' }}>
          <div style={{ color: '#f87171', fontWeight: 700 }}>▼ {d.drop_pct?.toFixed?.(1) ?? d.drop_pct}%</div>
          <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>Vol: {(d.volume_5m || 0).toLocaleString()}</div>
          <div style={{ color: '#22c55e', fontWeight: 700 }}>+{formatGP(d.est_profit)}</div>
        </div>
      </div>

      <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)', marginTop: 'auto' }}>
        {freeSlots > 0 ? (
          <button
            style={{
              width: '100%', padding: '7px 0', borderRadius: 6, fontSize: 11, fontWeight: 700, cursor: 'pointer',
              background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.35)', color: '#f87171',
            }}
            disabled={acceptingId === d.item_id}
            onClick={() => onAccept(d, null, { type: 'dump' })}
          >
            <Check size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />
            Accept dump
          </button>
        ) : (
          <select
            defaultValue=""
            onChange={e => e.target.value && onAccept(d, e.target.value, { type: 'dump' })}
            style={{ width: '100%', fontSize: 11, borderRadius: 6 }}
          >
            <option value="" disabled>Replace slot to accept</option>
            {activeTrades.map(t => (
              <option key={t.trade_id} value={t.trade_id}>Slot {t.slot_index}: {t.name}</option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}

/* ── Loading skeleton cards ──────────────────────────────────────────────── */

function SkeletonCard() {
  const shimmer = { background: 'rgba(255,255,255,0.05)', borderRadius: 6, animation: 'pulse 1.5s ease-in-out infinite' };
  return (
    <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
      <div style={{ padding: '12px 14px', display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ ...shimmer, width: 36, height: 36, borderRadius: 8 }} />
        <div style={{ flex: 1 }}>
          <div style={{ ...shimmer, height: 13, width: '70%', marginBottom: 6 }} />
          <div style={{ ...shimmer, height: 10, width: '45%' }} />
        </div>
        <div style={{ ...shimmer, width: 44, height: 44, borderRadius: '50%' }} />
      </div>
      <div style={{ height: 40, background: 'rgba(0,0,0,0.18)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', borderBottom: '1px solid var(--border)' }}>
        {[0, 1, 2, 3].map(i => (
          <div key={i} style={{ padding: '8px 10px', borderRight: i < 3 ? '1px solid var(--border)' : 'none' }}>
            <div style={{ ...shimmer, height: 9, width: '60%', marginBottom: 4 }} />
            <div style={{ ...shimmer, height: 12, width: '80%' }} />
          </div>
        ))}
      </div>
      <div style={{ padding: '8px 14px' }}>
        {[0, 1, 2, 3].map(i => (
          <div key={i} style={{ ...shimmer, height: 3, marginBottom: 6, width: `${80 - i * 10}%` }} />
        ))}
      </div>
    </div>
  );
}

/* ── Active Trades / Adjust Offers Panel ─────────────────────────────────── */

function ActiveTradesPanel({ trades, onAdjust, onReload }) {
  const [adjustingId, setAdjustingId] = useState(null);
  const [buyDraft, setBuyDraft]       = useState('');
  const [sellDraft, setSellDraft]     = useState('');
  const [saving, setSaving]           = useState(false);

  if (!trades || trades.length === 0) return null;

  const STATE_COLOR = {
    BUY_PENDING: '#f59e0b', BUYING: '#06b6d4',
    HOLDING: '#a78bfa', SELLING: '#22c55e',
  };

  const openAdjust = (t) => {
    setAdjustingId(t.trade_id);
    setBuyDraft(String(t.buy_target || ''));
    setSellDraft(String(t.sell_target || ''));
  };

  const saveAdjust = async (tradeId) => {
    setSaving(true);
    try {
      await onAdjust(tradeId, { buyTarget: parseGP(buyDraft), sellTarget: parseGP(sellDraft) });
      setAdjustingId(null);
      onReload();
    } catch (e) {
      window.alert(e.message || 'Failed to adjust offers');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: 12, padding: '12px 14px', marginBottom: 16,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-muted)', marginBottom: 10 }}>
        Active Slots ({trades.length})
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {trades.map(t => {
          const isAdj = adjustingId === t.trade_id;
          const sc = STATE_COLOR[t.state] || '#6b7280';
          return (
            <div key={t.trade_id} style={{
              background: 'rgba(0,0,0,0.18)', border: `1px solid ${isAdj ? sc : 'var(--border)'}`,
              borderRadius: 10, padding: '10px 12px', minWidth: 200, flex: '1 1 200px', maxWidth: 280,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <img src={IMG(t.item_id)} alt="" width={26} height={26}
                  style={{ imageRendering: 'pixelated', borderRadius: 4, background: 'rgba(0,0,0,0.3)', flexShrink: 0 }}
                  onError={e => { e.target.style.display = 'none'; }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.name}</div>
                  <span style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: `${sc}20`, color: sc }}>
                    {t.state}
                  </span>
                </div>
                {!isAdj ? (
                  <button onClick={() => openAdjust(t)} title="Adjust offers" style={{
                    padding: '4px 6px', borderRadius: 6, border: '1px solid var(--border)',
                    background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', flexShrink: 0,
                  }}>
                    <Edit2 size={11} />
                  </button>
                ) : (
                  <button onClick={() => setAdjustingId(null)} style={{
                    padding: '4px 6px', borderRadius: 6, border: '1px solid var(--border)',
                    background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', flexShrink: 0,
                  }}>
                    <X size={11} />
                  </button>
                )}
              </div>

              {!isAdj ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  <div>
                    <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 2 }}>BUY</div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#4ade80' }}>{formatGP(t.buy_target)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 2 }}>SELL</div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8' }}>{formatGP(t.sell_target)}</div>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                    <div>
                      <div style={{ fontSize: 9, color: '#4ade80', marginBottom: 3, fontWeight: 700 }}>BUY</div>
                      <input value={buyDraft} onChange={e => setBuyDraft(e.target.value)}
                        style={{ width: '100%', padding: '4px 6px', borderRadius: 5, border: '1px solid #4ade8055', background: 'rgba(0,0,0,0.3)', color: '#4ade80', fontSize: 12, fontWeight: 700, boxSizing: 'border-box' }} />
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: '#38bdf8', marginBottom: 3, fontWeight: 700 }}>SELL</div>
                      <input value={sellDraft} onChange={e => setSellDraft(e.target.value)}
                        style={{ width: '100%', padding: '4px 6px', borderRadius: 5, border: '1px solid #38bdf855', background: 'rgba(0,0,0,0.3)', color: '#38bdf8', fontSize: 12, fontWeight: 700, boxSizing: 'border-box' }} />
                    </div>
                  </div>
                  <button disabled={saving} onClick={() => saveAdjust(t.trade_id)} style={{
                    padding: '5px 0', borderRadius: 6, border: '1px solid rgba(34,197,94,0.4)',
                    background: 'rgba(34,197,94,0.12)', color: '#22c55e', fontSize: 11, fontWeight: 700, cursor: 'pointer',
                  }}>
                    {saving ? 'Saving…' : 'Save offers'}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────────────────────── */

export default function Opportunities() {
  const nav = useNavigate();
  const [category, setCategory] = useState('all');
  const [sortCol, setSortCol] = useState('flip_score');
  const [sortDir, setSortDir] = useState('desc');
  const [search, setSearch] = useState('');
  const [profile, setProfile] = useState('balanced');
  const [scoreMode, setScoreMode] = useState('balanced');
  const [valueMode, setValueMode] = useState('all');
  const [minFlipProfit, setMinFlipProfit] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [replaceForItem, setReplaceForItem] = useState(null);
  const [acceptingId, setAcceptingId] = useState(null);
  const [coinBalanceRaw, setCoinBalanceRaw] = useState('');
  const coinBalance = parseGP(coinBalanceRaw);

  const debugEnabled = import.meta.env.DEV || new URLSearchParams(window.location.search).get('debug') === '1';

  const opportunitiesParams = useMemo(
    () => ({
      limit: 200,
      profile,
      score_mode: scoreMode,
      min_price: 0,
      min_price_gp: 0,
      min_volume: 0,
      min_roi_pct: 0,
      min_profit_gp: 0,
      min_total_profit_gp: 0,
      value_mode: valueMode,
      min_profit_per_item_gp: 0,
    }),
    [profile, scoreMode, valueMode],
  );

  const { data: raw, loading, error, reload } = useApi(
    ({ signal }) => api.getOpportunities(opportunitiesParams, { signal, timeoutMs: 15000 }),
    [opportunitiesParams],
    autoRefresh ? 60_000 : null,
  );
  const { data: tradeData, reload: reloadTrades } = useApi(() => api.getActiveTrades(), [], 10000);
  const { data: dumpsRaw, loading: dumpsLoading, error: dumpsError, reload: reloadDumps } = useApi(
    () => api.getDumps(), [], 120000,
  );

  const opps      = useMemo(() => raw?.items || [], [raw]);
  const dumps     = useMemo(() => dumpsRaw?.items || [], [dumpsRaw]);
  const apiCount  = Number(raw?.count || 0);
  const lastUpdated     = timeAgo(raw?.generated_at);
  const filtersApplied  = raw?.filters_applied || null;
  const activeTrades    = tradeData?.items || [];
  const slotsUsed       = tradeData?.slots_used || 0;
  const slotsTotal      = tradeData?.slots_total || 8;
  const freeSlots       = Math.max(0, tradeData?.free_slots ?? (slotsTotal - slotsUsed));
  const activeScoreMode = raw?.score_mode || scoreMode;

  const acceptOpportunity = async (opp, replaceTradeId = null, overrides = {}) => {
    try {
      setAcceptingId(opp.item_id);
      await api.acceptTrade({
        item_id: opp.item_id,
        name: opp.name,
        buy_target: opp.buy_price || opp.instant_buy || opp.dump_price || 0,
        sell_target: opp.sell_price || opp.instant_sell || opp.ref_avg || 0,
        qty_target: Math.max(1, opp.position_sizing?.quantity || 1),
        max_invest_gp: Math.max(0, opp.position_sizing?.max_investment || (opp.buy_price || opp.dump_price || 0)),
        type: (opp.dump_signal || '').toLowerCase() === 'high' ? 'dump' : 'normal',
        volume_5m: opp.volume_5m || opp.volume,
        replace_trade_id: replaceTradeId,
        ...overrides,
      });
      setReplaceForItem(null);
      reloadTrades();
      if (category === 'dumps') reloadDumps();
    } catch (e) {
      window.alert(e.message || 'Failed to accept trade');
    } finally {
      setAcceptingId(null);
    }
  };

  /* ── Category counts (for badges) ── */
  const categoryCounts = useMemo(() => {
    const counts = { all: opps.length, pvm: 0, pvp: 0, supplies: 0, skilling: 0, merch: 0, dumps: dumps.length };
    opps.forEach(o => { counts[classifyItem(o)] = (counts[classifyItem(o)] || 0) + 1; });
    return counts;
  }, [opps, dumps]);

  /* ── Filter + sort ── */
  const filtered = useMemo(() => {
    if (category === 'dumps') return [];

    let items = [...opps];

    if (search) {
      const q = search.toLowerCase();
      items = items.filter(o => (o.name || o.item_name)?.toLowerCase().includes(q) || String(o.item_id).includes(q));
    }

    if (coinBalance > 0) {
      items = items.filter(o => {
        const cost = (o.buy_price || 0) * Math.max(1, o.qty_suggested || 1);
        return cost <= coinBalance;
      });
    }

    if (minFlipProfit > 0) {
      items = items.filter(o => (o.potential_profit ?? o.expected_profit ?? 0) >= minFlipProfit);
    }

    if (category !== 'all') {
      items = items.filter(o => classifyItem(o) === category);
    }

    const scoreKey = activeScoreMode === 'margin_hunter' ? 'margin_hunter_score' : 'flip_score';
    const sortKey  = sortCol === 'flip_score' ? scoreKey : sortCol;
    items.sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });

    return items;
  }, [opps, category, search, coinBalance, minFlipProfit, sortCol, sortDir, activeScoreMode]);

  /* ── Summary stats ── */
  const summaryStats = useMemo(() => {
    const src = category === 'dumps' ? dumps : filtered;
    if (!src.length) return null;
    if (category === 'dumps') {
      const totalProfit = dumps.reduce((s, d) => s + (d.est_profit || 0), 0);
      const best = dumps.reduce((b, d) => (d.stars || 0) > (b.stars || 0) ? d : b, dumps[0]);
      return { count: dumps.length, totalProfit, best: { name: best?.name, sub: `${best?.drop_pct?.toFixed(1)}% drop` } };
    }
    const scoreKey  = activeScoreMode === 'margin_hunter' ? 'margin_hunter_score' : 'flip_score';
    const avgScore  = filtered.reduce((s, o) => s + (o[scoreKey] ?? 0), 0) / filtered.length;
    const avgMargin = filtered.reduce((s, o) => s + (o.margin_pct ?? 0), 0) / filtered.length;
    const totalProfit = filtered.reduce((s, o) => s + (o.potential_profit ?? 0), 0);
    const best = filtered.reduce((b, o) => ((o[scoreKey] ?? 0) > (b[scoreKey] ?? 0)) ? o : b, filtered[0]);
    return {
      count: filtered.length, avgScore, avgMargin, totalProfit,
      best: { name: best?.name || best?.item_name, sub: `score ${(best?.[scoreKey] ?? 0).toFixed(0)}` },
    };
  }, [filtered, dumps, category, activeScoreMode]);

  /* ── Render ── */
  return (
    <div>
      {debugEnabled && (
        <div className="card" style={{ marginBottom: 12, border: '1px solid #f59e0b', background: 'rgba(245,158,11,0.08)' }}>
          <div style={{ padding: 12, fontFamily: 'monospace', fontSize: 11 }}>
            <strong style={{ color: '#f59e0b' }}>Debug</strong>{' '}
            count={raw?.count ?? '—'} profile={raw?.profile ?? '—'} score_mode={raw?.score_mode ?? '—'}{' '}
            api={API_BASE}
          </div>
        </div>
      )}

      <MarketSearchPanel />

      {/* ── Page header ── */}
      <div className="page-header">
        <div>
          <h2 className="page-title">Opportunities</h2>
          <p className="page-subtitle">
            {apiCount} items · {lastUpdated} · slots {slotsUsed}/{slotsTotal}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <select value={profile} onChange={e => setProfile(e.target.value)}
            style={{ padding: '7px 12px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12 }}>
            <option value="conservative">Conservative</option>
            <option value="balanced">Balanced</option>
            <option value="aggressive">Aggressive</option>
          </select>
          <select value={scoreMode} onChange={e => setScoreMode(e.target.value)}
            style={{ padding: '7px 12px', borderRadius: 20, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12 }}>
            <option value="balanced">Balanced</option>
            <option value="margin_hunter">Margin Hunter</option>
          </select>
          <button className={`pill ${autoRefresh ? 'active' : ''}`} onClick={() => setAutoRefresh(v => !v)} style={{ fontSize: 11 }}>
            {autoRefresh ? '⟳ Live' : '⟳ Paused'}
          </button>
          <button className="btn" onClick={() => category === 'dumps' ? reloadDumps() : reload()} disabled={loading}>
            <RefreshCw size={14} style={loading ? { animation: 'spin 1s linear infinite' } : {}} /> Refresh
          </button>
        </div>
      </div>

      {/* ── Category tabs ── */}
      <div style={{
        display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16,
        padding: '12px 14px', background: 'var(--bg-secondary)',
        border: '1px solid var(--border)', borderRadius: 12,
      }}>
        {CATEGORIES.map(({ key, label, Icon, color }) => {
          const active = category === key;
          const count  = categoryCounts[key] ?? 0;
          return (
            <button key={key} onClick={() => setCategory(key)} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '7px 14px', borderRadius: 8, cursor: 'pointer', fontSize: 12, fontWeight: 600,
              border: active ? `1px solid ${color}` : '1px solid var(--border)',
              background: active ? `${color}18` : 'transparent',
              color: active ? color : 'var(--text-muted)',
              transition: 'all 0.15s',
            }}>
              <Icon size={13} />
              {label}
              {count > 0 && (
                <span style={{
                  fontSize: 10, fontWeight: 700, padding: '1px 6px', borderRadius: 10,
                  background: active ? `${color}30` : 'rgba(255,255,255,0.07)',
                  color: active ? color : 'var(--text-muted)',
                }}>
                  {count}
                </span>
              )}
            </button>
          );
        })}

        {/* ── Right-side controls ── */}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            <Search size={12} style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input type="text" placeholder="Search…" value={search} onChange={e => setSearch(e.target.value)}
              style={{ padding: '7px 12px 7px 28px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card, var(--bg-secondary))', color: 'var(--text-primary)', fontSize: 12, width: 160 }} />
          </div>
        </div>
      </div>

      {/* ── Active trades / adjust offers ── */}
      <ActiveTradesPanel
        trades={activeTrades}
        onAdjust={(tradeId, offers) => api.adjustTradeOffers(tradeId, offers)}
        onReload={reloadTrades}
      />

      {/* ── Sub-filters (non-dumps) ── */}
      {category !== 'dumps' && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginBottom: 14 }}>
          {/* Coin balance filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderRadius: 8, border: '1px solid var(--border)', background: coinBalance > 0 ? 'rgba(34,197,94,0.06)' : 'transparent' }}>
            <Wallet size={12} color={coinBalance > 0 ? '#22c55e' : 'var(--text-muted)'} />
            <input
              type="text"
              placeholder="Your GP (e.g. 50M)"
              value={coinBalanceRaw}
              onChange={e => setCoinBalanceRaw(e.target.value)}
              style={{ width: 120, border: 'none', background: 'transparent', color: coinBalance > 0 ? '#22c55e' : 'var(--text-primary)', fontSize: 11, outline: 'none' }}
            />
            {coinBalance > 0 && (
              <button onClick={() => setCoinBalanceRaw('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0 }}>
                <X size={10} />
              </button>
            )}
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Value:</span>
          {['all', '1m', '10m'].map(m => (
            <button key={m} className={`pill ${valueMode === m ? 'active' : ''}`}
              onClick={() => setValueMode(m)} style={{ fontSize: 11 }}>
              {m === 'all' ? 'Any' : m === '1m' ? '1M+' : '10M+'}
            </button>
          ))}
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>Min profit/flip:</span>
          {[
            { label: 'Off',   gp: 0       },
            { label: '25K',   gp: 25_000  },
            { label: '50K',   gp: 50_000  },
            { label: '100K',  gp: 100_000 },
            { label: '200K',  gp: 200_000 },
            { label: '500K',  gp: 500_000 },
          ].map(({ label, gp }) => (
            <button key={gp} className={`pill ${minFlipProfit === gp ? 'active' : ''}`}
              onClick={() => setMinFlipProfit(gp)} style={{ fontSize: 11 }}>
              {label}
            </button>
          ))}
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>Sort:</span>
          {[
            { label: 'Score',  col: 'flip_score' },
            { label: 'Profit', col: 'potential_profit' },
            { label: 'ROI',    col: 'roi_pct' },
            { label: 'Volume', col: 'volume_5m' },
          ].map(({ label, col }) => (
            <button key={col} className={`pill ${sortCol === col ? 'active' : ''}`}
              onClick={() => { if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc'); else { setSortCol(col); setSortDir('desc'); } }}
              style={{ fontSize: 11 }}>
              {label}{sortCol === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''}
            </button>
          ))}
        </div>
      )}

      {/* ── Summary strip ── */}
      {summaryStats && (
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 10, marginBottom: 16,
        }}>
          {(category !== 'dumps' ? [
            { label: 'Showing',     value: summaryStats.count },
            { label: 'Avg Score',   value: `${summaryStats.avgScore?.toFixed(0)}/100` },
            { label: 'Avg Margin',  value: `${summaryStats.avgMargin?.toFixed(1)}%`, color: (summaryStats.avgMargin ?? 0) > 0 ? '#22c55e' : undefined },
            { label: 'Top Pick',    value: summaryStats.best?.name, sub: summaryStats.best?.sub, color: '#22c55e' },
          ] : [
            { label: 'Dumps',        value: summaryStats.count },
            { label: 'Total Est',    value: `+${formatGP(summaryStats.totalProfit)}`, color: '#22c55e' },
            { label: 'Top Dump',     value: summaryStats.best?.name, sub: summaryStats.best?.sub, color: '#ef4444' },
          ]).map(({ label, value, sub, color }) => (
            <div key={label} className="card" style={{ padding: '12px 14px' }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: sub ? 13 : 18, fontWeight: 700, color: color || 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{value ?? '—'}</div>
              {sub && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
            </div>
          ))}
        </div>
      )}

      {/* ── Dumps grid ── */}
      {category === 'dumps' ? (
        dumpsLoading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 14 }}>
            {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        ) : dumpsError ? (
          <div className="empty" style={{ color: '#ef4444' }}>
            <AlertTriangle size={24} style={{ marginBottom: 8 }} /><br />
            Failed to load dumps — {dumpsError.message || 'connection error'}
          </div>
        ) : dumps.length === 0 ? (
          <div className="empty">No dump candidates right now. Check back soon.</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 14 }}>
            {dumps.map(d => (
              <DumpCard key={d.item_id} d={d}
                freeSlots={freeSlots} activeTrades={activeTrades}
                acceptingId={acceptingId} onAccept={acceptOpportunity} />
            ))}
          </div>
        )
      ) : (
        /* ── Opportunities card grid ── */
        loading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 14 }}>
            {Array.from({ length: 9 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        ) : error ? (
          <div className="empty" style={{ color: '#ef4444' }}>
            <AlertTriangle size={24} style={{ marginBottom: 8 }} /><br />
            <strong>Failed to load opportunities</strong><br />
            <small className="text-muted">{error.message || 'Connection error'} — auto-retrying</small>
          </div>
        ) : apiCount === 0 ? (
          <div className="empty">
            <Filter size={24} style={{ marginBottom: 8, opacity: 0.5 }} /><br />
            {filtersApplied && (
              filtersApplied.value_mode !== 'all'
              || Number(filtersApplied.min_profit_per_item_gp || 0) > 0
            )
              ? `No items in ${filtersApplied.value_mode || 'this mode'} right now — try Any or adjust filters.`
              : 'No opportunities in cache yet.'}
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty">
            <Filter size={24} style={{ marginBottom: 8, opacity: 0.5 }} /><br />
            No {CATEGORIES.find(c => c.key === category)?.label || ''} items right now — try another category.
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 14 }}>
            {filtered.map((opp, i) => (
              <OpportunityCard
                key={opp.item_id ?? i}
                opp={opp}
                rank={i + 1}
                activeScoreMode={activeScoreMode}
                freeSlots={freeSlots}
                activeTrades={activeTrades}
                acceptingId={acceptingId}
                replaceForItem={replaceForItem}
                setReplaceForItem={setReplaceForItem}
                onAccept={acceptOpportunity}
                nav={nav}
              />
            ))}
          </div>
        )
      )}
    </div>
  );
}
