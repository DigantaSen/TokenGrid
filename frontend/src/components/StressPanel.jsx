import { useState } from 'react';

const ATTACKS = [
  { key: 'flood', name: 'Token Flood', desc: '500 tasks injected in one second' },
  { key: 'liars', name: 'Liar Swarm', desc: 'Half the agents inflate every prediction 3x' },
  { key: 'flash-drain', name: 'Flash Drain', desc: 'One agent requests 90% of capacity' },
  { key: 'collusion', name: 'Collusion Ring', desc: '5 agents cross-endorse to inflate reputation' },
  { key: 'correlated-failure', name: 'Correlated Failure', desc: 'All agents hit the same bad prompt at once' },
  { key: 'reset', name: 'Reset', desc: 'Clear all stress effects and return to baseline' },
];

export default function StressPanel({ apiBase }) {
  const [busyKey, setBusyKey] = useState(null);
  const [lastResult, setLastResult] = useState(null);

  async function fire(key) {
    setBusyKey(key);
    setLastResult(null);
    try {
      const res = await fetch(`${apiBase}/stress/${key}`, { method: 'POST' });
      if (res.ok) {
        setLastResult({ key, ok: true });
      } else {
        setLastResult({ key, ok: false, note: `HTTP ${res.status} — not wired up yet` });
      }
    } catch {
      setLastResult({ key, ok: false, note: 'backend unreachable' });
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <div className="panel">
      <div className="panel-title">
        <span>Stress suite</span>
      </div>
      <div className="stress-grid">
        {ATTACKS.map((a) => (
          <button
            key={a.key}
            className="stress-btn"
            disabled={busyKey === a.key}
            onClick={() => fire(a.key)}
          >
            <span className="name">{busyKey === a.key ? 'firing…' : a.name}</span>
            <span className="desc">{a.desc}</span>
          </button>
        ))}
      </div>
      {lastResult && (
        <div
          className="sub mono"
          style={{ marginTop: 10, color: lastResult.ok ? 'var(--accent)' : 'var(--text-dim)' }}
        >
          {lastResult.ok ? `${lastResult.key} fired — see event log` : `${lastResult.key}: ${lastResult.note}`}
        </div>
      )}
    </div>
  );
}
