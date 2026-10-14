import { useCallback, useEffect, useRef, useState } from 'react';
import HeroGraph from './components/HeroGraph';
import MetricsRow from './components/MetricsRow';
import SwarmPanel from './components/SwarmPanel';
import TokenSplit from './components/TokenSplit';
import AgentTable from './components/AgentTable';
import EventLog from './components/EventLog';
import StressPanel from './components/StressPanel';

const HOST = window.location.hostname || 'localhost';
// Port 8842, not 8000 — on some machines (this dev box included) something
// else (Docker Desktop's proxy, on macOS) already squats on 8000.
const API_BASE = import.meta.env.VITE_API_BASE || `http://${HOST}:8842`;
const WS_URL = import.meta.env.VITE_WS_URL || `ws://${HOST}:8842/ws`;

export default function App() {
  const [snapshot, setSnapshot] = useState(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer = null;

    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        if (!cancelled) setConnected(true);
      };
      ws.onmessage = (evt) => {
        if (cancelled) return;
        try {
          setSnapshot(JSON.parse(evt.data));
        } catch {
          // ignore malformed frame
        }
      };
      ws.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        reconnectTimer = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);

  const post = useCallback(async (path) => {
    try {
      await fetch(`${API_BASE}${path}`, { method: 'POST' });
    } catch {
      // backend unreachable — next WS reconnect will surface it
    }
  }, []);

  if (!snapshot) {
    return (
      <div className="app">
        <Header connected={connected} snapshot={null} onPost={post} />
        <div className="empty-hint">connecting to backend at {WS_URL}…</div>
      </div>
    );
  }

  const points = snapshot.metrics?.points || [];
  const latestPoint = points[points.length - 1];

  return (
    <div className="app">
      <Header connected={connected} snapshot={snapshot} onPost={post} />

      <HeroGraph metrics={snapshot.metrics} />

      <MetricsRow
        accounting={snapshot.accounting}
        cacheStats={snapshot.cache_stats}
        savings={snapshot.savings}
        latestPoint={latestPoint}
      />

      <div className="grid-2">
        <SwarmPanel swarm={snapshot.swarm} />
        <TokenSplit latestPoint={latestPoint} />
      </div>

      <div className="section">
        <AgentTable agents={snapshot.agents} roundSummary={snapshot.round_summary} />
      </div>

      <div className="grid-2 section">
        <StressPanel apiBase={API_BASE} />
        <EventLog events={snapshot.events} />
      </div>

      <div className="footer-note">
        tick {snapshot.swarm?.tick_count ?? 0} · {snapshot.agents.length} agents in swarm · TokenGrid demo
      </div>
    </div>
  );
}

function Header({ connected, snapshot, onPost }) {
  return (
    <div className="header">
      <div className="brand">
        <span className="logo" />
        <h1>TOKENGRID</h1>
        <span className="tagline">self-governing token economy for LLM agent swarms</span>
      </div>
      {snapshot && (
        <div className="header-controls">
          <span className="status-pill">
            <span className={`status-dot ${connected ? 'live' : ''}`} />
            {connected ? 'live' : 'reconnecting…'}
          </span>
          <span className="status-pill">
            <span className={`status-dot ${snapshot.dry_run ? 'dry' : 'live'}`} />
            {snapshot.dry_run ? 'DRY_RUN' : 'LIVE (Groq)'}
          </span>
          <button
            className="ctrl"
            onClick={() => onPost(snapshot.paused ? '/simulate/resume' : '/simulate/pause')}
          >
            {snapshot.paused ? '▶ resume' : '⏸ pause'}
          </button>
          <button className="ctrl danger" onClick={() => onPost('/simulate/reset')}>
            reset
          </button>
        </div>
      )}
    </div>
  );
}
