import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function SwarmPanel({ swarm }) {
  if (!swarm) return null;

  const curve = swarm.curve || [];
  // Downsample for a lighter chart — 200 points of a smooth curve don't need every one.
  const data = curve
    .filter((_, i) => i % 2 === 0)
    .map((c) => ({ n: c.n, cost: +(c.total_cost * 1e6).toFixed(2) }));

  const currentCost = swarm.cost_at_current_n ? +(swarm.cost_at_current_n.total_cost * 1e6).toFixed(2) : null;
  const optimalCost = swarm.cost_at_optimal_n ? +(swarm.cost_at_optimal_n.total_cost * 1e6).toFixed(2) : null;

  return (
    <div className="panel">
      <div className="panel-title">
        <span>Swarm sizer</span>
        <span className="mono" style={{ color: 'var(--text-dim)' }}>
          tick {swarm.tick_count}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 28, marginBottom: 12 }}>
        <div>
          <div className="label" style={{ fontSize: 10.5, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Current N
          </div>
          <div className="mono" style={{ fontSize: 28, fontWeight: 700 }}>
            {swarm.current_n}
          </div>
        </div>
        <div>
          <div className="label" style={{ fontSize: 10.5, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Optimal N*
          </div>
          <div className="mono" style={{ fontSize: 28, fontWeight: 700, color: 'var(--accent)' }}>
            {swarm.optimal_n}
          </div>
        </div>
      </div>

      {data.length < 2 ? (
        <div className="empty-hint">computing cost curve…</div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1b2030" strokeDasharray="3 3" />
              <XAxis dataKey="n" stroke="#4d5468" fontSize={10} tickLine={false} />
              <YAxis stroke="#4d5468" fontSize={10} tickLine={false} width={40} />
              <Tooltip
                contentStyle={{ background: '#151923', border: '1px solid #232838', fontSize: 11, borderRadius: 6 }}
                labelFormatter={(v) => `N=${v}`}
              />
              <Line type="monotone" dataKey="cost" stroke="#b088ff" strokeWidth={2} dot={false} isAnimationActive={false} />
              {currentCost !== null && <ReferenceDot x={swarm.current_n} y={currentCost} r={5} fill="#4da6ff" stroke="none" />}
              {optimalCost !== null && <ReferenceDot x={swarm.optimal_n} y={optimalCost} r={5} fill="#35d9a0" stroke="none" />}
            </LineChart>
          </ResponsiveContainer>
          <div className="legend" style={{ marginTop: 6 }}>
            <span>
              <span className="dot" style={{ background: '#4da6ff' }} />
              current N
            </span>
            <span>
              <span className="dot" style={{ background: '#35d9a0' }} />
              optimal N*
            </span>
          </div>
        </>
      )}
    </div>
  );
}
