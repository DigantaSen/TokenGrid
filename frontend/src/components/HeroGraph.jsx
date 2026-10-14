import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function HeroGraph({ metrics }) {
  const points = metrics?.points || [];
  const spikes = metrics?.spikes || [];

  const data = points.map((p) => ({
    tick: p.tick,
    withoutTG: +(p.cost_per_task_without * 1e6).toFixed(3),
    withTG: +(p.cost_per_task_with * 1e6).toFixed(3),
  }));

  return (
    <div className="panel">
      <div className="panel-title">
        <span>Cost per completed task (µ$ — millionths of a dollar)</span>
        <span className="legend">
          <span>
            <span className="dot" style={{ background: '#ff5470' }} />
            without TokenGrid
          </span>
          <span>
            <span className="dot" style={{ background: '#35d9a0' }} />
            with TokenGrid
          </span>
        </span>
      </div>
      {data.length < 2 ? (
        <div className="empty-hint">waiting for enough rounds to plot a trend…</div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 5, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1b2030" strokeDasharray="3 3" />
            <XAxis dataKey="tick" stroke="#4d5468" fontSize={11} tickLine={false} />
            <YAxis stroke="#4d5468" fontSize={11} tickLine={false} width={48} />
            <Tooltip
              contentStyle={{ background: '#151923', border: '1px solid #232838', fontSize: 12, borderRadius: 6 }}
              labelStyle={{ color: '#7d8496' }}
              labelFormatter={(v) => `tick ${v}`}
            />
            <Line type="monotone" dataKey="withoutTG" name="without TokenGrid" stroke="#ff5470" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="withTG" name="with TokenGrid" stroke="#35d9a0" strokeWidth={2} dot={false} isAnimationActive={false} />
            {spikes.map((s, i) => (
              <ReferenceLine
                key={i}
                x={s.tick}
                stroke="#ff5470"
                strokeDasharray="4 4"
                label={{ value: s.reason, position: 'insideTopLeft', fill: '#ff5470', fontSize: 10 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
