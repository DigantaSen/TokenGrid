function timeStr(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function EventLog({ events }) {
  return (
    <div className="panel">
      <div className="panel-title">
        <span>Event log</span>
        <span className="mono" style={{ color: 'var(--text-dim)' }}>
          what fired, when, why
        </span>
      </div>
      <div className="event-log">
        {events.length === 0 && <div className="empty-hint">no events yet</div>}
        {events.map((e) => (
          <div className={`event-row cat-${e.category}`} key={e.id}>
            <span className="time">{timeStr(e.timestamp)}</span>
            <span className="cat">{e.category}</span>
            <span className="msg">{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
