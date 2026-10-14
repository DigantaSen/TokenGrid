export default function TokenSplit({ latestPoint }) {
  const work = latestPoint?.work_tokens_cumulative ?? 0;
  const comm = latestPoint?.communication_tokens_cumulative ?? 0;
  const total = work + comm || 1;
  const workPct = (work / total) * 100;
  const commPct = 100 - workPct;

  return (
    <div className="panel">
      <div className="panel-title">
        <span>Token split — the communication tax</span>
      </div>
      {total <= 1 ? (
        <div className="empty-hint">no tokens yet</div>
      ) : (
        <>
          <div className="split-bar">
            <div className="seg" style={{ width: `${workPct}%`, background: '#35d9a0' }}>
              {workPct > 12 && `${workPct.toFixed(0)}%`}
            </div>
            <div className="seg" style={{ width: `${commPct}%`, background: '#f5a623' }}>
              {commPct > 12 && `${commPct.toFixed(0)}%`}
            </div>
          </div>
          <div className="legend">
            <span>
              <span className="dot" style={{ background: '#35d9a0' }} />
              work tokens ({Math.round(work).toLocaleString()})
            </span>
            <span>
              <span className="dot" style={{ background: '#f5a623' }} />
              communication tokens ({Math.round(comm).toLocaleString()})
            </span>
          </div>
          <div className="sub" style={{ marginTop: 8 }}>
            Communication is O(N²) standing overhead to keep the swarm in sync — it accrues every
            tick regardless of how much real work happens. A high share here means the swarm is
            paying more to coordinate than to think — exactly what the sizer panel is trying to fix.
          </div>
        </>
      )}
    </div>
  );
}
