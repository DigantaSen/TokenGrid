function repColor(rep) {
  if (rep >= 60) return '#35d9a0';
  if (rep >= 30) return '#f5a623';
  return '#ff5470';
}

export default function AgentTable({ agents, roundSummary }) {
  const entryMap = {};
  (roundSummary?.entries || []).forEach((e) => {
    entryMap[e.agent_id] = e;
  });

  return (
    <div className="panel">
      <div className="panel-title">
        <span>Agents ({agents.length})</span>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Strategy</th>
              <th>Reputation</th>
              <th>Bid score</th>
              <th>Status</th>
              <th>Tokens used</th>
              <th>Prediction accuracy</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((a) => {
              const e = entryMap[a.agent_id];
              return (
                <tr key={a.agent_id}>
                  <td>{a.agent_id}</td>
                  <td>
                    <span className="strategy-tag">{a.strategy}</span>
                  </td>
                  <td>
                    <div className="rep-bar-wrap">
                      <div className="rep-bar-track">
                        <div
                          className="rep-bar-fill"
                          style={{ width: `${a.reputation}%`, background: repColor(a.reputation) }}
                        />
                      </div>
                      <span>{a.reputation.toFixed(0)}</span>
                    </div>
                  </td>
                  <td>{e ? e.bid_score.toFixed(3) : '—'}</td>
                  <td>
                    <span className={`badge ${a.status}`}>{a.status}</span>
                  </td>
                  <td>{a.tokens_used_total.toLocaleString()}</td>
                  <td>{e && e.accuracy != null ? `${(e.accuracy * 100).toFixed(0)}%` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
