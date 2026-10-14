function fmtNum(n) {
  if (n === null || n === undefined) return '—';
  return Math.round(n).toLocaleString();
}

function fmtUsd(n) {
  if (n === null || n === undefined) return '—';
  return `$${n.toFixed(6)}`;
}

export default function MetricsRow({ accounting, cacheStats, savings, latestPoint }) {
  const totalTokens = accounting?.total_tokens ?? 0;
  const cacheTokensSaved = cacheStats?.tokens_saved ?? 0;
  const compressionTokensSaved = savings?.compression_tokens_saved_est ?? 0;
  const tokensSaved = cacheTokensSaved + compressionTokensSaved;

  const costSaved = latestPoint
    ? latestPoint.counterfactual_cost_cumulative - latestPoint.real_cost_cumulative
    : 0;

  const workTokens = latestPoint?.work_tokens_cumulative ?? 0;
  const compressionRatio = workTokens > 0 ? (workTokens + compressionTokensSaved) / workTokens : 1;

  const tiles = [
    { label: 'Total tokens', value: fmtNum(totalTokens) },
    {
      label: 'Tokens saved',
      value: fmtNum(tokensSaved),
      sub: `cache ${fmtNum(cacheTokensSaved)} · compress ${fmtNum(compressionTokensSaved)}`,
      cls: 'good',
    },
    { label: 'Cost saved', value: fmtUsd(costSaved), cls: 'good' },
    {
      label: 'Cache hit rate',
      value: `${((cacheStats?.hit_rate ?? 0) * 100).toFixed(1)}%`,
      sub: `${cacheStats?.hits ?? 0} hits · ${cacheStats?.cache_size ?? 0} entries`,
    },
    { label: 'Compression', value: `${compressionRatio.toFixed(2)}x` },
    {
      label: 'Rate-limit headroom',
      value: `${fmtNum(accounting?.rate_limit_remaining_requests)} req`,
      sub: `${fmtNum(accounting?.rate_limit_remaining_tokens)} tok`,
    },
  ];

  return (
    <div className="metrics-row">
      {tiles.map((t) => (
        <div className="stat-tile" key={t.label}>
          <div className="label">{t.label}</div>
          <div className={`value ${t.cls || ''}`}>{t.value}</div>
          {t.sub && <div className="sub">{t.sub}</div>}
        </div>
      ))}
    </div>
  );
}
