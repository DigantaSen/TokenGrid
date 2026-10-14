# TokenGrid

A self-governing token economy for LLM agent swarms. TokenGrid sits between a
swarm of AI agents and a real LLM API (Groq) and keeps cost-per-completed-task
flat as the swarm scales — by auctioning off scarce rate-limit capacity,
scoring agents on prediction honesty rather than task success, caching
semantically, compressing context, routing to the cheapest capable model, and
right-sizing the swarm to its cost-optimal N.

Built for a hackathon demo. In-memory only, no database, no blockchain.

## Evaluation: does the token auction actually work?

The live dashboard shows TokenGrid running; `backend/experiments/` tests
whether its claims hold, using real tasks, real token costs and baselines.
**For a plain-language walkthrough of what was done, why, and what was
found, see [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md).**

**Question.** When many LLM agents share one rate-limited API, does
TokenGrid's auction (bid = reputation × urgency / predicted tokens, with
reputation earned through accurate cost forecasts) get more correct work
done per dollar than standard schedulers? Does it stay robust when agents
lie about their costs? And does cost per completed task stay flat as the
swarm grows?

**Why the bid formula is principled.** Without reputation, the bid is
Smith's rule (weighted-shortest-job-first): order jobs by
weight / processing time, which minimizes weighted waiting time when job sizes
are known. TokenGrid only knows *predicted* sizes, from agents who may lie.
Reputation is the correction that discounts bad forecasters. The `wsjf`
baseline is exactly this rule without reputation, so the experiments isolate
what reputation adds.

**Method.**

| Piece | What it does |
|---|---|
| `app/tasks.py` | Real tasks with ground truth: GSM8K, MMLU, ARC-Challenge (500 seeded each), with answer grading. "Completed" means *correct*. |
| `experiments/collect_traces.py` | Runs every task through the real model once and logs actual tokens, cost and correctness. Resumable. |
| `app/token_predictor.py` | How an honest agent forecasts its own cost: MiniLM embedding + text features → ridge regression on log tokens. Cross-fitted, so no task is forecast by a model that saw it. |
| `experiments/simulator.py` | Trace-driven simulation. Runs TokenGrid's own `compute_capacity`, `run_auction` and `ReputationLedger` unchanged, against a fixed token rate limit, with honest agents, inflators (3×) and freeloaders (0.15–0.35×). |
| `experiments/run_experiments.py` | Forecast quality; **E1** TokenGrid vs. FIFO / random / urgency / WSJF / oracle (+ TokenGrid without the learned predictor); **E2** 0–50% liars; **E3** swarm size 5–80 at a fixed rate limit. Paired seeds, 95% CIs. |

Every policy clears bids through the same K, round budget and per-agent cap.
Only the ranking differs. Metrics: correct tasks per window (all agents and
honest agents only), cost per correct task, rate-limit utilization, 429
rate, p95 wait, honest agents' token share, liars and honest agents dropped,
and size bias (whether a policy looks cheap only by skipping expensive
tasks).

```bash
cd backend
pip install -r requirements.txt
python -m pytest -q tests                        # 27 tests

python -m experiments.collect_traces --n 500     # real traces (needs GROQ_API_KEY); free tier ~500/day, resumes
python -m experiments.run_experiments            # E1-E7 (~3 min) -> results/groq/report/summary.md
python -m experiments.verify                     # 22 checks on the real data and report

# no key? check the pipeline on synthetic traces (never report these numbers):
python -m experiments.collect_traces --mock && python -m experiments.run_experiments --mock
```

**Status:** 940 real tasks on `gpt-oss-120b` (GSM8K 499, MMLU 341, ARC 100);
27 unit tests and 22 data/result checks passing. Results (20 paired seeds,
95% CIs):

- **Liar protection.** With 25% lying agents, reputation keeps 99% of the
  budget with honest agents (FIFO: 72%), for +11.9 correct honest tasks per
  window [11.6–12.2], at a cost of <1% total throughput.
- **The fix (E4).** TokenGrid was leaving ~10% of the rate limit unused,
  caused by the "spend ≤50% of remaining headroom per round" rule. The fix
  releases headroom in the last round before the limit resets (live: from
  Groq's `x-ratelimit-reset-tokens` header), since anything held back then
  is lost.
  - Chosen among 4 rules on separate dev seeds, using the first 300 tasks.
  - Result: +3.9 correct tasks per window [3.8–4.1], liar protection
    unchanged, 429s about 1%.
  - Confirmed on the 640 tasks collected afterwards: +4.1 [3.9–4.3].
- **Forecasters (E5).** Ten forecasters plus online per-agent correction.
  - Throughput is nearly tied across learned models (+0.4 honest
    tasks/window over simple baselines). The real value shows in a
    near-cliff stress test: with extra forecast noise, the per-benchmark
    average loses 13 of 15 honest agents, the best learned forecasters
    about 3.
  - Best: **exact input tokens + predicted output** (MAE 106, ledger
    accuracy 0.811).
  - No better than a single model: kNN, cost buckets, an ensemble.
  - Worse: online per-agent correction, which doubles losses under stress
    because forecast errors are task-specific, not agent-specific.
- **Reputation cliff (E6–E7).**
  - "Safe" 90th-percentile bids cut 429s to 0.1% but cost honest agents
    reputation.
  - Deliberately worsened forecasts show that once honest agents' average
    reputation change per call turns negative, nearly all of them are
    removed. A more forgiving reputation rule is the next improvement.

Details: [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md).

**Simplifications to state when presenting results.** The rate limit uses
fixed windows (Groq's are rolling). One model executes all tasks (the
router, cache and compressor aren't in the simulation yet). On a 429, the
simulator leaves reputation unchanged; the live system scores it as a
0-token call.

## Architecture

```
backend/                FastAPI + WebSocket
  app/
    llm_provider.py      Groq adapter (OpenAI-compatible SDK) + DRY_RUN fake provider
    accounting.py         Token ledger — every call's real usage, once
    auction.py             The token auction: bid_score, K from live rate-limit headroom
    reputation.py          Prediction-accuracy ledger + collusion cycle detection
    semantic_cache.py      MiniLM embeddings, cosine similarity, cache hits
    context_compressor.py  Shared system-prompt dedup + history summarization
    model_router.py        Cheap real classification call decides the model
    swarm_sizer.py          N* via numeric search over a U-shaped cost curve
    agent_simulator.py     SimAgent + the auction-round orchestration loop
    stress.py               The 6 attack scenarios
    metrics_history.py      Time series behind the hero graph
    event_log.py            Shared reasoning log every mechanism writes to
  main.py                 FastAPI app, WebSocket broadcast, background tick loop

frontend/                React + Vite, single page, dark dashboard
  src/
    App.jsx                WebSocket client + layout
    components/             HeroGraph, MetricsRow, SwarmPanel, TokenSplit,
                            AgentTable, EventLog, StressPanel
```

The backend runs an autonomous auction loop in the background (one round
every ~1.5s) and broadcasts full state over `/ws` after every round — the
dashboard doesn't poll, it just renders whatever arrives.

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional — without this, the whole system runs in DRY_RUN (simulated
# token counts, zero network calls, safe for bad venue wifi):
cp .env.example .env   # then paste in GROQ_API_KEY

uvicorn main:app --reload --port 8842
```

**Why port 8842, not 8000:** on macOS with Docker Desktop running, something
in Docker's port-forwarding proxy was already bound to `:8000` on this dev
machine and silently intercepted requests instead of erroring — a real
example of an "obvious" default port not actually being free. Pick whatever
port is actually free on your machine and pass it to both `uvicorn --port`
and the frontend's `.env`.

Quick sanity check without a browser:

```bash
python run_demo.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed `localhost` URL. `frontend/.env` points it at the backend
(`VITE_API_BASE`, `VITE_WS_URL`) — update both if you changed the backend's
port.

## Models

Groq's catalog changed since this was speced (the originally-named
`llama-3.1-8b-instant` / `llama-3.3-70b-versatile` are retired). This build
verified live against `/openai/v1/models` and uses the current equivalent
capability spread: `openai/gpt-oss-20b` (simple/cheap) and
`openai/gpt-oss-120b` (complex/real work) — see `app/config.py`. Both are
reasoning models that will emit very long completions if left unbounded, so
every call passes an explicit `max_tokens` cap per complexity tier.

## 90-second demo script

1. **Open the dashboard.** Point at the hero graph first — two lines
   diverging hard: cost-per-completed-task *with* TokenGrid vs. what the same
   work *would have cost* without it. That gap is the whole pitch.
2. **Metrics row.** Real tokens saved, real dollars saved, cache hit rate
   climbing toward 90%+ as the swarm's overlapping work gets deduplicated,
   rate-limit headroom live from Groq's actual response headers.
3. **Swarm sizer.** Point at current N vs. optimal N* and the U-shaped cost
   curve — the swarm is visibly growing or shrinking toward the curve's
   minimum, not sitting at a number someone picked.
4. **Agent table.** Reputation bars moving in real time. Point at an
   `inflator` or `freeloader`-strategy agent: watch its bid score stay low
   (inflating your claimed cost tanks your own auction score) and its
   reputation fall when its prediction accuracy comes in under 40% —
   eventually dropped from the swarm entirely.
5. **Fire a stress test.** Click **Liar Swarm** — half the agents start
   claiming 3x their real cost. Nothing crashes. Watch the event log: the
   auction still clears bids off live rate-limit headroom, and the
   reputation ledger starts flagging and dropping the liars within a few
   rounds — the same mechanism already running, not a special case.
6. **Event log, close-up.** Read two or three lines aloud — each one names
   the mechanism, the numbers, and *why* it fired. That log is half the demo.

## Stress suite

Six adversarial scenarios, one button each, all survivable:

| Attack | What it does | Defense that fires |
|---|---|---|
| Token Flood | 500 synthetic bids in one round | Auction's K (from live rate-limit headroom) clears a fraction, queues the rest |
| Liar Swarm | Half the agents claim 3x true cost | Reputation ledger penalizes low prediction accuracy, drops repeat offenders |
| Flash Drain | One agent bids 90% of round capacity | Per-agent cap (20% of round) queues the oversized bid |
| Collusion Ring | 5 agents cross-endorse each other | Cycle detection over the endorsement graph flags the whole ring |
| Correlated Failure | Every agent hits the same adversarial prompt | Nothing hides it — the point is the swarm shares one base model, so failures cluster instead of averaging out like a diversified pool would predict |
| Reset | Clears all of the above | — |

## Constraints honored

- **DRY_RUN** auto-engages with no `GROQ_API_KEY` set (or force it with
  `DRY_RUN=true`) — the whole pipeline runs identically, just with simulated
  token counts instead of network calls.
- **In-memory only.** No database, no ORM, no blockchain — everything lives
  in process state and resets via `/simulate/reset`.
- **Real accounting.** Every dollar/token figure that claims to be "real" is
  computed from Groq's actual `usage` and rate-limit headers, never
  estimated — the one deliberate exception is the *counterfactual*
  ("without TokenGrid") line on the hero graph, which is necessarily an
  estimate since no second call was actually made to compare against.
