# TokenGrid evaluation: what we did and why

This guide explains, in plain language, how TokenGrid was turned from a hackathon
demo into a project with measured results. It covers what was built, why each piece
exists, what the real data showed, the improvement that came out of it, and how the
work was checked.

---

## 1. The short version

- **TokenGrid's idea:** many AI agents share one limited LLM budget. Each agent
  predicts how many tokens its task will cost and bids for a turn. An auction picks
  who goes, using `reputation × urgency ÷ predicted tokens`. Agents whose predictions
  turn out wrong lose reputation, and lose their turns.
- **The problem:** the hackathon version only *showed* this working. Nothing measured
  whether it actually works better than simpler approaches.
- **What we did:** recorded 940 real AI calls on real tasks, compared ten ways for
  agents to forecast their own costs, and replayed everything through TokenGrid's own
  auction code next to five other ways of sharing the budget.
- **What we found:**
  1. **Liar protection works.** With a quarter of agents lying, honest agents keep
     **99%** of the budget, versus **72%** under a simple queue, and complete **12 more
     correct tasks per window**.
  2. **A budget-waste problem, found and fixed.** TokenGrid was leaving about 10% of the
     budget unused. The fix adds **+3.9 correct tasks per window**, and it held up on
     640 tasks it was never tuned on.
  3. **Good forecasting is what keeps honest agents safe.** It barely changes
     throughput. Its real value shows up under stress: with somewhat worse forecasts,
     a simple average lets **14 of 15** honest agents get removed, while the best
     learned forecaster loses about **3**. The best forecaster counts the prompt's
     tokens exactly and predicts only the answer's length.
  4. **Some popular ideas don't help here.**
     - Nearest-neighbour lookup, cost buckets and ensembles are no better than single models.
     - "Safe" high bids are punished by the reputation system.
     - Agents correcting themselves from their own past errors makes things *worse*.
  5. **The reputation system has a cliff.** If honest agents' forecasts were moderately
     worse, almost **all** of them would eventually be removed. This is the clearest
     thing to improve next.
- **How we checked it:** 27 unit tests on the code, plus 22 checks on the real data
  and results. All pass.

---

## 2. Why evaluate at all?

The hackathon version had three weaknesses that a technical reviewer would spot
immediately:

| Weakness | Why it matters |
|---|---|
| Only 8 tasks, repeated forever | After the first round, every answer came from the cache. The auction and reputation system almost never actually ran. |
| Agents' cost predictions were a fixed formula | The whole auction depends on agents predicting costs. A made-up formula proves nothing. |
| No comparison to anything | "It works" means nothing without "compared to what?" |

So the goal became answering these questions with real numbers:

1. **Does the auction get more correct work done than simpler ways of sharing the budget?**
2. **Does it stay robust when some agents lie about their costs?**
3. **Does cost per completed task stay flat as the swarm grows?** This was TokenGrid's original pitch.
4. **Can the weaknesses we find be fixed?**
5. **How good does cost forecasting need to be, and which method should do it?**

---

## 3. The big picture

```
 real tasks ──► AI model (once) ──► recorded costs ──► cost forecaster ──► simulator ──► experiments ──► report
                                    (traces)           (agents' bids)     (TokenGrid +
                                                                           5 alternatives)
```

The key idea is to **call the real AI once, then replay**. Every task is sent to the
real model once, and its token cost and correctness are written down. All experiments
then replay those records. This has two benefits:

- **Cheap:** thousands of experiments for the price of one data collection ($0.21 at
  list prices for 940 tasks; $0 on Groq's free tier).
- **Fair:** every approach is tested on exactly the same task costs.

---

## 4. Each step: what and why

### Step 1: Real tasks with known answers
**File:** `backend/app/tasks.py`

**What:** tasks come from three public test sets:
- **GSM8K:** school math word problems
- **MMLU:** knowledge questions across 57 subjects, multiple choice
- **ARC-Challenge:** science reasoning, multiple choice

Each has a known correct answer. The model is asked to finish with `Answer: X`. We
read the last such line and compare it to the correct answer.

**Why:**
- Distinct tasks mean the cache can't hide everything.
- Known answers mean "completed" can mean **correct**, not just "got a turn".
- An answer we can't read counts as wrong, so the grader never flatters the results.

### Step 2: Record real costs
**File:** `backend/experiments/collect_traces.py`

**What:** sends each task to the real model (`gpt-oss-120b` on Groq) once and saves
one line per task: input tokens, output tokens, dollar cost, whether the answer was
correct, and the model's full reply.
- If the provider says "slow down" (a 429 error), it waits and retries.
- If the **daily** limit runs out, it stops cleanly instead of retrying all day.
  Rerunning the same command later continues where it stopped.
- If it crashes halfway, rerunning skips tasks already done.

**Free tier:** Groq's free tier allows 200K tokens a day for this model, which is
about 500 tasks.

**We stopped at 940 tasks:** GSM8K 499, MMLU 341, ARC 100. Collection runs in
benchmark order, so ARC has fewer tasks. One model was used throughout, the same one
the live TokenGrid runs on.

### Step 3: Teach agents to forecast their own cost
**Files:** `backend/app/token_predictor.py` (forecasting methods), `backend/experiments/simulator.py` (online correction)

**What:** we compare **ten ways of forecasting a task's token cost**, plus one strategy
where agents learn from their own track record:

| Forecaster | How it works | Why it's included |
|---|---|---|
| Constant | every task costs the overall average | the no-information baseline |
| Per-benchmark average | "a math task costs what math tasks cost on average" | **a strong simple baseline:** if machine learning can't beat this, it isn't worth it |
| Length-only | ridge regression on text length, word count, numbers in the text, and test set | the simplest real model |
| Embedding + ridge | adds a **sentence embedding** (MiniLM: 384 numbers capturing what the text means) | does understanding the text help? |
| Embedding + LightGBM | same features, gradient-boosted trees | can a non-linear model do better? |
| LightGBM, log-median | trained to predict the *typical* cost (median) instead of the *average* | the reputation score rewards being close in ratio terms, which matches the median, not the average |
| kNN (10 nearest) | average cost of the 10 most similar past tasks | simple, explainable, common in LLM-routing work |
| **Exact input + predicted output** | the prompt's tokens can be counted *before* the call; only the answer's length (including the model's thinking) is predicted, with LightGBM | removes part of the error for free |
| Ensemble | average (in log terms) of ridge, LightGBM and kNN | combining models is often more stable |
| Cost buckets | classify each task as short, medium or long; predict that group's typical cost | coarser, but possibly more robust |
| + Online per-agent correction | each honest agent tracks how far off its past forecasts were and scales its next bid to match | the most TokenGrid-specific idea: agents learn from experience |

Some details that matter:
- **Most models predict log(tokens).** Costs are lopsided: most tasks are cheap, a few
  are very expensive. Logs make that easier to learn. Average-targeting models apply a
  standard correction (Duan's smearing) when converting back. The median model
  deliberately doesn't, because that would push it back toward the average.
- **"Safe" forecasts (percentiles).** A model can also give a cost the task will
  *probably* stay under: "90% of tasks like this cost less than X". It does this by
  measuring its own typical errors on held-out training data.
- **Cross-fitting.** Tasks are split into 5 groups. The model is trained on 4 and
  predicts the 5th, rotating through all 5. **No task is ever predicted by a model that
  saw it.**

**Why:** TokenGrid only works if honest agents can predict their costs. This tests
whether they can and which method is best. Liars then distort the forecast:
- **inflators** claim 3× it
- **freeloaders** claim 15–35% of it

### Step 4: The simulator
**File:** `backend/experiments/simulator.py`

**What:** runs a swarm of agents for 300 rounds against a fixed rate limit (a token
budget that refills every 10 rounds). Each round:

1. TokenGrid's **own** capacity rule decides this round's budget and how many agents may go.
2. Every agent bids: its claimed cost and its urgency. Urgency rises the longer it waits.
3. TokenGrid's **own** auction picks the winners.
4. Winners "run" their task using the **real recorded cost**. If the real cost doesn't
   fit in the remaining budget, the call is rejected, just as a real API rejects it
   with a 429 error.
5. For TokenGrid, its **own** reputation ledger compares predicted and actual cost and
   updates reputation (+5 for an accurate forecast, −20 for a very wrong one). Agents
   below 25 are removed.

The simulator also records how the ledger judges honest agents' calls during play:
- **penalty rate:** the share of honest calls punished
- **reputation change per call:** an honest agent's average gain or loss per call

Below zero, honest agents drift toward removal. That number turned out to be the
best single measure of how safe honest agents are (Finding 3).

**What we compare:** six ways of choosing who goes. All use exactly the same budget
rules; **only the ordering differs**:

| Policy | Who goes first | Why it's included |
|---|---|---|
| fifo | whoever waited longest | the simplest fair queue |
| random | random order | the "no strategy" baseline |
| urgency | most urgent agent | ignores cost completely |
| wsjf | urgency ÷ claimed cost | **TokenGrid's formula without reputation**, a classic scheduling rule called "weighted shortest job first" |
| **tokengrid** | reputation × urgency ÷ claimed cost | the system being tested |
| oracle | urgency ÷ **true** cost | cheats by knowing real costs; shows the best possible ordering |

**Why it's built this way:**
- **It reuses TokenGrid's real code**, so we test the actual system, not a lookalike.
- **Same swarm for every policy.** Under a given random seed, every policy gets the
  same agents, the same liars, and the same task order. Differences come from the
  policy, not luck.
- **wsjf versus tokengrid** isolates exactly what reputation adds.

### Step 5: The experiments
**File:** `backend/experiments/run_experiments.py`

| Experiment | Question |
|---|---|
| **E1** | Which policy works best? (20 agents, 25% liars) |
| **E2** | What happens as liars increase from 0% to 50%? |
| **E3** | Does cost per task stay flat as the swarm grows from 5 to 80 agents? |
| **E4** | Does the budget fix work, including on tasks it never saw? (section 6) |
| **E5** | Which of the ten forecasters (plus online correction) is best, and how far does each keep honest agents from the reputation cliff, including a stress test with extra forecast noise? |
| **E6** | Do "safe" percentile bids help? |
| **E7** | How much does forecast quality matter? (deliberately worsened forecasts) |

**Rate limit:** set to a quarter of what 20 agents could use, so agents genuinely compete.

**How results are produced:**
- Every setting runs **20 times** with different random seeds.
- We report the average with a **95% confidence interval**: the range the true value
  very likely falls in.
- To compare two methods, we take the difference *seed by seed*, then build a
  confidence interval on those differences. **If it doesn't include zero, the
  difference is real.**

**Design choices are made on separate "development" seeds (100–119),** never on the
seeds we report:
- the budget rule (E4)
- the main forecaster (E5)

The forecaster is chosen by the largest *reputation change per honest call*. That
rule was fixed before the ten-way comparison was run, because the first comparison
had already shown throughput is tied between learned models, while safety from the
cliff is what differs.

### Step 6: Small changes to TokenGrid's code that support the experiments
- `run_auction()` now accepts an optional ranking rule, so the alternatives can reuse
  TokenGrid's exact auction. By default it uses TokenGrid's own formula, so the live
  system behaves exactly as before.
- The event log can be muted. Simulations run thousands of rounds, and logging each
  one would be slow and pointless.

---

## 5. What the real data showed

**Data:** 940 tasks on `gpt-oss-120b`.

| | |
|---|---|
| Accuracy | 91.8% overall (math 96.0%, knowledge 84.8%, science 95.0%) |
| Average cost | 434 tokens per task |
| Spread | some tasks cost about 4× others |

All results use the fixed budget rule (section 6) and the main forecaster (exact input
+ predicted output, chosen on development seeds) unless they say otherwise.

### Finding 1: Reputation protects honest agents from liars
With 25% liars:
- **Share of the budget going to honest agents:** TokenGrid **99.2%**; FIFO 72.2%
- **Correct tasks per window completed by honest agents:** TokenGrid **44.7**; FIFO 32.8.
  That's **+11.9** (95% confidence interval 11.6 to 12.2).
- **Total correct tasks per window:** TokenGrid 45.1; FIFO 45.4. TokenGrid does
  0.3 fewer (interval 0.1 to 0.5 fewer), **less than 1%**. That's the small price of
  removing liars, whose tasks were real work too.
- **Removals:** about 5 of 5 liars removed; almost no honest agents removed.

As liars increase, the honest agents' share holds up:

| Liars | 10% | 25% | 50% |
|---|---|---|---|
| FIFO | 84.1% | 72.2% | 51.6% |
| TokenGrid | **99.6%** | **99.2%** | **98.4%** |

### Finding 2: TokenGrid was wasting budget, now fixed (details in section 6)
- **Before:** about 89% of the rate limit used.
- **After:** about 98% used, giving **+3.9** correct tasks per window (interval 3.8 to 4.1).
- **On tasks the fix was never tuned on:** **+4.1** (interval 3.9 to 4.3).

### Finding 3: Good forecasting is what keeps honest agents safe (E5)

**Forecast quality** (all forecasts cross-fitted):

| Forecaster | Ranking quality (Spearman) | Typical error | Ledger accuracy | Honest calls punished | Reputation change per call |
|---|---|---|---|---|---|
| Constant | −0.03 | 142 | 0.747 | 1.6% | +1.19 |
| Per-benchmark average | 0.23 | 136 | 0.760 | 1.5% | +1.35 |
| Length-only | **0.64** | 110 | 0.802 | 1.2% | +1.74 |
| Embedding + ridge | 0.59 | 112 | 0.798 | 1.2% | +1.78 |
| Embedding + LightGBM | 0.58 | 111 | 0.801 | 1.2% | +1.91 |
| LightGBM, log-median | 0.60 | 110 | 0.806 | 1.3% | +1.93 |
| kNN (10 nearest) | 0.36 | 131 | 0.771 | 1.6% | +1.58 |
| **Exact input + predicted output** | 0.62 | **106** | **0.811** | 1.3% | **+1.97** |
| Ensemble | 0.57 | 113 | 0.796 | 1.2% | +1.78 |
| Cost buckets | 0.40 | 131 | 0.772 | **2.2%** | +1.44 |

**Normal conditions:** every learned forecaster gives about the same throughput (44.5–44.8
honest correct tasks per window, versus 44.3 for both simple baselines). The gains are
small but real, about +0.4 per window.

**The real difference shows under stress.** We gave every forecaster the same extra
random error, σ = 0.35, close to the cliff from Finding 5, and counted honest agents
removed:

| Forecaster | Reputation change per honest call | Honest agents removed (of 15) | Honest correct / window |
|---|---|---|---|
| Constant | −0.43 | **14.0** | 30.5 |
| Per-benchmark average | −0.30 | **13.3** | 34.6 |
| Cost buckets | +0.01 | 10.0 | 40.7 |
| kNN (10 nearest) | −0.02 | 10.2 | 41.2 |
| Length-only | +0.47 | 4.5 | 43.8 |
| Ensemble | +0.58 | 3.7 | 43.9 |
| LightGBM, log-median | +0.54 | 3.9 | 44.0 |
| Embedding + LightGBM | +0.55 | 3.3 | 43.9 |
| Embedding + ridge | +0.63 | **2.9** | 43.9 |
| **Exact input + predicted output** | +0.62 | **3.0** | 43.9 |
| Exact input + **online correction** | +0.30 | 7.0 | 43.6 |

**In plain words:**
- **Machine learning is clearly worth it here.** Compared with the per-benchmark
  average, the good learned forecasters keep about **10 more honest agents alive**
  under stress. Not because they're more accurate on average, but because they're
  wrong *badly* less often.
- **The best method: count the prompt exactly, predict only the answer.** It has the
  lowest error (106 tokens), the best ledger accuracy (0.811), and the largest safety
  margin under normal conditions. It was chosen as the main forecaster on the
  development seeds.
- **Predicting the median, as the reputation score suggests, helps a little.** Ledger
  accuracy rises from 0.801 to 0.806. But median bids are lower, so slightly more calls
  get rejected (0.9% vs 0.7%).
- **kNN and cost buckets fall short.** kNN ranks costs poorly (0.36), and buckets
  punish honest agents most often (2.2%). Both lose about 10 honest agents under stress.
- **The ensemble is no better** than its best member.
- **Online per-agent correction makes things worse.** It roughly doubles the agents lost
  under stress, from 3.0 to 7.0. Forecast errors are about the *task*, not the *agent*.
  An agent's last few errors don't predict its next one, so "correcting" for them
  just adds noise to its bids.

### Finding 4: "Safe" high bids backfire (E6)
What if honest agents bid a cost they'll *probably* stay under, instead of their best
guess? Rejected calls should drop. They do, but:

| Honest bid (Embedding + LightGBM) | Tasks within bid | Rejected calls | Honest correct / window | Honest agents removed |
|---|---|---|---|---|
| Best guess | 57% | 0.7% | **44.7** | 0 |
| 70th percentile | 71% | 0.4% | 44.4 | 0.05 |
| 80th percentile | 79% | 0.2% | 43.9 | 0.05 |
| 90th percentile | 90% | **0.1%** | 42.6 | **2.0** |

The percentile forecasts are accurately calibrated: the 90th covers 90.2% of tasks. But
TokenGrid's reputation ledger penalizes over-prediction as much as under-prediction.
So honest agents who bid safely lose reputation, win fewer turns, and at the 90th
percentile some get removed.

E6 uses Embedding + LightGBM because percentile forecasts need a model with a learned
error spread. The exact-input forecaster combines two pieces and doesn't provide one.

### Finding 5: The reputation system has a cliff (E7)
We deliberately made the main forecaster worse, adding random error of size σ, and
watched what happened:

| Added error σ | Honest calls punished | Honest agents removed (of 15) | Honest correct / window |
|---|---|---|---|
| 0 (main forecaster as is) | 1.3% | 0.05 | **44.7** |
| 0.25 | 2.9% | 0.55 | 44.3 |
| **0.5** | **10.6%** | **14.9** | **16.3** |
| 1.0 | 35.9% | 14.6 | 2.6 |

**What's happening:** a good forecast earns +5, a very wrong one costs −20, and
removal is permanent. One bad call cancels four good ones. When the average
reputation change per call turns **negative**, every honest agent is slowly pushed
toward removal, and over 300 rounds nearly all of them get there. Throughput then
collapses.

**Why it matters:** the defense against liars is the same mechanism that can wipe out
honest agents. Today's forecaster is on the safe side, but the stress test in Finding 3
shows how fast the margin disappears. A real deployment, with messier tasks than these
benchmarks, could sit closer to the cliff. **Making the reputation rule more forgiving
is the most valuable next improvement.** For example, use an agent's *average*
accuracy over recent calls instead of single-call penalties, or let reputation
recover over time.

### Finding 6: Cost per task is flat, but not because of TokenGrid (E3)
Cost per correct task stays around $0.00024 for **every** policy, at **every** swarm
size from 5 to 80. It's flat because the fixed rate limit caps spending, not because
of anything TokenGrid does. Only the oracle gets cheaper with big swarms, because
perfect knowledge lets it favor cheap tasks.

**What this means:** the original pitch, "TokenGrid keeps cost per task flat as the
swarm grows", is true but is not *caused* by TokenGrid. It should be stated carefully.

---

## 6. The improvement: not wasting budget that is about to expire

### The problem
The rate limit refills on a schedule; here, every 10 rounds. TokenGrid's original
rule let each round spend at most **50% of whatever budget was left**, keeping the
rest as a safety margin for later rounds. Each agent could also take at most 20% of a
round's budget.

Near the end of each window, that halving leaves budget too small for any honest
task. Then the window resets and **the leftover is simply lost**. Under the other
policies, liars' tiny under-bids happened to soak up those leftovers. TokenGrid
removes the liars, so the leftovers went to waste: about 10% of the whole budget.

### How we confirmed the cause before fixing it
- **With no liars at all, every policy wasted the same amount** (89–90% used). So the
  waste comes from the budget rule, not from reputation.
- **Letting rounds spend everything removed the waste** (89% → 98%).

### How we chose the fix
The rule was chosen when we had the first 300 tasks, using **separate "development"
seeds (100–119)**. Four rules were compared:

| Rule | What it does | Verdict |
|---|---|---|
| original | 50% of what's left, every round | wastes about 10% |
| spend all | 100% of what's left, every round | recovers the waste, but a simple queue gets about 3% rejections because there's no safety margin |
| **release at reset** ✅ | 50% normally; **100% in the last round before the reset** | recovers the waste, all liars still caught, rejections stay around 1% |
| even pacing | what's left ÷ rounds left | recovers the waste, but caught only 2 of 5 liars, and at 50% liars it let them take 94% of the budget under a simple queue |

**Release at reset** won. The logic is simple: **holding budget back only makes sense
if a later round can use it. In the last round before the reset, nothing later can,
so spend it.** The 50% safety margin stays in every other round.

### The result (TokenGrid, new rule minus original, paired by seed)

| Test | Budget used | Correct tasks per window | Rejected calls | Liars caught / honest removed |
|---|---|---|---|---|
| 0% liars | +9.3 points | **+4.3** [4.1, 4.4] | +1.1 points | no change |
| 25% liars | +8.6 points | **+3.9** [3.8, 4.1] | +0.8 points | no change |
| 50% liars | +9.2 points | **+4.3** [4.2, 4.5] | +0.8 points | no change |
| **Only the 640 tasks collected after the rule was chosen** | +9.2 points | **+4.1** [3.9, 4.3] | +0.9 points | no change |

The last row matters most. The fix was designed on the first 300 tasks, and it works
just as well on 640 tasks it never saw.

**The trade-off, stated honestly:** spending everything in the last round removes the
safety margin there. Rejected calls rose from 0% to about 1%, in exchange for about
10% more completed work.

### How it works in the live system
**Files:** `app/auction.py`, `app/llm_provider.py`, `app/accounting.py`, `app/agent_simulator.py`

- Groq reports how long until the token limit refills, in a response header
  (`x-ratelimit-reset-tokens`, for example `7.66s`).
- TokenGrid reads it, works out how many auction rounds remain before that moment,
  and in the last one releases the held-back budget.
- **Safety:** the estimate assumes each round is as short as possible (1.5 s). Real
  rounds are longer, so it can only release *too late* (no gain), never *too early*.
- If the header is missing, for example in dry-run mode, TokenGrid uses the original
  rule exactly.
- The per-agent 20% cap still applies in the release round, so one agent still can't
  grab a big share. This is tested directly.

---

## 7. How the work was checked

### Unit tests: `python -m pytest tests` (27 tests, all pass)
These check the code on small hand-made cases:

| Test file | What it protects |
|---|---|
| `test_tasks.py` | Answers are read and graded correctly; unreadable answers count as wrong |
| `test_auction_key.py` | TokenGrid's default auction is unchanged; a custom ranking only changes the order; muting the log works |
| `test_token_predictor.py` | The reputation score is fair both ways; forecasters find real patterns; percentile forecasts are calibrated and correctly ordered; LightGBM works; the per-benchmark, kNN and bucket forecasters give sensible answers; the median model skips the average-correction |
| `test_simulator.py` | The rate limit is never exceeded; same seed gives the same result; every policy gets the same agents; perfect knowledge never triggers rejections; reputation protects honest agents; online correction learns a consistent bias when one exists |
| `test_collect.py` | Per-minute limits are waited out and retried; the daily limit stops the run with no wasted calls and no failure rows; full replies are saved |
| `test_capacity_fix.py` | Unknown reset time means the original rule; the budget is held back until the last round, then fully spent; a 90% "flash drain" bid is still blocked in that round; Groq's reset header is read correctly; the live round count never releases early; the fix raises budget use |

### Data and result checks: `python -m experiments.verify` (22 checks, all pass)
These check the **real data and the real results**:

| # | Check | Why | Result |
|---|---|---|---|
| 1 | No task recorded twice | Duplicates would double-count tasks | 940 rows, 940 unique |
| 2 | Total tokens = input + output | Catches recording bugs | 0 mismatches |
| 3 | Cost = tokens × price | Catches pricing bugs | 0 mismatches |
| 4 | "Correct" matches answer vs. gold | Catches grading bugs | 0 disagreements |
| 5 | Answers not cut off by the length cap | Cut-off answers would look wrong for the wrong reason | 2 of 940 |
| 6 | **Saved replies re-grade to the same verdict** | Audits the grader against what the model actually wrote | 0 of 639 disagree |
| 7 | **No cheating by the forecaster** | If costs are shuffled randomly, a fair model should fail to predict them | shuffled −0.11 (fails, as it should); real 0.62 |
| 8–13 | Each policy respects the rate limit and gives identical results when rerun | The simulator must follow its own rules and be repeatable | all 6 pass |
| 14 | **Rerunning reproduces the saved report exactly** | Anyone can regenerate our numbers | 0 of 18 runs differ |
| 15 | Original rule, no liars: all policies waste the same | Confirms the waste isn't caused by reputation | 89–90% for all |
| 16 | Spending all remaining budget removes the waste | Confirms the cause | 89% → 98% |
| 17 | The fix recovers unused budget | The fix does what it claims | +8.7 points |
| 18 | The fix increases honest agents' correct work | The gain is real work | +4.0 tasks per window |
| 19 | The fix keeps liar protection | It mustn't weaken the main strength | honest share ≥ 98.6%; liars caught 49 (same as before); honest removed 0 (same as before) |
| 20 | The fix keeps rejections rare | The trade-off stays small | 0.90% |
| 21 | Percentile forecasts are calibrated | A "90% safe" bid should cover about 90% | q50 50.1%, q80 79.5%, q90 89.7% |
| 22 | The main forecaster beats a constant guess | The ML part must earn its place | 0.811 vs 0.747 ledger accuracy |

**A lesson from checking:** an earlier version of check 19 required "zero honest
agents removed". That failed on synthetic data, and investigating showed the synthetic
data's weaker forecasts get an honest agent removed **under both the old and the new
rule**: same agents, same seeds. So the fix wasn't the cause. The check was corrected
to the claim that matters: *the fix removes no more honest agents than before*. That
early clue, that bad forecasts get honest agents removed, is exactly what Findings 3
and 5 later measured properly.

---

## 8. Honest limitations

These should be stated whenever results are presented:

- **940 tasks, unbalanced across test sets** (GSM8K 499, MMLU 341, ARC 100). The
  results are pooled across all of them.
- **One model executes everything** (`gpt-oss-120b`). TokenGrid's router, cache and
  compressor aren't part of the simulation.
- **Forecasters were tuned lightly.** One reasonable setting each, not an exhaustive
  search. The differences between the good learned models are small, so a tuned
  version of any of them could come out on top.
- **"Exact input" assumes the prompt can be token-counted before the call.** With the
  model's tokenizer that's true. Here the recorded input count stands in for it.
- **The budget fix trades a few rejections for more work:** about 1% of calls rejected, up from 0%.
- **In the live system, the fix depends on Groq's reset header** and on estimating round
  timing. The estimate is deliberately cautious, so live gains may be smaller than
  simulated ones, but never harmful.
- **The rate-limit window is simplified.** It refills every 10 rounds; real providers use a rolling window.
- **Liars follow fixed strategies** (3× or 0.15–0.35×). A smarter liar that adapts to
  avoid detection hasn't been tested.
- **On a rejected call, the simulator leaves reputation unchanged.** The live system
  treats it as a 0-token call and penalizes the agent.
- **The first 300 tasks don't have the model's full reply saved.** The newer 640 do,
  and their grading was re-checked (check 6).

---

## 9. How to run everything

From the `backend/` folder:

```bash
pip install -r requirements.txt

python -m pytest tests                          # 27 unit tests
python -m experiments.collect_traces --n 500    # record real costs (needs GROQ_API_KEY in .env);
                                                # free tier: ~500 tasks/day, resumes where it stopped
python -m experiments.run_experiments           # all experiments E1-E7 (~3 min) -> results/groq/report/summary.md
python -m experiments.verify                    # 22 data and result checks

# without an API key (synthetic data, for testing the pipeline only):
python -m experiments.collect_traces --mock
python -m experiments.run_experiments --mock
python -m experiments.verify --mock
```

Older reports are kept for comparison:
- `results/groq/report_before_fix/`: 300 tasks, before the budget fix
- `results/groq/traces_backup_before_error_cleanup.jsonl`: the raw file before error rows were removed

---

## 10. Words used in this guide

| Term | Meaning |
|---|---|
| **Token** | A piece of text, roughly ¾ of a word. AI providers charge per token. |
| **Rate limit** | The most tokens or requests a provider allows per minute or per day. |
| **429 error** | The provider's "too many requests, try later" rejection. |
| **Trace** | One recorded real AI call: its cost and whether it was right. |
| **Embedding** | A list of numbers that captures what a piece of text means. |
| **Ridge regression** | A simple linear model with protection against overfitting. |
| **LightGBM** | A gradient-boosted tree model that can capture non-linear effects. |
| **kNN (k nearest neighbours)** | Predicting from the most similar past examples. |
| **Ensemble** | Combining several models' predictions into one. |
| **Median vs. average** | The typical value vs. the mean. With lopsided costs, a few huge tasks pull the average above the typical task. |
| **Percentile forecast** | A cost the task will probably stay under, e.g. "90% of tasks like this cost less than X". |
| **Calibrated** | A percentile forecast that's right as often as it claims (a 90% one covers about 90%). |
| **Cross-fitting** | Predicting each task with a model trained only on *other* tasks, so results aren't flattered. |
| **Spearman correlation** | How well predictions put things in the right order: 1 = perfect, 0 = random. |
| **Ledger accuracy** | TokenGrid's own score for a forecast: smaller ÷ larger of predicted and actual cost (1 = perfect). |
| **Reputation change per call** | An honest agent's average reputation gain or loss per call. Below zero means it drifts toward removal. |
| **Stress test** | Deliberately making conditions worse to see what breaks first. |
| **Baseline** | A simpler approach to compare against. |
| **Oracle** | A baseline that cheats by knowing the true answer; shows the best possible result. |
| **Ablation** | Removing one piece to see how much it mattered. |
| **Seed** | A number that fixes the randomness, so a run can be repeated exactly. |
| **Development vs. evaluation seeds** | Choose a design on one set of runs, report results on a different set, so the choice doesn't flatter the result. |
| **Held-out tasks** | Data that played no part in a design choice, used to check the choice still works. |
| **95% confidence interval** | The range the true value very likely falls in. |
| **Paired comparison** | Comparing two methods on the *same* random setup, so luck cancels out. |
| **WSJF** | "Weighted shortest job first": do valuable, quick jobs first. TokenGrid's formula without reputation. |

---

## 11. What's next

1. **Make the reputation rule forgiving (Findings 3 and 5).** Test rules based on average
   recent accuracy, or reputation that recovers over time. Choose on development seeds,
   then show the cliff moves out while liars are still caught.
2. **Write a short report** (4–6 pages) and put the project under version control (git).
