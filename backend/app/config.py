"""
Central config for TokenGrid backend.

Everything reads from env vars so the same code runs in DRY_RUN (no API key,
no network — safe for bad venue wifi) or LIVE mode against Groq.
"""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# DRY_RUN is explicit if set, otherwise we infer it: no key -> can't call Groq.
_dry_run_env = os.environ.get("DRY_RUN")
if _dry_run_env is not None:
    DRY_RUN = _dry_run_env.strip().lower() in ("1", "true", "yes")
else:
    DRY_RUN = GROQ_API_KEY == ""

# Model routing. This is a real capability spread, not two names for the same model.
# NOTE: llama-3.1-8b-instant / llama-3.3-70b-versatile (the models named in the original
# spec) have been retired from Groq's catalog as of this build (verified live against
# /openai/v1/models); these are the current 20B/120B pair in the same "cheap vs real work"
# shape. Swap here if Groq's catalog changes again — nothing else references model names.
MODEL_SIMPLE = "openai/gpt-oss-20b"     # classification, judging, cheap tasks
MODEL_COMPLEX = "openai/gpt-oss-120b"   # real work

# Hard cap on completion length per complexity tier. Some Groq models (the
# gpt-oss family especially) are reasoning models that will emit thousands
# of tokens for an open-ended prompt if left unbounded — capping this is
# what keeps a single call's cost predictable enough for the auction's
# token-budget math (and for agents' predictions) to mean anything.
MAX_OUTPUT_TOKENS = {"simple": 120, "complex": 350}

# Approximate Groq list pricing, USD per token (not per 1K/1M) so cost math is a
# plain multiply. These are for the demo's cost graph, not a billing source of truth.
PRICING_USD_PER_TOKEN = {
    MODEL_SIMPLE: {"input": 0.10e-6, "output": 0.50e-6},
    MODEL_COMPLEX: {"input": 0.15e-6, "output": 0.75e-6},
}

DEFAULT_NUM_AGENTS = 20
MAX_NUM_AGENTS = 200

# --- Auction ---
# What fraction of *remaining* rate-limit headroom we're willing to spend in
# a single round. Keeping this < 1.0 leaves slack for other rounds/safety
# margin instead of draining the whole window on one auction.
ROUND_REQUEST_FRACTION = 0.5
ROUND_TOKEN_FRACTION = 0.5
# No agent may take more than this fraction of a round's token capacity.
PER_AGENT_CAP_FRACTION = 0.20
# Used only before we've ever seen a real rate-limit header (cold start, or
# DRY_RUN with no headers yet) so the auction has *something* to derive K
# from instead of a hardcoded K.
FALLBACK_REMAINING_REQUESTS = 30
FALLBACK_REMAINING_TOKENS = 6000

# --- Reputation ---
REPUTATION_START = 50.0
REPUTATION_MIN = 0.0
REPUTATION_MAX = 100.0
REPUTATION_ACCURACY_HIGH = 0.85
REPUTATION_ACCURACY_LOW = 0.40
REPUTATION_BONUS = 5.0
REPUTATION_PENALTY = 20.0
REPUTATION_DROP_THRESHOLD = 25.0

# --- Agent strategies ---
AGENT_STRATEGIES = ["honest", "inflator", "freeloader", "efficient"]
INFLATOR_MULTIPLIER = 3.0        # matches the /stress/liars scenario
FREELOADER_UNDERBID_RANGE = (0.15, 0.35)  # fraction of true estimate they claim

# --- Semantic cache ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_SIMILARITY_THRESHOLD = 0.85

# --- Context compressor ---
HISTORY_COMPRESS_THRESHOLD_TOKENS = 800
HISTORY_KEEP_LAST_TURNS = 2  # user+assistant pairs kept verbatim, rest summarized

# --- Swarm sizer (N*) ---
# Total cost(N) = production_cost(N) + coordination_cost(N), a classic
# team-sizing U-curve (Brooks's-law shape): a fixed amount of work gets
# cheaper per task as the swarm specializes (production falls with N), but
# every pair of agents costs something to keep in sync (coordination grows
# O(N^2)). N* is found by numeric search over the curve, not a closed form,
# so the shape can come from live measured cost without re-deriving algebra.
SWARM_RECOMPUTE_INTERVAL_TICKS = 10   # re-derive N* only this often
SWARM_STEP_SIZE = 2                   # agents added/removed per tick toward N*
SWARM_TOTAL_TASKS_PER_TICK = 50       # exogenous task demand per tick, independent of N
SWARM_BASELINE_N = 10                 # reference swarm size for the specialization curve
SWARM_SPECIALIZATION_ALPHA = 0.6      # elasticity of per-task cost w.r.t. N (0..1)
SWARM_COMM_TOKENS_PER_PAIR = 3        # coordination tokens exchanged per agent pair per tick
SWARM_DEFAULT_AVG_COST_PER_TASK = 0.0002  # fallback before any real work call exists

# --- Model router ---
ROUTER_CLASSIFY_MAX_TOKENS = 5

# --- Live loop / dashboard ---
TICK_MIN_INTERVAL_SECONDS = 1.5  # floor between auction rounds, so the dashboard stays readable
