"""
TokenGrid backend.

Step 1: agent simulator + real Groq calls + token accounting.
Step 2: token auction + reputation ledger.
Step 3: semantic cache + context compressor.
Step 4: swarm sizer (N*) + model router.
Step 5: dashboard — this file now also runs a background auction loop and
        broadcasts full state over WebSocket after every round.

Run with:

    uvicorn main:app --reload

Then open the frontend (npm run dev in ../frontend), or poke it directly:

    curl -X POST "http://localhost:8000/simulate/reset?n=10"
    curl "http://localhost:8000/dashboard/snapshot"
    ws://localhost:8000/ws
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.accounting import ledger
from app.agent_simulator import run_round, spawn_agents, run_auction_round
from app.context_compressor import context_compressor
from app.event_log import event_log
from app.llm_provider import get_provider
from app.metrics_history import metrics_history
from app.reputation import reputation_ledger
from app.savings import savings
from app.semantic_cache import semantic_cache
from app.stress import (
    stress_collusion,
    stress_correlated_failure,
    stress_flash_drain,
    stress_flood,
    stress_liars,
)
from app.swarm_sizer import swarm_sizer

provider = get_provider()
agents = spawn_agents(config.DEFAULT_NUM_AGENTS, provider)

connected_sockets: set[WebSocket] = set()
paused = False


def agents_payload() -> list[dict]:
    return [
        {
            "agent_id": a.agent_id,
            "strategy": a.strategy,
            "status": a.status,
            "reputation": reputation_ledger.get(a.agent_id),
            "rounds_queued": a.rounds_queued,
            "tokens_used_total": a.tokens_used_total,
            "pending_task": a.pending_task[1] if a.pending_task else None,
        }
        for a in agents
    ]


def events_payload(n: int = 60) -> list[dict]:
    return [
        {"id": e.id, "timestamp": e.timestamp, "category": e.category, "message": e.message, "meta": e.meta}
        for e in event_log.recent(n)
    ]


def build_dashboard_snapshot(round_summary: dict | None = None) -> dict:
    return {
        "type": "tick",
        "paused": paused,
        "round_summary": round_summary,
        "agents": agents_payload(),
        "reputation": reputation_ledger.snapshot(),
        "events": events_payload(),
        "metrics": metrics_history.series(),
        "swarm": swarm_sizer.status(len(agents)),
        "cache_stats": semantic_cache.stats(),
        "savings": {**savings.totals(), "cache_tokens_saved": semantic_cache.stats()["tokens_saved"]},
        "accounting": ledger.totals(),
        "dry_run": config.DRY_RUN,
    }


async def broadcast(payload: dict):
    dead = []
    for ws in list(connected_sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_sockets.discard(ws)


async def tick_loop():
    loop = asyncio.get_event_loop()
    while True:
        try:
            if not paused and agents:
                summary = await loop.run_in_executor(None, run_auction_round, agents)
                await broadcast(build_dashboard_snapshot(summary))
        except Exception as e:
            event_log.add("system", f"tick loop error: {e}")
        await asyncio.sleep(config.TICK_MIN_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(tick_loop())
    yield
    task.cancel()


app = FastAPI(title="TokenGrid", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_sockets.add(websocket)
    try:
        await websocket.send_json(build_dashboard_snapshot())
        while True:
            await websocket.receive_text()  # keep-alive; client isn't expected to send anything meaningful
    except WebSocketDisconnect:
        pass
    finally:
        connected_sockets.discard(websocket)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dry_run": config.DRY_RUN,
        "model_simple": config.MODEL_SIMPLE,
        "model_complex": config.MODEL_COMPLEX,
        "num_agents": len(agents),
        "paused": paused,
    }


@app.get("/dashboard/snapshot")
def dashboard_snapshot():
    """REST fallback for the initial page load, before the WebSocket connects."""
    return build_dashboard_snapshot()


@app.post("/simulate/pause")
def simulate_pause():
    global paused
    paused = True
    event_log.add("system", "auction loop paused")
    return {"paused": paused}


@app.post("/simulate/resume")
def simulate_resume():
    global paused
    paused = False
    event_log.add("system", "auction loop resumed")
    return {"paused": paused}


@app.post("/simulate/reset")
def simulate_reset(n: int = config.DEFAULT_NUM_AGENTS):
    """Re-initialize the swarm: fresh agents, cleared reputation, cleared
    ledger and event log. (This is also the base for the /stress/reset
    endpoint added later.)"""
    global agents
    agents = spawn_agents(n, provider)
    reputation_ledger.__init__()
    ledger.records.clear()
    semantic_cache.reset()
    context_compressor.reset()
    savings.reset()
    swarm_sizer.reset()
    metrics_history.reset()
    event_log.clear()
    event_log.add("system", f"swarm reset: {n} fresh agents, all mechanisms cleared")
    return {"ok": True, "num_agents": len(agents)}


@app.post("/simulate/round")
def simulate_round():
    """Run one auction round on demand (bypasses the tick loop's cadence;
    used for scripted testing). The next scheduled tick still broadcasts
    normally to connected dashboards."""
    return run_auction_round(agents)


@app.post("/simulate/run")
def simulate_run_plain(n: int = config.DEFAULT_NUM_AGENTS):
    """Step-1-style plain round, bypassing the auction — every agent runs
    once. Useful for sanity-checking raw token accounting."""
    plain_agents = spawn_agents(n, provider)
    results = run_round(plain_agents)
    return {
        "dry_run": config.DRY_RUN,
        "results": [r.__dict__ for r in results],
        "totals": ledger.totals(),
    }


@app.post("/stress/flood")
def stress_flood_endpoint():
    return stress_flood(agents)


@app.post("/stress/liars")
def stress_liars_endpoint():
    return stress_liars(agents)


@app.post("/stress/flash-drain")
def stress_flash_drain_endpoint():
    return stress_flash_drain(agents)


@app.post("/stress/collusion")
def stress_collusion_endpoint():
    return stress_collusion(agents)


@app.post("/stress/correlated-failure")
def stress_correlated_failure_endpoint():
    return stress_correlated_failure(agents)


@app.post("/stress/reset")
def stress_reset_endpoint(n: int = config.DEFAULT_NUM_AGENTS):
    """Same as /simulate/reset — the dashboard's stress panel just calls
    this one under a name that matches the rest of the stress suite."""
    return simulate_reset(n)


@app.get("/agents")
def list_agents():
    return agents_payload()


@app.get("/reputation")
def reputation_snapshot():
    return reputation_ledger.snapshot()


@app.get("/cache/stats")
def cache_stats():
    return semantic_cache.stats()


@app.get("/savings")
def savings_totals():
    return {**savings.totals(), "cache_tokens_saved": semantic_cache.stats()["tokens_saved"]}


@app.get("/swarm/status")
def swarm_status():
    return swarm_sizer.status(len(agents))


@app.get("/metrics/history")
def metrics_history_series():
    return metrics_history.series()


@app.get("/accounting/totals")
def accounting_totals():
    return ledger.totals()


@app.get("/accounting/recent")
def accounting_recent(n: int = 20):
    return [r.__dict__ for r in ledger.recent(n)]


@app.get("/events/recent")
def events_recent(n: int = 50):
    return events_payload(n)
