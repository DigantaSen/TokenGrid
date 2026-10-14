"""The capacity fix: release headroom that would expire at the rate-limit reset."""
from dataclasses import replace

from app import config
from app.agent_simulator import rounds_until_token_reset
from app.auction import Bid, compute_capacity, run_auction
from app.event_log import event_log
from app.llm_provider import parse_duration_s
from experiments.simulator import SimConfig, simulate
from tests.test_simulator import pool


def cap(rr, rt, left=None):
    with event_log.muted():
        return compute_capacity(rr, rt, left)


def test_unknown_reset_keeps_original_rule():
    c = cap(100, 10_000)
    assert c.max_tokens == int(10_000 * config.ROUND_TOKEN_FRACTION)


def test_holds_back_until_the_last_round_then_spends_everything():
    assert cap(100, 10_000, left=3).max_tokens == 5_000
    assert cap(100, 10_000, left=1).max_tokens == 10_000
    assert cap(100, 10_000, left=1).per_agent_cap_tokens == 10_000 * config.PER_AGENT_CAP_FRACTION


def test_flash_drain_still_blocked_in_release_round():
    c = cap(100, 10_000, left=1)
    with event_log.muted():
        r = run_auction([Bid("drainer", 9_000, 1.0, 100.0), Bid("normal", 500, 0.5, 50.0)], c)
    assert [w.agent_id for w in r.winners] == ["normal"]
    assert [q.agent_id for q in r.queued] == ["drainer"]


def test_parse_groq_reset_header():
    assert parse_duration_s("7.66s") == 7.66
    assert abs(parse_duration_s("2m59.56s") - 179.56) < 1e-9
    assert parse_duration_s("120ms") == 0.12
    assert parse_duration_s("soon") is None and parse_duration_s(None) is None


def test_live_round_count_never_releases_early():
    tick = config.TICK_MIN_INTERVAL_SECONDS
    assert rounds_until_token_reset({"rate_limit_tokens_reset_at": None}) is None
    assert rounds_until_token_reset({"rate_limit_tokens_reset_at": 100.0}, now=100.0 - 0.5 * tick) == 1
    assert rounds_until_token_reset({"rate_limit_tokens_reset_at": 100.0}, now=100.0 - 1.5 * tick) == 2
    assert rounds_until_token_reset({"rate_limit_tokens_reset_at": 100.0}, now=150.0) == 1  # already reset


def test_fix_raises_utilization_in_simulation():
    cfg = SimConfig(n_agents=20, rounds=150, tokens_per_window=12_000, requests_per_window=200)
    new = simulate(pool(), "tokengrid", cfg)
    old = simulate(pool(), "tokengrid", replace(cfg, capacity_rule="original"))
    assert new["utilization"] > old["utilization"]
    assert new["honest_token_share"] > 0.95
