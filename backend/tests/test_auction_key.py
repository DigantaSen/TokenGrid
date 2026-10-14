from app.auction import Bid, compute_capacity, run_auction
from app.event_log import event_log


def bids():
    return [Bid("cheap", 100, 0.5, 50.0), Bid("pricey", 400, 0.9, 50.0), Bid("mid", 200, 0.6, 90.0)]


def test_default_key_is_tokengrid_score():
    with event_log.muted():
        r = run_auction(bids(), compute_capacity(4, 10_000))  # K=2
    # score = rep*urgency/tokens: cheap .25, mid .27, pricey .11
    assert [w.agent_id for w in r.winners] == ["mid", "cheap"]
    assert [q.agent_id for q in r.queued] == ["pricey"]


def test_custom_key_changes_only_the_ranking():
    with event_log.muted():
        r = run_auction(bids(), compute_capacity(4, 10_000), key=lambda b: b.urgency)
    assert [w.agent_id for w in r.winners] == ["pricey", "mid"]


def test_muted_log_records_nothing():
    before = len(event_log.events)
    with event_log.muted():
        compute_capacity(10, 1000)
    assert len(event_log.events) == before
    assert event_log.enabled
