"""Offline end-to-end tests for minisanity using the fixture source."""

from fastapi.testclient import TestClient

from minisanity.api import make_app
from minisanity.recommend import TfIdfIndex, tokenize
from minisanity.sources import FixtureSource
from minisanity.store import Paper, Store


def make_client():
    return TestClient(make_app(store=Store(), source=FixtureSource()))


def test_refresh_and_papers():
    c = make_client()
    r = c.post("/refresh")
    assert r.json() == {"ingested": 5}
    papers = c.get("/papers").json()
    assert len(papers) == 5 and papers[0]["arxiv_id"] == "2401.00001"


def test_recommend_lears_interest():
    c = make_client()
    c.post("/refresh")
    # no votes yet -> digest falls back to latest
    assert c.get("/digest").json()["mode"] == "latest"
    assert c.get("/recommend").json() == []

    # upvote the two NLP/LLM papers
    c.post("/vote/2401.00001", params={"up": True})
    c.post("/vote/2401.00005", params={"up": True})

    recs = c.get("/recommend", params={"k": 3}).json()
    assert len(recs) == 3
    # the remaining language-model paper should outrank the vision papers
    ids = [r["paper"]["arxiv_id"] for r in recs]
    assert ids[0] == "2401.00002", ids

    d = c.get("/digest").json()
    # 5 papers minus the 2 upvoted (excluded) = 3 recommendations
    assert d["mode"] == "recommended" and len(d["papers"]) == 3


def test_index_cache_invalidated_on_paper_revision():
    """v2 revision (same arxiv_id, new abstract) must refresh the tf-idf index."""
    from minisanity.recommend import Recommender
    from minisanity.sources import FixtureSource

    store = Store()
    for p in FixtureSource().fetch():
        store.upsert_paper(p)
    rec = Recommender(store)
    store.vote("2401.00001", True)  # interest: NLP / language models
    store.vote("2401.00005", True)

    rec.recommend(k=5)  # warm the cache
    s_before = {r["paper"]["arxiv_id"]: r["score"] for r in rec.recommend(k=5)}
    assert s_before["2401.00002"] > 0  # Attention paper matches the LM centroid

    # 2401.00002 "revised": abstract rewritten as a vision paper
    store.upsert_paper(Paper("2401.00002", "Attention Is All You Need",
                             "residual convolutional networks for visual "
                             "recognition of images", ["Vaswani"], ["cs.CL"]))
    s_after = {r["paper"]["arxiv_id"]: r["score"] for r in rec.recommend(k=5)}
    # stale cache would return the identical score; the fix rebuilds the index
    # so the revised paper no longer matches the LM centroid
    assert s_after["2401.00002"] < s_before["2401.00002"], \
        (s_before["2401.00002"], s_after["2401.00002"])


def test_similar():
    c = make_client()
    c.post("/refresh")
    out = c.get("/similar/2401.00003", params={"k": 2}).json()
    # the other vision paper should be the nearest neighbour
    assert out[0]["paper"]["arxiv_id"] == "2401.00004"
    assert 0 < out[0]["score"] < 1
    assert out[0]["score"] >= out[1]["score"]


def test_vote_unknown_404():
    c = make_client()
    c.post("/refresh")
    assert c.post("/vote/9999.99999").status_code == 404


def test_tokenize_stops_words():
    toks = tokenize("The attention mechanism of the transformer network")
    assert "the" not in toks and "attention" in toks and "of" not in toks


def test_cosine_sanity():
    idx = TfIdfIndex([])
    assert idx.cosine({"a": 1.0}, {"a": 1.0}) == 1.0
    assert idx.cosine({"a": 1.0}, {"b": 1.0}) == 0.0


if __name__ == "__main__":
    test_refresh_and_papers()
    test_recommend_lears_interest()
    test_index_cache_invalidated_on_paper_revision()
    test_similar()
    test_vote_unknown_404()
    test_tokenize_stops_words()
    test_cosine_sanity()
    print("all minisanity tests passed")
