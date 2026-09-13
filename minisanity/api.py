"""FastAPI application exposing the minisanity library."""

from fastapi import Depends, FastAPI, HTTPException

from .recommend import Recommender
from .sources import FixtureSource
from .store import Store


def make_app(store: Store | None = None, source=None) -> FastAPI:
    app = FastAPI(title="minisanity", version="0.1")
    store = store or Store()
    source = source or FixtureSource()

    def get_store():
        return store

    def get_recommender():
        return Recommender(store)

    @app.post("/refresh")
    def refresh(max_results: int = 100):
        papers = source.fetch(max_results=max_results)
        for p in papers:
            store.upsert_paper(p)
        return {"ingested": len(papers)}

    @app.get("/papers")
    def papers(store: Store = Depends(get_store)):
        return [p.__dict__ for p in store.all_papers()]

    @app.post("/vote/{arxiv_id}")
    def vote(arxiv_id: str, up: bool = True, store: Store = Depends(get_store)):
        known = {p.arxiv_id for p in store.all_papers()}
        if arxiv_id not in known:
            raise HTTPException(404, "unknown arxiv_id")
        store.vote(arxiv_id, up)
        return {"arxiv_id": arxiv_id, "up": up}

    @app.get("/similar/{arxiv_id}")
    def similar(arxiv_id: str, k: int = 5,
                rec: Recommender = Depends(get_recommender)):
        out = rec.similar(arxiv_id, k=k)
        if not out:
            raise HTTPException(404, "unknown arxiv_id")
        return out

    @app.get("/recommend")
    def recommend(k: int = 5, rec: Recommender = Depends(get_recommender)):
        return rec.recommend(k=k)

    @app.get("/digest")
    def digest(k: int = 5, rec: Recommender = Depends(get_recommender)):
        return rec.digest(k=k)

    return app


app = make_app()
