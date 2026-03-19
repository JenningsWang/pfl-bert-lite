from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    BaseModel = object

from pfl_bert_lite.nas.constraint_search import choose_candidate, default_candidates


class PredictionRequest(BaseModel):
    client_id: str
    text: str
    latency_budget_ms: float = 40.0
    parameter_budget_m: float = 12.0



def create_app() -> "FastAPI":
    if FastAPI is None:
        raise ImportError("fastapi is required to create the serving application.")

    app = FastAPI(title="PFL-BERT Lite", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: PredictionRequest) -> dict[str, object]:
        candidate = choose_candidate(
            candidates=default_candidates(),
            latency_budget_ms=payload.latency_budget_ms,
            parameter_budget_m=payload.parameter_budget_m,
            client_scale=0.5,
            topology_bonus=0.25,
        )
        return {
            "client_id": payload.client_id,
            "selected_subnetwork": candidate.name,
            "text_length": len(payload.text.split()),
            "message": "Attach a fine-tuned checkpoint for production inference.",
        }

    return app


app = create_app() if FastAPI is not None else None
