# The standalone FastAPI moderation service
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Req(BaseModel):
    text: str


UNSAFE_TRIGGERS = [
    "branch predictor",
    "branch-predictor",
    "prediction unit",
    "bpu",
    "l1 cache",
    "l2 cache",
    "reorder buffer",
    "rob",
    "rtl",
    "verilog",
    "microarchitecture",
    "die shot",
    "floorplan",
    "infinity fabric",
]


@app.post("/moderate")
def moderate(req: Req):
    t = (req.text or "").lower()
    verdict = "UNSAFE" if any(k in t for k in UNSAFE_TRIGGERS) else "SAFE"
    return {"verdict": verdict}