# safety_service/app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Req(BaseModel):
    text: str

# VERY simple placeholder:
UNSAFE_TRIGGERS = [
    "branch predictor", "branch-predictor", "prediction unit", "bpu",
    "l1 cache", "l2 cache", "reorder buffer", "rob", "rtl", "verilog",
    "microarchitecture", "die shot", "floorplan", "infinity fabric"
]

@app.post("/moderate")
def moderate(req: Req):
    t = (req.text or "").lower()
    # toy logic: if any trigger inside -> UNSAFE; otherwise SAFE
    verdict = "UNSAFE" if any(k in t for k in UNSAFE_TRIGGERS) else "SAFE"
    return {"verdict": verdict}
