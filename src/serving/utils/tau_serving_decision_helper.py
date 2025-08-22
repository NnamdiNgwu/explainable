import json
from pathlib import Path

def load_cascade_config(path: str | Path):
    cfg = json.loads(Path(path).read_text())
    tau  = float(cfg.get("tau", 0.2))
    tau2 = float(cfg.get("tau2", 0.5))
    return tau, tau2, cfg

def cascade_decision(p_rf: float, p_trans: float | None, tau: float, tau2: float):
    """
    Two-stage cascade decision rule.
    - If p_rf < tau → Benign (0).
    - Else, if p_trans is provided → Malicious iff p_trans >= tau2, otherwise Benign.
    - If p_trans is None (fallback) → Malicious iff p_rf >= 0.9, else Benign.
    """
    if p_rf < tau:
        return 0
    if p_trans is not None:
        return 1 if p_trans >= tau2 else 0
    return 1 if p_rf >= 0.9 else 0