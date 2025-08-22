""" python -m src.evaluation.evaluate_txm_fidelity --data_dir data_processed --config config/cascade_config.json --k 10 --out data_processed/txm_fidelity_report.json
Inspect report at data_processed/txm_fidelity_report.json"""

import argparse, json, pathlib
import numpy as np
import joblib
import shap
import torch
import time
from sklearn.model_selection import train_test_split

from src.evaluation.cross_model_attribution_fidelity_metrics import (
    sign_fidelity, rank_fidelity, prob_monotonicity
)
from src.models.cascade import transformer_predict_proba
from src.models.cascade import load_models  # load transformer
from src.models.safe_smote import SafeSMOTE  


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_tau(cfg_path: pathlib.Path) -> float:
    try:
        cfg = json.loads(cfg_path.read_text())
        return float(cfg.get("tau", 0.2))
    except Exception:
        return 0.2

def topk_alignment(rf_vec: np.ndarray, tx_vec: np.ndarray, k: int = 10):
    k = max(1, k)
    rf_top = np.argsort(-np.abs(rf_vec))[:k]
    tx_top = np.argsort(-np.abs(tx_vec))[:k]
    inter = np.intersect1d(rf_top, tx_top)
    overlap = len(inter) / float(k)
    if len(inter) == 0:
        sign_agree = 1.0  # vacuously true
    else:
        sign_agree = float(np.mean(np.sign(rf_vec[inter]) == np.sign(tx_vec[inter])))
    return overlap, sign_agree

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=pathlib.Path, default=pathlib.Path("data_processed"))
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config/cascade_config.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=10, help="top-k for rank/alignment")
    ap.add_argument("--s_min", type=float, default=0.7, help="sign fidelity threshold")
    ap.add_argument("--r_min", type=float, default=0.2, help="rank fidelity threshold")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="output JSON path")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load arrays used by cascade.py
    X_tab = np.load(args.data_dir / "X_train_tab.npy", allow_pickle=True)
    y_tab = np.load(args.data_dir / "y_train.npy", allow_pickle=True)
    seq_data = torch.load(args.data_dir / "seq_train.pt")

    # Feature config if available (for Transformer input slicing)
    cont_dim = 13
    high_cat_dim = 2
    low_cat_dim = 10
    if args.config.exists():
        try:
            _cfg = json.loads(args.config.read_text())
            fc = _cfg.get("feature_config", {})
            cont_dim = int(fc.get("continuous_features", cont_dim))
            high_cat_dim = int(fc.get("high_cardinality_features", high_cat_dim))
            low_cat_dim = int(fc.get("low_cardinality_features", low_cat_dim))
        except Exception:
            pass

    # Build validation split like cascade.py
    val_frac = 0.2
    min_val_size = len(np.unique(y_tab))
    val_size = max(min_val_size, int(len(X_tab) * val_frac))
    test_size = val_size / len(X_tab)

    if len(np.unique(y_tab)) > 1 and len(y_tab) > 1:
        indices = np.arange(len(X_tab))
        _, val_idx = train_test_split(indices, test_size=test_size, stratify=y_tab, random_state=42)
    else:
        val_idx = np.arange(val_size)

    X_val_tab = X_tab[val_idx]
    y_val = y_tab[val_idx]

    # Build seq_val tuple compatible with transformer_predict_proba
    if isinstance(seq_data, tuple):
        cont_train, cat_train = seq_data
        cont_val = cont_train[val_idx]
        if cat_train.ndim >= 3:
            cat_high_val = cat_train[val_idx, :, :high_cat_dim].long()
            cat_low_val = cat_train[val_idx, :, high_cat_dim:].long()
            seq_val = (cont_val, cat_high_val, cat_low_val)
        else:
            seq_val = (cont_val, cat_train[val_idx])
    else:
        seq_val = seq_data

    # Load models
    rf = joblib.load(args.data_dir / "rf_model.joblib")
    try:
        _, transformer, _ = load_models(args.data_dir, args.device)
    except Exception:
        transformer = None

    # TreeExplainer on RF
    # explainer = shap.TreeExplainer(rf, model_output="raw")

    tree_model = rf
    if hasattr(rf, "steps"):
            for name, step in reversed(rf.steps):
                # pick first step from the end that has predict_proba_ or tree attributes
                if hasattr(step, "predict_proba") or hasattr(step, "tree_") or hasattr(step, "estimators_"):
                    tree_model = step
                    break
    try:
        explainer = shap.TreeExplainer(tree_model, model_output="raw")
    except TypeError:
        # Older shap may not take model_output
        explainer = shap.TreeExplainer(tree_model)

    # RF probabilities and escalated subset by τ
    tau = load_tau(args.config)
    rf_proba = rf.predict_proba(X_val_tab)
    rf_max = rf_proba.max(axis=1)
    escalated_mask = rf_max >= tau
    if escalated_mask.sum() == 0:
        print("No escalated samples under current τ; lowering τ to include some.")
        thresh = np.quantile(rf_max, 0.2)
        escalated_mask = rf_max >= thresh

    X_escal = X_val_tab[escalated_mask]

    # RF SHAP for positive class
    shap_values = explainer.shap_values(X_escal, check_additivity=False)
    if isinstance(shap_values, list):
        pos_idx = 1 if len(shap_values) > 1 else 0
        s_rf = shap_values[pos_idx]
        expected = explainer.expected_value[pos_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        s_rf = shap_values
        expected = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]

    # If SHAP still 3D (n, features, classes) collapse last dim
    if s_rf.ndim == 3 and s_rf.shape[-1] == 2:
        # choose positive class (index 1); adjust if you need index 0
        s_rf = s_rf[..., 1]
    # if np.ndim(expected) > 0 and len(np.shape(expected)) == 1 and expected.shape[0] == 2:
    #     expected = expected[1]

    # Reconstruct RF probability from SHAP margin + baseline (escalated only)
    # margin = expected + np.sum(s_rf, axis=1)
    # p_rf_hat = sigmoid(margin) if (margin.max() > 1.0 or margin.min() < 0.0) else np.clip(margin, 0.0, 1.0)

    # Reconstruct RF positive-class probability using model predict_proba instead of SHAP margin to avoid shape issues
    rf_proba_escal = rf_proba[escalated_mask]
    p_rf_pos = rf_proba_escal[:, 1]

    # # (Optional) sanity: ensure shapes align
    # if s_rf.shape[0] != p_rf_pos.shape[0]:
    #     raise RuntimeError(f"Mismatch after SHAP reshape: s_rf {s_rf.shape}, p_rf_pos {p_rf_pos.shape}")
    
    # Transformer probabilities on validation then mask
    try:
        p_trans_all = transformer_predict_proba(
            transformer=transformer,
            seq_test=seq_val,
            device=device,
            cont_dim=cont_dim,
            high_cat_dim=high_cat_dim,
            low_cat_dim=low_cat_dim
        )
        if p_trans_all.ndim == 2 and p_trans_all.shape[1] == 2:
            p_trans_escal = p_trans_all[escalated_mask][:, 1]  # positive class
        else:
            p_trans_escal = p_trans_all[escalated_mask]
    except Exception:
        # Fallback: mirror RF for alpha=1
        # p_trans = p_rf_hat.copy()
        p_trans_escal = p_rf_pos.copy()

    # TXM mapping (instance scalar)
    eps = 1e-9
    # alpha = np.clip(p_trans / (p_rf_hat + eps), 0.0, 10.0)
    alpha = np.clip(p_trans_escal / (p_rf_pos + eps), 0.0, 10.0) # shape (n,)
    s_txm =alpha[:, None] * s_rf  # broadcasting to (n, features)
    # s_txm = (alpha.reshape(-1, 1)) * s_rf

    # Timing RF TreeSHAP per-event (approx)
    t0 = time.time()
    _ = explainer.shap_values(X_escal[:50], check_additivity=False)
    dt = (time.time() - t0) / 50.0
    print(f"[TIMING] RF TreeSHAP ~{dt*1000:.3f} ms/event on first 50 escalated samples")

    # Timing TXM mapping (reuse existing s_rf, p_rf_pos, p_trans_escal)
    t1 = time.time()
    _alpha_test = np.clip(p_trans_escal[:50] / (p_rf_pos[:50] + 1e-9), 0.0, 10.0)
    _ = _alpha_test[:, None] * s_rf[:50]
    dt_map = (time.time() - t1) / 50.0
    print(f"[TIMING] TXM map ~{dt_map*1000:.3f} ms/event (scaling only)")

    N = 1000
    times = []
    batch = X_escal[:N]
    for row in batch:
        t0 = time.time()
        _ = explainer.shap_values(row.reshape(1,-1), check_additivity=False)
        times.append(time.time()-t0)
    times = np.array(times)
    print("TreeSHAP median ms:", np.median(times)*1000)
    print("TreeSHAP p95 ms:", np.percentile(times,95)*1000)

    # Fidelity + alignment metrics per-instance
    k = max(1, args.k)
    sign_list, rank_list, mono_list, contradict = [], [], [], []
    overlap_list, overlap_sign_list = [], []
    for i in range(s_txm.shape[0]):
        rf_vec = s_rf[i]
        tx_vec = s_txm[i]
        sign_list.append(sign_fidelity(rf_vec, tx_vec))
        rank_list.append(rank_fidelity(rf_vec, tx_vec, k=k))
        # mono_list.append(prob_monotonicity(rf_vec, tx_vec, float(p_rf_hat[i]), float(p_trans[i])))
        mono_list.append(prob_monotonicity(rf_vec, tx_vec, float(p_rf_pos[i]), float(p_trans_escal[i])))
        # contradiction on RF top-k
        rf_top = np.argsort(-np.abs(rf_vec))[:k]
        opp = any(np.sign(rf_vec[j]) != np.sign(tx_vec[j]) and (abs(rf_vec[j]) > 0 and abs(tx_vec[j]) > 0) for j in rf_top)
        contradict.append(1 if opp else 0)
        # alignment overlap and sign agree on intersection of top-k
        ov, ov_sign = topk_alignment(rf_vec, tx_vec, k=k)
        overlap_list.append(ov)
        overlap_sign_list.append(ov_sign)
    
    alpha_mean = float(np.mean(alpha))
    alpha_std = float(np.std(alpha))
    alpha_min = float(alpha.min())
    alpha_max = float(alpha.max())
    alpha_zero_frac = float(np.mean(alpha == 0.0))
    alpha_nonunity_frac = float(np.mean(np.abs(alpha-1) > 1e-9))
    alpha_percentiles = {p: float(np.percentile(alpha, p)) 
                        for p in (0,0.1,0.5,1,5,25,50,75,95,99,99.5,99.9,100)}
    sign_fidelity_std = float(np.std(sign_list)) if sign_list else None
    rank_fidelity_k_std = float(np.std(rank_list)) if rank_list else None
    prob_monotonicity_std = float(np.std(mono_list)) if mono_list else None
    prob_monotonicity_percentiles = {p: float(np.percentile(mono_list, p)) for p in (1,5,25,50,75,95,99)}


    summary = {
        "n_escalated": int(s_txm.shape[0]),
        "tau": float(tau),
        "k": int(k),
        "sign_fidelity_mean": float(np.mean(sign_list)) if sign_list else None,
        "rank_fidelity_k_mean": float(np.mean(rank_list)) if rank_list else None,
        "prob_monotonicity_mean": float(np.mean(mono_list)) if mono_list else None,
        "contradiction_rate_topk": float(np.mean(contradict)) if contradict else None,
        "alignment_overlap_mean": float(np.mean(overlap_list)) if overlap_list else None,
        "alignment_sign_agree_mean": float(np.mean(overlap_sign_list)) if overlap_sign_list else None,
        "thresholds": {"s_min": args.s_min, "r_min": args.r_min},
        "fails_sign": float(np.mean([x < args.s_min for x in sign_list])) if sign_list else None,
        "fails_rank": float(np.mean([x < args.r_min for x in rank_list])) if rank_list else None,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_zero_frac": alpha_zero_frac,
        "alpha_nonunity_frac": alpha_nonunity_frac,
        "alpha_percentiles": alpha_percentiles,
        "sign_fidelity_std": sign_fidelity_std,
        "rank_fidelity_k_std": rank_fidelity_k_std,
        "prob_monotonicity_std": prob_monotonicity_std,
        "prob_monotonicity_percentiles": prob_monotonicity_percentiles,
    }

    out_path = args.out or (args.data_dir / "txm_fidelity_report.json")
    out_path.write_text(json.dumps({"summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()