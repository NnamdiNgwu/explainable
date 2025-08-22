"""python -m src.evaluation.feature_importance_Shap_concentration_TXM_and_IG_overlap --plot"""

import json, pathlib, joblib, numpy as np, shap, pandas as pd, torch
from sklearn.model_selection import train_test_split
from src.models.safe_smote import SafeSMOTE
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: Captum IG (will skip if not installed or mapping not implemented)
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False

from src.models.cascade import load_models, transformer_predict_proba

REPORT_NAME = "cross_model_feature_importance_report.json"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_tau(cfg_path: pathlib.Path) -> float:
    if not cfg_path.exists():
        return 0.2
    try:
        cfg = json.loads(cfg_path.read_text())
        return float(cfg.get("tau", 0.2))
    except Exception:
        return 0.2

def prepare_seq_subset(seq_data, idx, high_cat_dim, low_cat_dim):
    if isinstance(seq_data, tuple):
        cont_train, cat_train = seq_data
        cont_val = cont_train[idx]
        if cat_train.ndim >= 3:
            cat_high_val = cat_train[idx, :, :high_cat_dim].long()
            cat_low_val  = cat_train[idx, :, high_cat_dim:].long()
            return (cont_val, cat_high_val, cat_low_val)
        return (cont_val, cat_train[idx])
    return seq_data

def instance_alpha(rf_probs, trans_probs, eps=1e-8, clip_max=10.0):
    return np.clip(trans_probs / (rf_probs + eps), 0.0, clip_max)

def try_compute_ig_feature_vector(transformer, seq_subset, device):
    """
    Placeholder: attempts to compute IG and aggregate to RF feature space.
    Without a defined mapping from sequence positions to RF feature indices
    this returns None. Extend with your custom aggregation if available.
    """
    if not (CAPTUM_AVAILABLE and transformer):
        return None
    try:
        transformer.eval()
        # Expect seq_subset = (cont, cat_high, cat_low)
        cont, cat_high, cat_low = seq_subset
        cont = cont.to(device)
        cat_high = cat_high.to(device)
        cat_low = cat_low.to(device)

        def forward_fn(c, ch, cl):
            out = transformer(c, ch, cl)
            logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
            # binary positive logit
            return logits[:, 1]

        ig = IntegratedGradients(lambda c, ch, cl: forward_fn(c, ch, cl))
        baseline_cont = torch.zeros_like(cont)
        baseline_ch   = torch.zeros_like(cat_high)
        baseline_cl   = torch.zeros_like(cat_low)
        atts = ig.attribute((cont, cat_high, cat_low),
                            baselines=(baseline_cont, baseline_ch, baseline_cl),
                            n_steps=32, internal_batch_size=None)
        # atts is a tuple; only continuous tensor can be trivially reduced
        cont_attr = atts[0].detach().cpu().numpy()  # shape: (N, T, C_cont)
        # Simple reduction: mean over time steps → per-cont feature vector
        ig_vec = np.mean(cont_attr, axis=1)  # (N, C_cont)
        # NOTE: This does NOT align to full RF feature taxonomy (categoricals omitted)
        # Return mean absolute as placeholder; user must extend mapping.
        return np.mean(np.abs(ig_vec), axis=0)
    except Exception:
        return None


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    s = x.sum()
    if s == 0 or x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # Gini = (2 * sum_i i*x_i)/(n*sum x) - (n+1)/n
    return float( (2.0 * np.sum((np.arange(1, n+1) * x_sorted)))/(n * s) - (n + 1.0)/n )

def herfindahl(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s == 0 or x.size == 0:
        return 0.0
    p = x / s
    return float(np.sum(p * p))


def compute_ig_mean_abs(transformer, seq_escal, device, cont_dim, batch_size=64):
    """
    Compute mean |IG| per original continuous feature (aggregated over time & samples).
    Categorical features not attributed here (set to 0); extend if you implement embedding hooks.
    """
    if not (CAPTUM_AVAILABLE and transformer):
        return None
    try:
        transformer.eval()
        cont, ch, cl = seq_escal
        N = cont.size(0)
        ig = IntegratedGradients(lambda c, chh, cll: (
            transformer(c, chh, cll)["logits"][:, 1]
            if isinstance(transformer(c, chh, cll), dict)
            else (transformer(c, chh, cll)[0][:, 1] if isinstance(transformer(c, chh, cll), tuple)
                  else transformer(c, chh, cll)[:, 1])
        ))
        acc = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            c_batch = cont[start:end].to(device)
            ch_batch = ch[start:end].to(device)
            cl_batch = cl[start:end].to(device)
            atts = ig.attribute((c_batch, ch_batch, cl_batch),
                                baselines=(torch.zeros_like(c_batch),
                                           torch.zeros_like(ch_batch),
                                           torch.zeros_like(cl_batch)),
                                n_steps=32)
            cont_attr = atts[0].detach().cpu().numpy()  # (b, T, cont_dim)
            acc.append(np.abs(cont_attr).mean(axis=(0,1)))  # (cont_dim,)
        mean_cont = np.mean(acc, axis=0)  # (cont_dim,)
        # Build full vector length = total RF feature count; assume continuous first
        # Zero padding for categorical features
        pad_len = len(feature_names) - cont_dim
        if pad_len < 0:
            # Truncate if mismatch
            mean_cont = mean_cont[:len(feature_names)]
            pad_len = 0
        ig_full = np.concatenate([mean_cont, np.zeros(pad_len, dtype=float)], axis=0)
        return ig_full
    except Exception:
        return None


def plot_concentration(feature_names, mean_abs_rf, mean_abs_txm, ig_full, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cumulative curves
    def curve(m):
        s = m.sum()
        return np.cumsum(np.sort(m)[::-1]) / (s if s else 1)
    rf_curve = curve(mean_abs_rf)
    txm_curve = curve(mean_abs_txm)
    plt.figure(figsize=(5,4))
    plt.plot(rf_curve[:50], label="RF SHAP")
    plt.plot(txm_curve[:50], label="TXM mapped")
    if ig_full is not None:
        plt.plot(curve(ig_full)[:50], label="IG (cont)")
    plt.xlabel("Top-k features")
    plt.ylabel("Cumulative |importance| fraction")
    plt.title("Importance concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "importance_concentration_curve.png", dpi=150)
    plt.close()

    # Top-20 bar (RF vs TXM)
    top_k = 20
    order_rf = np.argsort(-mean_abs_rf)[:top_k]
    order_txm = np.argsort(-mean_abs_txm)[:top_k]
    union = list(dict.fromkeys([*order_rf, *order_txm]))[:top_k]
    x_labels = [feature_names[i] for i in union]
    rf_vals = mean_abs_rf[union]
    txm_vals = mean_abs_txm[union]
    width = 0.4
    x = np.arange(len(union))
    plt.figure(figsize=(0.5*len(union)+1,4))
    plt.bar(x - width/2, rf_vals, width, label="RF SHAP")
    plt.bar(x + width/2, txm_vals, width, label="TXM mapped")
    if ig_full is not None:
        ig_vals = ig_full[union]
        plt.plot(x, ig_vals, 'x-', label="IG (cont)")
    plt.xticks(x, x_labels, rotation=75, ha='right', fontsize=8)
    plt.ylabel("Mean |importance|")
    plt.title("Top feature importance comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "top_feature_importance_comparison.png", dpi=150)
    plt.close()



# ...existing code above...

def resolve_feature_names(rf, X_tab, data_dir: pathlib.Path) -> list:
    """
    Returns list of feature names length = X_tab.shape[1].
    Order of preference:
      1. rf.feature_names_in_ (or estimator step)
      2. recovered file rf_feature_names_resolved.json (exact length match)
      3. semantic feature_lists.json (exact length match)
      4. Fallback generic f{i}
    Logs diagnostics; writes a mapping report.
    """
    n = X_tab.shape[1]
    chosen = None
    source = None

    def grab_names_from_model(model):
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "named_steps"):
            for s in model.named_steps.values():
                if hasattr(s, "feature_names_in_"):
                    return list(s.feature_names_in_)
        if hasattr(model, "steps"):
            for _, s in model.steps:
                if hasattr(s, "feature_names_in_"):
                    return list(s.feature_names_in_)
        return None

    # 1 direct/pipeline
    direct = grab_names_from_model(rf)
    if direct and len(direct) == n:
        chosen, source = direct, "model"
    # 2 recovered file
    if chosen is None:
        rec_path = data_dir / "rf_feature_names_resolved.json"
        if rec_path.exists():
            try:
                rec = json.loads(rec_path.read_text())
                if len(rec) == n:
                    chosen, source = rec, "recovered_file"
            except Exception as e:
                print(f"[names] failed reading recovered file: {e}")
    # 3 semantic
    if chosen is None:
        fl_path = data_dir / "feature_lists.json"
        if fl_path.exists():
            try:
                fl = json.loads(fl_path.read_text())
                semantic = (
                    fl.get("CONTINUOUS_USED", []) +
                    fl.get("BOOLEAN_USED", []) +
                    fl.get("LOW_CAT_USED", []) +
                    fl.get("HIGH_CAT_USED", [])
                )
                if len(semantic) == n:
                    chosen, source = semantic, "semantic_feature_lists"
                else:
                    print(f"[names] semantic length mismatch semantic={len(semantic)} matrix={n}")
            except Exception as e:
                print(f"[names] semantic load error: {e}")
    # 4 fallback
    if chosen is None:
        chosen, source = [f"f{i}" for i in range(n)], "generic"

    print(f"[names] using source={source} count={len(chosen)}")
    # Write mapping report
    mapping = [{"index": i, "name": nm, "source": source} for i, nm in enumerate(chosen)]
    (data_dir / "feature_name_mapping.json").write_text(json.dumps(mapping, indent=2))
    return chosen


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="Generate PNG plots")
    args, _ = ap.parse_known_args()
    
    data_dir = pathlib.Path("data_processed")
    config_path = pathlib.Path("config/cascade_config.json")

    rf = joblib.load(data_dir / "rf_model.joblib")
    
    # def get_cols(model):
    #     # try direct
    #     if hasattr(model, "feature_names_in_"):
    #         return list(model.feature_names_in_)
    #     # try sklearn pipeline steps
    #     if hasattr(model, "named_steps"):
    #         for s in model.named_steps.values():
    #             if hasattr(s, "feature_names_in_"):
    #                 return list(s.feature_names_in_)
    #             # try generic step
    #     if hasattr(model, "steps"):
    #         for _, s in model.steps:
    #             if hasattr(s, "feature_names_in_"):
    #                 return list(s.feature_names_in_)
    #     return []
    # cols = get_cols(rf)
    # if cols:
    #     (data_dir / "rf_feature_names_resolved.json").write_text(json.dumps(cols, indent=2))
    #     print(f"[recover] Wrote {len(cols)} raw model feature names.")
    # else:
    #     print("[recover] No feature_names_in_ exposed; skipping empty write.")
    
    X_tab = np.load(data_dir / "X_train_tab.npy", allow_pickle=True)
    feature_names = resolve_feature_names(rf, X_tab, data_dir)
    # Hierarchical feature names resolution
    # feature_names = None
    # try:
    #     # direct
    #     feature_names = getattr(rf, "feature_names_in_", None)
    # except Exception:
    #     pass
    # #2 last estimator in pipeline
    # if feature_names is None and hasattr(rf, "steps"):
    #     # Try last step (usually the estimator)
    #     last_est = rf.steps[-1][1]
    #     # if hasattr(last_est, "feature_names_in_"):
    #     feature_names = getattr(last_est, "feature_names_in_", None)
    # # if feature_names is None:
    # #     # Final fallback
    # #     feature_names = [f"f{i}" for i in range(X_tab.shape[1])]
    # recovered = data_dir / "rf_feature_names_resolved.json"
    # if (feature_names is None or len(feature_names) == 0) and recovered.exists():
    #     try:
    #         rec = json.loads(recovered.read_text())
    #         if len(rec) == X_tab.shape[1]:
    #             feature_names = rec
    #             print(f"[names] using recovered names ({len(feature_names)})")
    #     except Exception as e:
    #         print(f"[recover] Failed to use recovered names: {e}")
    #   # semantic featurelists.json only if lenght match
    # feature_lists_path = data_dir / "feature_lists.json"
    # if feature_lists_path.exists():
    #     try:
    #         fl = json.loads(feature_lists_path.read_text())
    #         semantic_names = (
    #             fl.get("CONTINUOUS_USED", []) +
    #             fl.get("BOOLEAN_USED", []) +
    #             fl.get("LOW_CAT_USED", []) +
    #             fl.get("HIGH_CAT_USED", [])
    #         )
    #         if len(semantic_names) == X_tab.shape[1]:
    #             feature_names = semantic_names
    #             print(f"[names] Applied semantic feature names ({len(feature_names)})")
    #         else:
    #             print(f"[names] Semantic length mismatch: semantic={len(semantic_names)} rf_matrix={X_tab.shape[1]} (keeping previous).")
    #     except Exception as e:
    #         print(f"[names] Failed semantic load: {e}")
    # else:
    #     print(f"[feature_lists] File not found at {feature_lists_path}; using fallback names.")
    
    # 5. Fallback
    if feature_names is None or len(feature_names) != X_tab.shape[1]:
        feature_names = [f"f{i}" for i in range(X_tab.shape[1])]
        print(f"[names] Using generic names ({len(feature_names)})")

    y_tab = np.load(data_dir / "y_train.npy", allow_pickle=True)
    seq_data = torch.load(data_dir / "seq_train.pt")

    # Simple temporal/stratified validation split mimic
    val_frac = 0.2
    val_size = max(len(np.unique(y_tab)), int(len(X_tab) * val_frac))
    test_size = val_size / len(X_tab)
    if len(np.unique(y_tab)) > 1 and len(y_tab) > 1:
        idx_all = np.arange(len(X_tab))
        _, val_idx = train_test_split(idx_all, test_size=test_size, stratify=y_tab, random_state=42)
    else:
        val_idx = np.arange(val_size)

    X_val = X_tab[val_idx]
    y_val = y_tab[val_idx]

    # TreeSHAP (RF)
    # explainer = shap.TreeExplainer(rf, model_output="raw", feature_dependence="independent")
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

    # Sample up to 5000 for SHAP stats
    rng = np.random.default_rng(0)
    sample_idx = val_idx if len(val_idx) <= 5000 else rng.choice(val_idx, 5000, replace=False)
    X_sample = X_tab[sample_idx]
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(shap_values, list):
        pos_idx = 1 if len(shap_values) > 1 else 0
        s_rf_sample = shap_values[pos_idx]
        rf_expected = explainer.expected_value[pos_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        s_rf_sample = shap_values
        rf_expected = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
    # Handle 3D (N, F, C) → select positive class
    if s_rf_sample.ndim == 3 and s_rf_sample.shape[-1] >= 2:
        s_rf_sample = s_rf_sample[..., 1]

    mean_abs_rf = np.mean(np.abs(s_rf_sample), axis=0)
    order_rf = np.argsort(-mean_abs_rf)
    cum_rf = np.cumsum(np.sort(mean_abs_rf)[::-1]) / mean_abs_rf.sum()
    top10_cover = float(np.sum(np.sort(mean_abs_rf)[::-1][:10]) / mean_abs_rf.sum())
    top20_cover = float(np.sum(np.sort(mean_abs_rf)[::-1][:20]) / mean_abs_rf.sum()) if len(mean_abs_rf) >= 20 else None

    # Compute RF probabilities on validation and select escalated subset
    tau = load_tau(config_path)
    rf_probs_val = rf.predict_proba(X_val)
    rf_max = rf_probs_val.max(axis=1)
    escalated_mask = rf_max >= tau
    if escalated_mask.sum() == 0:
        # fallback quantile
        thr = np.quantile(rf_max, 0.8)
        escalated_mask = rf_max >= thr
    X_escal = X_val[escalated_mask]

    # Recompute SHAP on escalated subset (for mapped TXM stats)
    shap_escal = explainer.shap_values(X_escal, check_additivity=False)
    if isinstance(shap_escal, list):
        pos_idx2 = 1 if len(shap_escal) > 1 else 0
        s_rf_escal = shap_escal[pos_idx2]
        rf_expected_escal = explainer.expected_value[pos_idx2] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        s_rf_escal = shap_escal
        rf_expected_escal = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
    if s_rf_escal.ndim == 3 and s_rf_escal.shape[-1] >= 2:
        s_rf_escal = s_rf_escal[..., 1]

    # Load transformer for probabilities
    try:
        _, transformer, _ = load_models(data_dir, "cpu")
    except Exception:
        transformer = None

    # Feature config (fallback defaults)
    cont_dim = 13
    high_cat_dim = 2
    low_cat_dim = 10
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            fc = cfg.get("feature_config", {})
            cont_dim = int(fc.get("continuous_features", cont_dim))
            high_cat_dim = int(fc.get("high_cardinality_features", high_cat_dim))
            low_cat_dim = int(fc.get("low_cardinality_features", low_cat_dim))
        except Exception:
            pass

    # Prepare escalated seq subset indices (map val_idx to escalated indices)
    val_positions = np.where(np.isin(val_idx, val_idx))[0]  # identity but explicit
    escal_positions = np.where(escalated_mask)[0]
    # Build sequence tensors for entire validation then subset
    seq_val_full = prepare_seq_subset(seq_data, val_idx, high_cat_dim, low_cat_dim)
    # Subset for escalated
    if isinstance(seq_val_full, tuple):
        cont_v, ch_v, cl_v = seq_val_full
        seq_escal = (cont_v[escal_positions], ch_v[escal_positions], cl_v[escal_positions])
    else:
        seq_escal = seq_val_full

    # Transformer probabilities (positive class)
    if transformer:
        p_trans_all = transformer_predict_proba(transformer=transformer,
                                                seq_test=seq_val_full,
                                                device=torch.device("cpu"),
                                                cont_dim=cont_dim,
                                                high_cat_dim=high_cat_dim,
                                                low_cat_dim=low_cat_dim)
        p_trans = p_trans_all[escalated_mask]
    else:
        # Fallback: mirror RF positive prob for alpha=1 baseline
        p_trans = rf_probs_val[escalated_mask, 1] if rf_probs_val.shape[1] > 1 else rf_probs_val[escalated_mask, 0]

    # RF positive prob (for escalated subset)
    p_rf_escal = rf_probs_val[escalated_mask, 1] if rf_probs_val.shape[1] > 1 else rf_probs_val[escalated_mask, 0]

    # TXM mapped (instance-probability scaling)
    alpha_inst = instance_alpha(p_rf_escal, p_trans, eps=1e-8, clip_max=10.0)
    # s_txm = alpha_inst.reshape(-1, 1) * s_rf_escal # (N, F)
    s_txm = alpha_inst[:, None] * s_rf_escal  # (N, F)
    mean_abs_txm = np.mean(np.abs(s_txm), axis=0)
    order_txm = np.argsort(-mean_abs_txm)
    txm_top10_cover = float(np.sum(np.sort(mean_abs_txm)[::-1][:10]) / mean_abs_txm.sum())
    txm_top20_cover = float(np.sum(np.sort(mean_abs_txm)[::-1][:20]) / mean_abs_txm.sum()) if len(mean_abs_txm) >= 20 else None

    # Optional IG true (continuous) attributions
    ig_full = None
    if transformer and isinstance(seq_escal, tuple):
        ig_full = compute_ig_mean_abs(transformer, seq_escal, torch.device("cpu"), cont_dim=cont_dim)

    # Metrics helper
    def concentration_stats(mass_vec: np.ndarray, label: str):
        m = np.asarray(mass_vec, dtype=float)
        total = m.sum()
        if total <= 0:
            return {
                "sum": float(total),
                "top10_mass_fraction": None,
                "top20_mass_fraction": None,
                "gini": None,
                "herfindahl": None
            }
        sorted_abs = np.sort(m)[::-1]
        return {
            "sum": float(total),
            "top10_mass_fraction": float(sorted_abs[:10].sum()/total) if m.size >= 10 else float(sorted_abs.sum()/total),
            "top20_mass_fraction": float(sorted_abs[:20].sum()/total) if m.size >= 20 else None,
            "gini": gini(m),
            "herfindahl": herfindahl(m)
        }
    rf_stats   = concentration_stats(mean_abs_rf, "rf_shap")
    txm_stats  = concentration_stats(mean_abs_txm, "txm_mapped")
    ig_stats   = concentration_stats(ig_full, "ig") if ig_full is not None else None

    # Rank overlaps
    def rank_order(v): return np.argsort(-v)
    order_rf = rank_order(mean_abs_rf)
    order_txm = rank_order(mean_abs_txm)
    overlap_top10_txm = len(set(order_rf[:10]) & set(order_txm[:10]))/10.0
    overlap_top20_txm = len(set(order_rf[:20]) & set(order_txm[:20]))/20.0 if len(order_rf) >= 20 else None

    ig_overlap_top10 = ig_overlap_top20 = None
    if ig_full is not None:
        order_ig = rank_order(ig_full)
        ig_overlap_top10 = len(set(order_rf[:10]) & set(order_ig[:10]))/10.0
        ig_overlap_top20 = len(set(order_rf[:20]) & set(order_ig[:20]))/20.0 if len(order_rf) >= 20 else None

    report = {
        "rf_shap": {
            **rf_stats,
            "top10_features": [feature_names[i] for i in order_rf[:10]],
            "cumulative_mass_curve_first_25": (np.cumsum(np.sort(mean_abs_rf)[::-1])/ (mean_abs_rf.sum() or 1))[:25].tolist(),
            "raw_mean_abs": mean_abs_rf.tolist()
        },
        "all_escalated": bool(escalated_mask.sum() == len(X_val)),
        "txm_mapped": {
            **txm_stats,
            "top10_features": [feature_names[i] for i in order_txm[:10]],
            "top10_overlap_with_rf": overlap_top10_txm,
            "top20_overlap_with_rf": overlap_top20_txm,
            "raw_mean_abs": mean_abs_txm.tolist()
        },
        "ig_continuous": {
            "available": ig_full is not None,
            **(ig_stats or {}),
            "top10_overlap_with_rf": ig_overlap_top10,
            "top20_overlap_with_rf": ig_overlap_top20,
            "note": "IG covers continuous features only; categorical positions zero-padded.",
            "raw_mean_abs": (ig_full.tolist() if ig_full is not None else None)
        },
         "meta": {
            "tau_used": tau,
            "n_validation": int(len(X_val)),
            "n_escalated": int(escalated_mask.sum()),
            "captum_available": CAPTUM_AVAILABLE,
            "all_escalated": bool(escalated_mask.sum() == len(X_val))
        }
    }


    # # Rank overlap RF SHAP vs TXM (top-k)
    # def topk_overlap(a_idx, b_idx, k):
    #     return len(set(a_idx[:k]) & set(b_idx[:k])) / float(k)

    # overlap_top10_txm = topk_overlap(order_rf, order_txm, 10)
    # overlap_top20_txm = topk_overlap(order_rf, order_txm, 20) if len(order_rf) >= 20 else None

    # # Optional IG mean abs (placeholder aggregation)
    # ig_mean_abs = None
    # ig_overlap_top10 = None
    # if transformer:
    #     ig_mean_abs = try_compute_ig_feature_vector(transformer, seq_escal, torch.device("cpu"))
    #     if ig_mean_abs is not None and ig_mean_abs.shape[0] == mean_abs_rf.shape[0]:
    #         order_ig = np.argsort(-ig_mean_abs)
    #         ig_overlap_top10 = topk_overlap(order_rf, order_ig, 10)

    # report = {
    #     "rf_shap": {
    #         "top10_mass_fraction": top10_cover,
    #         "top20_mass_fraction": top20_cover,
    #         "cumulative_mass_curve_first_25": cum_rf[:25].tolist(),
    #         "top10_features": [feature_names[i] for i in order_rf[:10]]
    #     },
    #     "txm_mapped": {
    #         "top10_mass_fraction": txm_top10_cover,
    #         "top20_mass_fraction": txm_top20_cover,
    #         "top10_features": [feature_names[i] for i in order_txm[:10]],
    #         "top10_overlap_with_rf": overlap_top10_txm,
    #         "top20_overlap_with_rf": overlap_top20_txm
    #     },
    #     "ig_optional": {
    #         "available": ig_mean_abs is not None,
    #         "top10_overlap_rf": ig_overlap_top10
    #     },
    #     "meta": {
    #         "tau_used": tau,
    #         "n_validation": int(len(X_val)),
    #         "n_escalated": int(escalated_mask.sum()),
    #         "captum_available": CAPTUM_AVAILABLE
    #     }
    # }

    out_path = data_dir / REPORT_NAME
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Report written to {out_path}")

    if args.plot:
        plot_concentration(feature_names, mean_abs_rf, mean_abs_txm, ig_full, data_dir)
    print("Feature importance analysis completed.")

if __name__ == "__main__":
    main()