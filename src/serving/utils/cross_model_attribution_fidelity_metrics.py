import numpy as np
from scipy.stats import kendalltau

def sign_fidelity(rf_s: np.ndarray, txm_s: np.ndarray) -> float:
    same = np.sign(rf_s) == np.sign(txm_s)
    return float(np.mean(same))

def rank_fidelity(rf_s: np.ndarray, txm_s: np.ndarray, k: int = 10) -> float:
    k = min(k, rf_s.size)
    rf_top = np.argsort(-np.abs(rf_s))[:k]
    txm_top = np.argsort(-np.abs(txm_s))[:k]
    # Kendall τ on union of indices
    idx = np.unique(np.concatenate([rf_top, txm_top]))
    tau, _ = kendalltau(np.abs(rf_s[idx]), np.abs(txm_s[idx]))
    return float(0.0 if np.isnan(tau) else tau)

# def prob_monotonicity(rf_s: np.ndarray, txm_s: np.ndarray, p_rf: float, p_trans_escal: float) -> float:
#     rel_change = (np.sum(np.abs(txm_s)) - np.sum(np.abs(rf_s))) / (np.sum(np.abs(rf_s)) + 1e-9)
#     return float(rel_change if p_trans_escal >= p_rf else -rel_change)

# L1 vs prob ratio alignment (normalized)
def prob_monotonicity(rf_s, txm_s, p_rf, p_trans):
    # Use L1 attribution scaling consistency with probability ratio
    l1_rf = np.sum(np.abs(rf_s)) + 1e-9
    l1_tx = np.sum(np.abs(txm_s)) + 1e-9
    rel_change_attr = (l1_tx - l1_rf) / l1_rf
    rel_change_prob = (p_trans - p_rf) / (p_rf + 1e-9)
    # Return signed agreement score in [-2,2]; normalize to [-1,1]
    score = rel_change_attr * rel_change_prob
    # Optionally clamp
    return float(np.tanh(score))

def test_example():
    # toy vectors
    rf = np.array([0.6, -0.2, 0.1, 0.0, 0.3])
    txm = 2.0 * rf
    assert sign_fidelity(rf, txm) >= 0.99
    assert rank_fidelity(rf, txm, k=3) > 0.9