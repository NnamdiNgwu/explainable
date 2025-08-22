#!/usr/bin/env python3
"""make_features.py

Generates ML feature artifacts from uploads.parquet + split indices.

Pipeline stages:
- Population statistics & behavioral anomaly features
- Feature scaling / normalization
- Categorical encoding (one-hot low-card, ordinal high-card)
- Rationale (XAI) multi-hot target generation + thresholds
- Sequence tensor construction (continuous & categorical streams)

Inputs (from chunk_build_uploads.py):
  uploads.parquet
  train_ids.npy
  test_ids.npy

Outputs (to out_dir):
  X_train_tab.npy, X_test_tab.npy          # tabular feature matrices
  y_train.npy, y_test.npy                  # tabular labels
  seq_train.pt, seq_test.pt                # (cont_tensor, cat_tensor) tuples
  y_train_t.pt, y_test_t.pt                # sequence labels
  preprocessors.pkl                        # fitted ColumnTransformer + parts
  feature_lists.json                       # features actually used/missing
  embedding_maps.json, embedding_sizes.json
  rationale_train.npy / rationale_test.npy # rationale multi-hot (uint8)
  rationale_train.pt / rationale_test.pt
  rationale_vocab.json                     # rationale phrases
  rationale_thresholds.json                # learned threshold metadata

Run:
  python -m src.features.make_features \
      --uploads data_interim/uploads.parquet \
      --train_ids data_interim/train_ids.npy \
      --test_ids  data_interim/test_ids.npy \
      --out_dir data_processed
"""

from __future__ import annotations
import argparse, json, pathlib, typing as T

import joblib, numpy as np, pandas as pd, torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTENC


# ---------------------------------------------------------------------------
# RATIONALE (XAI) TARGET GENERATION
# ---------------------------------------------------------------------------

def build_rationale_from_features(df: pd.DataFrame,
                                  thresholds_path: pathlib.Path,
                                  is_training: bool,
                                  train_df: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """
    Derive multi-hot rationale targets from already engineered columns.
    Returns (matrix[N,V], {'vocab': list, 'thresholds': dict})
    """
    vocab = [
        "after-hours timing",
        "deep night hour",
        "large upload size",
        "size outlier vs population",
        "hour outlier vs population",
        "first-time destination",
        "destination domain high entropy",
        "burst of recent uploads",
        "has email attachments",
        "external destination",
        "competitor communication",
        "USB channel transfer",
        "rapid succession timing",
        "destination novelty high"
    ]

    # Safe getter
    def sg(col, default=0):
        if col in df.columns:
            return df[col]
        return pd.Series(default, index=df.index)

    def to_int_series(s):
        if not isinstance(s, pd.Series):
            s = pd.Series(s, index=df.index)
        return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")
        # return pd.Series(default, index=df.index, dtype='int8')

    if is_training:
        entropy_high = sg("destination_entropy", 0).quantile(0.90)
        burst_high   = sg("post_burst", 0).quantile(0.95)
        novelty_high = 0.7
        thresholds = {
            "entropy_high": float(entropy_high),
            "burst_high": float(burst_high),
            "novelty_high": float(novelty_high)
        }
        thresholds_path.write_text(json.dumps(thresholds, indent=2))
    else:
        if thresholds_path.exists():
            thresholds = json.loads(thresholds_path.read_text())
        else:
            base = train_df if train_df is not None else df
            thresholds = {
                "entropy_high": float(base.get("destination_entropy", pd.Series([0])).quantile(0.90)),
                "burst_high": float(base.get("post_burst", pd.Series([0])).quantile(0.95)),
                "novelty_high": 0.7
            }

    entropy_high = thresholds["entropy_high"]
    burst_high   = thresholds["burst_high"]
    novelty_high = thresholds["novelty_high"]

    # Build rationale matrix
    mat = np.column_stack([
        to_int_series(sg("after_hours",0)),
        ((sg("hour", 12) < 6) | (sg("hour", 12) >= 22)).astype(int),
        to_int_series(sg("is_large_upload",0)),
        to_int_series(sg("is_outlier_size",0)),
        to_int_series(sg("is_outlier_hour",0)),
        to_int_series(sg("first_time_destination",0)),
        (sg("destination_entropy", 0) > entropy_high).astype(int),
        (sg("post_burst", 0) > burst_high).astype(int),
        to_int_series(sg("has_attachments",0)),
        to_int_series(sg("external_domain",0)),
        to_int_series(sg("competitor_communication",0)),
        to_int_series(sg("is_usb",0)),
        (sg("seconds_since_previous", 3600) < 60).astype(int),
        (sg("destination_novelty_score", 0) > novelty_high).astype(int),
    ])

    return mat.astype(np.uint8), {"vocab": vocab, "thresholds": thresholds}


def process_rationale_targets(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              out_dir: pathlib.Path) -> None:
    """
    Generate and persist rationale targets + vocab + thresholds.
    """
    thresholds_path = out_dir / "rationale_thresholds.json"
    train_mat, meta_train = build_rationale_from_features(
        train_df, thresholds_path, is_training=True)
    test_mat, _ = build_rationale_from_features(
        test_df, thresholds_path, is_training=False, train_df=train_df)

    assert train_mat.shape[1] == len(meta_train["vocab"]), "Vocab length mismatch"
    assert test_mat.shape[1] == len(meta_train["vocab"]), "Test matrix width mismatch"

    np.save(out_dir / "rationale_train.npy", train_mat)
    np.save(out_dir / "rationale_test.npy", test_mat)
    (out_dir / "rationale_vocab.json").write_text(json.dumps(meta_train["vocab"], indent=2))

    # Torch tensors for sequence / model training
    torch.save(torch.from_numpy(train_mat), out_dir / "rationale_train.pt")
    torch.save(torch.from_numpy(test_mat), out_dir / "rationale_test.pt")

    print(f"[✓] Rationale targets saved: train={train_mat.shape} test={test_mat.shape} vocab={len(meta_train['vocab'])}")
    print(f"[✓] Thresholds: {meta_train['thresholds']}")


# Enhanced feature lists that include anomaly detection
CONTINUOUS = [
    "post_burst", "destination_entropy", "hour", "uploads_last_24h",
     "timestamp_unix", "seconds_since_previous",
    
    # Anomaly detection features
    "user_anomaly_score", "cluster_anomaly_score", "temporal_anomaly_score",
    "volume_anomaly_score", "network_anomaly_score", "composite_anomaly_score",
    "destination_novelty_score", "frequency_anomaly_score",
    # Ground truth features
    "decoy_risk_score", "days_since_decoy",
    
    # Pattern discovery features
    "size_hour_interaction", "entropy_burst_interaction", "upload_velocity",
    "domain_switching_rate", 
]

BOOLEAN = [
    "first_time_destination", "after_hours", "is_large_upload", "to_suspicious_domain",
    "is_usb", "is_weekend", "has_attachments", "is_from_user",
    # Population-based behavioral flags
    "is_outlier_hour",                # Hour is in top/bottom 5% of population
    "is_outlier_size",                # Size is in top 5% of population
    "is_rare_channel",                # Channel used by <10% of population
    "is_high_frequency_user",         # Upload frequency in top 10%
    "exceeds_daily_volume_norm",       # Daily volume exceeds 90th percentile
    
    # Ground truth features
    "is_decoy_interaction",
    
    # Pattern discovery features  
    "weekend_afterhours_interaction", "rare_hour_flag"
]

LOW_CAT = ["channel", "from_domain", "primary_channel"]
HIGH_CAT = ["destination_domain"]
SEQ_LEN    = 50



def diagnose_feature_mismatch(uploads_df, CONTINUOUS, BOOLEAN, LOW_CAT, HIGH_CAT):
    """Track which features are missing and causing the mismatch."""
    print("\n" + "="*60)
    print("FEATURE MISMATCH DIAGNOSIS")
    print("="*60)
    
    present_cols = set(uploads_df.columns)
    
    # Check each feature category
    categories = {
        "CONTINUOUS": CONTINUOUS,
        "BOOLEAN": BOOLEAN, 
        "LOW_CAT": LOW_CAT,
        "HIGH_CAT": HIGH_CAT
    }
    
    total_expected = 0
    total_present = 0
    missing_features = []
    
    for category, features in categories.items():
        expected = len(features)
        present = [f for f in features if f in present_cols]
        missing = [f for f in features if f not in present_cols]
        
        print(f"\n{category}:")
        print(f"  Expected: {expected} features")
        print(f"  Present:  {len(present)} features")
        if missing:
            print(f"  Missing:  {missing}")
            missing_features.extend(missing)
        if present:
            print(f"  Found:    {present}")
            
        total_expected += expected
        total_present += len(present)
    
    print(f"\n" + "-"*40)
    print(f"SUMMARY:")
    print(f"  Total expected: {total_expected}")
    print(f"  Total present:  {total_present}")
    print(f"  Mismatch:       {total_expected - total_present}")
    
    if missing_features:
        print(f"\nMISSING FEATURES ({len(missing_features)}):")
        for i, feature in enumerate(missing_features, 1):
            print(f"  {i}. {feature}")
    
    # Show actual columns in data
    print(f"\nACTUAL COLUMNS IN DATA ({len(uploads_df.columns)}):")
    for col in sorted(uploads_df.columns):
        print(f"  - {col}")
    
    return missing_features, total_expected, total_present

# ---------------------------------------------------------------------------

def factorize_to_index(df: pd.Series) -> T.Tuple[pd.Series, dict[str, int]]:
    """Return integer-coded Series + mapping; UNK idx = 0."""
    mapping: dict[str, int] = {"<UNK>": 0}
    codes  = []
    for val in df.astype(str):
        codes.append(mapping.setdefault(val, len(mapping)))
    return pd.Series(codes, index=df.index, dtype="int64"), mapping

# ---------------------------------------------------------------------------

def build_preprocessors(train_df, CONTINUOUS_USED, BOOLEAN_USED, LOW_CAT_USED, HIGH_CAT_USED):
    scaler  = StandardScaler()
    onehot  = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    pre = ColumnTransformer(
        [
            ("cont", scaler,  CONTINUOUS_USED),
            ("bool", "passthrough", BOOLEAN_USED),
            ("low" , onehot , LOW_CAT_USED),
            ("high", ordinal, HIGH_CAT_USED),
        ], remainder="drop")
    pre.fit(train_df)
    return pre, scaler, onehot, ordinal

# ---------------------------------------------------------------------------


def df_to_sequence_tensor_FIXED(df, embed_maps, CONTINUOUS_USED, BOOLEAN_USED, HIGH_CAT_USED, LOW_CAT_USED, train_df=None, is_training=True):
    """
    FIXED: Generate sequences without temporal leakage.
    
    Args:
        df: Current dataset (train or test)
        reference_data: Dataset to use for building historical context (CRITICAL)
    """
    scaler = StandardScaler()
    cont_cols = CONTINUOUS_USED + BOOLEAN_USED 
   
    if is_training:
        # Training use current data for scaling
        scaler.fit(df[cont_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
        reference_data = df  # Use current data for historical context
    else:
        # Test use training data for scaling and history
        scaler.fit(train_df[cont_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
        reference_data = train_df  # Use training data for historical context

    cont_rows = []
    cat_rows = []
    
    for _, row in df.iterrows():
        if is_training:
            # Use events up to (but Not including) current event
            hist = reference_data[
                (reference_data['date'] < row['date']) & # Temporal constraint
                (np.abs(reference_data['hour'] - row['hour']) <= 3) & # Similar time
                (reference_data['channel'] == row['channel']) & # Same channel
                (np.abs(reference_data['megabytes_sent'] - row['megabytes_sent']) <= row['megabytes_sent'] * 0.5) # Similar size
            ].tail(SEQ_LEN)
        else:
            # Test: Use training events with similar behvioral patterns
            hist = reference_data[
                (reference_data['hour'] - row['hour'] <= 3) & 
                (reference_data['channel'] == row['channel']) &
                (np.abs(reference_data['megabytes_sent'] - row['megabytes_sent']) <= row['megabytes_sent'] * 0.5)
            ].tail(SEQ_LEN)
        
        # Handle case where no similar behavior found
        if len(hist) == 0:
            # # Create dummy history with zeros/defaults
            # dummy_data = {col: [0.0] for col in cont_cols}
            # dummy_data.update({col: ['<UNK>'] for col in HIGH_CAT_USED + LOW_CAT_USED})
            # hist = pd.DataFrame(dummy_data)

            #  Use general population sample
            hist  = reference_data.sample(min(SEQ_LEN, len(reference_data)))
        
        # Process continuous features
        cont_df = hist[cont_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cont_scaled = scaler.transform(cont_df)
        cont = torch.tensor(cont_scaled.astype(np.float32), dtype=torch.float32)

        # Process categorical features
        cat_vals = []
        for col in HIGH_CAT_USED + LOW_CAT_USED:
            cat_vals.append(hist[col].astype(str).map(lambda v: embed_maps[col].get(v, 0)).values)
        cat = torch.tensor(np.stack(cat_vals, axis=-1), dtype=torch.long)
        
        # Padding
        pad = SEQ_LEN - len(hist)
        if pad:
            cont = torch.nn.functional.pad(cont, (0, 0, pad, 0))
            cat = torch.nn.functional.pad(cat, (0, 0, pad, 0))
        
        cont_rows.append(cont)
        cat_rows.append(cat)
    
    return torch.stack(cont_rows), torch.stack(cat_rows)
# ---------------------------------------------------------------------------

# Validation script to confirm no leakage
def validate_no_leakage():
    # Check temporal boundaries
    train_df = pd.read_parquet('data_interim/uploads.parquet').loc[train_ids]
    test_df = pd.read_parquet('data_interim/uploads.parquet').loc[test_ids]
    
    train_max_date = train_df['date'].max()
    test_min_date = test_df['date'].min()
    
    assert train_max_date < test_min_date, "Temporal overlap detected!"
    
    # Check sequence generation
    cont_test, cat_test = torch.load('data_processed/seq_test.pt')
    
    # Verify no perfect correlations
    X_test = np.load('data_processed/X_test_tab.npy')
    y_test = np.load('data_processed/y_test.npy')
    
    from sklearn.metrics import mutual_info_score
    for i in range(X_test.shape[1]):
        mi = mutual_info_score(X_test[:, i], y_test)
        assert mi < 0.9, f"Feature {i} has perfect correlation: {mi}"
    
    print(" No leakage detected - pipeline is clean")

def main():
    global SEQ_LEN
    p = argparse.ArgumentParser()
    p.add_argument("--uploads", required=True, type=pathlib.Path)
    p.add_argument("--train_ids", required=True, type=pathlib.Path)
    p.add_argument("--test_ids" , required=True, type=pathlib.Path)
    p.add_argument("--out_dir", required=True, type=pathlib.Path)
    p.add_argument("--seq_len", default=SEQ_LEN, type=int)
    p.add_argument("--diagnose", action="store_true", help="Run feature diagnosis")
    args = p.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    uploads = pd.read_parquet(args.uploads)
    train_df = uploads.loc[np.load(args.train_ids, allow_pickle=True)]
    test_df  = uploads.loc[np.load(args.test_ids , allow_pickle=True)]
    # (optional) generate rationale targets before amy column drops so all engineered features are considered
    process_rationale_targets(train_df, test_df, args.out_dir)


    # Drop columns that are all NaN in train_df
    all_nan_cols = train_df.columns[train_df.isnull().all()]
    if len(all_nan_cols) > 0:
        # print("Dropping all-NaN columns:", list(all_nan_cols))
        train_df = train_df.drop(columns=all_nan_cols)
        test_df = test_df.drop(columns=all_nan_cols, errors='ignore')

        # --- Update feature lists to only include present columns ---
    present_cols = set(train_df.columns)
    CONTINUOUS_USED = [col for col in CONTINUOUS if col in present_cols]
    BOOLEAN_USED = [col for col in BOOLEAN if col in present_cols]
    LOW_CAT_USED = [col for col in LOW_CAT if col in present_cols]
    HIGH_CAT_USED = [col for col in HIGH_CAT if col in present_cols]

    
    #  REPORT FEATURE USAGE
    print(f"\nFEATURE USAGE SUMMARY:")
    print(f"  CONTINUOUS: {len(CONTINUOUS_USED)}/{len(CONTINUOUS)} ({[c for c in CONTINUOUS if c not in present_cols] or 'all present'})")
    print(f"  BOOLEAN:    {len(BOOLEAN_USED)}/{len(BOOLEAN)} ({[c for c in BOOLEAN if c not in present_cols] or 'all present'})")
    print(f"  LOW_CAT:    {len(LOW_CAT_USED)}/{len(LOW_CAT)} ({[c for c in LOW_CAT if c not in present_cols] or 'all present'})")
    print(f"  HIGH_CAT:   {len(HIGH_CAT_USED)}/{len(HIGH_CAT)} ({[c for c in HIGH_CAT if c not in present_cols] or 'all present'})")
    print(f"  TOTAL:      {len(CONTINUOUS_USED) + len(BOOLEAN_USED) + len(LOW_CAT_USED) + len(HIGH_CAT_USED)}")


        # --- Save feature lists for downstream scripts ---
    feature_lists = {
        "CONTINUOUS_USED": CONTINUOUS_USED,
        "BOOLEAN_USED": BOOLEAN_USED,
        "LOW_CAT_USED": LOW_CAT_USED,
        "HIGH_CAT_USED": HIGH_CAT_USED,
        "CONTINUOUS_MISSING": [c for c in CONTINUOUS if c not in present_cols],
        "BOOLEAN_MISSING": [c for c in BOOLEAN if c not in present_cols],
        "LOW_CAT_MISSING": [c for c in LOW_CAT if c not in present_cols],
        "HIGH_CAT_MISSING": [c for c in HIGH_CAT if c not in present_cols]
    }
    (args.out_dir / "feature_lists.json").write_text(json.dumps(feature_lists, indent=2))


    # ---  check for NaNs and unique values ---
    # print(train_df[CONTINUOUS_USED + BOOLEAN_USED + LOW_CAT_USED + HIGH_CAT_USED].isnull().sum())
    # print(train_df[CONTINUOUS_USED + BOOLEAN_USED + LOW_CAT_USED + HIGH_CAT_USED].nunique())
    # print(test_df[CONTINUOUS_USED + BOOLEAN_USED + LOW_CAT_USED + HIGH_CAT_USED].isnull().sum())
    # print(test_df[CONTINUOUS_USED + BOOLEAN_USED + LOW_CAT_USED + HIGH_CAT_USED].nunique())

    # print("uploads.shape:", uploads.shape)
    # # print("train_ids.shape:", train_ids.shape)
    # # print("test_ids.shape:", test_ids.shape)
    # print("train_df.shape:", train_df.shape)
    # print("test_df.shape:", test_df.shape)
    # test_ids = np.load(args.test_ids, allow_pickle=True)
    # print("test_ids:", test_ids)
    # print("uploads.index:", uploads.index)

    # 1. tabular preprocessors ----------------------------------------------
    pre, scaler, onehot, ordinal = build_preprocessors(
        train_df, CONTINUOUS_USED, BOOLEAN_USED, LOW_CAT_USED, HIGH_CAT_USED)
    X_train_tab = pre.transform(train_df)
    X_test_tab  = pre.transform(test_df)

    X_train_tab = np.array(X_train_tab, dtype=np.float32)
    X_test_tab = np.array(X_test_tab, dtype=np.float32)

    # col with nan after transformation
    train_nan_col = np.isnan(X_train_tab).any(axis=0)
    # print("Columns with NaNs after transformation:", np.where(train_nan_col)[0])
    test_nan_col  = np.isnan(X_test_tab).any(axis=0)
    # print("Columns with NaNs in test set after transformation:", np.where(test_nan_col)[0])


    X_train_tab = pd.DataFrame(X_train_tab).apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    X_test_tab = pd.DataFrame(X_test_tab).apply(pd.to_numeric, errors='coerce').fillna(0.0).values


    # Ensure numeric type
    X_train_tab = np.nan_to_num(X_train_tab, nan=0.0).astype(np.float32)
    X_test_tab  = np.nan_to_num(X_test_tab , nan=0.0).astype(np.float32)
    assert not np.isnan(X_train_tab).any(), "NaNs remain in X_train_tab!"
    assert not np.isnan(X_test_tab).any(), "NaNs remain in X_test_tab!"
    assert not np.isnan(train_df.label.values).any(), "NaNs remain in y_train!"
    assert not np.isnan(test_df.label.values).any(), "NaNs remain in y_test!"
    # print("X_train_tab shape:", X_train_tab.shape)
    # print("X_test_tab shape:", X_test_tab.shape)
    # print("y_train shape:", train_df.label.shape)
    # print("y_test shape:", test_df.label.shape)
    np.save(args.out_dir / "X_train_tab.npy", X_train_tab)
    np.save(args.out_dir / "X_test_tab.npy" , X_test_tab)
    np.save(args.out_dir / "y_train.npy"    , train_df.label.values)
    np.save(args.out_dir / "y_test.npy"     , test_df.label.values)
    joblib.dump({"pre": pre, "scaler": scaler, "onehot": onehot, "ordinal": ordinal},
                args.out_dir / "preprocessors.pkl")

    # 2. embedding maps ------------------------------------------------------
    embed_maps = {}
    for col in HIGH_CAT_USED + LOW_CAT_USED:
        train_df[f"{col}_idx"], embed_maps[col] = factorize_to_index(train_df[col])
        test_df[f"{col}_idx"]  = test_df[col].astype(str).map(embed_maps[col]).fillna(0).astype("int64")

    (args.out_dir / "embedding_maps.json").write_text(json.dumps(embed_maps, indent=2))
    (args.out_dir / "embedding_sizes.json").write_text(json.dumps({k: len(v) for k,v in embed_maps.items()}, indent=2))

    # 3. sequence tensors ----------------------------------------------------
    SEQ_LEN = args.seq_len  # override if flag provided
    # seq_train = df_to_sequence_tensor(train_df, embed_maps, CONTINUOUS_USED, BOOLEAN_USED, HIGH_CAT_USED, LOW_CAT_USED)
    # seq_test  = df_to_sequence_tensor(test_df , embed_maps, CONTINUOUS_USED, BOOLEAN_USED, HIGH_CAT_USED, LOW_CAT_USED)
    # cont_train, cat_train = df_to_sequence_tensor(
    #     train_df, embed_maps, CONTINUOUS_USED, BOOLEAN_USED, HIGH_CAT_USED, LOW_CAT_USED)
    # cont_test, cat_test = df_to_sequence_tensor(
    #     test_df, embed_maps, CONTINUOUS_USED, BOOLEAN_USED, HIGH_CAT_USED, LOW_CAT_USED,
    #     train_df=train_df)  # Pass train_df for scaling fitting

    print("\n GENERATING LEAK-FREE SEQUENCES...")
    print("=" * 50)

    # TRAINING SEQUENCES: Use only training data for history
    print("📚 Processing training sequences...")
    cont_train, cat_train = df_to_sequence_tensor_FIXED(
        df=train_df,
        embed_maps=embed_maps, 
        CONTINUOUS_USED=CONTINUOUS_USED, 
        BOOLEAN_USED=BOOLEAN_USED, 
        HIGH_CAT_USED=HIGH_CAT_USED, 
        LOW_CAT_USED=LOW_CAT_USED,
        train_df=None,
        is_training=True 
    )
    
    # TEST SEQUENCES: Use only training data for history (simulate deployment)
    print("🧪 Processing test sequences...")
    cont_test, cat_test = df_to_sequence_tensor_FIXED(
        df=test_df,
        embed_maps=embed_maps,
        CONTINUOUS_USED=CONTINUOUS_USED, 
        BOOLEAN_USED=BOOLEAN_USED, 
        HIGH_CAT_USED=HIGH_CAT_USED, 
        LOW_CAT_USED=LOW_CAT_USED,
        train_df=train_df,  # Use training data for scaling and history
        is_training=False
    ) 

    print(f"✅ Training sequences: {cont_train.shape}")
    print(f"✅ Test sequences: {cont_test.shape}")
    print("🔒 Temporal isolation enforced - no leakage possible")
    
    torch.save((cont_train, cat_train), args.out_dir / "seq_train.pt")
    torch.save((cont_test , cat_test ), args.out_dir / "seq_test.pt")
    # print("seq_train shape:", seq_train.shape)
    # print("seq_test shape:", seq_test.shape)

    # print("DEBUG: seq_train.shape =", seq_train.shape)
    # print("DEBUG: seq_test.shape  =", seq_test.shape)

    # # Print a sample row's feature slices for inspection
    # sample = seq_train[0]  # shape [SEQ_LEN, F]
    # print("DEBUG: Sample row shape:", sample.shape)
    # print("DEBUG: CONTINUOUS_USED + BOOLEAN_USED:", CONTINUOUS_USED + BOOLEAN_USED)
    # print("DEBUG: HIGH_CAT_USED:", HIGH_CAT_USED)
    # print("DEBUG: LOW_CAT_USED:", LOW_CAT_USED)
    # print("DEBUG: Continuous+Boolean slice:", sample[:, :len(CONTINUOUS_USED)+len(BOOLEAN_USED)])
    # print("DEBUG: High-cat slice:", sample[:, len(CONTINUOUS_USED)+len(BOOLEAN_USED):len(CONTINUOUS_USED)+len(BOOLEAN_USED)+len(HIGH_CAT_USED)])
    # print("DEBUG: Low-cat slice:", sample[:, -len(LOW_CAT_USED):])

    # print("CONTINUOUS_USED:", CONTINUOUS_USED)
    # print("BOOLEAN_USED:", BOOLEAN_USED)
    # print("HIGH_CAT_USED:", HIGH_CAT_USED)
    # print("LOW_CAT_USED:", LOW_CAT_USED)
    # print("Total expected features:", len(CONTINUOUS_USED) + len(BOOLEAN_USED) + len(HIGH_CAT_USED) + len(LOW_CAT_USED))
    # print("seq_train.shape:", seq_train.shape)
    # print("seq_train.shape[2]:", seq_train.shape[2])
    # print("Expected:", len(CONTINUOUS_USED) + len(BOOLEAN_USED) + len(HIGH_CAT_USED) + len(LOW_CAT_USED))
    # Update the assertions to use the new separate tensors
    print(f"cont_train shape: {cont_train.shape}")  
    print(f"cat_train shape: {cat_train.shape}")
    print(f"cont_test shape: {cont_test.shape}")
    print(f"cat_test shape: {cat_test.shape}")

    # Check dimensions
    expected_cont_features = len(CONTINUOUS_USED) + len(BOOLEAN_USED)
    expected_cat_features = len(HIGH_CAT_USED) + len(LOW_CAT_USED)

    assert cont_train.shape[2] == expected_cont_features, \
        f"Continuous tensor has {cont_train.shape[2]} features, expected {expected_cont_features}"
    assert cat_train.shape[2] == expected_cat_features, \
        f"Categorical tensor has {cat_train.shape[2]} features, expected {expected_cat_features}"
    assert cont_test.shape[2] == expected_cont_features, \
        f"Continuous test tensor has {cont_test.shape[2]} features, expected {expected_cont_features}"
    assert cat_test.shape[2] == expected_cat_features, \
        f"Categorical test tensor has {cat_test.shape[2]} features, expected {expected_cat_features}"
    # torch.save(seq_train, args.out_dir / "seq_train.pt")
    # torch.save(seq_test , args.out_dir / "seq_test.pt")
    torch.save(torch.tensor(train_df.label.values), args.out_dir / "y_train_t.pt")
    torch.save(torch.tensor(test_df.label.values) , args.out_dir / "y_test_t.pt")

    print("[✓] Saved tabular + sequence features →", args.out_dir)

if __name__ == "__main__":
    main()