# """
# Enhanced ground truth labeling that combines multiple risk signals.
# """
import pandas as pd
import numpy as np
from typing import Dict, Any, Set, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from src.utils.correlation_utils import prepare_corr_frame as _prepare_corr_frame

logger = logging.getLogger(__name__)

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    num = num.dropna(axis=1, how='all')
    std = num.std(ddof=0)
    num = num.loc[:, std > 0]
    return num
class GroundTruthLabeler:
    """Multi-signal risk labeling using ground truth and anomaly detection."""
    
    def __init__(self, decoy_file_path: str = None):
        self.decoy_file_path = decoy_file_path
        self.decoy_pcs = set()
        self.anomaly_detectors = {}
        self.population_stats = {}
        self.quantile_thresholds = None  # store (q_low, q_high) from training
        self.signal_correlations: Dict[str, float] = {}
        
    def load_decoy_data(self) -> Set[str]:
        """Load decoy file interactions as definitive ground truth."""
        if not self.decoy_file_path:
            return set()
            
        try:
            decoy_df = pd.read_csv(self.decoy_file_path)
            self.decoy_pcs = set(decoy_df['pc'].unique())
            logger.info(f"Loaded {len(self.decoy_pcs)} decoy PCs from ground truth")
            return self.decoy_pcs
        except Exception as e:
            logger.warning(f"Could not load decoy data: {e}")
            return set()
    

    def identify_decoy_interactions(self, uploads_df: pd.DataFrame) -> pd.Series:
        if 'pc' not in uploads_df.columns:
            return pd.Series(0.0, index=uploads_df.index)
        # Ensure decoy set loaded
        if not self.decoy_pcs:
            self.load_decoy_data()
        df = uploads_df[['pc','date']].copy()
        df = df.sort_values(['pc','date'])
        # Mark decoy events
        df['is_decoy'] = df['pc'].isin(self.decoy_pcs)
        # Last decoy date per pc forward-filled
        df['last_decoy_date'] = df['date'].where(df['is_decoy'])
        df['last_decoy_date'] = df.groupby('pc')['last_decoy_date'].ffill()
        # Days since last decoy
        days_since = (df['date'] - df['last_decoy_date']).dt.days
        # Where no prior decoy -> inf
        days_since = days_since.fillna(np.inf)
        # Scoring
        score = np.select(
            [
                df['is_decoy'],
                days_since <= 1,
                days_since <= 7,
                days_since <= 30
            ],
            [1.0, 0.8, 0.6, 0.4],
            default=0.0
        )
        out = pd.Series(score, index=df.index)
        # Reindex to original order
        return out.reindex(uploads_df.index).fillna(0.0)
    
    def detect_behavioral_anomalies(self, uploads_df: pd.DataFrame, is_training: bool = True) -> Dict[str, pd.Series]:
        """Use unsupervised methods to detect novel behavioral patterns."""
        
        # Create user-level behavioral features
        user_features = uploads_df.groupby('user').agg({
            'megabytes_sent': ['mean', 'std', 'max', 'sum'],
            'hour': ['mean', 'std', 'min', 'max'],
            'destination_domain': 'nunique',
            'after_hours': 'mean',
            'is_weekend': 'mean',
            'channel': lambda x: x.value_counts().iloc[0] if len(x) > 0 else 'UNKNOWN',
            'post_burst': ['mean', 'max'],
            'destination_entropy': ['mean', 'std'],
            'first_time_destination': 'mean'
        }).fillna(0)
        
        # Flatten column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Select numeric features for anomaly detection
        numeric_features = user_features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            uploads_df['user_anomaly_score'] = 0.0
            uploads_df['cluster_anomaly_score'] = 0.0
            return {'user_anomaly_score': uploads_df['user_anomaly_score'],
                    'cluster_anomaly_score': uploads_df['cluster_anomaly_score']}

        if is_training or 'behavior_iso' not in self.anomaly_detectors:
            iso = IsolationForest(contamination=0.1, random_state=42)
        # self.anomaly_detectors['behavior_iso'] = iso:
        #     # Fit anomaly detectors on training data
        #     self.anomaly_detectors['isolation_forest'] = IsolationForest(
        #         contamination=0.1, random_state=42
        #     ).fit(numeric_features)
            iso.fit(numeric_features)
            self.anomaly_detectors['behavior_iso'] = iso

        iso = self.anomaly_detectors['behavior_iso']
        # iso_pred = iso.predict(numeric_features)            # -1 outlier, 1 inlier
        # isolation_flag = (iso_pred == -1).astype(float)     # 1 = anomaly
        isolation_flag = (iso.predict(numeric_features) == -1).astype(float)
        # if 'dbscan' in self.anomaly_detectors:
        #     cluster_labels = self.anomaly_detectors['dbscan'].fit_predict(numeric_features) if is_training else self.anomaly_detectors['dbscan'].fit_predict(numeric_features)
        #     cluster_flag = (cluster_labels == -1).astype(float)
        # else:
        #     cluster_flag = np.zeros(len(numeric_features), dtype=float)

        user_isolation_scores = pd.Series(isolation_flag, index=numeric_features.index)
        # user_cluster_scores   = pd.Series(cluster_flag,   index=numeric_features.index)
        uploads_df['user_anomaly_score']    = uploads_df['user'].map(user_isolation_scores).fillna(0.0)
        uploads_df['cluster_anomaly_score'] = 0.0 #uploads_df['user'].map(user_cluster_scores).fillna(0.0)
        return {
            'user_anomaly_score': uploads_df['user_anomaly_score'],
            'cluster_anomaly_score': uploads_df['cluster_anomaly_score'],
        }
            

    def detect_temporal_anomalies(self, uploads_df: pd.DataFrame) -> pd.Series:
        hour_counts = uploads_df['hour'].value_counts()
        rare_hours = hour_counts[hour_counts < hour_counts.quantile(0.1)].index
        h = uploads_df['hour']
        is_rare = h.isin(rare_hours).astype(float) * 0.3
        deep_night = ((h >= 2) & (h <= 5)).astype(float) * 0.4
        weekend = uploads_df['is_weekend'].astype(float) * 0.2
        after_hours = uploads_df['after_hours'].astype(float) * 0.1
        score = (is_rare + deep_night + weekend + after_hours).clip(0, 1.0)
        return score
    # def detect_temporal_anomalies(self, uploads_df: pd.DataFrame) -> pd.Series:
    #     """Detect unusual temporal patterns."""
        
    #     # Hour-based anomalies
    #     hour_counts = uploads_df['hour'].value_counts()
    #     rare_hours = hour_counts[hour_counts < hour_counts.quantile(0.1)].index
        
    #     # Weekend anomalies
    #     weekend_uploads = uploads_df[uploads_df['is_weekend'] == True]
    #     weekend_anomaly = len(weekend_uploads) / len(uploads_df) if len(uploads_df) > 0 else 0
        
    #     # Calculate temporal anomaly scores
    #     temporal_scores = []
    #     for _, row in uploads_df.iterrows():
    #         score = 0.0
            
    #         # Rare hour penalty
    #         if row['hour'] in rare_hours:
    #             score += 0.3
            
    #         # Deep night penalty (2-5 AM)
    #         if 2 <= row['hour'] <= 5:
    #             score += 0.4
            
    #         # Weekend penalty
    #         if row['is_weekend']:
    #             score += 0.2
            
    #         # After hours penalty
    #         if row['after_hours']:
    #             score += 0.1
                
    #         temporal_scores.append(min(score, 1.0))
        
    #     return pd.Series(temporal_scores, index=uploads_df.index)
    
    def detect_volume_anomalies(self, uploads_df: pd.DataFrame) -> pd.Series:
        """Detect unusual upload volumes."""
        
        # Calculate volume percentiles
        volume_data = uploads_df['megabytes_sent'].fillna(0)
        pct = (volume_data.rank(pct=True))
        # smooth continuous volume component (0-0.5 weight)
        cont_component = pct.clip(upper=1.0) * 0.5

        # Isolation Forest for volume anomalies
        if len(volume_data) > 10:
            volume_iso = IsolationForest(contamination=0.05, random_state=42)
            volume_anomalies = volume_iso.fit_predict(volume_data.values.reshape(-1, 1))
            volume_scores = (volume_anomalies == -1).astype(float)
        else:
            volume_scores = np.zeros(len(volume_data))
        
        # Large file penalty
        large_file_threshold = volume_data.quantile(0.95)
        large_file_scores = (volume_data > large_file_threshold).astype(float) * 0.5
        
        # Combine scores
        combined_scores = np.maximum(volume_scores, large_file_scores)
        # blend with continuous (take max keeps tail emphasis)
        combined_scores = np.maximum(combined_scores, cont_component)
        return pd.Series(combined_scores, index=uploads_df.index)

    def detect_network_anomalies(self, uploads_df: pd.DataFrame) -> pd.Series:
        dom = uploads_df['destination_domain'].astype(str).str.lower()
        # comp_pat = r'(lockheedmartin\.com|northropgrumman\.com|boeing\.com|raytheon\.com|generaldynamics\.com)'
        # competitor = dom.str.contains(comp_pat, regex=True).astype(float) * 0.8
        comp_pat = r'(?:lockheedmartin\.com|northropgrumman\.com|boeing\.com|raytheon\.com|generaldynamics\.com)'
        competitor = dom.str.contains(comp_pat, regex=True).astype(float) * 0.8
        first_time = uploads_df.get('first_time_destination', 0).astype(float) * 0.3
        high_entropy = (uploads_df.get('destination_entropy', 0) > 3.0).astype(float) * 0.2
        score = (competitor + first_time + high_entropy).clip(0, 1.0)
        return score
    

    # def create_composite_labels(self, uploads_df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.Series, pd.Series]:
    #     """
    #     Build orthogonal composite risk score:
    #       Signals (each 0–1):
    #         decoy_scores
    #         anomaly_block (mean of user & cluster anomaly flags)
    #         temporal_scores
    #         volume_scores
    #         network_scores
    #       Weights sum to 1.0: 0.25, 0.20, 0.20, 0.20, 0.15
    #     Train: compute 0.80 / 0.95 quantile thresholds and cache.
    #     Test: reuse cached thresholds (no leakage).
    #     """
    #     logger.info("Creating composite risk labels (orthogonal weighting)...")
    #     self.load_decoy_data()

    #     decoy_scores   = self.identify_decoy_interactions(uploads_df)
    #     anomaly_scores = self.detect_behavioral_anomalies(uploads_df, is_training)
    #     temporal_scores = self.detect_temporal_anomalies(uploads_df)
    #     volume_scores   = self.detect_volume_anomalies(uploads_df)
    #     network_scores  = self.detect_network_anomalies(uploads_df)

    #     anomaly_block = 0.5 * anomaly_scores['user_anomaly_score'] + 0.5 * anomaly_scores['cluster_anomaly_score']

    #     composite_score = (
    #         0.25 * decoy_scores +
    #         0.20 * anomaly_block +
    #         0.20 * temporal_scores +
    #         0.20 * volume_scores +
    #         0.15 * network_scores
    #     )

    #     # Sanity
    #     composite_score = composite_score.clip(0, 1)
    #     # assert not composite_score.isnull().any(), "Composite score NaNs"
    #     # assert ((composite_score >= 0) & (composite_score <= 1)).all(), "Composite score out of bounds"

    #     if composite_score.nunique() < 3:
    #         # Force spread to avoid identical q80=q95
    #         noise = (np.random.RandomState(42).rand(len(composite_score)) * 1e-6)
    #         composite_score = (composite_score + noise).clip(0,1)

    #     if is_training or self.quantile_thresholds is None:
    #         q80, q95 = composite_score.quantile([0.80, 0.95]).values
    #         if q80 >= q95:  # collapse safeguard
    #             q95 = min(1.0, q80 + 1e-6)
    #         self.quantile_thresholds = (float(q80), float(q95))
    #         logger.info(f"Training thresholds q80={q80:.4f} q95={q95:.4f}")
    #     else:
    #         q80, q95 = self.quantile_thresholds
    #         logger.info(f"Using stored thresholds q80={q80:.4f} q95={q95:.4f}")

    #     risk_labels = pd.cut(
    #         composite_score,
    #         bins=[-np.inf, q80, q95, np.inf],
    #         labels=[0, 1, 2],
    #         include_lowest=True
    #     ).astype(int)

    #     dist = risk_labels.value_counts().sort_index().to_dict()
    #     logger.info(f"Risk label distribution: {dist}")
    #     logger.info(f"Composite stats: min={composite_score.min():.4f} max={composite_score.max():.4f} mean={composite_score.mean():.4f}")
    #     uploads_df['ml_risk_score'] = composite_score
    #     return risk_labels, composite_score


    def create_composite_labels(self, uploads_df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.Series, pd.Series]:
        """
        Build orthogonal composite risk score:
          Signals (each 0–1):
            decoy_scores
            anomaly_block (mean of user & cluster anomaly flags)
            temporal_scores
            volume_scores
            network_scores
          Weights sum to 1.0: 0.25, 0.20, 0.20, 0.20, 0.15
        Train: compute 0.80 / 0.95 quantile thresholds and cache.
        Test: reuse cached thresholds (no leakage).
        """
        logger.info("Creating composite risk labels (orthogonal weighting)...")
        self.load_decoy_data()

        decoy_scores   = self.identify_decoy_interactions(uploads_df)
        anomaly_scores = self.detect_behavioral_anomalies(uploads_df, is_training)
        temporal_scores = self.detect_temporal_anomalies(uploads_df)
        volume_scores   = self.detect_volume_anomalies(uploads_df)
        network_scores  = self.detect_network_anomalies(uploads_df)

        anomaly_block = 0.5 * anomaly_scores['user_anomaly_score'] + 0.5 * anomaly_scores['cluster_anomaly_score']

        composite_score = (
            0.25 * decoy_scores +
            0.20 * anomaly_block +
            0.20 * temporal_scores +
            0.20 * volume_scores +
            0.15 * network_scores
        )

        composite_score = composite_score.clip(0, 1)

        if composite_score.nunique() < 3:
            noise = (np.random.RandomState(42).rand(len(composite_score)) * 1e-6)
            composite_score = (composite_score + noise).clip(0,1)

        # --- Signal correlation (for diagnostics) ---
        try:
            signals_df = pd.DataFrame({
                'decoy_score': decoy_scores,
                'anomaly_block': anomaly_block,
                'temporal_score': temporal_scores,
                'volume_score': volume_scores,
                'network_score': network_scores,
                'composite': composite_score
            })
            # Prepare (drops zero-variance automatically)
            prep = _prepare_corr_frame(signals_df)
            if 'composite' in prep.columns and prep['composite'].std(ddof=0) > 0:
                self.signal_correlations = {
                    c: float(prep[c].corr(prep['composite']))
                    for c in prep.columns if c != 'composite' and prep[c].std(ddof=0) > 0
                }
                logger.debug(f"Signal correlations: {self.signal_correlations}")
        except Exception as e:
            logger.warning(f"Could not compute signal correlations: {e}")

        if is_training or self.quantile_thresholds is None:
            q80, q95 = composite_score.quantile([0.80, 0.95]).values
            if q80 >= q95:
                q95 = min(1.0, q80 + 1e-6)
            self.quantile_thresholds = (float(q80), float(q95))
            logger.info(f"Training thresholds q80={q80:.4f} q95={q95:.4f}")
        else:
            q80, q95 = self.quantile_thresholds
            logger.info(f"Using stored thresholds q80={q80:.4f} q95={q95:.4f}")

        risk_labels = pd.cut(
            composite_score,
            bins=[-np.inf, q80, q95, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(int)

        dist = risk_labels.value_counts().sort_index().to_dict()
        logger.info(f"Risk label distribution: {dist}")
        logger.info(f"Composite stats: min={composite_score.min():.4f} max={composite_score.max():.4f} mean={composite_score.mean():.4f}")
        uploads_df['ml_risk_score'] = composite_score
        return risk_labels, composite_score
    

    def save_thresholds(self, path):
        if self.quantile_thresholds:
            import json
            with open(path,'w') as f: json.dump({'q80': self.quantile_thresholds[0],'q95': self.quantile_thresholds[1]}, f)

    def load_thresholds(self, path):
        import json, os
        if os.path.exists(path):
            with open(path) as f: d=json.load(f)
            self.quantile_thresholds=(d['q80'], d['q95'])

    
    # def create_composite_labels(self, uploads_df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.Series, pd.Series]:
    #     """Create comprehensive risk labels from multiple signals."""
        
    #     logger.info("Creating composite risk labels from multiple signals...")
        
    #     # Load decoy ground truth
    #     self.load_decoy_data()
        
    #     # Get all risk signals
    #     decoy_scores = self.identify_decoy_interactions(uploads_df)
    #     anomaly_scores = self.detect_behavioral_anomalies(uploads_df, is_training)
    #     temporal_scores = self.detect_temporal_anomalies(uploads_df)
    #     volume_scores = self.detect_volume_anomalies(uploads_df)
    #     network_scores = self.detect_network_anomalies(uploads_df)
        
    #     # Weighted composite score
    #     composite_score = (
    #         0.15 * decoy_scores +  # Highest weight to ground truth
    #         0.1 * anomaly_scores['user_anomaly_score'] +
    #         0.1 * anomaly_scores['cluster_anomaly_score'] +
    #         0.2 * temporal_scores +
    #         0.2 * volume_scores +
    #         0.1 * network_scores +
    #         0.15 * anomaly_scores['composite_anomaly_score'] 
    #     )
    #     #Check for NaNs or out-of-bounds values
    #     assert not composite_score.isnull().any(), "Composite score contains NaN values!"
    #     assert ((composite_score >= 0) & (composite_score <= 1)).all(), "Composite score out of expected [0,1] range!"

    #     # Print distribution for debugging
    #     logger.info(f"Composite score stats: min={composite_score.min()}, max={composite_score.max()}, mean={composite_score.mean()}")
    #     logger.info(f"Composite score quantiles: {composite_score.quantile([0.7, 0.9]).to_dict()}")

    #     # Print a sample of composite scores
    #     logger.info(f"Sample composite scores: {composite_score.sample(10, random_state=42).tolist()}")

    #     # Create risk labels with multiple thresholds
    #     # risk_labels = pd.cut(
    #     #     composite_score, 
    #     #     bins=[-np.inf, 0.2, 0.3, 0.6, np.inf],
    #     #     labels=[0, 1, 2, 3],  # low, medium, high, critical
    #     #     include_lowest=True
    #     # ).astype(int)

    #     quantiles = composite_score.quantile([0.8, 0.95]).values # 80% minor, 15% major, 5% critical
    #     risk_labels = pd.cut(
    #         composite_score,
    #         bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
    #         labels=[0, 1, 2], # low, medium, High
    #         include_lowest=True
    #     ).astype(int)

    #     # Log distribution
    #     label_dist = risk_labels.value_counts().sort_index()
    #     logger.info(f"Risk label distribution: {label_dist.to_dict()}")
        
    #     return risk_labels, composite_score


# class GroundTruthLabeler:
#     """Clean, simple ground truth labeling using decoy interactions."""
    
#     def __init__(self, decoy_file_path: str = None):
#         self.decoy_file_path = decoy_file_path
#         self.decoy_pcs = set()
        
#     def load_decoy_data(self) -> Set[str]:
#         """Load decoy PCs from file."""
#         if not self.decoy_file_path:
#             return set()
            
#         try:
#             decoy_df = pd.read_csv(self.decoy_file_path)
#             self.decoy_pcs = set(decoy_df['pc'].unique())
#             logger.info(f"Loaded {len(self.decoy_pcs)} decoy PCs")
#             return self.decoy_pcs
#         except Exception as e:
#             logger.warning(f"Could not load decoy data: {e}")
#             return set()
    
#     def identify_decoy_interactions(self, uploads_df: pd.DataFrame) -> pd.Series:
#         """Simple binary decoy detection."""
#         self.load_decoy_data()
        
#         if 'pc' not in uploads_df.columns:
#             return pd.Series(0.0, index=uploads_df.index)
            
#         # Binary: 1 if decoy PC, 0 otherwise
#         decoy_scores = uploads_df['pc'].isin(self.decoy_pcs).astype(float)
#         return decoy_scores
    
#     def create_composite_labels(self, uploads_df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.Series, pd.Series]:
#         """Create simple binary labels: decoy vs normal."""
        
#         logger.info("Creating binary decoy-based labels...")
        
#         # Get decoy interactions (ground truth)
#         decoy_scores = self.identify_decoy_interactions(uploads_df)
        
#         # Binary classification: decoy interaction = 1, normal = 0
#         risk_labels = (decoy_scores > 0).astype(int)
        
#         # Log distribution
#         label_dist = risk_labels.value_counts().sort_index()
#         logger.info(f"Label distribution: {label_dist.to_dict()}")
        
#         positive_rate = label_dist.get(1, 0) / len(risk_labels) * 100
#         logger.info(f"Positive class rate: {positive_rate:.2f}%")
        
#         return risk_labels, decoy_scores