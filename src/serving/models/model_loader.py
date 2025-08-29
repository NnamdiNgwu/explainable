"""loads models and creates explainers"""
import json
import joblib
import torch
import pathlib
import shap
import logging
from sklearn.ensemble import RandomForestClassifier

from models.to_be_submitted.cybersecurity_transformer import (
    build_cybersecurity_transformer_from_maps
)
from ..utils.feature_mapping import TransformerExplanationMapper
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline

def _extract_rf_estimator(obj):
    """Return (bare_rf, rf_pipeline_or_None)."""
    from sklearn.ensemble import RandomForestClassifier
    if isinstance(obj, RandomForestClassifier):
        return obj, None
    if isinstance(obj, (ImbPipeline, SkPipeline)):
        last = obj.steps[-1][1] if obj.steps else None
        if isinstance(last, RandomForestClassifier):
            return last, obj
    return None, None

class ModelLoader:
    @staticmethod
    def _create_explainers(rf_model, rf_estimator,
                            transformer_model, embed_maps,
                            feature_lists, feature_names, 
                            preprocessor=None, rf_pipeline=None):
        explainers = {}
        # RF TreeSHAP on the estimator (post-preprocess space)
        logging.info("Creating SHAP TreeExplainer for RF estimator…")
        # RF SHAP explainer on the *bare* RF
        explainers['rf_explainer'] = shap.TreeExplainer(rf_estimator)

        # Try to detect a full pipeline if rf_model is a pipeline
        _, rf_pipeline = _extract_rf_estimator(rf_model)

        explainers['transformer_explainer'] = ModelLoader._create_explanation_mapper(
            rf_explainer=explainers['rf_explainer'],
            transformer_model=transformer_model,
            embed_maps=embed_maps,
            feature_lists=feature_lists,
            features_order=feature_names,
            preprocessor=preprocessor,         # <-- ensure passed
            rf_estimator=rf_estimator,
            rf_pipeline=rf_pipeline,           # <-- ensure passed
        )
        return explainers
    

    @staticmethod
    def _create_explanation_mapper(
        rf_explainer,
        transformer_model,
        embed_maps,
        feature_lists,
        features_order,
        preprocessor=None,
        rf_estimator=None,
        rf_pipeline=None,
    ):
        return TransformerExplanationMapper(
            rf_explainer=rf_explainer,
            transformer_model=transformer_model,
            feature_lists=feature_lists,
            features_order=features_order,
            embed_maps=embed_maps,
            preprocessor=preprocessor,     # <-- set on mapper
            rf_pipeline=rf_pipeline,       # <-- set on mapper
            rf_estimator=rf_estimator,     # <-- set on mapper
        )


    @staticmethod
    def load_all_models(model_dir):
        model_dir = pathlib.Path(model_dir)
        logging.info(f"Loading models from: {model_dir}")

        cascade_config = json.loads((model_dir / "cascade_config.json").read_text())
        embed_maps     = json.loads((model_dir / "embedding_maps.json").read_text())
        feature_lists  = json.loads((model_dir / "feature_lists.json").read_text())

        CONT = feature_lists["CONTINUOUS_USED"]
        BOOL = feature_lists["BOOLEAN_USED"]
        HI   = feature_lists["HIGH_CAT_USED"]
        LO   = feature_lists["LOW_CAT_USED"]
        FEATURES_ORDER = CONT + BOOL + HI + LO

        # --- build feature_defaults (used by handle_missing_features) ---
        feature_defaults = {}
        for f in CONT:
            feature_defaults[f] = 0.0
        for f in BOOL:
            feature_defaults[f] = False
        for f in (HI + LO):
            feature_defaults[f] = "unknown"

        # RF objects
        rf_model = joblib.load(cascade_config["rf_model_path"])
        preprocessors = joblib.load(model_dir / "preprocessors.pkl")
        pre = preprocessors['pre']

        # Extract bare RF and optional pipeline
        rf_estimator, _ = _extract_rf_estimator(rf_model)
        if rf_estimator is None:
            raise ValueError("RandomForestClassifier not found for SHAP.")
        
        # Derive RF feature names after preprocessing (if available)
        # try:
        #     rf_feature_names = list(preprocessors['pre'].get_feature_names_out(FEATURES_ORDER))
        # except Exception:
        #     # Fallback if the preprocessor doesn't expose names
        #     n = getattr(rf_estimator, "n_features_in_", None) or 57
        #     rf_feature_names = [f"feature_{i}" for i in range(int(n))]
        # logging.info(f"RF feature names count: {len(rf_feature_names)}")

        # Build transformer
        cont_dim = len(feature_lists["CONTINUOUS_USED"]) + len(feature_lists["BOOLEAN_USED"])
        transformer_model = build_cybersecurity_transformer_from_maps(embed_maps, continuous_dim=cont_dim, num_classes=3)
        state_dict = torch.load(cascade_config["transformer_model_path"], map_location="cpu")
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        transformer_model.load_state_dict(state_dict)
        transformer_model.eval()

       # Create explainers (pass preprocessor explicitly)
        explainers = ModelLoader._create_explainers(
            rf_model, rf_estimator, 
            transformer_model, embed_maps,
            feature_lists, FEATURES_ORDER,
            preprocessor=preprocessors['pre'],
            rf_pipeline=rf_model  # <-- if rf_model is a Pipeline, this lets TXM call predict_proba(X_df)
        )

        # Feature defaults (used by handle_missing_features)
        feature_defaults = {f: 0.0 for f in feature_lists["CONTINUOUS_USED"]}
        feature_defaults.update({f: False for f in feature_lists["BOOLEAN_USED"]})
        feature_defaults.update({f: "unknown" for f in (feature_lists["HIGH_CAT_USED"] + feature_lists["LOW_CAT_USED"])})

        return {
            'rf': rf_model,                           # pipeline OR estimator; we’ll pass rf_estimator separately
            'rf_estimator': rf_estimator,             # sklearn RandomForestClassifier trained on transformed matrix
            'transformer': transformer_model,
            'preprocessor': preprocessors['pre'],     # ColumnTransformer
            'embed_maps': embed_maps,
            'feature_lists': feature_lists,
            'feature_names': FEATURES_ORDER,      # raw input order (encoders.py uses this)
            # 'rf_feature_names': rf_feature_names,    # expanded, matches SHAP lenght
            'feature_defaults': feature_defaults,
            'tau': float(cascade_config['tau']),
            'tau2': float(cascade_config['tau2']),
            'device': "cuda:0" if torch.cuda.is_available() else "cpu",
            'cascade_config': cascade_config,
            **explainers,
        }

       