import json
import joblib
import torch
import pathlib
import shap
import logging
import numpy as np
from src.models.cybersecurity_transformer import build_cybersecurity_transformer_from_maps
from sklearn.ensemble import RandomForestClassifier
from captum.attr import IntegratedGradients, GradientShap, DeepLift
from ..models.encoders import dict_to_transformer_tensors, encode_sequence_flat_for_shap
from ..utils.transformer_wrapper import TransformerShapWrapper
from ..utils.feature_mapping import TransformerExplanationMapper
from ..config.settings import Config


class ModelLoader:
    @staticmethod
    def load_all_models(model_dir):
        """Load all models and components with comprehensive explainers."""
        import gc
        gc.collect()  # Clean up memory before loading

        model_dir = pathlib.Path(model_dir)
        logging.info(f"Loading models from: {model_dir}")

        try:
            # Load configuration files
            cascade_config = json.loads((model_dir / "cascade_config.json").read_text())
            embed_maps = json.loads((model_dir / "embedding_maps.json").read_text())
            feature_lists = json.loads((model_dir / "feature_lists.json").read_text())
            
            # Extract feature order
            CONTINUOUS_USED = feature_lists["CONTINUOUS_USED"]
            BOOLEAN_USED = feature_lists["BOOLEAN_USED"]
            HIGH_CAT_USED = feature_lists["HIGH_CAT_USED"] 
            LOW_CAT_USED = feature_lists["LOW_CAT_USED"]
            FEATURES_ORDER = CONTINUOUS_USED + BOOLEAN_USED + HIGH_CAT_USED + LOW_CAT_USED
            
            # Generate feature defaults dynamically
            feature_defaults = {}
            for feature in CONTINUOUS_USED:
                feature_defaults[feature] = 0.0
            for feature in BOOLEAN_USED:
                feature_defaults[feature] = False
            for feature in HIGH_CAT_USED + LOW_CAT_USED:
                feature_defaults[feature] = "unknown"

            # Load models
            rf_model = joblib.load(cascade_config["rf_model_path"])
            preprocessors = joblib.load(model_dir / "preprocessors.pkl")
            
            # Extract RF estimator for feature importance
            rf_estimator = ModelLoader._extract_rf_estimator(rf_model)

            if rf_estimator is None:
                raise ValueError("RandomForestClassifier not found in pipeline for SHAP.")
            print("rf_estimator type:", type(rf_estimator))
            
            # Load transformer
            cont_dim = len(CONTINUOUS_USED) + len(BOOLEAN_USED)
            transformer_model = build_cybersecurity_transformer_from_maps(
                embed_maps, continuous_dim=cont_dim, num_classes=2
            )
            
            # Load transformer state dict
            state_dict = torch.load(cascade_config["transformer_model_path"], map_location="cpu")
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
                    new_state_dict[new_key] = value
                state_dict = new_state_dict

            transformer_model.load_state_dict(state_dict)
            transformer_model.eval()
            
            # Create explainers
            explainers = ModelLoader._create_explainers(
                rf_model, rf_estimator, transformer_model, 
                embed_maps, feature_lists, FEATURES_ORDER
            )
            
            logging.info("✅ All models and explainers loaded successfully")

            return {
                'rf': rf_model,
                'rf_estimator': rf_estimator,
                'transformer': transformer_model,
                'preprocessor': preprocessors['pre'],
                'embed_maps': embed_maps,
                'feature_lists': feature_lists,
                'feature_names': FEATURES_ORDER,
                'feature_defaults': feature_defaults, 
                'tau': float(cascade_config['tau']),
                'tau2': float(cascade_config['tau2']),
                'device': "cuda:0" if torch.cuda.is_available() else "cpu",
                'cascade_config': cascade_config,
                **explainers  # Add all explainers
            }
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    @staticmethod
    def _extract_rf_estimator(pipeline):
        """Extract RandomForest from pipeline."""
        
        if isinstance(pipeline, RandomForestClassifier):
            return pipeline
        if hasattr(pipeline, "named_steps"):
            for step in pipeline.named_steps.values():
                found = ModelLoader._extract_rf_estimator(step)
                if found is not None:
                    return found
        return None
        

    @staticmethod
    def _create_explainers(rf_model, rf_estimator, transformer_model, embed_maps, feature_lists, feature_names):
        """Create all explainers for both models."""
        explainers = {}

        # Debug RF components
        logging.info(f"RF Model type: {type(rf_model)}")
        logging.info(f"RF Estimator type: {type(rf_estimator)}")
        logging.info(f"RF Estimator has feature_importances_: {hasattr(rf_estimator, 'feature_importances_')}")
    
        
        # debug 
        try:
            # Check if rf_estimator is valid
            if rf_estimator is None:
                raise ValueError("RF estimator is None, cannot create SHAP explainer")
            
            # Try creating SHAP explainer for RF
            logging.info("Creating SHAP TreeExplainer...")
            explainers['rf_explainer'] = shap.TreeExplainer(rf_estimator) #, check_additivity=False)
            
            # Debug SHAP structure
            expected_val = explainers['rf_explainer'].expected_value
            print(f"SHAP expected_value type: {type(expected_val)}")
            logging.info(f"SHAP expected_value type: {type(expected_val)}")
            logging.info(f"SHAP expected_value shape: {expected_val.shape if hasattr(expected_val, 'shape') else 'no shape'}")
            logging.info(f"SHAP expected_value content: {expected_val}")
                
            logging.info(" RF SHAP explainer created")
                
        except Exception as e:
            logging.error(f"Failed to create RF SHAP explainer: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"RF estimator details: {rf_estimator}")
      
        try:
            # 2. Transformer SHAP and Captum Explainer
            # explainers['transformer_explainer'] = ModelLoader._create_transformer_shap_explainer(
            #     transformer_model, embed_maps, feature_lists, feature_names
            # )
            # logging.info("✅ Transformer SHAP and Captum explainer created")
            #  logging.info("✅ RF SHAP explainer created")
            
            # Create explanation mapper (much simpler!)
            explainers['transformer_explainer'] = ModelLoader._create_explanation_mapper(
                explainers['rf_explainer'], transformer_model, embed_maps, feature_lists, feature_names
            )
            logging.info("✅ Transformer explanation mapper created")
        
        except Exception as e:
            logging.warning(f"Failed to create Transformer explainer: {e}")
            explainers['rf_explainer'] = None
            explainers['transformer_explainer'] = None
        
        return explainers
    
    @staticmethod
    def _create_explanation_mapper(rf_explainer, transformer_model, embed_maps, feature_lists, features_order):
        """Create a mapper that translates RF explanations to Transformer explanations."""
        return TransformerExplanationMapper(
            rf_explainer=rf_explainer,
            transformer_model=transformer_model,
            feature_lists=feature_lists,
            features_order=features_order,
            embed_maps=embed_maps,
        )
    
    @staticmethod
    def _create_transformer_shap_explainer(transformer_model, embed_maps, feature_lists, features_order):
        """Create explanation mapper instead of complex SHAP setup."""
        return {
            'transformer_explainer': None,  # Will be set by mapper
            'transformer_captum': None,
            'captum_gradshap': None, 
            'captum_deeplift': None
        }
    
    # @staticmethod
    # def _create_transformer_shap_explainer(transformer_model, embed_maps, feature_lists, features_order):
    #     """Create SHAP and Captum explainer for transformer."""
    #     explainer = {}
    #     device = 'cpu'  # Use CPU for SHAP and Captum

    #     try:

    #         # Create wrapper for SHAP compatibility
    #         transformer_predict_wrapper = TransformerShapWrapper(
    #             transformer_model=transformer_model,
    #             cont_dim=len(feature_lists['CONTINUOUS_USED']) + len(feature_lists['BOOLEAN_USED']),
    #             high_cat_dim=len(feature_lists['HIGH_CAT_USED']),
    #             low_cat_dim=len(feature_lists['LOW_CAT_USED']),
    #             feature_lists=feature_lists,
    #             embed_maps=embed_maps,
    #         )
    #         # background_events = Config.BACKGROUND_EVENTS

    #         background_events = [
    #         # Normal user activity
    #         {
    #             "post_burst": 1, "destination_entropy": 2.1, "hour": 14, "megabytes_sent": 2.4,
    #             "uploads_last_24h": 3, "user_upload_count": 15, "user_mean_upload_size": 5.2,
    #             "user_std_upload_size": 1.8, "user_unique_destinations": 4, "user_destination_count": 8,
    #             "attachment_count": 0, "bcc_count": 0, "cc_count": 1, "size": 2.4,
    #             "first_time_destination": False, "after_hours": False, "is_large_upload": False,
    #             "to_suspicious_domain": False, "is_usb": False, "is_weekend": False,
    #             "has_attachments": False, "is_from_user": True, "is_outlier_hour": False,
    #             "is_outlier_size": False, "destination_domain": "internal.com", "user": "normal_user",
    #             "channel": "HTTP", "from_domain": "company.com"
    #         },
    #         # After-hours activity
    #         {
    #             "post_burst": 2, "destination_entropy": 3.5, "hour": 22, "megabytes_sent": 8.1,
    #             "uploads_last_24h": 1, "user_upload_count": 8, "user_mean_upload_size": 7.3,
    #             "user_std_upload_size": 2.4, "user_unique_destinations": 2, "user_destination_count": 3,
    #             "attachment_count": 1, "bcc_count": 0, "cc_count": 0, "size": 8.1,
    #             "first_time_destination": False, "after_hours": True, "is_large_upload": False,
    #             "to_suspicious_domain": False, "is_usb": False, "is_weekend": False,
    #             "has_attachments": True, "is_from_user": True, "is_outlier_hour": True,
    #             "is_outlier_size": False, "destination_domain": "partner.org", "user": "late_worker",
    #             "channel": "HTTPS", "from_domain": "company.com"
    #         },
    #         # High-risk scenario
    #         {
    #             "post_burst": 5, "destination_entropy": 6.2, "hour": 23, "megabytes_sent": 25.8,
    #             "uploads_last_24h": 12, "user_upload_count": 3, "user_mean_upload_size": 18.5,
    #             "user_std_upload_size": 8.2, "user_unique_destinations": 8, "user_destination_count": 12,
    #             "attachment_count": 3, "bcc_count": 2, "cc_count": 0, "size": 25.8,
    #             "first_time_destination": True, "after_hours": True, "is_large_upload": True,
    #             "to_suspicious_domain": True, "is_usb": True, "is_weekend": True,
    #             "has_attachments": True, "is_from_user": True, "is_outlier_hour": True,
    #             "is_outlier_size": True, "destination_domain": "suspicious.net", "user": "threat_user",
    #             "channel": "USB", "from_domain": "company.com"
    #         }
    #     ]
    #         # Create background dataset
    #         # background_tensors = []
    #         # for evt in background_events:
    #         #     # Fill missing features
    #         #     complete_evt = {feat: evt.get(feat, 0) for feat in features_order}
    #         #     flat_tensor = encode_sequence_flat_for_shap(complete_evt, feature_lists, embed_maps).squeeze(0)
    #         #     background_tensors.append(flat_tensor)
            
    #         # background = torch.stack(background_tensors)

    #         background_arrays = []
    #         for evt in background_events:
    #             complete_evt = {feat: evt.get(feat, 0) for feat in features_order}
    #             flat_tensor = encode_sequence_flat_for_shap(complete_evt, feature_lists, embed_maps).squeeze(0)
                
    #             # Convert to numpy for SHAP
    #             if hasattr(flat_tensor, 'detach'):
    #                 flat_array = flat_tensor.detach().cpu().numpy()
    #             else:
    #                 flat_array = np.array(flat_tensor, dtype=np.float32)
                
    #             background_arrays.append(flat_array)

    #         background = np.array(background_arrays) 

    #         def transformer_predict_wrapper_for_shap(x):
    #             """Wrapper that handles numpy input from SHAP."""
    #             # Convert numpy to tensor if needed
    #             if isinstance(x, np.ndarray):
    #                 x = torch.tensor(x, dtype=torch.float32, device='cpu')
                
    #             # Use the transformer wrapper
    #             with torch.no_grad():
    #                 output = transformer_predict_wrapper(x)
    #                 # Ensure output is numpy for SHAP
    #                 if hasattr(output, 'detach'):
    #                     return output.detach().cpu().numpy()
    #                 return output
            
    #         # Try DeepExplainer first, fallback to GradientExplainer
    #         try:
    #             # shap_explainer = shap.DeepExplainer(transformer_predict_wrapper, background)
    #             shap_explainer = shap.KernelExplainer(transformer_predict_wrapper_for_shap, background)
    #             explainer['transformer_explainer'] = shap_explainer
    #             # logging.info("✅ Transformer SHAP DeepExplainer created successfully")
    #             logging.info("✅ Transformer SHAP DeepExplainer created successfully")
    #         except Exception as deep_e:
    #             logging.warning(f"DeepExplainer failed: {deep_e}, trying GradientExplainer")
    #             # shap_explainer = shap.GradientExplainer(transformer_predict_wrapper, background)
    #             # explainer['transformer_explainer'] = shap_explainer
    #             explainer['transformer_explainer'] = None
    #             logging.info("✅ Transformer SHAP GradientExplainer created successfully")
            
    #         # Transformer Captum Explainer
    #         explainer['transformer_captum'] = IntegratedGradients(transformer_predict_wrapper.semantic_forward)
    #         explainer['captum_gradshap'] = GradientShap(transformer_predict_wrapper.semantic_forward) 
    #         explainer['captum_deeplift'] = DeepLift(transformer_predict_wrapper.semantic_forward)
       
    #         logging.info("✅ Transformer Captum explainer created successfully")
            
    #     except Exception as e:
    #         logging.error(f"Failed to create Transformer explainers: {e}")
    #         explainer['transformer_explainer'] = None
    #         explainer['transformer_captum'] = None
    #         explainer['captum_gradshap'] = None
    #         explainer['captum_deeplift'] = None
    
    #     return explainer
            
    #         # Create background dataset
    #         # background_data = np.zeros((10, len(feature_names)))
            
    #         # # Use KernelExplainer for model-agnostic explanations
    #         # explainer = shap.KernelExplainer(transformer_predict_wrapper, background_data)
            
    #         # return explainer