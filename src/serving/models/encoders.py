import torch
import numpy as np
import pandas as pd
from flask import current_app
from typing import Dict, Tuple
import logging

# def encode_tabular(event: Dict) -> np.ndarray:
#     """Return tabular vector ready for RF."""
#     models = current_app.ml_models

#     feature_names = models['feature_names']
#     preprocessor = models['preprocessor']
    
#     # df = {k: [event.get(k, 0)] for k in feature_names}
#     # X_raw = preprocessor.transform(pd.DataFrame.from_dict(df))
#     # return X_raw[0]  # return single row as 1D array

#      # Create feature vector in correct order
#     feature_vector = [event[name] for name in feature_names]
#     feature_array = np.array(feature_vector).reshape(1, -1)
    
#     # Apply preprocessing
#     X_processed = preprocessor.transform(feature_array)

#     # Handle case where preprocessor returns tuple or sparse matrix
#     if isinstance(X_processed, tuple):
#         X_processed = X_processed[0]  # Take first element
    
#     # Convert sparse matrix to dense if needed
#     if hasattr(X_processed, 'toarray'):
#         X_processed = X_processed.toarray()
    
#     # Return as 1D array for single sample
#     # return X_processed[0] if X_processed.ndim > 1 else X_processed
#     # Return as 1D array for single sample
#     return X_processed.flatten() if X_processed.ndim > 1 else X_processed


# def encode_tabular(event: Dict) -> np.ndarray:
#     """Return tabular vector ready for RF - following serve_flask.py pattern."""
#     models = current_app.ml_models
    
#     # Use exact feature order from training (like serve_flask.py)
#     features_order = models['feature_names']
#     preprocessor = models['preprocessor']
    
#     # Create DataFrame with exact same structure as training
#     df_dict = {k: [event.get(k)] for k in features_order}
#     df = pd.DataFrame.from_dict(df_dict)
    
#     # Transform using preprocessor
#     X_processed = preprocessor.transform(df)
    
#     # Handle various return types
#     if isinstance(X_processed, tuple):
#         X_processed = X_processed[0]
#     if hasattr(X_processed, 'toarray'):
#         X_processed = X_processed.toarray()
    
#     return X_processed[0] 

def encode_tabular(event: Dict) -> np.ndarray:
    """Return tabular vector ready for RF."""
    models = current_app.ml_models
    
    features_order = models['feature_names']
    preprocessor = models['preprocessor']

    df = {k: [event.get(k)] for k in features_order} 
    X_raw = preprocessor.transform(pd.DataFrame.from_dict(df))
    return X_raw[0]  # return single row as 1D array

# def encode_sequence_semantic(event: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Encode preserving semantic structure for model compatibility."""
#     models = current_app.ml_models
#     feature_lists = models['feature_lists']
#     embed_maps = models['embed_maps']
#     device = models['device']
    
#     # continuous_used = feature_lists['CONTINUOUS_USED']
#     # boolean_used = feature_lists['BOOLEAN_USED']
#     # high_cat_used = feature_lists['HIGH_CAT_USED']
#     # low_cat_used = feature_lists['LOW_CAT_USED']

#     # Continuous + Boolean features (maintain float32)
#     cont_features = feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]
#     cont_values = [event.get(c, 0.0) for c in cont_features]
#     cont = torch.tensor([[cont_values]], dtype=torch.float32, device=device)  # [1, 1, cont_dim]
    
#     # # High-cardinality categorical (convert to LONG indices)
#     # cat_high_values = [embed_maps[c].get(str(event.get(c, '')), 0) for c in feature_lists["HIGH_CAT_USED"]]
#     # cat_high = torch.tensor([[cat_high_values]], dtype=torch.long, device=device)  # [1, 1, high_cat_dim]
    
#     # # Low-cardinality categorical (convert to LONG indices)  
#     # cat_low_values = [embed_maps[c].get(str(event.get(c, '')), 0) for c in feature_lists["LOW_CAT_USED"]]
#     # cat_low = torch.tensor([[cat_low_values]], dtype=torch.long, device=device)  # [1, 1, low_cat_dim]

#     # High-cardinality categorical
#     high_cat_values = []
#     for c in feature_lists["HIGH_CAT_USED"]:
#         val = embed_maps[c].get(str(event.get(c, '')), 0)
#         high_cat_values.append(val)
#     cat_high = torch.tensor([[high_cat_values]], dtype=torch.long, device=device)
    
#     # Low-cardinality categorical  
#     low_cat_values = []
#     for c in feature_lists["LOW_CAT_USED"]:
#         val = embed_maps[c].get(str(event.get(c, '')), 0)
#         low_cat_values.append(val)
#     cat_low = torch.tensor([[low_cat_values]], dtype=torch.long, device=device)
    
#     return cont, cat_high, cat_low

def encode_sequence_semantic(event: Dict, feature_lists: dict, embed_maps: dict = None, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode single event as sequence for transformer compatibility."""

    if device is None or isinstance(device, dict):
        device = 'cpu'

    if feature_lists is None or embed_maps is None:
        try:
            models = current_app.ml_models
            feature_lists = feature_lists or models['feature_lists']
            embed_maps = embed_maps or models['embed_maps']
            device = device or models.get( 'device', 'cpu')
        except RuntimeError:
            # Outside Flask context - parameters must be provided
            if feature_lists is None or embed_maps is None:
                raise ValueError("feature_lists and embed_maps must be provided when outside Flask context")
        
    
    # Sequence parameters (match training)
    seq_len = 10  # Reasonable length for inference
    
    # Continuous + Boolean features
    cont_features = feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]
    cont_values = [event.get(c, 0.0) for c in cont_features]
    
    # Create sequence: latest event + padding
    cont_seq = [[0.0] * len(cont_values)] * (seq_len - 1) + [cont_values]  # Pad with zeros, end with actual event
    cont = torch.tensor([cont_seq], dtype=torch.float32, device=device)  # [1, seq_len, cont_dim]
    
    # High-cardinality categorical
    high_cat_values = [embed_maps[c].get(str(event.get(c, '')), 0) for c in feature_lists["HIGH_CAT_USED"]]
    high_cat_seq = [[0] * len(high_cat_values)] * (seq_len - 1) + [high_cat_values]
    cat_high = torch.tensor([high_cat_seq], dtype=torch.long, device=device)  # [1, seq_len, high_cat_dim]
    
    # Low-cardinality categorical
    low_cat_values = [embed_maps[c].get(str(event.get(c, '')), 0) for c in feature_lists["LOW_CAT_USED"]]
    low_cat_seq = [[0] * len(low_cat_values)] * (seq_len - 1) + [low_cat_values]
    cat_low = torch.tensor([low_cat_seq], dtype=torch.long, device=device)  # [1, seq_len, low_cat_dim]
    try:
        return cont.to(device), cat_high.to(device), cat_low.to(device)
    except Exception as e:
        logging.warning(f"Device conversion failed, using CPU: {e}")
        return cont.to('cpu'), cat_high.to('cpu'), cat_low.to('cpu')

def encode_sequence_flat_for_shap(event: Dict, feature_lists=None, embed_maps=None) -> torch.Tensor: # np.ndarray: #torch.Tensor:
    """Convert event to flat tensor for SHAP compatibility."""
    # try to get flask context first, 
    if feature_lists is None or embed_maps is None:
        try:
            models = current_app.ml_models
            feature_lists = models['feature_lists']
            embed_maps = models['embed_maps']
        except RuntimeError:
             # Outside Flask context - parameters must be provided
            if feature_lists is None or embed_maps is None:
                raise ValueError("feature_lists and embed_maps must be provided when outside Flask context")


    arr = []
    
    # continuous_used = feature_lists['CONTINUOUS_USED']
    # boolean_used = feature_lists['BOOLEAN_USED']
    # high_cat_used = feature_lists['HIGH_CAT_USED']
    # low_cat_used = feature_lists['LOW_CAT_USED']
    
    # features = []
    # # Continuous features
    # features.extend([event.get(c, 0.0) for c in continuous_used + boolean_used])
    # # Categorical as float indices (semantic loss for SHAP compatibility)
    # features.extend([float(embed_maps[c].get(str(event.get(c, '')), 0)) for c in high_cat_used])
    # features.extend([float(embed_maps[c].get(str(event.get(c, '')), 0)) for c in low_cat_used])
    
    # return torch.tensor([features], dtype=torch.float32)
    
    # Continuous and boolean features
    for c in feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]:
        arr.append(float(event.get(c, 0.0)))
    
    # High-cardinality categorical features
    for c in feature_lists["HIGH_CAT_USED"]:
        arr.append(float(embed_maps[c].get(str(event.get(c, '')), 0)))
    
    # Low-cardinality categorical features
    for c in feature_lists["LOW_CAT_USED"]:
        arr.append(float(embed_maps[c].get(str(event.get(c, '')), 0)))
    
    # return np.array([arr], dtype=np.float32)
    return torch.tensor([arr], dtype=torch.float32)


def transformer_tensors_to_flat(cont: torch.Tensor, cat_high: torch.Tensor, cat_low: torch.Tensor, feature_lists: Dict) -> np.ndarray:
    """Convert transformer tensors back to flat array for feature mapping."""
    # Extract last timestep (actual event, not padding)
    cont_values = cont[0, -1, :].cpu().numpy()  # Last timestep
    cat_high_values = cat_high[0, -1, :].cpu().numpy()
    cat_low_values = cat_low[0, -1, :].cpu().numpy()
    
    # Combine all features in the same order as feature_names
    all_values = np.concatenate([cont_values, cat_high_values, cat_low_values])
    return all_values

def dict_to_transformer_tensors(event_dict, feature_lists, embed_maps, device='cpu'):
    """Convert event dictionary to transformer input tensors."""
    cont, cat_high, cat_low = encode_sequence_semantic(
        event_dict,
        feature_lists,
        embed_maps,
        device)
    return cont.to(device), cat_high.to(device), cat_low.to(device)