from ._shap_values import individual_shap_values
import numpy as np


def shap_feature_importance(instances, model, x, is_categorical):
    shap_values = individual_shap_values(
        instances,
        model,
        x,
        is_categorical,
    )
    feature_importance = np.abs(shap_values).mean(axis=0)
    return feature_importance
