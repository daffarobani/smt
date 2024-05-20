from ._shap_values import individual_shap_values_2
import numpy as np


def shap_feature_importance_2(instances, model, x):
    shap_values = individual_shap_values_2(
        instances,
        model,
        x,
    )
    feature_importance = np.abs(shap_values).mean(axis=0)
    return feature_importance
