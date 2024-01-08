import numpy as np
from ._partial_dependence import partial_dependence

def pd_feature_importance_computation(
        features,
        is_categorical,
        pd_results, 
):
    importances = []
    for i, feature in enumerate(features):
        pd = pd_results[i]['average']
        if is_categorical[feature]:
            importance = (np.max(pd) - np.min(pd)) / 4
        else:
            k = len(pd)
            mean_pd = np.mean(pd)
            importance = np.power(np.sum((pd - mean_pd)**2) / (k-1), 0.5)
        importances.append(importance)
    return importances


def pd_feature_importance(
        model, 
        X, 
        features, 
        *, 
        sample_weight=None,
        categorical_features=None,
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        # uniform="true",
        method="uniform",
        ratio_samples=None
):
    pd_results = partial_dependence(
        model, 
        X, 
        features,
        sample_weight=sample_weight,
        categorical_features=categorical_features,
        percentiles=percentiles,
        grid_resolution=grid_resolution,
        # uniform=uniform,
        method=method,
        kind="average",
        ratio_samples=ratio_samples,
    )

    is_categorical = [0] * X.shape[1]
    if categorical_features is not None:
        for feature_index in categorical_features:
            is_categorical[feature_index] = 1

    importances = pd_feature_importance_computation(
        features, 
        is_categorical,
        pd_results,
    )
    return importances