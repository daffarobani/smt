"""Features interaction based on partial dependence"""
# Authors: Muhammad Daffa Robani

import numpy as np
from ._partial_dependence import partial_dependence


def pd_pairwise_interaction(
        feature_pairs,
        X,
        model,
        *,
        ratio_samples=None,
        categorical_features=None,
):
    if ratio_samples is None:
        X_eval = X.copy()
    else:
        num_samples = int(ratio_samples*len(X))
        indexes = np.random.choice(
            X.shape[0],
            size=num_samples,
            replace=False
        )
        X_eval = X[indexes]

    h_scores = list()
    for feature_pair in feature_pairs:
        feature_i = feature_pair[0]
        feature_j = feature_pair[1]
        pd_features = [
            feature_i,
            feature_j,
            (feature_i, feature_j),
        ]

        pd_results = partial_dependence(
            model,
            X_eval,
            pd_features,
            categorical_features=categorical_features,
            method="sample",
            kind="average",
        )
        average_i = pd_results[0]["average"]
        average_j = pd_results[1]["average"]
        average_ij = pd_results[2]["average"]
        h_score = compute_h_score(
            average_ij,
            average_i,
            average_j,
        )

        h_scores.append(h_score)
    return h_scores


def pd_overall_interaction(
        features,
        X,
        model,
        *,
        ratio_samples=None,
        categorical_features=None,
):
    if ratio_samples is None:
        X_eval = X.copy()
    else:
        num_samples = int(ratio_samples * len(X))
        indexes = np.random.choice(
            X.shape[0],
            size=num_samples,
            replace=False
        )
        X_eval = X[indexes]

    h_scores = list()
    for feature in features:
        other_features = [f for f in range(X_eval.shape[1]) if f != feature]
        pd_features = [
            feature,
            other_features,
        ]

        pd_results = partial_dependence(
            model,
            X_eval,
            pd_features,
            categorical_features=categorical_features,
            method="sample",
            kind="average",
        )
        average_on_feature = pd_results[0]["average"]
        average_on_other = pd_results[1]["average"]
        y_pred = model.predict_values(X_eval).reshape(-1, )
        h_score = compute_h_score(
            y_pred,
            average_on_feature,
            average_on_other,
        )
        h_scores.append(h_score)

    return h_scores


def compute_h_score(
        ref_values,
        explainer_first,
        explainer_second,
):
    # center to make mean equals zero
    ref_values = ref_values - ref_values.mean()
    explainer_first = explainer_first - explainer_first.mean()
    explainer_second = explainer_second - explainer_second.mean()

    # compute h score
    numerator = ref_values - explainer_first - explainer_second
    numerator = (numerator ** 2).sum()
    denominator = (ref_values ** 2).sum()
    h_score = numerator / denominator
    return h_score
