"""Partial dependence for classification and regression models"""
# Authors: Muhammad Daffa Robani
import numpy as np
from scipy.stats.mstats import mquantiles


def grid_from_x(
        X, 
        features, 
        percentiles, 
        grid_resolution, 
        is_categorical, 
        method,
        # uniform,
):
    # method: ["uniform", "unique", "sample"]
    if type(features) is int:
        features = tuple([features])
    grid_values = []
    for i in features:
        # if is_categorical[i] or not uniform:
        #     axis = np.unique(X[:, i])
        # else:
        #     emp_percentiles = mquantiles(X[:, i], prob=percentiles, axis=0)
        #     axis = np.linspace(emp_percentiles[0], emp_percentiles[1], num=grid_resolution, endpoint=True)
        if method == "sample":
            axis = X[:, i]
        else:
            if is_categorical[i] or method == "unique":
                axis = np.unique(X[:, i])
            else:
                emp_percentiles = mquantiles(X[:, i], prob=percentiles, axis=0)
                axis = np.linspace(emp_percentiles[0], emp_percentiles[1], num=grid_resolution, endpoint=True)

        grid_values.append(axis)

    if method in ["unique", "uniform"]:
        grid_2d = cartesian(grid_values)
    else:
        grid_2d = non_cartesian(grid_values)
    # return cartesian(grid_values), grid_values
    return grid_2d, grid_values


def _partial_dependence_brute(
    model, 
    grid_cartesian, 
    grid_values, 
    features, 
    X,
    method,
    sample_weight=None, 
    ratio_samples=None,
):    
    if type(features) is int:
        features = tuple([features])

    nsamp = len(X)
    lengths = [len(grid_value) for grid_value in grid_values]
    predictions = []
    averaged_predictions = []
    if ratio_samples is None:
        nsamples = nsamp
        X_samp = X.copy()
    else:
        nsamples = int(ratio_samples * nsamp)
        index = np.random.choice(nsamp, nsamples, replace=False) 
        X_samp = X[index]

    for new_values in grid_cartesian:
        X_eval = X_samp.copy()
        for i, feature in enumerate(features):
            X_eval[:, feature] = new_values[i]
        try:
            pred = model.predict_values(X_eval) 
        except AttributeError:
            pred = model.predict(X_eval)
        averaged_pred = np.average(pred, weights=sample_weight)

        predictions.append(pred)
        averaged_predictions.append(averaged_pred)
        
    predictions = np.array(predictions).T
    averaged_predictions = np.array(averaged_predictions).T

    if method != "sample":
        predictions = predictions.reshape([nsamples]+lengths)
        averaged_predictions = averaged_predictions.reshape(lengths)
    
    return averaged_predictions, predictions


def partial_dependence(
        model,
        X,
        features,
        *,
        sample_weight=None,
        categorical_features=None,
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        kind="average",
        # uniform="true",
        method="uniform",
        ratio_samples=None,
        inverse_categories_map=None,
):
    """
    Partial dependence.
    
    Parameters
    ----------
    - model
    - 
    
    Returns 
    ----------
    
    
    """
    # to do: check if the model is fitted
    pass
    # to do: check if the model in SMT supports multi class multi label classification
    pass
    # to do: check sample weight

    if sample_weight is not None:
        pass

    for i, feature in enumerate(features):
        if type(feature) in [tuple, list]:
            if len(feature) == 1:
                features[i] = feature[0]
            elif len(feature) == 2:
                features[i] = tuple(feature)
                # raise ValueError("Interaction of features hasn't been developed")
            else:
                if method != "sample":
                    raise ValueError("Interaction features can't be more than two.")

    is_categorical = [0] * X.shape[1]
    if categorical_features is not None:
        for feature_index in categorical_features:
            is_categorical[feature_index] = 1
     
    pdp_results = []
    for feature in features:
        pdp_result = {}
        # create grid 
        grid_cartesian, grid_values = grid_from_x(
            X, 
            feature, 
            percentiles, 
            grid_resolution, 
            is_categorical, 
            # uniform,
            method,
            )
        
        # predictions
        averaged_predictions, predictions = _partial_dependence_brute(
            model, 
            grid_cartesian, 
            grid_values, 
            feature, 
            X,
            method,
            sample_weight,
            ratio_samples,
            )
        
        if type(feature) is int:
            is_categories = [is_categorical[feature]]
        else:
            is_categories = [is_categorical[f] for f in feature]
        has_categories = np.max(is_categories) == 1

        if has_categories == 1:
            grid_categories = []
            for i, (is_category, grid_values_) in enumerate(zip(is_categories, grid_values)):
                if is_category == 1:
                    if inverse_categories_map is not None:
                        if type(feature) is int:
                            f = feature
                        else:
                            f = feature[i]
                        grid_categories_ = [inverse_categories_map[f][i] for i in grid_values_]
                    else:
                        grid_categories_ = [value for value in grid_values_]
                else:
                    grid_categories_ = []

                grid_categories_ = np.array(grid_categories_)
                grid_categories.append(grid_categories_)
        
        # store
        pdp_result['grid_values'] = grid_values
        if has_categories == 1:
            pdp_result['grid_categories'] = grid_categories
        if kind == "average":
            pdp_result['average'] = averaged_predictions
        elif kind == "individual":
            pdp_result['individual'] = predictions
        else:
            pdp_result['average'] = averaged_predictions
            pdp_result['individual'] = predictions
        pdp_results.append(pdp_result)
    return pdp_results


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray of shape (M, len(arrays)), default=None
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray of shape (M, len(arrays))
        Array containing the cartesian products formed of input arrays.
        If not provided, the `dtype` of the output array is set to the most
        permissive `dtype` of the input arrays, according to NumPy type
        promotion.

        .. versionadded:: 1.2
           Add support for arrays of different types.

    Notes
    -----
    This function may not be used on more than 32 arrays
    because the underlying numpy functions do not support it.

    Examples
    --------
    >>> from sklearn.utils.extmath import cartesian
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        dtype = np.result_type(*arrays)  # find the most permissive dtype
        if dtype.str[:2] != '<U':
            out = np.empty_like(ix, dtype=dtype)
        else:
            out = np.empty_like(ix, dtype='object')

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def non_cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    
    ref = np.zeros((len(arrays[0]), len(arrays)))
    if out is None:
        out = np.empty_like(ref, dtype="object")

    for n, array in enumerate(arrays):
        out[:, n] = array
    return out
