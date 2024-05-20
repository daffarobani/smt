from .. import shap_feature_importance
import numpy as np


class ShapFeatureImportanceDisplay:
    def __init__(self, feature_importance, feature_names):
        if feature_names is None:
            num_features = len(feature_importance)
            feature_names = [
                fr'$x_{i}$' for i in range(num_features)
            ]

        self.feature_importance = feature_importance
        self.feature_names = feature_names

    @classmethod
    def from_surrogate_model(
            cls,
            model,
            x,
            *,
            feature_names=None,
            figsize=None,
            sort=False,
            categorical_features=None,
    ):
        num_features = x.shape[1]
        # boolean flags for categorical variable indicator
        is_categorical = [0] * num_features
        if categorical_features is not None:
            for feature_idx in categorical_features:
                is_categorical[feature_idx] = 1
        # compute feature importance
        feature_importance = shap_feature_importance(
            x,
            model,
            x,
            is_categorical,
        )
        display = ShapFeatureImportanceDisplay(feature_importance, feature_names)
        return display.plot(figsize=figsize, sort=sort)

    def plot(self, *, figsize=None, sort=False):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": "cmr10",
            "axes.formatter.use_mathtext": True,
        })

        num_features = len(self.feature_importance)
        feature_names = np.array(self.feature_names)
        feature_importance = np.array(self.feature_importance)

        if figsize is None:
            length = max(5, int(num_features * 0.6))
            width = 4
        else:
            length = figsize[0]
            width = figsize[1]

        if sort:
            vis_feature_names = feature_names[np.argsort(feature_importance * -1)]
            vis_feature_importance = feature_importance[np.argsort(feature_importance * -1)]
        else:
            vis_feature_names = feature_names
            vis_feature_importance = feature_importance

        indexes = np.arange(num_features)
        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        ax.bar(
            indexes,
            vis_feature_importance,
            color="blue",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_xticks(indexes)
        ax.set_xticklabels(vis_feature_names, fontsize=14)
        ax.set_ylabel("Feature Importance", fontsize=14)
        ax.grid(color="black", alpha=0.2)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_axisbelow(True)
        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        fig.tight_layout()

        self.fig = fig

        return self
