from .. import pd_feature_importance
import numpy as np


class PDFeatureImportanceDisplay:
    def __init__(self, importances, feature_names):
        self.importances = importances
        self.feature_names = feature_names

    @classmethod
    def from_surrogate_model(
        cls,
        model,
        x,
        *, 
        features=None,
        feature_names=None,
        sample_weight=None,
        categorical_features=None, 
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        # uniform=True,
        method="uniform",
        sort=False,
        ratio_samples=None,
        figsize=None,
    ):
        if features is None:
            features = [i for i in range(x.shape[1])]

        importances = pd_feature_importance(
            model,
            x,
            features,
            sample_weight=sample_weight, 
            categorical_features=categorical_features, 
            percentiles=percentiles,
            grid_resolution=grid_resolution,
            # uniform=uniform,
            method=method,
            ratio_samples=ratio_samples,
        )
        display = PDFeatureImportanceDisplay(
            importances, 
            feature_names,
        )
        return display.plot(
            sort=sort,
            figsize=figsize,
        )
    
    def plot(
            self,
            *, 
            figsize=None,
            sort=False,
    ):
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": "cmr10",
            "axes.formatter.use_mathtext": True,
        })

        if figsize is None:
            length = max(
                5,
                int(len(self.importances) * 0.6)
            )
            width = 4
        else:
            length = figsize[0]
            width = figsize[1]

        if self.feature_names is None:
            feature_names = [fr'$x_{i}$' for i in range(len(self.importances))]
        else:
            feature_names = self.feature_names
        feature_names = np.array(feature_names)
        importances = np.array(self.importances)

        if sort:
            vis_feature_names = feature_names[np.argsort(importances*-1)]
            vis_importances = importances[np.argsort(importances*-1)]
        else:
            vis_feature_names = feature_names
            vis_importances = importances

        indexes = np.arange(len(vis_importances))
        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        ax.bar(indexes, vis_importances, color="blue", edgecolor='black', linewidth=0.8)
        ax.set_xticks(indexes)
        ax.set_xticklabels(vis_feature_names, fontsize=14)
        ax.set_ylabel('Feature Importance', fontsize=14)
        ax.grid(color="black", alpha=0.2)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_axisbelow(True)
        fig.tight_layout()

        self.fig = fig

        return self
