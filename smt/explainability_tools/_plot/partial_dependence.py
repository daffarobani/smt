from .. import partial_dependence

import numpy as np
from scipy.stats.mstats import mquantiles

class PartialDependenceDisplay:
    def __init__(self, pd_results, *, features, deciles, random_state = None):
        self.pd_results = pd_results
        self.features = features
        self.deciles = deciles
        self.random_state = random_state

    @classmethod
    def from_surrogate_model(
            cls,
            model,
            X,
            features,
            *,
            sample_weight=None,
            categorical_features=None,
            feature_names=None,
            percentiles=(0.05, 0.95),
            grid_resolution=100,
            kind="average", 
            centered = False
    ):
        pd_results = partial_dependence(
            model, X, features, sample_weight=sample_weight, 
            categorical_features=categorical_features, percentiles=percentiles, 
            grid_resolution=grid_resolution, kind=kind)

        target_features = set()
        for feature in features:
            if type(feature) is int:
                target_features.add(feature)
            else:
                for f in feature:
                    target_features.add(f)

        deciles = {}
        for feature in target_features:
            deciles[feature] = mquantiles(X[:, feature], prob=np.arange(0.1, 1.0, 0.1))
            
        display = PartialDependenceDisplay(
            pd_results, 
            features = features, 
            deciles = deciles,
            )
        return display.plot(
            centered=centered
        )

    def _plot_ice_lines(
        self,
        preds,
        feature_values,
        n_ice_to_plot,
        ax,
        pd_plot_idx,
        n_total_lines_by_plot,
        individual_line_kw
    ):
        if self.random_state is None:
            rng = np.random.mtrand._rand
        else:
            rng = np.random.RandomState(self.random_state)
        # subsample ICE
        ice_lines_idx = rng.choice(
            preds.shape[0],
            n_ice_to_plot,
            replace=False
        )
        ice_lines_subsampled = preds[ice_lines_idx, :]
        # plot the subsampled ICE
        for ice_idx, ice in enumerate(ice_lines_subsampled):
            # line_idx = np.unravel_index(
            #     pd_plot_idx * n_total_lines_by_plot + ice_idx, self.lines_.shape
            # )
            # self.lines_[line_idx] = ax.plot(
            #     feature_values, ice.ravel(), **individual_line_kw
            # )[0]
            ax.plot(feature_values[0], ice.ravel(), **individual_line_kw)

    def _plot_average_dependence(
        self, 
        avg_preds,
        feature_values,
        ax,
        pd_line_idx,
        line_kw
    ):
        categorical = False
        if categorical:
            pass
        else:
            ax.plot(
                feature_values[0], 
                avg_preds,
                **line_kw
            )

    def _plot_one_way_partial_dependence(
        self, 
        kind, 
        preds,
        avg_preds,
        feature_values, 
        feature_idx, 
        n_ice_lines,
        ax,
        n_cols, 
        plot_idx, 
        n_lines, 
        ice_lines_kw,
        pd_line_kw,
        bar_kw, 
        pdp_lim
    ):
        from matplotlib import transforms

        if kind in ["individual", "both"]:
            self._plot_ice_lines(
                preds,
                feature_values,
                n_ice_lines,
                ax,
                plot_idx,
                n_lines,
                ice_lines_kw
            )

        if kind in ("average", "both"):
            # the average is stored as the last line
            if kind == "average":
                pd_line_idx = plot_idx
            else:
                pd_line_idx = plot_idx * n_lines + n_ice_lines
            self._plot_average_dependence(
                avg_preds.ravel(),
                feature_values,
                ax,
                pd_line_idx,
                pd_line_kw
            )
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
         # create the decile line for the vertical axis
        vlines_idx = np.unravel_index(plot_idx, self.deciles_vlines_.shape)
        if self.deciles.get(feature_idx, None) is not None:
            self.deciles_vlines_[vlines_idx] = ax.vlines(
                self.deciles[feature_idx],
                0,
                0.05,
                transform=trans,
                color="k",
            )
        # reset ylim which was overwritten by vlines
        pass
        # min_val = min(val[0] for val in pdp_lim.values())
        # max_val = max(val[1] for val in pdp_lim.values())
        # ax.set_ylim([min_val, max_val])
        if kind in ["individual", "both"]:
            ax.set_ylim([1*preds.min(), 1*preds.max()])
        else:
            ax.set_ylim([1*avg_preds.min(), 1*avg_preds.max()])

        # Set xlabel if it is not already set
        # if not ax.get_xlabel():
        #     ax.set_xlabel(self.feature_names[feature_idx])
        ax.set_xlabel(f'x_{feature_idx}')

        if n_cols is None or plot_idx % n_cols == 0:
            if not ax.get_ylabel():
                ax.set_ylabel("Partial dependence")
        else:
            # ax.set_yticklabels([])
            pass


        # if pd_line_kw.get("label", None) and kind != "individual" and not categorical:
        #     ax.legend()
        if kind == "both":
            ax.legend()
        
    
    def _plot_two_way_partial_dependence(
            self,
            kind,
            avg_preds,
            feature_values,
            feature_idx,
            ax,
            Z_level,
            contour_kw,
            heatmap_kw
    ):
        categorical = False
        if kind == "individual":
            pass
        else:
            if categorical:
                pass
            else:
                from matplotlib import transforms
                XX, YY = np.meshgrid(feature_values[0], feature_values[1])
                Z = avg_preds.T
                CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5, colors="k")
                # contour_idx = np.unravel_index(pd_plot_idx, self.contours_.shape)
                ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1], vmin=Z_level[0], **contour_kw)
                ax.clabel(CS, fmt="%2.2f", colors="k", fontsize=10, inline=True)

                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                # create the decile line for the vertical axis
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                ax.vlines(
                    self.deciles[feature_idx[0]],
                    0,
                    0.05,
                    transform=trans,
                    color="k",
                )
                # create the decile line for the horizontal axis
                ax.hlines(
                    self.deciles[feature_idx[1]],
                    0,
                    0.05,
                    transform=trans,
                    color="k",
                )
                # reset xlim and ylim since they are overwritten by hlines and
                # vlines
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(f'x_{feature_idx[0]}')
                ax.set_ylabel(f'x_{feature_idx[1]}')
            
    def plot(
        self, 
        *,
        n_cols=3, 
        ax = None,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
        bar_kw=None,
        heatmap_kw=None,
        centered = False,
        pdp_lim = None,
        max_num_ice_lines = 250,
    ):
        import matplotlib.pyplot as plt  
        from matplotlib.gridspec import GridSpecFromSubplotSpec 

        kind = []
        # for pd_result in self.pd_results:
        #     if len(pd_result['grid_values']) > 1:
        #         kind.append('average')
        #     else:
        #         keys = pd_result.keys()
        #         if ('average' in keys) and ('individual' in keys):
        #             kind.append('both')
        #         elif ('average' in keys) and ('individual' not in keys):
        #             kind.append('average')
        #         else:
        #             kind.append('individual')
        for pd_result in self.pd_results:
            keys = pd_result.keys()
            if (len(pd_result['grid_values'])>1) & ('average' in keys):
                kind.append('average')
            else:
                if ('average' in keys) and ('individual' in keys):
                    kind.append('both')
                elif ('average' in keys) and ('individual' not in keys):
                    kind.append('average')
                else:
                    kind.append('individual')

        n_results = len(self.pd_results)
        if ax is None:
            _, ax = plt.subplots()

        if not centered:
            pd_results_ = self.pd_results
        else:
            pd_results_ = []
            for kind_plot, pd_result in zip(kind, self.pd_results):
                current_results = {"grid_values": pd_result["grid_values"]}

                if kind_plot in ("individual", "both"):
                    preds = pd_result["individual"]
                    preds = preds - preds[:, 0, None]
                    current_results["individual"] = preds

                if kind_plot in ("average", "both"):
                    avg_preds = pd_result["average"]
                    avg_preds = avg_preds - avg_preds[0, None]
                    current_results["average"] = avg_preds

                pd_results_.append(current_results)

        if pdp_lim is None:
            pdp_lim = {}
            for kind_plot, pd_result in zip(kind, pd_results_):
                values = pd_result["grid_values"]
                preds = pd_result["average"] if kind_plot == "average" else pd_result["individual"]
                min_pd = preds.min()
                max_pd = preds.max()

                # expand the limits to account so that the plotted lines do not touch
                # the edges of the plot
                span = max_pd - min_pd
                min_pd -= 0.05 * span
                max_pd += 0.05 * span

                n_features = len(values)
                old_min_pd, old_max_pd = pdp_lim.get(n_features, (min_pd, max_pd))
                min_pd = min(min_pd, old_min_pd)
                max_pd = max(max_pd, old_max_pd)
                pdp_lim[n_features] = (min_pd, max_pd)
        

        if line_kw is None:
            line_kw = {}
        if ice_lines_kw is None:
            ice_lines_kw = {}
        if pd_line_kw is None:
            pd_line_kw = {}
        if bar_kw is None:
            bar_kw = {}
        if heatmap_kw is None:
            heatmap_kw = {}
        if contour_kw is None:
            contour_kw = {}
        default_contour_kws = {"alpha": 0.75}
        contour_kw = {**default_contour_kws, **contour_kw}

        is_average_plot = [kind_plot == "average" for kind_plot in kind]
        if all(is_average_plot):
            # only average plots are requested
            n_ice_lines = 0
            n_lines = 1
        else:
            # we need to determine the number of ICE samples computed
            ice_plot_idx = is_average_plot.index(False)
            n_ice_lines = pd_results_[ice_plot_idx]["individual"].shape[0]
            n_ice_lines = min(n_ice_lines, max_num_ice_lines)
            
            if any([kind_plot == "both" for kind_plot in kind]):
                n_lines = n_ice_lines + 1  # account for the average line
            else:
                n_lines = n_ice_lines

        if isinstance(ax, plt.Axes):
            ax.set_axis_off()
            self.bounding_ax_ = ax
            self.figure_ = ax.figure

            n_cols = min(n_cols, n_results)
            n_rows = int(np.ceil(n_results / float(n_cols)))
            self.axes_ = np.empty((n_rows, n_cols), dtype=object)
            self.figure_.set_size_inches(n_cols * 7, n_rows * 5)

            axes_ravel = self.axes_.ravel()

            gs = GridSpecFromSubplotSpec(
                n_rows, n_cols, subplot_spec=ax.get_subplotspec()
            )
            for i, spec in zip(range(n_results), gs):
                axes_ravel[i] = self.figure_.add_subplot(spec)

        # create contour levels for two-way plots
        if 2 in pdp_lim:
            Z_level = np.linspace(*pdp_lim[2], num=8)
        self.deciles_vlines_ = np.empty_like(self.axes_, dtype=object)
        self.deciles_hlines_ = np.empty_like(self.axes_, dtype=object)

        for plot_idx, (axi, pd_result, kind_plot, feature_idx) in enumerate(
            zip(
                self.axes_.ravel(),
                pd_results_, 
                kind, 
                self.features
            )
        ):
            # print(feature_idx)
            avg_preds = None
            preds = None
            feature_values = pd_result["grid_values"]

            if kind_plot == "individual":
                preds = pd_result["individual"]
            elif kind_plot == "average":
                avg_preds = pd_result["average"]
            else: # kind_plot == "both"
                preds = pd_result["individual"]
                avg_preds = pd_result["average"]
            
            if len(feature_values) == 1:
                # define the line-style for the current plot
                default_line_kws = {
                    "color": "C0",
                    "label": "average" if kind_plot == "both" else None,
                }
                if kind_plot == "individual":
                    default_ice_lines_kws = {"alpha": 0.3, "linewidth": 0.5}
                    default_pd_lines_kws = {}
                elif kind_plot == "both":
                    # by default, we need to distinguish the average line from
                    # the individual lines via color and line style
                    default_ice_lines_kws = {
                        "alpha": 0.3,
                        "linewidth": 0.5,
                        "color": "tab:blue",
                    }
                    default_pd_lines_kws = {
                        "color": "tab:orange",
                        "linestyle": "--",
                    }
                else:
                    default_ice_lines_kws = {}
                    default_pd_lines_kws = {}

                ice_lines_kw = {
                    **default_line_kws,
                    **default_ice_lines_kws,
                    **line_kw,
                    **ice_lines_kw,
                }
                del ice_lines_kw["label"]

                pd_line_kw = {
                    **default_line_kws,
                    **default_pd_lines_kws,
                    **line_kw,
                    **pd_line_kw,
                }

                default_bar_kws = {"color": "C0"}
                bar_kw = {**default_bar_kws, **bar_kw}

                default_heatmap_kw = {}
                heatmap_kw = {**default_heatmap_kw, **heatmap_kw}

                self._plot_one_way_partial_dependence(
                    kind_plot, 
                    preds,
                    avg_preds,
                    feature_values, 
                    feature_idx, 
                    n_ice_lines,
                    axi,
                    n_cols, 
                    plot_idx, 
                    n_lines, 
                    ice_lines_kw,
                    pd_line_kw,
                    bar_kw, 
                    pdp_lim
                )
                
            else:
                self._plot_two_way_partial_dependence(
                    kind_plot,
                    avg_preds,
                    feature_values,
                    feature_idx,
                    axi,
                    Z_level,
                    contour_kw,
                    heatmap_kw
                )

        return self


        