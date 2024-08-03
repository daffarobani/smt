from smt.utils.sm_test_case import SMTestCase
from smt.problems import WingWeight
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG

from smt.applications.explainability_tools import ShapDisplay

from smt.problems import MixedCantileverBeam
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
from smt.applications.mixed_integer import MixedIntegerKrigingModel
from smt.surrogate_models import (
    KPLS,
    MixIntKernelType,
    MixHrcKernelType,
)
import numpy as np
import unittest


class NumericalTestProblem:
    def __init__(self, num_samples):
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
        x = sampling(num_samples)
        y = fun(x)

        feature_names = [
            r'$S_{w}$', r'$W_{fw}$', r'$A$', r'$\Delta$',
            r'$q$', r'$\lambda$', r'$t_{c}$', r'$N_{z}$',
            r'$W_{dg}$', r'$W_{p}$',
        ]

        sm = KRG(
            theta0=[1e-2] * x.shape[1],
            print_prediction=False
        )
        sm.set_training_values(x, y)
        sm.train()

        self.model = sm
        self.x = x
        self.feature_names = feature_names


class MixedTestProblem:
    def __init__(self, num_samples):
        fun = MixedCantileverBeam()
        # Design space
        ds = DesignSpace([
            CategoricalVariable(values=[str(i + 1) for i in range(12)]),
            FloatVariable(10.0, 20.0),
            FloatVariable(1.0, 2.0),
        ])
        x = fun.sample(num_samples)
        y = fun(x)

        # Name of the features
        feature_names = [r'$\tilde{I}$', r'$L$', r'$S$']
        # Index for categorical features
        categorical_feature_indices = [0]
        # create mapping for the categories
        categories_map = dict()
        for feature_idx in categorical_feature_indices:
            categories_map[feature_idx] = {
                i: value for i, value in enumerate(ds._design_variables[feature_idx].values)
            }

        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                design_space=ds,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ARC_KERNEL,
                theta0=np.array([4.43799547e-04, 4.39993134e-01, 1.59631650e+00]),
                corr="squar_exp",
                n_start=1,
                cat_kernel_comps=[2],
                n_comp=2,
                print_global=False,
            ),
        )
        sm.set_training_values(x, np.array(y))
        sm.train()

        self.model = sm
        self.x = x
        self.feature_names = feature_names
        self.categorical_feature_indices = categorical_feature_indices
        self.categories_map = categories_map


class TestShap(SMTestCase):
    def setUp(self):
        self.num_samples_numerical = 300
        self.num_samples_mixed = 100
        self.index_for_individual_plot = 0
        self.feature_pairs_for_numerical_problem = [(0, 1), (2, 3)]
        self.feature_pairs_for_mixed_problem = [(0, 1), (2, 0), (1, 2)]

    def test_shap_numerical(self):
        test_problem = NumericalTestProblem(self.num_samples_numerical)
        instances = test_problem.x
        ref_x = test_problem.x
        shap_explainer = ShapDisplay.from_surrogate_model(
            instances,
            test_problem.model,
            ref_x,
            feature_names=test_problem.feature_names,
        )
        individual_plot = shap_explainer.individual_plot(index=self.index_for_individual_plot)
        dependence_plot = shap_explainer.dependence_plot([i for i in range(test_problem.x.shape[1])])
        interaction_plot = shap_explainer.interaction_plot(self.feature_pairs_for_numerical_problem)
        summary_plot = shap_explainer.summary_plot()
        assert shap_explainer.shap_values.shape == (self.num_samples_numerical, test_problem.x.shape[1])

    def test_shap_mixed(self):
        test_problem = MixedTestProblem(self.num_samples_mixed)
        instances = test_problem.x
        ref_x = test_problem.x
        shap_explainer = ShapDisplay.from_surrogate_model(
            instances,
            test_problem.model,
            ref_x,
            feature_names=test_problem.feature_names,
            categorical_feature_indices=test_problem.categorical_feature_indices,
            categories_map=test_problem.categories_map,
        )
        assert shap_explainer.shap_values.shape == (self.num_samples_mixed, test_problem.x.shape[1])
        individual_plot = shap_explainer.individual_plot(index=self.index_for_individual_plot)
        dependence_plot = shap_explainer.dependence_plot([i for i in range(test_problem.x.shape[1])])
        interaction_plot = shap_explainer.interaction_plot(self.feature_pairs_for_mixed_problem)
        summary_plot = shap_explainer.summary_plot()


if __name__ == "__main__":
    unittest.main()
