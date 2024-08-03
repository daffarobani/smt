from smt.utils.sm_test_case import SMTestCase
from smt.problems import WingWeight
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG

from smt.applications.explainability_tools import PDFeatureInteractionDisplay

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
import itertools
import random


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


class TestPDInteraction(SMTestCase):
    def setUp(self):
        self.num_samples_numerical = 300
        self.num_samples_mixed = 100
        self.ratio_samples = 0.2
        self.num_feature_pairs = 3

    def test_pd_overall_interaction_numerical(self):
        test_problem = NumericalTestProblem(self.num_samples_numerical)
        overall_pd_interaction = PDFeatureInteractionDisplay.overall_interaction(
            test_problem.model,
            test_problem.x,
            feature_names=test_problem.feature_names,
        )
        fig = overall_pd_interaction.plot()
        assert len(overall_pd_interaction.h_scores) == test_problem.x.shape[1]

    def test_pd_pairwise_interaction_numerical(self):
        test_problem = NumericalTestProblem(self.num_samples_numerical)
        feature_pairs = list(itertools.combinations([i for i in range(test_problem.x.shape[1])], 2))
        random.shuffle(feature_pairs)
        feature_pairs = feature_pairs[:self.num_feature_pairs]

        pairwise_pd_interaction = PDFeatureInteractionDisplay.pairwise_interaction(
            test_problem.model,
            test_problem.x,
            feature_pairs,
            feature_names=test_problem.feature_names,
        )
        fig = pairwise_pd_interaction.plot()
        assert len(pairwise_pd_interaction.h_scores) == len(feature_pairs)

    def test_pd_overall_interaction_numerical_with_ratio_samples(self):
        test_problem = NumericalTestProblem(self.num_samples_numerical)
        overall_pd_interaction = PDFeatureInteractionDisplay.overall_interaction(
            test_problem.model,
            test_problem.x,
            feature_names=test_problem.feature_names,
            ratio_samples=self.ratio_samples
        )
        fig = overall_pd_interaction.plot()
        assert len(overall_pd_interaction.h_scores) == test_problem.x.shape[1]

    def test_pd_pairwise_interaction_numerical_with_ratio_samples(self):
        test_problem = NumericalTestProblem(self.num_samples_numerical)
        feature_pairs = list(itertools.combinations([i for i in range(test_problem.x.shape[1])], 2))
        random.shuffle(feature_pairs)
        feature_pairs = feature_pairs[:self.num_feature_pairs]

        pairwise_pd_interaction = PDFeatureInteractionDisplay.pairwise_interaction(
            test_problem.model,
            test_problem.x,
            feature_pairs,
            feature_names=test_problem.feature_names,
            ratio_samples=self.ratio_samples
        )
        fig = pairwise_pd_interaction.plot()
        assert len(pairwise_pd_interaction.h_scores) == len(feature_pairs)

    def test_pd_overall_interaction_mixed(self):
        test_problem = MixedTestProblem(self.num_samples_mixed)
        overall_pd_interaction = PDFeatureInteractionDisplay.overall_interaction(
            test_problem.model,
            test_problem.x,
            feature_names=test_problem.feature_names,
            categorical_feature_indices=test_problem.categorical_feature_indices
        )
        fig = overall_pd_interaction.plot()
        assert len(overall_pd_interaction.h_scores) == test_problem.x.shape[1]

    def test_pd_pairwise_interaction_mixed(self):
        test_problem = MixedTestProblem(self.num_samples_mixed)
        feature_pairs = list(itertools.combinations([i for i in range(test_problem.x.shape[1])], 2))
        random.shuffle(feature_pairs)
        feature_pairs = feature_pairs[:self.num_feature_pairs]

        pairwise_pd_interaction = PDFeatureInteractionDisplay.pairwise_interaction(
            test_problem.model,
            test_problem.x,
            feature_pairs,
            feature_names=test_problem.feature_names,
            categorical_feature_indices=test_problem.categorical_feature_indices,
        )
        fig = pairwise_pd_interaction.plot()
        assert len(pairwise_pd_interaction.h_scores) == len(feature_pairs)

    def test_pd_overall_interaction_mixed_with_ratio_samples(self):
        test_problem = MixedTestProblem(self.num_samples_mixed)
        overall_pd_interaction = PDFeatureInteractionDisplay.overall_interaction(
            test_problem.model,
            test_problem.x,
            feature_names=test_problem.feature_names,
            categorical_feature_indices=test_problem.categorical_feature_indices,
            ratio_samples=self.ratio_samples
        )
        fig = overall_pd_interaction.plot()
        assert len(overall_pd_interaction.h_scores) == test_problem.x.shape[1]

    def test_pd_pairwise_interaction_mixed_with_ratio_samples(self):
        test_problem = MixedTestProblem(self.num_samples_mixed)
        feature_pairs = list(itertools.combinations([i for i in range(test_problem.x.shape[1])], 2))
        random.shuffle(feature_pairs)
        feature_pairs = feature_pairs[:self.num_feature_pairs]

        pairwise_pd_interaction = PDFeatureInteractionDisplay.pairwise_interaction(
            test_problem.model,
            test_problem.x,
            feature_pairs,
            feature_names=test_problem.feature_names,
            categorical_feature_indices=test_problem.categorical_feature_indices,
            ratio_samples=self.ratio_samples,
        )
        fig = pairwise_pd_interaction.plot()
        assert len(pairwise_pd_interaction.h_scores) == len(feature_pairs)


if __name__ == "__main__":
    unittest.main()
