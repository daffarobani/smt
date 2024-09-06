from smt.utils.sm_test_case import SMTestCase
from smt.problems import WingWeight
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
from smt.applications.explainability_tools import PartialDependenceDisplay

import unittest


class TestPartialDependenceNumerical(SMTestCase):
    def setUp(self):
        nsamples = 300
        grid_resolution_1d = 100
        grid_resolution_2d = 25
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
        x = sampling(nsamples)
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
        self.nsamples = nsamples
        self.grid_resolution_1d = grid_resolution_1d
        self.grid_resolution_2d = grid_resolution_2d

    def test_ice_one_dimension(self):
        features = [i for i in range(self.x.shape[1])]
        pdd = PartialDependenceDisplay.from_surrogate_model(
            self.model,
            self.x,
            features,
            feature_names=self.feature_names,
            grid_resolution=self.grid_resolution_1d,
            kind='individual',
        )
        pdd_fig = pdd.plot(centered=True)
        pd_results = pdd.pd_results

        assert len(pd_results) == len(features)
        for i in range(len(pd_results)):
            assert set(pd_results[i].keys()) == {"grid_values", "individual"}
            assert len(pd_results[i]["grid_values"]) == 1
            assert pd_results[i]["grid_values"][0].shape == (self.grid_resolution_1d, )
            assert pd_results[i]["individual"].shape == (self.nsamples, self.grid_resolution_1d)

    def test_pdp_one_dimension(self):
        features = [i for i in range(self.x.shape[1])]
        pdd = PartialDependenceDisplay.from_surrogate_model(
            self.model,
            self.x,
            features,
            feature_names=self.feature_names,
            grid_resolution=self.grid_resolution_1d,
            kind='average',
        )
        pdd_fig = pdd.plot(centered=True)
        pd_results = pdd.pd_results

        assert len(pd_results) == len(features)
        for i in range(len(pd_results)):
            assert set(pd_results[i].keys()) == {"grid_values", "average"}
            assert len(pd_results[i]["grid_values"]) == 1
            assert pd_results[i]["grid_values"][0].shape == (self.grid_resolution_1d,)
            assert pd_results[i]["average"].shape == (self.grid_resolution_1d, )

    def test_pdp_ice_one_dimension(self):
        features = [i for i in range(self.x.shape[1])]
        pdd = PartialDependenceDisplay.from_surrogate_model(
            self.model,
            self.x,
            features,
            feature_names=self.feature_names,
            grid_resolution=self.grid_resolution_1d,
            kind='both',
        )
        pdd_fig = pdd.plot(centered=True)
        pd_results = pdd.pd_results

        assert len(pd_results) == len(features)
        for i in range(len(pd_results)):
            assert set(pd_results[i].keys()) == {"grid_values", "average", "individual"}
            assert len(pd_results[i]["grid_values"]) == 1
            assert pd_results[i]["grid_values"][0].shape == (self.grid_resolution_1d,)
            assert pd_results[i]["average"].shape == (self.grid_resolution_1d,)
            assert pd_results[i]["individual"].shape == (self.nsamples, self.grid_resolution_1d)

    def test_pdp_two_dimension(self):
        features = [(0, 1), (2, 3)]
        pdd = PartialDependenceDisplay.from_surrogate_model(
            self.model,
            self.x,
            features,
            feature_names=self.feature_names,
            grid_resolution=self.grid_resolution_2d,
        )
        pdd_fig = pdd.plot(centered=True)
        pd_results = pdd.pd_results

        assert len(pd_results) == len(features)
        for i in range(len(pd_results)):
            assert set(pd_results[i].keys()) == {"grid_values", "average"}
            assert len(pd_results[i]["grid_values"]) == 2
            for j in range(len(pd_results[i]["grid_values"])):
                assert pd_results[i]["grid_values"][j].shape == (self.grid_resolution_2d,)

            assert pd_results[i]["average"].shape == (self.grid_resolution_2d, self.grid_resolution_2d)

    @staticmethod
    def run_pd_numerical_example():
        from smt.problems import WingWeight
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG
        from smt.applications.explainability_tools import PartialDependenceDisplay

        nsamples = 100
        grid_resolution_2d = 10
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
        x = sampling(nsamples)
        y = fun(x)

        feature_names = [
            r'$S_{w}$', r'$W_{fw}$', r'$A$', r'$\Delta$',
            r'$q$', r'$\lambda$', r'$t_{c}$', r'$N_{z}$',
            r'$W_{dg}$', r'$W_{p}$',
        ]

        model = KRG(
            theta0=[1e-2] * x.shape[1],
            print_prediction=False
        )
        model.set_training_values(x, y)
        model.train()

        features = [(0, 1), (2, 3)]
        pdd = PartialDependenceDisplay.from_surrogate_model(
            model,
            x,
            features,
            feature_names=feature_names,
            grid_resolution=grid_resolution_2d,
        )
        pdd_fig = pdd.plot(centered=True)
        #
        # pdd_fig


if __name__ == "__main__":
    unittest.main()
