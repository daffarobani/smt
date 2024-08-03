"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

import unittest
from smt.utils.sm_test_case import SMTestCase
import numpy as np
from scipy import special
from smt.applications import PODI
from smt.sampling_methods import LHS


def cos_coeff(i: int, x: np.ndarray):
    """Generates the i-th coefficient for the one-dimension problem."""

    a = 2 * i % 2 - 1
    return a * x[:, 0] * np.cos(i * x[:, 0])


def cos_coeff_nd(i: int, x: np.ndarray):
    """Generates the i-th coefficient for the multi dimensions problem."""
    a = 2 * i % 2 - 1
    return a * sum(x.T) * np.cos(i * sum(x.T))


def Legendre(i: int, t: np.ndarray):
    """Generates the i-th Legendre's polynom and returns its values at input values."""

    return special.legendre(i)(t)


class Test(SMTestCase):
    """Class to test Proper Orthogonal Decomposition and Interpolation (PODI) surrogate models based."""

    def setUp(self):
        """Sets up the test case for the tests."""

        self.full_database = self.pb_1d()
        self.database = self.full_database[:, : self.nt]

    @staticmethod
    def gram_schmidt(input_array: np.ndarray) -> np.ndarray:
        """Static method that performs the  Gram-Schmidt's algorithm."""

        basis = np.zeros_like(input_array)
        for i in range(len(input_array)):
            basis[i] = input_array[i]
            for j in range(i):
                basis[i] -= (
                    np.dot(input_array[i], basis[j])
                    / np.dot(basis[j], basis[j])
                    * basis[j]
                )
            basis[i] /= np.linalg.norm(basis[i])
        return basis

    @staticmethod
    def check_projection(
        basis_original: np.ndarray, basis_pod: np.ndarray
    ) -> np.ndarray:
        """
        Computes the residue's norm of the projection of the pod basis on the subspace generated by the problem basis.

        Parameters
        ----------
        basis_original : np.ndarray

        basis_pod : np.ndarray

        Returns
        -------
        norm_residue : np.ndarray
            norm of the left residue
        """

        norm_residue = np.zeros(basis_pod.shape[1])

        projection = basis_original.dot(np.dot(basis_original.T, basis_pod))

        for i in range(projection.shape[1]):
            proj = projection[:, i]
            norm_residue[i] = np.linalg.norm(basis_pod[:, i] - proj)
        return norm_residue

    def pb_1d(self, nt=40, nv=20) -> np.ndarray:
        """
        Constructs the one-dimension problem

        Parameters
        ----------
        nt : int
            Number of training values desired (number of snapshot).

        Returns
        -------
        database : np.ndarray
            Snapshot matrix, each row corresponds to the values of our problem at a specific snapshot.
        """
        self.seed = 42

        self.ny = 100
        self.t = np.linspace(-1, 1, self.ny)
        self.n_modes_test = 10

        xlimits = np.array([[0, 4]])
        sampling = LHS(xlimits=xlimits, random_state=self.seed)
        self.nt = nt
        self.xt = sampling(self.nt)
        self.nn = 15
        self.xn = sampling(self.nn)
        self.nv = nv
        sampling_new = LHS(xlimits=xlimits, random_state=self.seed + 1)
        self.xv = sampling_new(self.nv)
        self.x = np.concatenate((self.xt, self.xv))

        u0 = np.zeros((self.ny, 1))

        alpha = np.zeros((self.x.shape[0], self.n_modes_test))
        for i in range(self.n_modes_test):
            alpha[:, i] = cos_coeff(i, self.x)

        V_init = np.zeros((self.ny, self.n_modes_test))
        for i in range(self.n_modes_test):
            V_init[:, i] = Legendre(i, self.t)

        V = Test.gram_schmidt(V_init.T).T
        database = u0 + np.dot(V, alpha.T)
        self.basis_original = V

        return database

    def pb_2d_local(self, nt1=10, nt2=10) -> np.ndarray:
        """
        Constructs the two-dimension problem adapted for local POD (can be used for global POD as well).

        Parameters
        ----------
        nt1 : int
            Number of values for the first dimension of the inputs.
            Each one of these values will correspond to a line in the DoE.
        nt2 : int
            Number of values for the second dimension of the inputs.

        Returns
        -------
        database : np.ndarray
            Snapshot matrix, each row corresponds to the values of our problem at a specific snapshot.
        """
        self.seed = 42

        self.ny = 100
        self.t = np.linspace(-1, 1, self.ny)
        self.n_modes_test = 10

        xlimits = [[0, 1], [0, 4]]
        sampling_x1 = LHS(xlimits=np.array([xlimits[0]]), random_state=self.seed)
        sampling_x2 = LHS(xlimits=np.array([xlimits[1]]), random_state=self.seed + 1)

        self.nt1 = nt1
        self.nt2 = nt2
        self.nt = self.nt1 * self.nt2
        self.xt1 = sampling_x1(self.nt1)
        self.xt2 = sampling_x2(self.nt)
        self.xt = np.zeros((self.nt, 2))
        self.xt[:, 1] = self.xt2[:, 0]
        for i, elt in enumerate(self.xt1):
            self.xt[i * self.nt2 : (i + 1) * self.nt2, 0] = elt

        sampling_new = LHS(xlimits=np.array(xlimits), random_state=self.seed)

        self.nn = 15
        self.xn = sampling_new(self.nn)
        self.nv = 10
        self.xv = sampling_new(self.nv)
        self.x = np.concatenate((self.xt, self.xv))

        u0 = np.zeros((self.ny, 1))

        alpha = np.zeros((self.x.shape[0], self.n_modes_test))
        for i in range(self.n_modes_test):
            alpha[:, i] = cos_coeff_nd(i, self.x)

        V_init = np.zeros((self.ny, self.n_modes_test))
        for i in range(self.n_modes_test):
            V_init[:, i] = Legendre(i, self.t)

        V = Test.gram_schmidt(V_init.T).T
        database = u0 + np.dot(V, alpha.T)
        self.basis_original = V

        return database

    def test_predict(self):
        """Tests the predict methods."""

        sm = PODI()

        sm.compute_pod(self.database, tol=1, seed=self.seed)
        sm.set_interp_options("KRG")
        sm.set_training_values(self.xt)

        for predict_method in [sm.predict_values, sm.predict_variances]:
            error_msg = f"It should not be possible to call {predict_method.__name__} before the training."
            with self.assertRaises(RuntimeError, msg=error_msg):
                predict_method(self.xn)

        for predict_method in [sm.predict_derivatives, sm.predict_variance_derivatives]:
            error_msg = f"It should not be possible to call {predict_method.__name__} before the training."
            with self.assertRaises(RuntimeError, msg=error_msg):
                predict_method(self.xn, 0)

        sm.train()

        error_msg = "It should not be possible to make a prediction with incorrect dimension input."
        for predict_method in [sm.predict_values, sm.predict_variances]:
            with self.assertRaises(ValueError, msg=error_msg):
                predict_method(np.ones((1, self.xn.shape[1] + 1)))

        for predict_method in [sm.predict_derivatives, sm.predict_variance_derivatives]:
            with self.assertRaises(ValueError, msg=error_msg):
                predict_method(np.ones((1, self.xn.shape[1] + 1)), 0)

        error_msg = "It should not be possible to predict a derivative out of the input's dimensions."
        for predict_method in [sm.predict_derivatives, sm.predict_variance_derivatives]:
            with self.assertRaises(ValueError, msg=error_msg):
                predict_method(self.xn, self.xn.shape[1] + 1)

        var_xt = sm.predict_variances(self.xt)

        np.testing.assert_allclose(var_xt, np.zeros(var_xt.shape), atol=1e-6)

        mean_xn = sm.predict_values(self.xn)
        var_xn = sm.predict_variances(self.xn)
        deriv_xn = sm.predict_derivatives(self.xn, 0)
        var_deriv_xn = sm.predict_variance_derivatives(self.xn, 0)

        self.assertEqual(mean_xn.shape, (self.ny, self.nn))
        self.assertEqual(var_xn.shape, (self.ny, self.nn))
        self.assertEqual(deriv_xn.shape, (self.ny, self.nn))
        self.assertEqual(var_deriv_xn.shape, (self.ny, self.nn))

        mean_xv = sm.predict_values(self.xv)

        diff = self.full_database[:, self.nt :] - mean_xv
        rms_error = [np.sqrt(np.mean(diff[:, i] ** 2)) for i in range(diff.shape[1])]

        np.testing.assert_allclose(rms_error, np.zeros(self.nv), atol=1e-2)

    def test_set_options(self):
        """Tests the method that sets the interpolations options settings."""

        sm = PODI()

        error_msg = "It should not be possible to call 'set_interp_options' before 'compute_pod'."
        with self.assertRaises(RuntimeError, msg=error_msg):
            sm.set_interp_options()

        sm.compute_pod(self.database, n_modes=2, seed=self.seed)

        error_msg = (
            "It should not be possible to initialize an unavailable surrogate model."
        )
        with self.assertRaises(ValueError, msg=error_msg):
            sm.set_interp_options("non existing surrogate model")

        error_msg = (
            "It should not be possible to use a non-valid size for the list of options."
        )
        with self.assertRaises(ValueError, msg=error_msg):
            sm.set_interp_options("KRG", [{}, {}, {}])

        options_global = [
            {
                "poly": "quadratic",
                "corr": "matern32",
                "pow_exp_power": 0.38,
                "theta0": [1e-1],
            }
        ]
        sm.set_interp_options("KRG", options_global)

        sm_list = sm.get_interp_coeff()
        for interp_coeff in sm_list:
            for key in options_global[0].keys():
                self.assertEqual(interp_coeff.options[key], options_global[0][key])

        options_local = [{"poly": "quadratic"}, {"corr": "matern52"}]
        sm.set_interp_options("KRG", options_local)

        sm_list = sm.get_interp_coeff()
        for i, interp_coeff in enumerate(sm_list):
            for key in options_local[i].keys():
                self.assertEqual(interp_coeff.options[key], options_local[i][key])

    def test_global_pod(self):
        """Tests the computing of the global pod."""

        sm = PODI()

        error_msg = "It should not be possible to execute compute_pod without tol or n_modes argument."
        with self.assertRaises(ValueError, msg=error_msg):
            sm.compute_pod(self.database, seed=self.seed)

        error_msg = "It should not be possible to execute compute_pod with both tol and n_modes arguments."
        with self.assertRaises(ValueError, msg=error_msg):
            sm.compute_pod(self.database, tol=0.1, n_modes=1, seed=self.seed)

        error_msg = "It should not be possible to execute compute_pod with more mods than data values."
        with self.assertRaises(ValueError, msg=error_msg):
            sm.compute_pod(self.database, n_modes=self.nt + 1, seed=self.seed)

        sm.compute_pod(self.database, tol=1, seed=self.seed)
        self.assertEqual(sm.get_ev_ratio(), 1)

        n_modes = sm.get_n_modes()
        self.assertLessEqual(n_modes, self.n_modes_test)

        basis_pod = sm.get_singular_vectors()

        self.assertEqual(basis_pod.shape, (self.ny, min(self.nt, self.ny)))

        singular_values = sm.get_singular_values()
        self.assertEqual(len(singular_values), min(self.nt, self.ny))
        np.testing.assert_allclose(
            singular_values[n_modes:],
            np.zeros(min(self.nt, self.ny) - n_modes),
            atol=1e-6,
        )

        norm_residue = Test.check_projection(self.basis_original, basis_pod)
        np.testing.assert_allclose(norm_residue[:n_modes], np.zeros(n_modes), atol=1e-6)

        # local pod
        error_msg = "It should not be possible to compute the local pod with a local basis with incorrect dimensions."
        with self.assertRaises(ValueError, msg=error_msg):
            sm.compute_pod(
                database=self.database, pod_type="local", local_basis=basis_pod[1:, :]
            )
        sm.compute_pod(database=self.database, pod_type="local", local_basis=basis_pod)

    def test_interp_subspaces(self):
        xt1 = np.array([[1], [2]])

        m1 = np.array([[1, 1], [1, 2]])
        m2 = np.array([[1, 1], [2, 1]])
        input_matrices = [m1, m2]

        xn1 = np.array([[1.5]])

        error_msg = "It should not be possible to interpolate subspaces if the bases don't have the same dimensions."
        with self.assertRaises(ValueError, msg=error_msg):
            PODI.interp_subspaces(xt1=xt1, input_matrices=[m1, m2[:1]], xn1=xn1)

        error_msg = "It should not be possible to interpolate if there are not as much bases than assiociated values."
        with self.assertRaises(ValueError, msg=error_msg):
            PODI.interp_subspaces(xt1=xt1[0], input_matrices=input_matrices, xn1=xn1)

        output_list = PODI.interp_subspaces(
            xt1=xt1, input_matrices=input_matrices, xn1=xn1, print_global=False
        )
        self.assertEqual(len(output_list), xn1.shape[0])
        for basis in output_list:
            self.assertEqual(basis.shape, input_matrices[0].shape)

        # with realizations
        n_realizations = 10
        output_list, realization_list = PODI.interp_subspaces(
            xt1=xt1,
            input_matrices=input_matrices,
            xn1=xn1,
            compute_realizations=True,
            n_realizations=n_realizations,
            print_global=False,
        )
        self.assertEqual(len(output_list), xn1.shape[0])
        for basis in output_list:
            self.assertEqual(basis.shape, input_matrices[0].shape)
        self.assertEqual(len(realization_list), xn1.shape[0])
        for realization in realization_list:
            self.assertEqual(len(realization), n_realizations)
            for basis in realization:
                self.assertEqual(basis.shape, input_matrices[0].shape)

    def test_set_training_train(self):
        """Tests the set_training_values and train methods."""
        sm = PODI()

        error_msg = (
            "It should not be possible to set training values before computing the pod."
        )
        with self.assertRaises(RuntimeError, msg=error_msg):
            sm.set_training_values(self.xt)

        error_msg = (
            "It should not be possible to train the model before computing the pod."
        )
        with self.assertRaises(RuntimeError, msg=error_msg):
            sm.train()

        sm.compute_pod(self.database, n_modes=1, seed=self.seed)

        error_msg = (
            "It should not be possible to set training values with incorrect size."
        )
        with self.assertRaises(ValueError, msg=error_msg):
            sm.set_training_values(self.xt[1:])

        error_msg = (
            "It should not be possible to train the model before setting train values."
        )
        with self.assertRaises(RuntimeError, msg=error_msg):
            sm.train()

    @staticmethod
    def run_podi_example_1d():
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications import PODI
        from smt.sampling_methods import LHS

        light_pink = np.array((250, 233, 232)) / 255

        p = 100
        t = np.linspace(-1, 1, p)
        n_modes_test = 10

        def function_test_1d(x, t, n_modes_test, p):
            import numpy as np  # Note: only required by SMT doc testing toolchain

            def cos_coeff(i: int, x: np.ndarray):
                a = 2 * i % 2 - 1
                return a * x[:, 0] * np.cos(i * x[:, 0])

            def Legendre(i: int, t: np.ndarray):
                from scipy import special

                return special.legendre(i)(t)

            def gram_schmidt(input_array: np.ndarray) -> np.ndarray:
                """To perform the  Gram-Schmidt's algorithm."""

                basis = np.zeros_like(input_array)
                for i in range(len(input_array)):
                    basis[i] = input_array[i]
                    for j in range(i):
                        basis[i] -= (
                            np.dot(input_array[i], basis[j])
                            / np.dot(basis[j], basis[j])
                            * basis[j]
                        )
                    basis[i] /= np.linalg.norm(basis[i])
                return basis

            u0 = np.zeros((p, 1))

            alpha = np.zeros((x.shape[0], n_modes_test))
            for i in range(n_modes_test):
                alpha[:, i] = cos_coeff(i, x)

            V_init = np.zeros((p, n_modes_test))
            for i in range(n_modes_test):
                V_init[:, i] = Legendre(i, t)

            V = gram_schmidt(V_init.T).T
            database = u0 + np.dot(V, alpha.T)

            return database

        seed_sampling = 42
        xlimits = np.array([[0, 4]])
        sampling = LHS(xlimits=xlimits, random_state=seed_sampling)

        nt = 40
        xt = sampling(nt)

        nv = 400
        xv = sampling(nv)

        x = np.concatenate((xt, xv))
        dbfull = function_test_1d(x, t, n_modes_test, p)

        # Training data
        dbt = dbfull[:, :nt]

        # Validation data
        dbv = dbfull[:, nt:]

        podi = PODI()
        seed_pod = 42
        podi.compute_pod(dbt, tol=0.9999, seed=seed_pod)
        podi.set_training_values(xt)
        podi.train()

        values = podi.predict_values(xv)
        variances = podi.predict_variances(xv)

        # Choosing a value from the validation inputs
        i = nv // 2

        diff = dbv[:, i] - values[:, i]
        rms_error = np.sqrt(np.mean(diff**2))
        plt.figure(figsize=(8, 5))
        light_pink = np.array((250, 233, 232)) / 255
        plt.fill_between(
            np.ravel(t),
            np.ravel(values[:, i] - 3 * np.sqrt(variances[:, i])),
            np.ravel(values[:, i] + 3 * np.sqrt(variances[:, i])),
            color=light_pink,
            label="confiance interval (99%)",
        )
        plt.scatter(
            t,
            values[:, i],
            color="r",
            marker="x",
            s=15,
            alpha=1.0,
            label="prediction (mean)",
        )
        plt.scatter(
            t,
            dbv[:, i],
            color="b",
            marker="*",
            s=5,
            alpha=1.0,
            label="reference",
        )
        plt.plot([], [], color="w", label="rms = " + str(round(rms_error, 9)))

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)

        plt.ylabel("u(x = " + str(xv[i, 0])[:4] + ")")
        plt.title("Estimation of u at x = " + str(xv[i, 0])[:4])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()