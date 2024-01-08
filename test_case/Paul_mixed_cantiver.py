#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:46:41 2022

@author: psaves
"""
import unittest
import numpy as np
import matplotlib

matplotlib.use("Agg")
from smt.applications.mixed_integer import (
    MixedIntegerSurrogateModel,
    MixedIntegerKrigingModel,
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
)
from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import (
    KRG,
    QP,
    KPLS,
)
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
    HAS_CONFIG_SPACE,
)
import smt.utils.design_space as ds

from smt.sampling_methods import LHS
from smt.surrogate_models import (
    KRG,
    KPLS,
    QP,
    MixIntKernelType,
    MixHrcKernelType,
)
import itertools

listI = [
    0.0833,
    0.139,
    0.380,
    0.0796,
    0.133,
    0.363,
    0.0859,
    0.136,
    0.360,
    0.0922,
    0.138,
    0.369,
]


def y(X):
    I = np.int64(X[0])
    L = X[1]
    S = X[2]
    Ival = listI[I]
    E = 200e9
    P = 50e3
    y = (P * L**3) / (3 * E * S**2 * Ival)
    return y


f_obj = y


n_doe = 42


ds = DesignSpace(
     [
         CategoricalVariable(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]),
         FloatVariable(10.0, 20.0),
         FloatVariable(1.0, 2.0),
     ]
 )

mixint = MixedIntegerContext(ds)

xdoe, is_acting_sampled = ds.sample_valid_x(100)
# print(xdoe)

y_doe = [f_obj(xdoe[i]) for i in range(len(xdoe))]
# Surrogate

sm = mixint.build_kriging_model(KRG(print_prediction=False))

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
        ),
)

import time

start_time = time.time()


sm.set_training_values(xdoe, np.array(y_doe))
sm.train()
print("run time (s):", time.time() - start_time)


RMSE = np.zeros(12)
for j in range(12):
    npt = 30
    x0 = np.zeros((npt * npt, 3))
    x0[:, 0] = j
    a = np.linspace(10, 20, npt)
    b = np.linspace(1, 2, npt)
    u = np.array(list(itertools.product(a, b)))[:, 0]
    v = np.array(list(itertools.product(a, b)))[:, 1]
    x0[:, 1] = u
    x0[:, 2] = v
    y0 = sm.predict_values(x0)
    x1 = x0
    y1 = [f_obj(x1[i]) for i in range(len(x1))]
    RMSE[j] = np.linalg.norm(np.abs((y1 - y0.T)[0]))

print(np.linalg.norm(RMSE) * 100)
# print(sm._surrogate.coeff_pls)
print(np.shape(sm._surrogate.optimal_theta))
print(sm._surrogate.optimal_theta)
print(sm._surrogate.optimal_rlf_value)
