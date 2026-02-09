import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
#import arviz as az
#import xarray as xr
#import matplotlib.pyplot as plt

from pymc.pytensorf import collect_default_updates
from statsmodels.tsa.ar_model import AutoReg

import utils as funcs
from tools import DataIn


def get_eb_estimates(data: DataIn, model, return_model=False):
    df = data.get_df4

    if model == "statsmodel":
        model_fit = AutoReg(df.sales, lags=1).fit()
        params_list = np.array(model_fit.params)   # [intercept, rho]
        sigma = np.sqrt(model_fit.sigma2)
        eta = params_list[0] / (1 - params_list[1])

        if return_model:
            return model_fit
        return {"coefs": params_list, "eta": eta, "sigma": sigma}

    elif model == "pymc":
        with pm.Model() as AR1:
            coefs = pm.Normal("coefs", 0, 2, size=2)
            init = pm.Normal.dist(1, size=1)
            sigma = pm.Exponential("sigma", 1)
            ar1 = pm.AR("ar1", coefs, sigma=sigma, init_dist=init, constant=True, observed=df.sales)
            trace = pm.find_MAP()
        eta = trace["coefs"][0] / (1 - trace["coefs"][1])
        trace["eta"] = eta
        return trace



def ar_dist(ar_init, eta, rho, sigma, nsteps, size=None):
    def ar_step(x_tm1, eta, rho, sigma):
        mu = eta * (1 - rho) + x_tm1 * rho
        eps = pm.Normal.dist(0, 1)
        x = mu + eps * sigma
        return x, collect_default_updates(outputs=[x])

    ar_innov, _ = pytensor.scan(
        fn=ar_step,
        outputs_info=[{"initial": ar_init, "taps": range(-1, 0)}],
        non_sequences=[eta, rho, sigma],
        n_steps=nsteps,
        strict=True,
    )
    return ar_innov


def multiple_did_outcome(b0, b1, b2, b3, t, group, treated):
    return b0 + (b1 * group) + b2 * t + (b3 * treated)


class BaseDidAr:
    def fit(self, data: DataIn, eb_model="statsmodel", draws=1000):
        self.df = data.get_df3
        self.df4 = data.get_df4
        self.initial_point = data.get_last_diff
        self.eb_estimates = get_eb_estimates(data, model=eb_model)

        coords = {"obs_idx": self.df.index.values}
        self.model = pm.Model()
        self.model.add_coords(coords=coords)

        with self.model:
            t = pm.intX(pm.MutableData("t", self.df["period"].values, dims="obs_idx"))
            treated = pm.MutableData("treated", self.df["treated"].values, dims="obs_idx")
            group = pm.MutableData("locale", self.df["locale"].values, dims="obs_idx")

            _b0 = pm.HalfNormal("b0", 8)
            _b1 = pm.HalfNormal("b1", 2)
            _b2 = pm.Normal("b2", 0, 0.2)
            sigma = pm.HalfNormal("sigma", 5)

            _att = pm.Normal("att", 0, 2.5)

            self._summ_xi = self.get_summ_xi(self.model, t, data)

            if bool(pt.eq(self._summ_xi.ndim, 0).eval()):
                _b3 = pm.Deterministic("b3", _att)
                _psi = pm.Deterministic("psi", _b3 - self._summ_xi)
            else:
                _b3 = pm.Deterministic("b3", _att)
                _psi = pm.Deterministic("psi", _b3 - self._summ_xi[t])

            mu = pm.Deterministic(
                "mu",
                multiple_did_outcome(_b0, _b1, _b2, _b3, t, group, treated),
                dims="obs_idx",
            )

            pm.Normal("obs", mu, sigma, observed=self.df["sales"].values)

            idata = pm.sample(
                draws=draws,
                chains=4,
                nuts_sampler="nutpie",
                progressbar=True,
            )
            idata.attrs["name"] = self.name
            idata.attrs["title"] = self.title

            return pm.sample_posterior_predictive(idata, extend_inferencedata=True)


class ParallelTrendsHold(BaseDidAr):
    def __init__(self, name, title):
        self.name = name
        self.title = title

    def get_summ_xi(self, model, t, data):
        with model:
            _xi = pt.constant(0.0, name="xi")
            _summ_xi = pm.Deterministic("_sum_xi", _xi)
        return _summ_xi


class FixedRhoSigma(BaseDidAr):
    def __init__(self, name, title, rho_value=0.95, sigma_value=0.001):
        self.name = name
        self.title = title
        self.rho_value = rho_value
        self.sigma_value = sigma_value

    def get_summ_xi(self, model, t, data):
        with model:
            eta = pm.Uniform("eta", 0.1, 0.9)
            rho = pt.constant(self.rho_value, name="rho")
            ar_sigma = pt.constant(self.sigma_value, name="ar_sigma")

            sigma_pre = self.df4["sales"].std()
            last_pre_value = self.df4["sales"].values[-1]
            ar_init = pm.Normal("ar_init", mu=last_pre_value, sigma=sigma_pre, shape=(1,))

            ar_innov = pm.CustomDist("ar_dist", ar_init, eta, rho, ar_sigma, 14, dist=ar_dist)

            _xi_pre = pt.as_tensor_variable(self.df4["sales"].values)
            _xi_post = pm.Deterministic("xi_post", pt.concatenate([_xi_pre, ar_innov], axis=-1))

            _xi = pm.Deterministic("xi", _xi_post[1:] - _xi_post[:-1])
            _summ_xi = pm.Deterministic("_sum_xi", pt.cumsum(_xi))
        return _summ_xi

    
class UniEta(BaseDidAr):
    def __init__(self, name, title, sigma_scale=1.0):
        self.name = name
        self.title = title
        self.sigma_scale = sigma_scale

    def get_summ_xi(self, model, t, data):
        with model:
            eta = pm.Uniform("eta", 0.1, 0.9)
            rho = pm.Beta("rho", 2, 2)
            ar_sigma = pm.HalfNormal("ar_sigma", 1 * self.sigma_scale)

            sigma_pre = self.df4["sales"].std()
            last_pre_value = self.df4["sales"].values[-1]
            ar_init = pm.Normal("ar_init", mu=last_pre_value, sigma=sigma_pre, shape=(1,))

            ar_innov = pm.CustomDist("ar_dist", ar_init, eta, rho, ar_sigma, 14, dist=ar_dist)

            _xi_pre = pt.as_tensor_variable(self.df4["sales"].values)
            _xi_post = pm.Deterministic("xi_post", pt.concatenate([_xi_pre, ar_innov], axis=-1))

            _xi = pm.Deterministic("xi", _xi_post[1:] - _xi_post[:-1])
            _summ_xi = pm.Deterministic("_sum_xi", pt.cumsum(_xi))
        return _summ_xi



class EBays(BaseDidAr):
    def __init__(self, name, title, eta_value, rho_value, sigma_value=0.166):
        self.name = name
        self.title = title
        self.eta_value = eta_value
        self.rho_value = rho_value
        self.sigma_value = sigma_value

    def get_summ_xi(self, model, t, data):
        with model:
            eta = pt.constant(self.eta_value, name="eta")
            rho = pt.constant(self.rho_value, name="rho")
            ar_sigma = pt.constant(self.sigma_value, name="ar_sigma")

            sigma_pre = self.df4["sales"].std()
            last_pre_value = self.df4["sales"].values[-1]
            ar_init = pm.Normal("ar_init", mu=last_pre_value, sigma=sigma_pre, shape=(1,))

            ar_innov = pm.CustomDist("ar_dist", ar_init, eta, rho, ar_sigma, 14, dist=ar_dist)

            _xi_pre = pt.as_tensor_variable(self.df4["sales"].values)
            _xi_post = pm.Deterministic("xi_post", pt.concatenate([_xi_pre, ar_innov], axis=-1))

            _xi = pm.Deterministic("xi", _xi_post[1:] - _xi_post[:-1])
            _summ_xi = pm.Deterministic("_sum_xi", pt.cumsum(_xi))
            return _summ_xi
