
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.blocking import RaveledVars
from scipy.stats import t, norm
import itertools
import scipy


def create_one_sales_column(df):
    yr2016 = df[['ID', 'city', 'period',
                 'PreTaxTarget']].rename(columns={
                     'city': 'locale',
                     'PreTaxTarget': 'sales'
                 })
    yr2017 = df[['ID', 'city', 'period',
                 'PostTaxTarget']].rename(columns={
                     'city': 'locale',
                     'PostTaxTarget': 'sales'
                 })

    yr2017['period'] = yr2017.period + 13

    comb = pd.concat([yr2016, yr2017], axis=0)
    return comb


def add_treated_column(df):
    res = df.copy()
    res['treated'] = np.where(
        (df.locale.isin(['Philadelphia', 'Border Counties'])) &
        (df.period >= 14), 1, 0)
    return res

def data_preprocess_no_average(input_data):
    temp = create_one_sales_column(input_data)
    temp = add_treated_column(temp)
    
    balt_philly = temp[temp.locale.isin(['Baltimore', 'Philadelphia'])]
    border = temp[temp.locale.isin(
        ['Border Counties', 'Non-Border Counties'])]

    # create new baltimore + non-border column before averaging
    temp3 = temp[temp != 'Philadelphia']
    temp3['locale'] = np.where(
        temp3.locale.isin(['Baltimore', 'Non-Border Counties']),
        'Baltimore & non-border', temp3.locale)
    border_v_non_border_and_balt = temp3

    return balt_philly, border, border_v_non_border_and_balt

def did(df: pd.DataFrame) -> float:
    diff_control = (
    df.loc[(df["period"] == 1) & (df["locale"] == 0)]["sales"].mean()
    - df.loc[(df["period"] == 0) & (df["locale"] == 0)]["sales"].mean()
    )
    print(f"Pre/post difference in Baltimore = {diff_control:.2f}")

    diff_treat = (
        df.loc[(df["period"] == 1) & (df["locale"] == 1)]["sales"].mean()
        - df.loc[(df["period"] == 0) & (df["locale"] == 1)]["sales"].mean()
    )

    print(f"Pre/post difference in Philadelphia = {diff_treat:.2f}")

    diff_in_diff = diff_treat - diff_control
    print(f"Difference in differences = {diff_in_diff:.2f}")
    return diff_in_dff


def sample_from_model(df: pd.DataFrame, xi_sd: float) -> az.InferenceData:

    def outcome(t, control_intercept, treat_intercept_delta, trend, group, treated, b3):
        return control_intercept + (treat_intercept_delta * group) + (t * trend) + (b3 * treated)
    
    with pm.Model() as model:
        # data
        t = pm.MutableData("t", df["period"].values, dims="obs_idx")
        treated = pm.MutableData("treated", df["treated"].values, dims="obs_idx")
        group = pm.MutableData("locale", df["locale"].values, dims="obs_idx")
        # priors
        _control_intercept = pm.HalfNormal("control_intercept", 15)
        _treat_intercept_delta = pm.Normal("philadelphia_effect", 0, 2.5)
        _trend = pm.Normal("2017_time_effect", 0, 2.5)
        _att = pm.Normal("att", 0, 2.5)
        _xi = pm.Normal("xi", 0, xi_sd) if xi_sd > 0 else pm.math.constant(0.0, name="xi")
        _b3 = pm.Deterministic("b3", _xi + _att)

        sigma = pm.HalfNormal("sigma", 10)
        
        
        # expectation
        mu = pm.Deterministic(
            "mu",
            outcome(t, _control_intercept, _treat_intercept_delta, _trend, group, treated, _b3),
            dims="obs_idx",
        )
        # likelihood
        pm.Normal("obs", mu, sigma, observed=df["sales"].values, dims="obs_idx")


    class NormalProposal:
        def __init__(self, loc: float, scale: float) -> None:
            self.loc = loc
            self.scale = scale

        def __call__(self, rng=None, size=()) -> np.ndarray:            
            if rng is None:
                rng = np.random
            return rng.normal(self.loc, scale=self.scale, size=size)


    class FixedDistSample(ArrayStepShared):
        name = "fixed_dist_sample"

        generates_stats = False

        def __init__(self, vars, proposal_kwarg_dict, model=None):
            model = pm.modelcontext(model)
            initial_values = model.initial_point()

            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = pm.inputvars(vars)
            initial_values_shape = [initial_values[v.name].shape for v in vars]
            self.ndim = int(sum(np.prod(ivs) for ivs in initial_values_shape))
            self.proposal_dist = NormalProposal(**proposal_kwarg_dict)

            shared = pm.make_shared_replacements(initial_values, vars, model)
            super().__init__(vars, shared)

        def astep(self, q0: RaveledVars) -> RaveledVars:
            point_map_info = q0.point_map_info
            q0 = q0.data
            q = self.proposal_dist(size=self.ndim)

            return RaveledVars(q, point_map_info), []

    # Now draw samples from the model we built
    with model:
        fixed_step = FixedDistSample([_xi], {'loc': 0, 'scale': xi_sd})
        idata = pm.sample(step=fixed_step, draws=20000, chains=4)
        return pm.sample_posterior_predictive(idata, extend_inferencedata=True)

def plot_posterior_sds(results: list, xi_sds: list) -> float:
    parameters = ["xi", "att"]#, "y_obs"]
    names = ["Standard deviation of the prior for xi (sigma_xi)", "The standard deviation of the posterior distribution of the ATT, \nsqrt(V(P(psi|D)))"]#, "predicted sales"]
    for parameter, name in zip(parameters, names):
        if parameter == "xi":
            #posterior_means = [0]
            posterior_stds = [0]
            #posterior_means += [az.summary(idata, round_to=2)['mean'].loc[parameter] for idata in results[1:]]
            posterior_stds += [az.summary(idata, round_to=2)['sd'].loc[parameter] for idata in results[1:]]
        elif parameter == "y_obs":
            posterior_stds = [az.summary(idata.posterior_predictive)["sd"] for idata in results]
        else:
            #posterior_means = [az.summary(idata, round_to=2)['mean'].loc[parameter] for idata in results]
            posterior_stds = [az.summary(idata, round_to=2)['sd'].loc[parameter] for idata in results]
        
        #posterior_means = np.array(posterior_means)
        posterior_stds = np.array(posterior_stds)
        if parameter == "att":
            slope, intercept = np.polyfit(xi_sds, posterior_stds, 1)
            plt.plot(xi_sds.tolist(), posterior_stds, label=f"{name}, slope={slope:.2f}")
        else:
            plt.plot(xi_sds.tolist(), posterior_stds, label=f"{name}")
        # plt.fill_between(xi_variances, posterior_means - posterior_stds, posterior_means + posterior_stds, \
                        # alpha=0.2)

    #plt.annotate(f'Slope = {slope:.2f}', (0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
    plt.title("Standard deviation of the ATT estimate for different levels of informativeness of the deviation from parallel trends")
    plt.xlabel('Standard deviation of the prior for xi (sigma_xi)')
    plt.ylabel('Effect on Sales')
    plt.legend()
    plt.show()
    return slope

def plot_att_posteriors(results: list) -> None:
    for i in (1,2,4,8):
        # Extracting samples from the traces
        att_pt = results[0].posterior['att'].values.flatten()
        att_npt = results[i].posterior['att'].values.flatten()

        # Plotting histograms of the two distributions
        plt.hist(att_pt, bins=30, alpha=0.5, label='Parallel Trends holds', density=True)
        plt.hist(att_npt, bins=30, alpha=0.5, label=f'Parallel Trends doesn\'t hold \nxi ~ N(0, {i/2.0})', density=True)

        # Adding labels and legend
        plt.title("Posterior distribution of the ATT with and without parallel trends")
        plt.xlabel('ATT in sales')
        plt.ylabel('Density')
        plt.legend()

        plt.show()

# def pairwise_wasserstein_distance(inference_data_list, parameter_name):
#     # Initialize an empty dictionary to store pairwise distances
#     pairwise_distances = {}
    
#     # Iterate over all pairs of InferenceData objects
#     for idx1, idx2 in itertools.combinations(range(len(inference_data_list)), 2):
#         data1 = inference_data_list[idx1]
#         data2 = inference_data_list[idx2]
        
#         # Extract posterior samples for the parameter of interest from each InferenceData object
#         posterior1 = data1.posterior[parameter_name].values.flatten()
#         posterior2 = data2.posterior[parameter_name].values.flatten()
        
#         # Calculate the Wasserstein distance between the two posterior distributions
#         wasserstein_distance = scipy.stats.wasserstein_distance(posterior1, posterior2)
        
#         # Store the calculated distance in the pairwise_distances dictionary
#         pair_key = (idx1, idx2)
#         pairwise_distances[pair_key] = wasserstein_distance
        
#     return pairwise_distances

# # Calculate pairwise Wasserstein distances
# pairwise_distances = pairwise_wasserstein_distance(results, param)

# # Create a matrix to store the distances
# num_objects = len(results)
# distance_matrix = np.zeros((num_objects, num_objects))

# # Fill the matrix with pairwise distances
# for pair, distance in pairwise_distances.items():
#     distance_matrix[pair[0], pair[1]] = distance
#     distance_matrix[pair[1], pair[0]] = distance  # Symmetric matrix

# mask = np.triu(np.ones_like(distance_matrix, dtype=bool))
# # Create a heatmap
# fig, ax = plt.subplots(figsize=(10,8)) 
# sns.heatmap(distance_matrix, annot=True, cmap='viridis', square=True, mask=mask, fmt='.1',\
#             cbar_kws={'label': 'Wasserstein distance'})
# plt.title('Pairwise Wasserstein Distance Between Posterior Distributions of the ATT with Different Priors on the Deviation from Parallel Trends')
# plt.xlabel('Standard deviation of the prior for xi (sigma_xi)')
# plt.ylabel('Standard deviation of the prior for xi (sigma_xi)')
# plt.xticks(ticks=np.arange(num_objects) + 0.5, labels=xi_sds, rotation=45, ha='right')
# plt.yticks(ticks=np.arange(num_objects) + 0.5, labels=xi_sds)
# #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1'))  # Round x-axis ticks to one decimal place
# #plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1'))  # Round y-axis ticks to one decimal place
# plt.show()

