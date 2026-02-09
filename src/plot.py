from tools import DataIn,ModelSensitivityPlotter,DataViz
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd 
import arviz as az


def plot_counter_factuals(trace, data, file_name=None):
    plot_name = "On Outcome Scale"
    posterior = trace.posterior

    b0_post = posterior.b0
    b1_post = posterior.b1
    b2_post = posterior.b2    
    xi_post = posterior.xi.rename({'xi_dim_0': '_dim'}) 
    b3_post = posterior.b3
    
    periods_plot = np.arange(26)
    locale_plot = np.ones((26,))
    treated_plot = np.hstack((np.zeros(13,), np.ones(13)))

    periods_arr = xr.DataArray(periods_plot, dims=["_dim"])
    locale_arr = xr.DataArray(locale_plot, dims=["_dim"])
    treated_arr = xr.DataArray(treated_plot, dims=["_dim"])
    
    try:        
        sum_xi_post = posterior._sum_xi.rename({"_sum_xi_dim_0": "_dim"})
        y_post_modified = b0_post + b1_post * locale_arr + b2_post * periods_arr + xi_post.cumsum(dim="_dim")
    except AttributeError:
        y_post_modified = b0_post + b1_post * locale_arr + b2_post * periods_arr + xi_post
    print((xi_post.cumsum(dim="_dim") - sum_xi_post).mean())

    y_post_modified_hdi = az.hdi(y_post_modified, hdi_prob=0.95).x
    y_post_locale0 = b0_post + b2_post * periods_arr
    y_post_locale1 = b0_post + b1_post * locale_arr + b2_post * periods_arr
    y_post_locale1_treated = b0_post + b1_post * locale_arr + b2_post * periods_arr + b3_post * treated_arr

    # Make the plot smaller by adjusting the figure size
    fig, axs = plt.subplots(figsize=(4, 2))  # Smaller figure size
    
    axs.plot(
        periods_plot,
        y_post_locale0.mean(dim=["chain", "draw"]),
        color="#3A6EA5", ls='--', label="Control Group"
    )
    az.plot_hdi(
        periods_plot, y_post_locale0, hdi_prob=0.95, fill_kwargs={"alpha": 0.2, "color": "#3A6EA5"}, ax=axs
    )
    
    axs.plot(
        periods_plot,
        y_post_locale1_treated.mean(dim=["chain", "draw"]),
        color="#C44E52", ls='--', label="Treatment Group"
    )


    axs.plot(
        periods_plot,
        y_post_locale1.mean(dim=["chain", "draw"]),
        color="#8E4585", ls='--', label="Treatment Group (Counterfactual)"
    )
    
    az.plot_hdi(
        periods_plot, y_post_locale1, hdi_prob=0.95, fill_kwargs={"alpha": 0.2, "color": "#8E4585"}, ax=axs
    )
    
    axs.scatter(
        periods_plot,
        y_post_modified.mean(dim=["chain", "draw"]),
        color="black", edgecolor='black', facecolors='none', s=15
    )
    
    y_modified_hdi_estimates = y_post_modified_hdi.T.values
    axs.fill_between(
        periods_plot, y_modified_hdi_estimates[0, :], y_modified_hdi_estimates[1, :],
        alpha=0.2, color="black"
    )
    axs.plot(
        periods_plot[0:26],
        y_post_modified.mean(dim=["chain", "draw"])[0:26],
        color="black", ls="--", alpha=0.3, label="Modified trend with xi"
    )
    axs.axvline(12.1, ls="--", color="black", alpha=0.5)
    #axs.set_title("Total Average Sales (ATT)", fontsize=10) 
    
    axs.set_xticks([5, 12.1, 18])
    axs.set_xticklabels(["2016", "Tax Implemented", "2017"], fontsize=12, rotation=0, ha='center') 
    axs.tick_params(axis='both', which='major', labelsize=10)  # Increase x and y tick size
    
    axs.set_ylabel("Total Average Sales (ATT)", fontsize=10)  # Increase y-label font size
    
    # Adjust the x-axis limits to reduce the gap on the sides
    axs.set_xlim(periods_plot[0] - 0.5, periods_plot[-1] + 0.5)
    #axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure only major ticks are shown
    axs.tick_params(axis='x', which='both', bottom=False)  # Remove ticks on x-axis

    
    DataViz(data).plot_data(axs_obj=axs)
    
    # Remove title and legend
    #axs.set_title("")
    axs.set_ylabel("")
    axs.set_xlabel("")
    
    #axs.set_xticklabels("")
    legend = axs.get_legend()
    if legend:
        legend.remove()
    
    # Save the plot with a generated file name if not provided
    if file_name is None:
        file_name = f"{trace.attrs['name']}_plot.pdf"
    plt.ylim(0, 6)
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.show()


def plot_xi_dists_inside_model(trace, data, file_name=None):
    posterior = trace.posterior
    b0_post = posterior.b0
    xi_post = posterior.xi.rename({'xi_dim_0': '_dim'})
    
    xi_post_p = xi_post.stack(sample=("chain", "draw"))[:, :200]
    xi_post_post = posterior.xi_post
    xi_post_post_p = xi_post_post.stack(sample=("chain", "draw"))[:, :200]
    xi_sum_post = posterior._sum_xi
    xi_sum_post_p = xi_sum_post.stack(sample=("chain", "draw"))[:, :200]

    data_points = data.get_df4['sales'].values
    data_points_before_diff = data_points
    data_points_after_diff = data_points[1:] - data_points[:-1]
    data_points_summ = np.cumsum(data_points_after_diff, axis=-1)

    # Set smaller figure size
    fig, axs = plt.subplots(3, 1, figsize=(3.9, 4.5))

    axs[0].scatter(np.arange(13), data_points_before_diff, color="grey", edgecolor="grey", facecolor="none", s=24)
    axs[1].scatter(np.arange(12), data_points_after_diff, color="grey", edgecolor="grey", facecolor="none", s=24)
    axs[2].scatter(np.arange(12), data_points_summ, color="grey", edgecolor="grey", facecolor="none", s=24)

    axs[0].plot(np.arange(27), xi_post_post_p, color="grey", alpha=0.1)
    axs[1].plot(np.arange(26), xi_post_p, color="grey", alpha=0.1)
    axs[2].plot(np.arange(26), xi_sum_post_p, color="grey", alpha=0.1)

    for ax in axs.flat:
        ax.axvline(12.1, ls="--", color="black", alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=9)  # Increase x and y tick label size
        #ax.set_ylim(-3, 3)

        ax.set_xticks([])

    #axs[0].set_ylabel("Difference in Sales")
    #axs[1].set_ylabel("Differences in Differences")
    #axs[2].set_ylabel("Sum: Differences in Differences")
    ax.set_xlim(-0.5, 25.2) 

    # Set x-ticks and labels
    
    for ax in axs:
        ax.set_xticks([5, 12.1, 18])
        ax.set_xticklabels(["2016", "Tax Implemented", "2017"], fontsize=9)
        ax.set_xlim(-0.5, 25.2) 
        
    ax.set_xlim(-0.5, 25.2)    
    # Remove titles
    axs[0].set_title("")
    axs[1].set_title("")
    axs[2].set_title("")

    #axs.set_xticklabels("")

    fig_middle, ax_middle = plt.subplots(figsize=(3.9, 1.5))
    ax_middle.scatter(np.arange(12), data_points_after_diff, color="grey", edgecolor="grey", facecolor="none", s=24)
    ax_middle.plot(np.arange(26), xi_post_p, color="grey", alpha=0.1)
    ax_middle.axvline(12.1, ls="--", color="black", alpha=0.5)
    ax_middle.tick_params(axis='both', which='major', labelsize=9)
    ax_middle.set_ylim(-3, 3)
    
    #ax_middle.set_xticks([])
    ax_middle.set_xticks([5, 12.1, 18])
    ax_middle.set_xticklabels(["2016", "Tax Implemented", "2017"], fontsize=9)
    ax_middle.set_xlim(-0.5, 25.2)
    
    post_intervention_start = 12  # Assuming post-intervention starts at index 12

    xi_post_post_mean = xi_post_p[:, post_intervention_start:].mean(axis=1)  # Mean
    xi_post_post_max = xi_post_p[:, post_intervention_start:].max(axis=1)  # Maximum
    xi_post_post_min = xi_post_p[:, post_intervention_start:].min(axis=1)  # Minimum
    
    # Print or save these values for analysis
    print("Mean of post-intervention AR(1) process:", xi_post_post_mean[13:27].mean())
    print("Max of post-intervention AR(1) process:", xi_post_post_max[13:27].mean())
    print("Min of post-intervention AR(1) process:", xi_post_post_min[13:27].mean())


    # Save the plot as a PDF file if a file name is provided
    if file_name is not None:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.show()


# posterior_ds = trace8.posterior

def plot_distribution_xi(trace,title_prefix = None,ylim=None):
    plot_name ="xi disribution over time(t)- 95% hdi"
    posterior = trace.posterior
    b0_post= posterior.b0

    b1_post = posterior.b1
    b2_post = posterior.b2    
    xi_post = posterior.xi
    xi_post = xi_post.rename({'xi_dim_0': '_dim'})
    b3_post = posterior.b3
    # b3_post =b3_post.isel(b3_dim_0=slice(0,26)).rename({"b3_dim_0":"_dim"})

    periods_plot = np.arange(26)
    locale_plot = np.ones( (26,))
    periods_arr = xr.DataArray(periods_plot,dims = ["_dim"])
    locale_arr = xr.DataArray(locale_plot,dims = ["_dim"])
    treated_plot = np.hstack((np.zeros(13,),np.ones(13)))
    treated_arr = xr.DataArray(treated_plot,dims=["_dim"])
    y_post_modified = b0_post + b1_post*locale_arr + b2_post*periods_arr + xi_post.cumsum(dim="_dim")

    y_post_modified_hdi =az.hdi(y_post_modified).x
    y_post_locale0 = b0_post +  b2_post*periods_arr
    y_post_locale1 = b0_post + b1_post*locale_arr + b2_post*periods_arr
    y_post_locale1_treated =b0_post + b1_post*locale_arr + b2_post*periods_arr + b3_post*treated_arr

    xi_filtered = xi_post.isel(_dim=slice(12,26))

    fig,axs = plt.subplots(figsize=(10,5))
    xi_post_modified_hdi =az.hdi(xi_filtered,hdi_prob=0.95).xi


    axs.plot(
        np.full((2,14),np.arange(12,26)),
        xi_post_modified_hdi.T,color="k",alpha=0.3,marker="_",
    )
    axs.plot(
        np.arange(12,26),
        xi_filtered.mean(dim=["chain","draw"]),color="k",ls="--",alpha=0.3,
    )
    title_sub_text=trace.attrs["name"]
    if title_prefix is not None:
        title=f"{title_prefix} {title_sub_text} : {plot_name}"
    else:
        title = f"{title_sub_text} :"

    axs.axvline(12.0,ls="--",alpha=0.3),
    axs.set(

        ylim=ylim,
        title = title,
        xticks = np.arange(12,26),
        xticklabels = ["2017"  if i == 18 else "" for i in range(12, 26)],
        # xlabel="After Tax Implementation",
        
        
        
    )
#     print(xi_post_modified_hdi)
    plt.show()


def report_table(trace,caption=None):
    posterior_ds = trace.posterior
    constant_ds = trace.constant_data
    posterior_ds=posterior_ds.assign(
        fe_control = posterior_ds.b0,
        fe_treatment = posterior_ds.b0 + posterior_ds.b1,
        trend = posterior_ds.b2,
        psi_marginalized = posterior_ds.psi.isel(psi_dim_0=slice(13,26)).mean(dim="psi_dim_0")
    )

    b0_post= posterior_ds.b0
    b1_post = posterior_ds.b1
    b2_post = posterior_ds.b2    
    xi_post = posterior_ds.xi
    # xi_post = xi_post.rename({'xi_dim_0': '_dim'}) 
    b3_post = posterior_ds.b3
    
    periods_plot = np.arange(26)
    locale_plot = np.ones( (26,))
    periods_arr = xr.DataArray(periods_plot,dims = ["_dim"])
    locale_arr = xr.DataArray(locale_plot,dims = ["_dim"])
    treated_plot = np.hstack((np.zeros(13,),np.ones(13)))
    treated_arr = xr.DataArray(treated_plot,dims=["_dim"])
           
    sum_xi_post = posterior_ds._sum_xi.rename({"_sum_xi_dim_0": "_dim"})
    y_post_modified = b0_post + b1_post*locale_arr + b2_post*periods_arr + sum_xi_post
    y_post_modified= y_post_modified.isel(_dim=slice(13,26))
    # print(y_post_modified.mean("_dim"))
    y_post_locale1 = b0_post + b1_post*locale_arr + b2_post*periods_arr
    y_post_locale1 = y_post_locale1.isel(_dim=slice(13,26))

    y_post_locale1_treated =b0_post + b1_post*locale_arr + b2_post*periods_arr + b3_post*treated_arr
    y_post_locale1_treated=y_post_locale1_treated.isel(_dim=slice(13,26))
    
    posterior_ds= posterior_ds.assign(
        modified_trend = y_post_modified.mean("_dim"),
        counterfactual_trend = y_post_locale1.mean("_dim"),
        observed_trend = y_post_locale1_treated.mean("_dim"),
    )


    rename_dict = {
        'fe_control': 'Fixed Effect : Control',
        'fe_treatment': 'Fixed Effect : Treatment',
        'trend': 'Trend',
        'psi_marginalized': 'PSI under modified trend',
        "modified_trend": "Modified Trend",
        "att":"ATT under Parallel Trend",
        "counterfactual_trend":"Counterfactual Trend",
        "observed_trend":"Observed Trend"
    }


    df_report_base=az.summary(
        posterior_ds,var_names = ["fe_control","fe_treatment","trend","counterfactual_trend","modified_trend","observed_trend",
                                  "att","psi_marginalized",],
        kind="stats",hdi_prob=0.95
        ).rename(
            index=rename_dict
        )


    #     hdi_columns = df_report.columns[df_report.columns.str.startswith("hdi_")]
    df_report = df_report_base.assign(
        CI_95 =lambda x: x.filter(regex='^hdi_')
                                   .astype(str)
                                   .agg(','.join, axis=1)
    ).drop(df_report_base.filter(regex='^hdi_').columns, axis=1)

    ar_params = ["eta","rho","ar_sigma"]

    posterior_params = list(posterior_ds.data_vars)

    df_report2 = pd.DataFrame(columns=df_report.columns)
    

    if "eta" in posterior_params:

        df_report2_eta=az.summary(
            posterior_ds,var_names = ["eta"],
        kind="stats",hdi_prob=0.95,round_to=3
        )

        df_report2_eta = df_report2_eta.assign(
        CI_95 =lambda x: x.filter(regex='^hdi_')
                                   .astype(str)
                                   .agg(','.join, axis=1)
        ).drop(df_report2_eta.filter(regex='^hdi_').columns, axis=1)

    else:
        df_report2_eta = pd.DataFrame(columns=df_report.columns,index=["eta"])
        # df_report2_eta.loc["eta","mean"]=constant_ds.eta_constant.values[0].round(3)
        df_report2_eta.loc["eta","mean"] = getattr(trace.attrs, "eta", 1.598)#.round(3)



    if "rho" in posterior_params:

        df_report2_rho=az.summary(
            posterior_ds,var_names = ["rho"],
        kind="stats",hdi_prob=0.95,round_to=3
        )

        df_report2_rho = df_report2_rho.assign(
        CI_95 =lambda x: x.filter(regex='^hdi_')
                                   .astype(str)
                                   .agg(','.join, axis=1)
        ).drop(df_report2_rho.filter(regex='^hdi_').columns, axis=1)

    else:
        df_report2_rho = pd.DataFrame(columns=df_report.columns,index=["rho"])
        # df_report2_rho.loc["rho","mean"]=constant_ds.rho_constant.values[0].round(3)
        df_report2_rho.loc["rho","mean"] = getattr(trace.attrs, "rho", 0.371)#.round(3)



    if "ar_sigma" in posterior_params:
        df_report2_sigma=az.summary(
            posterior_ds,var_names = ["ar_sigma"],
        kind="stats",hdi_prob=0.95,round_to=3
        )

        df_report2_sigma = df_report2_sigma.assign(
        CI_95 =lambda x: x.filter(regex='^hdi_')
                                   .astype(str)
                                   .agg(','.join, axis=1)
        ).drop(df_report2_sigma.filter(regex='^hdi_').columns, axis=1)

    else:
        df_report2_sigma = pd.DataFrame(columns=df_report.columns,index=["ar_sigma"])
        # df_report2_sigma.loc["ar_sigma","mean"]=constant_ds.ar_sigma_constant.values[0].round(3)
        df_report2_sigma.loc["ar_sigma","mean"] = getattr(trace.attrs, "ar_sigma", 0.166)#.round(3)


    df_report2=pd.concat([df_report2_eta,df_report2_rho,df_report2_sigma])
    
    df_report_final=pd.concat([df_report,df_report2])
#     df_report_final["mean"] = pd.to_numeric(df_report_final["mean"])
#     df_report_final["mean"]=df_report_final["mean"].round(3)

    if caption:
        return df_report_final.style.set_caption(trace.attrs["name"]).format(precision=3)
    else: 
        return df_report_final.style.set_caption(trace.attrs["name"]).format(precision=3)
