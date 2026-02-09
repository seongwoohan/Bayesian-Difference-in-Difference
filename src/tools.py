
import pandas as pd
from dataclasses import dataclass
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@dataclass
class DataIn:
    df: pd.DataFrame
        
    @property
    def get_raw(self):
        return self.df
    
    
    @property
    def get_df2(self):
        return (self.df.copy().
                sort_values(by=["locale","period","ID"])
       .assign(period = lambda d: d["period"]-1)
      )
    
    @property
    def get_df3(self):
        return (
            self.get_df2.copy()
            .groupby(["locale","period"],as_index=False)
            .agg(treated=("treated","mean"),sales=("sales","mean"))
            )
    
    
    @property
    def get_df4(self):
        return (self.get_df3.query("period <=12")
         .pivot(index='period', columns='locale', values='sales')
         .reset_index()
         .assign(sales=lambda d:d[1]-d[0])
            )
    
    @property
    def get_last_diff(self):
        return self.get_df4.sales.values[-1]
@dataclass
class DataViz():
    data:DataIn
    
    def plot_data(self,axs_obj=None,legend=None):
        if axs_obj ==None:
            fig, axs_obj = plt.subplots(figsize = (10,5))
            axs_obj.set(xlabel="Period",ylabel="Total Average Sales")

        # custom_palette = {
        #     "Philadelphia": "#4C72B0",  # Replace with your custom color for "Philadelphia"
        #     "Baltimore": "#C44E52"     # Replace with your custom color for "Baltimore"
        # }
        
        #self.data.get_df2['locale'] = self.data.get_df2['locale'].replace({0: "Philadelphia", 1: "Baltimore"})

        custom_palette = {
            0: "#3A6EA5",  # Custom color for locale 0
            1: "#C44E52"   # Custom color for locale 1
        }
        
        (self.data.get_df2
        .groupby(["period","locale"],as_index=False)
         .agg(total_sales_avg=("sales","mean"))
         .pipe(
             (sns.scatterplot,"data"),
             x="period",
             y="total_sales_avg",
            hue="locale",
            # edgecolor='none', facecolors='none', 
            s=30,
            palette=custom_palette,
            legend=None,
            ax=axs_obj,
            
         )
        )
        
                
        (self.data.get_df2
        .groupby(["period","locale"],as_index=False)
         .agg(total_sales_avg=("sales","mean"))
         .pipe(
             (sns.lineplot,"data"),
             x="period",
             y="total_sales_avg",
             hue="locale",
             ax=axs_obj,
             palette=custom_palette,
             legend=legend
    
         )
        )
        
    def __call__(self,axs_obj=None):
        self.plot_data(axs_obj)

                   
@dataclass
class ModelPlotter:
    model_names : list
    model_results :list


class ModelSensitivityPlotter(ModelPlotter):
    def __init__(self,model_names,model_results):
        if model_names is None:
            model_names = [trace.attrs['name'] for trace in model_results]
        super().__init__(model_names,model_results)


        
    def plot_att_posteriors(self,var_name):
        if var_name=="psi_new":
            for i in range(1,len(self.model_results)):
                        
                # Extracting samples from the traces
                # att_pt = self.model_results[0].posterior[var_name].values.flatten()
                

                att_pt = self.model_results[0].posterior["att"].values.flatten()
                att_npt1 = self.model_results[i].posterior["att"].values.flatten()
                
                att_npt=(
                    self.model_results[i].posterior.b3.isel(b3_dim_0=slice(0,26)).values - self.model_results[i].posterior.xi.isel(xi_dim_0=slice(0,26)).cumsum(dim='xi_dim_0').values
                ).flatten()

                # Plotting histograms of the two distributions
                plt.hist(att_pt, bins=30, alpha=0.5, label=fr"Parallel Trends holds' {self.model_names[0]}", density=True)
                plt.hist(att_npt, bins=30, alpha=0.5, 
                        label=f'Parallel Trends doesn\'t hold \n-{self.model_names[i]} - psi', density=True,color="orange")
                # plt.hist(att_npt1, bins=30, alpha=0.5, 
                #         label=f'Parallel Trends doesn\'t hold \n-{self.model_names[i]} - att', density=True,color="green")

                # Adding labels and legends
                plt.title("Posterior distribution of the ATT")
                plt.xlabel('ATT in sales')
                plt.ylabel('Density')
                plt.xlim((-20,8))
                # plt.xlim((-8,8))

                plt.legend()
                plt.axvline(0.0,ls="--",alpha=0.1,color="k")

                plt.show()
        elif var_name == "psi":
            for i in range(1,len(self.model_names)):
                        
                # Extracting samples from the traces
                att_pt = self.model_results[0].posterior["att"].values.flatten()              
                # att_npt = self.model_results[i].posterior[var_name].mean(dim="psi_dim_0").values.flatten()
                att_npt = self.model_results[i].posterior[var_name][:,:,13:26].mean(dim="psi_dim_0").values.flatten()
                # print("this")

                plt.figure(figsize=(2.5, 2.5))

                # Plotting histograms of the two distributions
                # plt.hist(att_pt, bins=30, alpha=0.5, label=fr"Parallel Trends holds' {self.model_names[0]}", density=True)
                # plt.hist(att_npt, bins=30, alpha=0.5, 
                        #label=f'Parallel Trends doesn\'t hold \n-{self.model_names[i]}', density=True,color="orange")

                plt.hist(att_pt, bins=30, alpha=0.5, density=True, color="blue")
                plt.hist(att_npt, bins=30, alpha=0.5, density=True, color="red")
                #plt.gca().legend().set_visible(False)
               

                # Adding labels and legends
                #plt.title("Posterior distribution of the ATT")
                plt.xlabel('Total average sales (ATT)',fontsize=11)
                plt.ylabel('Density',fontsize=11)
                plt.xticks(fontsize=10)                               
                plt.yticks(fontsize=10)
                plt.xlim((-4.5,3))
                # plt.xlim((-5,5))
                #plt.legend()
                plt.axvline(0.0,ls="--",alpha=0.1,color="k")

                file_name = f"att_posterior_phar44_plot_{i}.pdf"
                plt.savefig(file_name, bbox_inches="tight", dpi=300)

                plt.show()
            

            
    def plot_att_posteriors_box(self,var_name,plot_obj=None):
        fig,axs = plt.subplots(figsize=(6,6))
        df=pd.DataFrame(columns=["eX","eY"])
        label = ['Parallel Trends holds' if i==0 else  f'{self.model_names[i]}' for i in [0,1,2,3]]
        for i,idx in zip((0,1,2,3),[0,1,2,3]):
        # Extracting samples from the traces
    #         att_pt = results[0].posterior['att'].values.flatten()
            att_npt = self.model_results[i].posterior[var_name].values.flatten()        
            color= "orange" if i > 0 else "deepskyblue"

            ylabel="att" if i==0 else ""

            df_add =pd.DataFrame()
            df_add=df_add.assign(
                eX=np.repeat(idx,len(att_npt)),
                eY=att_npt
            )

            df=pd.concat([df,df_add],axis=0)

        df.pipe(
            (plot_obj,"data"),
                 x="eX",
               y="eY",
            ax=axs,
            width=0.4,
    #         hue="eX",
            palette=["deepskyblue","orange","orange","orange"],
            whiskerprops={'alpha': 1,"color":"silver"},
            boxprops={"alpha":0.8},
            flierprops={'alpha': 0.1}, 
            capprops={'alpha': 0.5},
            medianprops={'alpha': 0.5},
            showfliers=False
        )
        axs.axhline(0,ls="--",alpha = 0.1,color="k")
    #     axs.axvline(5.5,ls="--")
        axs.set(xticklabels=label,ylabel="ATT \nPosterior",xlabel="")
        font_props = FontProperties(size=10)
        for label in axs.get_xticklabels():
            label.set_fontproperties(font_props)
        axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')
        plt.show()
        
    