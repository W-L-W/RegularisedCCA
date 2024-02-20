# primarily intended for parametric bootstrap
# but can also use beyond this scenario also
import pandas as pd
import matplotlib.pyplot as plt

from src.scaffold.wrappers import get_cv_obj_from_data
from src.scaffold.incoming import algo_labels

cmap = plt.get_cmap('turbo')
cmap_oracle = lambda z: cmap(0.65*z)

def panel_plot(data,Kmax=5):
    fig, axs = plt.subplots(nrows=6,ncols=4,figsize=(26,20),sharey='row',gridspec_kw={'height_ratios':[1.5,1.5,1,1,2,2]})
    # allow larger font sizes for the labels and titles
    label_size = 15
    title_size = 20
    # Define the labels for each row
    y_labels = ['r2sk', 'R2sk', 'wt_uk', 'vt_uk', 'wt_Uk', 'vt_Uk']
    kwarg_dict = {'ridge': {'logx':False,'xlim':[0,1]},
                    'wit': {'logx':True},
                    'suo': {'logx':True, 'xlim':[10**-4,20]},
                    'gglasso': {'logx':True, 'xlim':[10**-4,20]},
                    }


    for idx,algo in enumerate(['wit','suo','gglasso','ridge',]):
        solp = get_cv_obj_from_data(data,algo).full_path
        try:
            df = solp.load_dffull() # if already processed: 
        except:
            df = solp.process_estimates() 
        df = df.set_index('pen').sort_index()
        kwargs = kwarg_dict[algo]
        kwargs['cmap'] = cmap_oracle
        
        # compute rhosum2s
        R = solp.K
        rho_cols = df[[f'rho{r}' for r in range(1,R+1)]]
        renamer = {f'rho{i}': f'r2s{i}' for i in range(1,10+1)}
        new_cols = (rho_cols**2).cumsum(axis=1).rename(columns = renamer)
        df = pd.concat([df,new_cols])
        
        df[['r2s'+str(k+1) for k in range(Kmax)]].plot(**kwargs,ax=axs[0,idx]).legend(loc='upper right')
        df[['R2s'+str(k+1) for k in range(Kmax)]].plot(**kwargs,ax=axs[1,idx]).legend(loc='upper right')
        df[['wt_u'+str(k+1) for k in range(Kmax)]].plot(**kwargs,ax=axs[2,idx],ylim=[0,1.1]).legend(loc='upper right')
        df[['vt_u'+str(k+1) for k in range(Kmax)]].plot(**kwargs,ax=axs[3,idx],ylim=[0,1.1]).legend(loc='upper right')
        df[['wt_U'+str(k+1) for k in range(Kmax)]].plot(**kwargs,ax=axs[4,idx],ylim=[0,Kmax]).legend(loc='upper right')
        df[['vt_U'+str(k+1) for k in range(Kmax)]].plot(**kwargs,ax=axs[5,idx],ylim=[0,Kmax]).legend(loc='upper right')
        
        # Labels and titles
        # Add y-axis label to the first column and increase label size
        if idx == 0:
            for i in range(6):
                axs[i,idx].set_ylabel(y_labels[i], fontsize=label_size)
        # Remove x-axis label from all but the bottom row
        for i in range(5):
            axs[i,idx].set_xlabel('')
        # Increase size of x-axis label in the bottom row
        axs[5,idx].set_xlabel('pen', fontsize=label_size)
        # Large title row
        axs[0,idx].set_title(algo_labels[algo], fontsize=title_size)

    _ = [ax.grid(True, which='major', axis='both') for ax in axs.flatten()]   
    print(solp.data.folder)
    return fig