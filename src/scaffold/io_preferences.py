import os

import matplotlib.pyplot as plt



dir_path_this_script = os.path.dirname(os.path.realpath(__file__))
real_data_dir = dir_path_this_script+'/../../real_data'
base_output_dir = dir_path_this_script+'/../../expmts/output'
plot_dir = base_output_dir + '/plots/'



def output_folder(rel_path: str, mode = 'detail') -> str:
    """Folder name for data objects, to determine where the estimates and summary statistics are saved
    mode = {'detail', 'processed'}"""
    return base_output_dir + '/' + mode + '/' + rel_path + '/'

def rel_path_real_data(dataset: str) -> str:
    """Relative path for real data objects"""
    return 'real/' + dataset + '/'

def rel_path_mvn(path_stem: str, n: int, rs: int):
    return f'{path_stem}/n{n}/rs{rs}/'



# def output_folder_real_data(dataset: str, mode='detail') -> str:
#     """Folder name for data objects, to determine where the estimates and summary statistics are saved
#     mode = {'detail', 'processed'}"""
#     return base_output_dir + '/' + mode + '/real/' + dataset + '/'


def save_mplib(fig: plt.Figure, name: str):
    fig.savefig(plot_dir + name + '.pdf', bbox_inches='tight')

def save_plotly(fig, name: str):
    fig.write_image(plot_dir + name + '.pdf')



