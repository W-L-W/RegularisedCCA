import os
import matplotlib.pyplot as plt

this_file_dir_path = os.path.dirname(os.path.realpath(__file__))
plot_dir = this_file_dir_path + '/'

def save_mplib(fig: plt.Figure, name: str):
    fig.savefig(plot_dir + name + '.pdf', bbox_inches='tight')

def save_plotly(fig, name: str):
    fig.write_image(plot_dir + name + '.pdf')