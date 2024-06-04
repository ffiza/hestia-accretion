import matplotlib.pyplot as plt


def figure_setup():
    params = {'axes.labelsize': 10.0,
              'axes.titlesize': 8.0,
              'text.usetex': True,
              'figure.dpi': 500,
              'figure.facecolor': 'white',
              'font.size': 8.0,
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'font.family': 'serif',
              'xtick.top': 'on',
              "xtick.labelsize": 8.0,
              'xtick.major.width': 0.5,
              'xtick.major.size': 1.5,
              'xtick.minor.width': 0.25,
              'xtick.minor.size': 1.5,
              'ytick.right': 'on',
              'ytick.major.width': 0.5,
              'ytick.major.size': 1.5,
              'ytick.minor.width': 0.25,
              'ytick.minor.size': 1.5,
              "ytick.labelsize": 8.0,
              'savefig.dpi': 500,
              'savefig.bbox': 'tight',
              'savefig.pad_inches': 0.02}
    plt.rcParams.update(params)