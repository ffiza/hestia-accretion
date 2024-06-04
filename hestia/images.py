import matplotlib.pyplot as plt
import yaml

CONFIG = yaml.safe_load(open("configs/figures.yml"))


def figure_setup():
    params = {"axes.labelsize": CONFIG["AXES_LABELSIZE"],
              "axes.titlesize": CONFIG["AXES_TITLESIZE"],
              "text.usetex": bool(CONFIG["TEXT_USETEX"]),
              "figure.dpi": CONFIG["FIGURE_DPI"],
              "figure.facecolor": CONFIG["FIGURE_FACECOLOR"],
              "font.size": CONFIG["FONT_SIZE"],
              "font.serif": [],
              "font.sans-serif": [],
              "font.monospace": [],
              "font.family": CONFIG["FONT_FAMILY"],
              "xtick.top": CONFIG["XTICK_TOP"],
              "xtick.labelsize": CONFIG["XTICK_LABELSIZE"],
              "xtick.major.width": CONFIG["XTICK_MAJOR_WIDTH"],
              "xtick.major.size": CONFIG["XTICK_MAJOR_SIZE"],
              "xtick.minor.width": CONFIG["XTICK_MINOR_WIDTH"],
              "xtick.minor.size": CONFIG["XTICK_MINOR_SIZE"],
              "ytick.right": CONFIG["XTICK_RIGHT"],
              "ytick.major.width": CONFIG["YTICK_MAJOR_WIDTH"],
              "ytick.major.size": CONFIG["YTICK_MAJOR_SIZE"],
              "ytick.minor.width": CONFIG["YTICK_MINOR_WIDTH"],
              "ytick.minor.size": CONFIG["YTICK_MINOR_SIZE"],
              "ytick.labelsize": CONFIG["YTICK_LABELSIZE"],
              "savefig.dpi": CONFIG["SAVEFIG_DPI"],
              "savefig.bbox": CONFIG["SAVEFIG_BBOX"],
              "savefig.pad_inches": CONFIG["SAVEFIG_PAD_INCHES"]}
    plt.rcParams.update(params)
