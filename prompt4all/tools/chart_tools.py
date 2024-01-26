import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
from prompt4all.utils.io_utils import download_file
from prompt4all import context

cxt = context._context()


#
def prepare_fonts():
    download_file('https://fonts.gstatic.com/s/notosanstc/v35/-nF7OG829Oofr2wohFbTp9iFPysLA_ZJ1g.ttf',
                  cxt.get_prompt4all_dir(), 'NotoSansTC-Thin.ttf')
    font_path = os.path.join(cxt.get_prompt4all_dir(), 'NotoSansTC-Thin.ttf')
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans TC Thin'
    plt.rcParams['axes.unicode_minus'] = False


def toy():
    # Import Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    df_select = df.loc[df.cyl.isin([4, 8]), :]
    fig = plt.figure()
    # Plot
    sns.set_style("white")
    gridobj = sns.lmplot(x="displ", y="hwy", hue="cyl", data=df_select,
                         height=7, aspect=1.6, robust=True, palette='tab10',
                         scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

    # Decorations
    gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
    plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
    # plt.show()
    html_str = mpld3.fig_to_html(fig)
    plt.savefig('./generate_images/chart_test.png')
    return '![{0}]({1}  "{2}")'.format(
        '由mermaid生成', './generate_images/chart_test.png', 'chat')


def fig2html(fig):
    html_str = mpld3.fig_to_html(fig)
