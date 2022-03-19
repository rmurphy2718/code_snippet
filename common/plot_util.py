import time

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=False)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PlotUtil:
    @classmethod
    def plot(cls, plt_lmb, send_param=False, tight=True, fig_width=13.0,
             fig_height=4.0, plot_file=None, pdf=None, save=True, show=False):
        plt.clf()
        if send_param:
            plt_lmb(plt, sns)
        else:
            plt_lmb()
        fig = plt.gcf()
        sns.despine(fig, left=True)
        if save:
            fig.set_size_inches(fig_width, fig_height)
            if plot_file is None:
                plot_file = f"../plots/{time.time()}.pdf"
            if pdf is None:
                if tight:
                    fig.savefig(plot_file, dpi=300, transparent=True, bbox_inches='tight')
                else:
                    fig.savefig(plot_file, dpi=300, transparent=True)
            else:
                if tight:
                    pdf.savefig(fig, dpi=300, transparent=True, bbox_inches='tight')
                else:
                    pdf.savefig(fig, dpi=300, transparent=True)
            return plot_file
        else:
            plt.show()

    @classmethod
    def create_pdf(cls, plot_file=None):
        if plot_file is None:
            plot_file = f"../plots/{time.time()}.pdf"
        pdf = PdfPages(plot_file)
        return pdf

    @classmethod
    def rotate_xlabels(cls, ax, rot):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rot)
