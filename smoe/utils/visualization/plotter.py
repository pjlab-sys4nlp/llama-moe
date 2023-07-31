import math
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")

# fmt: off
class plotter:
    def __init__(self, cmap='Set2'):
        super().__init__()
        self.data = {}
        self.plt = plt
        plt.set_cmap(cmap)
        plt.cla()

    def add_figure(self, figure_name, xlabel="", ylabel="", title="", legend="best"):
        if not figure_name in self.data:
            self.data[figure_name] = {}, xlabel, ylabel, title, legend

    def add_label(self, figure_name, label, linewidth=2, linestyle="-", markersize=8, marker=".", dot=False, annotate=None):
        self.add_figure(figure_name)
        if not label in self.data[figure_name][0]:
            self.data[figure_name][0][label] = [], [], linewidth, linestyle, markersize, marker, dot, annotate

    def add_data(self, figure_name, label, x, y):
        self.add_label(figure_name, label)
        self.data[figure_name][0][label][0].append(x)
        self.data[figure_name][0][label][1].append(y)

    def draw(self, show=False, save=False, path="", name_prefix=None, format="png", dpi=320):
        for figure_name in self.data.keys():
            all_label_infos, xlabel, ylabel, title, legend = self.data[figure_name]
            fig = plt.figure(figure_name)

            for label in all_label_infos.keys():
                label_info = all_label_infos[label]
                x_list, y_list, linewidth, linestyle, markersize, marker, dot, annotate = label_info

                if dot == True:
                    plt.scatter(x_list, y_list, label=label, s=markersize * markersize, marker=marker)
                else:
                    plt.plot(x_list, y_list, label=label, linewidth=linewidth, linestyle=linestyle, markersize=markersize, marker=marker)

                if annotate is not None:
                    if annotate == "max":
                        pos = np.argmax(y_list)
                        plt.scatter(x_list[pos], y_list[pos], s=markersize * markersize, marker="*", color="black", zorder=100)
                        plt.annotate(str(y_list[pos]) + "\n", (x_list[pos], y_list[pos]), ha="center", va="center", weight="bold", c="black")
                    elif annotate == "min":
                        pos = np.argmin(y_list)
                        plt.scatter(x_list[pos], y_list[pos], s=markersize * markersize, marker="*", color="black", zorder=100)
                        plt.annotate("\n" + str(y_list[pos]), (x_list[pos], y_list[pos]), ha="center", va="center", weight="bold", c="black")
                    else:
                        raise ValueError

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend(loc=legend)

            if show:
                fig.show()
            if save:
                if name_prefix is None:
                    fig.savefig(os.path.join(path, figure_name + "." + format), dpi=dpi)
                else:
                    fig.savefig(os.path.join(path, name_prefix + "-" + figure_name + "." + format), dpi=dpi)

    def show(self):
        self.draw(show=True)

    def save(self, **kwargs):
        self.draw(save=True, **kwargs)


if __name__ == "__main__":
    p = plotter()

    p.add_figure("fig1", xlabel="x", ylabel="y", title="fig1", legend="best")
    p.add_figure("fig2", xlabel="x", ylabel="y", title="fig2", legend="best")

    p.add_label("fig1", "test1", dot=True, annotate="max")
    p.add_label("fig1", "test2", dot=True, annotate="max")
    p.add_label("fig2", "test1", marker="", annotate="min")
    p.add_label("fig2", "test2", marker="", annotate="min")

    for i in range(10):
        p.add_data("fig1", "test1", i, i + 5)
        p.add_data("fig1", "test2", i, i * i)

    for i in range(10):
        p.add_data("fig2", "test1", i, i + 5 + math.sqrt(i) - 25)
        p.add_data("fig2", "test2", i, i * i - 0.5 * i - 15)

    print(p.data)

    p.save(path="../")
