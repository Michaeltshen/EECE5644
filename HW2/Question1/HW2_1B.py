import csv
import pandas as pd
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def fisher(data):
    label_0, label_1 = [], []
    tau = -10
    # Calculate LDA Fisher stuff
    for num in range(0, len(data)):
        value = data.iloc[num].to_numpy()[:-1]
        if data["class_prior"][num] == 0:
            label_0.append(value)
        elif data["class_prior"][num] == 1:
            label_1.append(value)

        label0numpy = np.array(label_0)
        label1numpy = np.array(label_1)

        covariance1 = np.cov(label0numpy.T)
        m0 = np.mean(label0numpy, axis=0)
        covariance2 = np.cov(label1numpy.T)
        m1 = np.mean(label1numpy, axis=0)
        Sw = covariance1 + covariance2
        mu = m0 - m1
        w = np.matmul(np.linalg.inv(Sw), mu.T)
        print(w)

    # true_positive_rate = true_positive / (true_positive + false_negative)
    # false_positive_rate = false_positive / (false_positive + true_negative)
    # error_rate = false_positive_rate * \
    #     0.65 + (1 - true_positive_rate) * 0.35

    # true_positive_list.append(true_positive_rate)
    # false_positive_list.append(false_positive_rate)
    # error_rate_list.append(error_rate)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.plot(x_list, w_list, 'b.', markersize=2)
    # ax.set_xlabel("False Positive Rates")
    # ax.set_ylabel("True Positive Rates")
    # ax.set_title("ERM classification using the knowledge of true data pdf")
    # legend_elements = [Line2D([0], [0], color='tab:red', lw=3, label='Line'),
    #                    Line2D([0], [0], color='tab:green', lw=3, label='Line')]

    # # Create the figure
    # labels = ['Estimated Optimal Point', 'Theoretical Optimal Point']

    # # Put a legend below current axis
    # ax.legend(handles=legend_elements, labels=labels, bbox_to_anchor=(0.65, 0.2),
    #           loc='upper center', fancybox=True)
    plt.savefig("1B.png")


def main():
    data = read_csv("Datafile1.csv")

    fisher(data)


if __name__ == "__main__":
    main()
