import csv
import pandas as pd
from pandas import *
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def roc_curve(m01, m02, c01, c02, m1, c1, data):
    false_positive_list, true_positive_list, error_rate_list = [], [], []
    gamma = 0
    # Calculate Gamma values based on curated data
    while gamma < 100:
        true_negative, false_negative, false_positive, true_positive = 0, 0, 0, 0
        for num in range(0, len(data)):
            value = data.iloc[num].to_numpy()[:-1]
            gamma_val = (multivariate_normal.pdf(value, m1, c1) / (0.5 * multivariate_normal.pdf(
                value, m01, c01) + 0.5 * multivariate_normal.pdf(value, m02, c02)))

            if gamma_val > gamma:
                decision = 1
            else:
                decision = 0

            if data["class_prior"][num] == 0 and decision == 0:
                true_negative += 1
            elif data["class_prior"][num] == 1 and decision == 0:
                false_negative += 1
            elif data["class_prior"][num] == 0 and decision == 1:
                false_positive += 1
            elif data["class_prior"][num] == 1 and decision == 1:
                true_positive += 1

        true_positive_rate = true_positive / (true_positive + false_negative)
        false_positive_rate = false_positive / (false_positive + true_negative)
        error_rate = false_positive_rate * \
            0.65 + (1 - true_positive_rate) * 0.35

        true_positive_list.append(true_positive_rate)
        false_positive_list.append(false_positive_rate)
        error_rate_list.append(error_rate)

        gamma += 0.25

    # Optimal probability of error estimate
    min_error_rate = 100000000
    min_index = 0
    for num, error_rate in enumerate(error_rate_list):
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_index = num

    print("The estimated minimum error rate is : " + str(min_error_rate))
    print("The gamma value at that point is: " + str(min_index * 0.25))

    # Theoretical miniumum probability of errorr
    true_negative, false_negative, false_positive, true_positive = 0, 0, 0, 0
    for num in range(0, len(data)):
        value = data.iloc[num].to_numpy()[:-1]
        gamma_val = (multivariate_normal.pdf(value, m1, c1) / (0.5 * multivariate_normal.pdf(
            value, m01, c01) + 0.5 * multivariate_normal.pdf(value, m02, c02)))

        if gamma_val > 13/7:
            decision = 1
        else:
            decision = 0

        if data["class_prior"][num] == 0 and decision == 0:
            true_negative += 1
        elif data["class_prior"][num] == 1 and decision == 0:
            false_negative += 1
        elif data["class_prior"][num] == 0 and decision == 1:
            false_positive += 1
        elif data["class_prior"][num] == 1 and decision == 1:
            true_positive += 1

    theoretical_true_positive_rate = true_positive / \
        (true_positive + false_negative)
    theoretical_false_positive_rate = false_positive / \
        (false_positive + true_negative)
    theoretical_error_rate = theoretical_false_positive_rate * \
        0.65 + (1 - theoretical_true_positive_rate) * 0.35

    print("The theoretical minimum error rate is : " +
          str(theoretical_error_rate))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(false_positive_list, true_positive_list, 'b.', markersize=2)
    ax.plot(false_positive_list[min_index], true_positive_list[min_index],
            'r.', markersize=10)
    ax.plot(theoretical_false_positive_rate, theoretical_true_positive_rate,
            'g.', markersize=10)
    ax.set_xlabel("False Positive Rates")
    ax.set_ylabel("True Positive Rates")
    ax.set_title("ERM classification using the knowledge of true data pdf")
    legend_elements = [Line2D([0], [0], color='tab:red', lw=3, label='Line'),
                       Line2D([0], [0], color='tab:green', lw=3, label='Line')]

    # Create the figure
    labels = ['Estimated Optimal Point', 'Theoretical Optimal Point']

    # Put a legend below current axis
    ax.legend(handles=legend_elements, labels=labels, bbox_to_anchor=(0.65, 0.2),
              loc='upper center', fancybox=True)
    plt.savefig("1A2.png")


def main():
    m01 = [3, 0]
    c01 = [[2, 0], [0, 1]]
    m02 = [0, 3]
    c02 = [[1, 0], [0, 2]]
    m1 = [2, 2]
    c1 = [[1, 0], [0, 1]]

    data = read_csv("Datafile1.csv")

    roc_curve(m01, m02, c01, c02, m1, c1, data)


if __name__ == "__main__":
    main()
