import csv
import pandas as pd
from pandas import *
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np


def determine_decision(m01, c01, m02, c02, m03, c03, data, loss_matrix, output_name):
    correct1, correct2, correct3 = 0, 0, 0
    correct1list, correct2list, correct3list = [], [], []
    incorrect1, incorrect2, incorrect3 = 0, 0, 0
    incorrect1list, incorrect2list, incorrect3list = [], [], []

    confusion_matrix = np.zeros((3, 3))

    for num in range(0, len(data)):
        value = data.iloc[num].to_numpy()[:-1]
        class_prior = data.iloc[num].to_numpy()[-1]

        multivariate_1 = multivariate_normal.pdf(value, m01, c01)
        multivariate_2 = multivariate_normal.pdf(value, m02, c02)
        multivariate_3 = multivariate_normal.pdf(value, m03, c03)
        multivariate_list = [multivariate_1, multivariate_2, multivariate_3]
        multivariate_array = np.array(multivariate_list)

        new_matrix = np.matmul(multivariate_array, loss_matrix)
        smallest = 10000000000
        decision = 0
        for num, x in enumerate(new_matrix):
            if x < smallest:
                decision = num + 1
                smallest = x

        confusion_matrix[int(class_prior) - 1, decision - 1] += 1

        if class_prior == 1 and decision == 1:
            correct1 += 1
            correct1list.append(value)
        elif class_prior == 1 and decision != 1:
            incorrect1 += 1
            incorrect1list.append(value)
        elif decision == 2 and class_prior == 2:
            correct2 += 1
            correct2list.append(value)
        elif decision != 2 and class_prior == 2:
            incorrect2 += 1
            incorrect2list.append(value)
        elif decision == 3 and class_prior == 3:
            correct3 += 1
            correct3list.append(value)
        elif decision != 3 and class_prior == 3:
            incorrect3 += 1
            incorrect3list.append(value)

    confusion_matrix /= 10000
    confusion_matrix *= 100

    for i in range(0, 3):
        for j in range(0, 3):
            print(confusion_matrix[i][j], end=" ")
        print("\n")

    # stuff for ploting graph
    correct1x, correct1y, correct1z = [], [], []
    correct2x, correct2y, correct2z = [], [], []
    correct3x, correct3y, correct3z = [], [], []
    for i in range(0, len(correct1list)):
        correct1x.append(correct1list[i][0])
        correct1y.append(correct1list[i][1])
        correct1z.append(correct1list[i][2])
    for i in range(0, len(correct2list)):
        correct2x.append(correct2list[i][0])
        correct2y.append(correct2list[i][1])
        correct2z.append(correct2list[i][2])
    for i in range(0, len(correct3list)):
        correct3x.append(correct3list[i][0])
        correct3y.append(correct3list[i][1])
        correct3z.append(correct3list[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(correct1x), np.array(correct1y),
               np.array(correct1z), marker='o', color='g')
    ax.scatter(np.array(correct2x), np.array(correct2y),
               np.array(correct2z), marker='s', color='g')
    ax.scatter(np.array(correct3x), np.array(correct3y),
               np.array(correct3z), marker='^', color='g')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title_name = output_name[:-4] + " Loss Matrix Classification Data"
    ax.set_title(title_name)

    incorrect1x, incorrect1y, incorrect1z = [], [], []
    incorrect2x, incorrect2y, incorrect2z = [], [], []
    incorrect3x, incorrect3y, incorrect3z = [], [], []
    for i in range(0, len(incorrect1list)):
        incorrect1x.append(incorrect1list[i][0])
        incorrect1y.append(incorrect1list[i][1])
        incorrect1z.append(incorrect1list[i][2])
    for i in range(0, len(incorrect2list)):
        incorrect2x.append(incorrect2list[i][0])
        incorrect2y.append(incorrect2list[i][1])
        incorrect2z.append(incorrect2list[i][2])
    for i in range(0, len(incorrect3list)):
        incorrect3x.append(incorrect3list[i][0])
        incorrect3y.append(incorrect3list[i][1])
        incorrect3z.append(incorrect3list[i][2])

    ax.scatter(np.array(incorrect1x), np.array(incorrect1y),
               np.array(incorrect1z), marker='o', color='r')
    ax.scatter(np.array(incorrect2x), np.array(incorrect2y),
               np.array(incorrect2z), marker='s', color='r')
    ax.scatter(np.array(incorrect3x), np.array(incorrect3y),
               np.array(incorrect3z), marker='^', color='r')
    plt.savefig(output_name)


def main():
    m01 = [0, 0, 0]
    c01 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    m02 = [3, -3, 0]
    c02 = [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
    m03 = [-3, 3, 0]
    c03 = [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
    data = read_csv("Datafile2.csv")

    loss_matrix_a10 = np.array([[0, 1, 1], [1, 0, 1], [10, 10, 0]])
    determine_decision(m01, c01, m02, c02, m03, c03,
                       data, loss_matrix_a10, "a10.png")
    loss_matrix_a100 = np.array([[0, 1, 1], [1, 0, 1], [100, 100, 0]])
    determine_decision(m01, c01, m02, c02, m03, c03,
                       data, loss_matrix_a100, "a100.png")


if __name__ == "__main__":
    main()
