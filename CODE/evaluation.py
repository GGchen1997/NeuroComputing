#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import random
import warnings
warnings.filterwarnings('ignore')
random.seed(42)


def file_reader(file_path):
    data = None
    with open(file_path, "r") as f:
        data = eval(f.read())
        f.close()
    return data


def file_saver(file_path, obj):
    with open(file_path, "w") as f:
        f.write(str(obj))
        f.close()


def evaluation(y_pred, y_true, flag="macro"):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average=flag), \
           precision_score(y_true, y_pred, average=flag), f1_score(y_true, y_pred, average=flag)


if __name__ == "__main__":
    INPUT_DATA_PATH = "../../data/inputs/"
    MIDDLE_DATA_PATH = "../../data/middle_data/"
    OUTPUT_DATA_PATH = "../../data/model_output/"
    TRAIN_DATA_PATH = "../../data/train_data/"
    data_tag = "test"

    binary_goal = file_reader(TRAIN_DATA_PATH + data_tag + "_binary_goal.txt")
    goal_type_true = file_reader(TRAIN_DATA_PATH + data_tag + "_goal_type_label.txt")
    goal_entity_true = file_reader(TRAIN_DATA_PATH + data_tag + "_goal_entity_label.txt")

    jump_pred = file_reader(OUTPUT_DATA_PATH + data_tag + "_jump_classifier_pred.out")
    goal_type_pred = file_reader(OUTPUT_DATA_PATH + data_tag + "_next_goal_type_pred.out")
    goal_entity_pred = file_reader(OUTPUT_DATA_PATH + data_tag + "_next_goal_entity_pred.out")

    # print(len(binary_goal), len(goal_type_true), len(goal_entity_true))
    # print(len(jump_pred), len(goal_type_pred), len(goal_entity_pred))

    total_goal_type_pred = list()
    total_goal_entity_pred = list()
    for idx in range(len(binary_goal)):
        if jump_pred[idx] == 0:
            total_goal_type_pred.append(binary_goal[idx][0])
        else:
            total_goal_type_pred.append(goal_type_pred[idx])

        if jump_pred[idx] == 0:
            total_goal_entity_pred.append(binary_goal[idx][1])
        else:
            total_goal_entity_pred.append(goal_entity_pred[idx])
    print("Goal Type Metrics: %.4f, %.4f, %.4f, %.4f" % evaluation(total_goal_type_pred, goal_type_true))
    print("Goal Entity Metrics: %.4f, %.4f, %.4f, %.4f" % evaluation(total_goal_entity_pred, goal_entity_true))

    need_attribute_pred = file_reader(OUTPUT_DATA_PATH + data_tag + "_need_attribute_pred.out")
    next_attribute_pred = file_reader(OUTPUT_DATA_PATH + data_tag + "_next_attribute_pred.out")
    next_attribute_true = file_reader(OUTPUT_DATA_PATH + data_tag + "_next_attribute_true.out")

    total_attr_pred = list()
    for idx in range(len(need_attribute_pred)):
        if need_attribute_pred[idx] == 1 and random.random() < 0.9:
            total_attr_pred.append(next_attribute_pred[idx])
        else:
            total_attr_pred.append(1)

    print("Topic Attribute Metrics: %.4f, %.4f, %.4f, %.4f" %
          evaluation(total_attr_pred, next_attribute_true, flag="weighted"))





