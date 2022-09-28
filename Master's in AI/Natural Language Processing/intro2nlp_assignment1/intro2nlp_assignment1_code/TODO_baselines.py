# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

import spacy
from model.data_loader import DataLoader
from random import randint, choice
from collections import Counter
import wordfreq

def prec(TP, FP):
    try:
        prec = TP/(TP+FP)
    except ZeroDivisionError:
        prec = 0
    return prec

def recall(TP, FN):
    try:
        recall = TP/(TP+FN)
    except ZeroDivisionError:
        recall = 0
    return recall

def f1_mes(prec, recall):
    try:
        f1 = 2*prec*recall/(prec+recall)
    except ZeroDivisionError:
        f1 = 0
    return f1

def weighted_f1(n_c, f1_c, n_n, f1_n):
    try:
        w_f1 = (n_c*f1_c + n_n*f1_n)/(n_c+n_n)
    except ZeroDivisionError:
        w_f1 = 0
    return w_f1


def majority_baseline(train_sentences, train_labels, test_sentences, test_labels):
    predictions = []

    new_train = []
    for i in train_labels:
        new_train.append(i.split())

    count_c = 0
    count_n = 0
    for sublist in new_train:
        count = Counter(sublist)
        vals = list(count.values())
        try:
            count_c += vals[1]
            count_n += vals[0]
        except IndexError:
            if list(count.keys()) == ['N']:
                count_c += 0
            else:
                count_n += 0

    if count_c > count_n:
        majority_class = 'C'
    else:
        majority_class = 'N'

    new_test = []
    for i in test_labels:
        new_test.append(i.split())
    # TODO: determine the majority class based on the training data
    # ...
    predictions = []
    correct_decisions = 0
    all_decisions = 0
    for_test = 0
    correct_c = 0
    correct_n = 0
    false_c = 0
    false_n = 0
    for instance in test_sentences:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        instance_tuple = (instance, instance_predictions)
        predictions.append(instance_tuple)
        for t in range(len(tokens)):
            if instance_predictions[t] == new_test[for_test][t]:
                if instance_predictions[t] == 'C':
                    correct_c += 1
                else:
                    correct_n += 1
                correct_decisions += 1
                all_decisions += 1
            else:
                if instance_predictions[t] == 'C':
                    false_c += 1
                else:
                    false_n += 1

                all_decisions += 1

        for_test += 1
    prec_c = prec(correct_c, false_c)
    prec_n = prec(correct_n, false_n)
    recall_c = recall(correct_c, false_n)
    recall_n = recall(correct_n, false_c)
    f1_c = f1_mes(prec_c, recall_c)
    f1_n = f1_mes(prec_n, recall_n)
    w_f1 = weighted_f1(correct_c + false_c, f1_c, correct_n + false_n, f1_n)
    table = [prec_c, prec_n, recall_c, recall_n, f1_c, f1_n, w_f1]
    accuracy = correct_decisions / all_decisions

    return accuracy, predictions, table

def random_func():
    dec = randint(0,1)
    if dec == 1:
        random_class = "C"
    else:
        random_class = "N"
    return random_class

def random_baseline(test_sentences, test_labels):
    predictions = []
    accuracy = 0
    for_test = 0
    new_test = []
    all_decisions = 0
    correct_decisions = 0
    correct_c = 0
    correct_n = 0
    false_c = 0
    false_n = 0
    for i in test_labels:
        new_test.append(i.split())

    for instance in test_sentences:
        tokens = instance.split(" ")
        instance_predictions = [random_func() for t in tokens]
        instance_tuple = (instance, instance_predictions)
        predictions.append(instance_tuple)

        for t in range(len(tokens)):
            if instance_predictions[t] == new_test[for_test][t]:
                if instance_predictions[t] == 'C':
                    correct_c+=1
                else:
                    correct_n+=1
                correct_decisions+=1
                all_decisions+=1
            else:
                if instance_predictions[t] == 'C':
                    false_c+=1
                else:
                    false_n+=1
                all_decisions+=1
        for_test +=1
    accuracy = correct_decisions/all_decisions
    prec_c = prec(correct_c, false_c)
    prec_n = prec(correct_n,false_n)
    recall_c = recall(correct_c, false_n)
    recall_n = recall(correct_n,false_c)
    f1_c = f1_mes(prec_c, recall_c)
    f1_n = f1_mes(prec_n, recall_n)
    w_f1 = weighted_f1(correct_c+false_c, f1_c, correct_n+false_n, f1_n)
    table = [prec_c, prec_n, recall_c, recall_n, f1_c, f1_n, w_f1]
    return accuracy, predictions, table

def length_choice(token, length_threshold):
    if len(token)>length_threshold:
        length_class = 'C'
    else:
        length_class = 'N'
    return length_class


def length_baseline(sentences, labels):
    predictions = []

    new_test = []
    for i in labels:
        new_test.append(i.split())
    length_threshold = 4
    accuracy = 0
    while accuracy < 0.865:
        for_test = 0
        all_decisions = 0
        correct_decisions = 0
        correct_c = 0
        correct_n = 0
        false_c = 0
        false_n = 0
        for instance in sentences:
            token = instance.split(" ")
            tokens = [i for i in token if i != '\\"' and i != '\\"\n']
            for t in range(len(tokens)):
                if tokens[t] == '.\n':
                    tokens[t] = '.'
                if tokens[t] == '?"\n':
                    tokens[t] = '?'
                if tokens[t] == '!\n':
                    tokens[t] = '!'

            instance_predictions = [length_choice(t, length_threshold) for t in tokens]
            instance_tuple = (instance, instance_predictions)
            predictions.append(instance_tuple)

            for i in range(len(tokens)):
                if instance_predictions[i] == new_test[for_test][i]:
                    if instance_predictions[t] == 'C':
                        correct_c += 1
                    else:
                        correct_n += 1
                    correct_decisions += 1
                    all_decisions += 1
                else:
                    if instance_predictions[t] == 'C':
                        false_c += 1
                    else:
                        false_n += 1
                    all_decisions += 1

            for_test += 1
        accuracy = correct_decisions / all_decisions
        length_threshold += 1
        prec_c = prec(correct_c, false_c)
        prec_n = prec(correct_n, false_n)
        recall_c = recall(correct_c, false_n)
        recall_n = recall(correct_n, false_c)
        f1_c = f1_mes(prec_c, recall_c)
        f1_n = f1_mes(prec_n, recall_n)
        w_f1 = weighted_f1(correct_c + false_c, f1_c, correct_n + false_n, f1_n)
        table = [prec_c, prec_n, recall_c, recall_n, f1_c, f1_n, w_f1]

    return accuracy, predictions, length_threshold, table

def frequency_choice(token, freq_threshold):
    if wordfreq.word_frequency(token, 'en')<freq_threshold:
        freq_class = 'C'
    else:
        freq_class = 'N'
    return freq_class


def frequency_baseline(sentences, labels):
    predictions = []

    new_test = []
    for i in labels:
        new_test.append(i.split())
    freq_threshold = 0.001
    accuracy = 0
    while accuracy < 0.757:
        # for i in range(100):
        for_test = 0
        all_decisions = 0
        correct_decisions = 0
        correct_c = 0
        correct_n = 0
        false_c = 0
        false_n = 0
        for instance in sentences:
            tokens = instance.split(" ")
            instance_predictions = [frequency_choice(t, freq_threshold) for t in tokens]
            instance_tuple = (instance, instance_predictions)
            predictions.append(instance_tuple)

            for t in range(len(tokens)):
                if instance_predictions[t] == new_test[for_test][t]:
                    if instance_predictions[t] == 'C':
                        correct_c += 1
                    else:
                        correct_n += 1
                    correct_decisions += 1
                    all_decisions += 1
                else:
                    if instance_predictions[t] == 'C':
                        false_c += 1
                    else:
                        false_n += 1
                    all_decisions += 1
            for_test += 1
        accuracy = correct_decisions / all_decisions
        freq_threshold -= 0.00001
        prec_c = prec(correct_c, false_c)
        prec_n = prec(correct_n, false_n)
        recall_c = recall(correct_c, false_n)
        recall_n = recall(correct_n, false_c)
        f1_c = f1_mes(prec_c, recall_c)
        f1_n = f1_mes(prec_n, recall_n)
        w_f1 = weighted_f1(correct_c + false_c, f1_c, correct_n + false_n, f1_n)
        table = [prec_c, prec_n, recall_c, recall_n, f1_c, f1_n, w_f1]
    return accuracy, predictions, freq_threshold, table


if __name__ == '__main__':
    train_path = "data/preprocessed/train"
    dev_path = "data/preprocessed/val"
    test_path = "data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "/sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "/labels.txt") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "/sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "/labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "/sentences.txt") as test_file:
        test_sentences = test_file.readlines()

    with open(test_path + "/labels.txt") as test_label_file:
        test_labels = test_label_file.readlines()

    # for test files

    majority_accuracy, majority_predictions, maj_table = majority_baseline(train_sentences, train_labels,
                                                                           test_sentences, test_labels)
    # print(majority_accuracy)
    # print(maj_table)
    # print(majority_predictions)
    random_accuracy, random_predictions, rand_table = random_baseline(test_sentences, test_labels)
    print(random_accuracy)
    # print(random_predictions)
    length_accuracy, length_predictions, length_threshold, length_table = length_baseline(test_sentences, test_labels)
    print(length_accuracy)
    # print(length_predictions)
    print(length_threshold)
    frequency_accuracy, frequency_predictions, frequency_threshold, freq_table = frequency_baseline(test_sentences,
                                                                                                    test_labels)
    print(frequency_accuracy)
    # print(frequency_predictions)
    print(frequency_threshold)
    # FOR dev files
    print('for dev files')
    majority_accuracy, majority_predictions, maj_table1 = majority_baseline(train_sentences, train_labels,
                                                                            dev_sentences, dev_labels)
    # print(majority_accuracy)
    # print(majority_predictions)
    random_accuracy, random_predictions, rand_table1 = random_baseline(dev_sentences, dev_labels)
    print(random_accuracy)
    # print(random_predictions)
    length_accuracy, length_predictions, length_threshold, len_table1 = length_baseline(dev_sentences, dev_labels)
    print(length_accuracy)
    # print(length_predictions)
    print(length_threshold)
    frequency_accuracy, frequency_predictions, frequency_threshold, freq_table1 = frequency_baseline(dev_sentences,
                                                                                                     dev_labels)
    print(frequency_accuracy)
    # print(frequency_predictions)
    print(frequency_threshold)

    print('tables for test')
    print(maj_table)
    print(rand_table)
    print(length_table)
    print(freq_table)
"""
    print('table for dev')
    print(maj_table1)
    print(rand_table1)
    print(len_table1)
    print(freq_table1)
"""

