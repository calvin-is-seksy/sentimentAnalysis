#!/usr/bin/env python3
from collections import Counter

data_dir = "../data/"
filenames = ["training_pos.txt",
             "training_neg.txt",
             "test_pos_public.txt",
             "test_neg_public.txt"]

def lines(infile):
    with open(infile, 'r') as fp:
        for line in fp:
            yield line

def countTrainBOW():
    """
    takes file and returns the counter object (word, n_appearance) of whole document
    :param filenames:
    :return:
    """
    cnt_pos = Counter()
    cnt_neg = Counter()
    for line in lines(data_dir + filenames[0]):
        words = line.strip().split()
        cnt_pos.update(words)

    for line in lines(data_dir + filenames[1]):
        words = line.strip().split()
        cnt_neg.update(words)

    # TODO: Build an overall vocab set?
    # vocab = list(myCounter.keys())

    for word in cnt_pos:
        cnt_pos[word] += 1

    for word in cnt_neg:
        cnt_neg[word] += 1

    return cnt_pos, cnt_neg

def countTrainTFIDF():
    """
        takes file and returns the counter object (word, n_appearance) of whole document
        ADDITIONALLY: returns a dictionary of the # reviews the term appears in
    """
    cnt_pos = Counter()
    cnt_neg = Counter()
    occurrence = {}
    numReviews = 0
    for line in lines(data_dir + filenames[0]):
        words = line.strip().split()
        cnt_pos.update(words)
        for word in words:
            if word in occurrence:
                occurrence[word] += 1
            else:
                occurrence[word] = 1

    for line in lines(data_dir + filenames[1]):
        numReviews += 1
        words = line.strip().split()
        cnt_neg.update(words)
        for word in words:
            if word in occurrence:
                occurrence[word] += 1
            else:
                occurrence[word] = 1

    for word in cnt_pos:
        cnt_pos[word] += 1

    for word in cnt_neg:
        cnt_neg[word] += 1

    return cnt_pos, cnt_neg, occurrence, numReviews

def count(pol_dict):
    # returns the count of total number of words in the dataset

    counter = 0
    for entry in pol_dict:
        counter += int(pol_dict[entry])
    return counter
