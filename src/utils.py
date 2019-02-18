#!/usr/bin/env python3
from collections import Counter
import math

data_dir = "../data/"
# filenames = ["training_pos.txt",
#              "training_neg.txt",
#              "test_pos_public.txt",
#              "test_neg_public.txt"]

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

# TODO: Refactor and potentially merge into 1 function with above^
def countTrainTFIDF():
    """
        takes file and returns the counter object (word, n_appearance) of whole document
        ADDITIONALLY: returns a dictionary of the # reviews the term appears in
    """
    cnt_pos = Counter()
    cnt_neg = Counter()
    occurrencePos = {}
    occurrenceNeg = {}
    numReviews = 0
    for line in lines(data_dir + filenames[0]):
        numReviews += 1
        words = line.strip().split()
        cnt_pos.update(words)
        for word in set(words):
            if word in occurrencePos:
                occurrencePos[word] += 1
            else:
                occurrencePos[word] = 1

    for line in lines(data_dir + filenames[1]):
        numReviews += 1
        words = line.strip().split()
        cnt_neg.update(words)
        for word in set(words):
            if word in occurrenceNeg:
                occurrenceNeg[word] += 1
            else:
                occurrenceNeg[word] = 1

    for word in cnt_pos:
        cnt_pos[word] += 1

    for word in cnt_neg:
        cnt_neg[word] += 1

    return cnt_pos, cnt_neg, occurrencePos, occurrenceNeg, numReviews

def countTraining_gnb_bow(filenames):
    """
        takes file and returns the counter object (word, n_appearance) of whole document
        ADDITIONALLY: returns a dictionary of the # reviews the term appears in
    """
    cnt_pos = Counter()
    cnt_neg = Counter()
    occurPerReviewPos = {}
    occurPerReviewNeg = {}
    occurrencePos = {}
    occurrenceNeg = {}
    numReviews = 0
    for line in lines(data_dir + filenames[0]):
        tempCounter = Counter()
        numReviews += 1
        words = line.strip().split()
        cnt_pos.update(words)
        tempCounter.update(words)
        for word, numOccur in tempCounter.items():
            if word in occurPerReviewPos:
                occurPerReviewPos[word].append(numOccur)
            else:
                occurPerReviewPos[word] = [numOccur]
        for word in set(words):
            if word in occurrencePos:
                occurrencePos[word] += 1
            else:
                occurrencePos[word] = 1

    for line in lines(data_dir + filenames[1]):
        tempCounter = Counter()
        numReviews += 1
        words = line.strip().split()
        cnt_neg.update(words)
        tempCounter.update(words)
        for word, numOccur in tempCounter.items():
            if word in occurPerReviewNeg:
                occurPerReviewNeg[word].append(numOccur)
            else:
                occurPerReviewNeg[word] = [numOccur]
        for word in set(words):
            if word in occurrenceNeg:
                occurrenceNeg[word] += 1
            else:
                occurrenceNeg[word] = 1

    for word in cnt_pos:
        cnt_pos[word] += 1

    for word in cnt_neg:
        cnt_neg[word] += 1

    return cnt_pos, cnt_neg, occurPerReviewPos, occurPerReviewNeg, numReviews, occurrencePos, occurrenceNeg

def count(pol_dict):
    # returns the count of total number of words in the dataset

    counter = 0
    for entry in pol_dict:
        counter += int(pol_dict[entry])
    return counter

def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
	separated = [0,1]
	summaries = {}
	for label in separated:
		summaries[label] = summarize(instances)
	return summaries

