#!/usr/bin/env python3
from collections import Counter

def lines(infile):
    with open(infile, 'r') as fp:
        for line in fp:
            yield line

def train_prep(filenames):
    """
        takes file and returns the counter object (word, n_appearance) of whole document
        ADDITIONALLY: returns a dictionary of the # reviews the term appears in
    """
    pCounter = Counter()
    nCounter = Counter()
    occurrencePos = {}
    occurrenceNeg = {}
    numReviews = 0
    for line in lines(filenames[0]):
        numReviews += 1
        words = line.strip().split()
        pCounter.update(words)
        for word in set(words):
            if word in occurrencePos:
                occurrencePos[word] += 1
            else:
                occurrencePos[word] = 1

    for line in lines(filenames[1]):
        numReviews += 1
        words = line.strip().split()
        nCounter.update(words)
        for word in set(words):
            if word in occurrenceNeg:
                occurrenceNeg[word] += 1
            else:
                occurrenceNeg[word] = 1

    for word in pCounter:
        pCounter[word] += 1

    for word in nCounter:
        nCounter[word] += 1

    return pCounter, nCounter, numReviews, occurrencePos, occurrenceNeg

def count(pol_dict):
    # returns the count of total number of words in the dataset
    counter = 0
    for entry in pol_dict:
        counter += int(pol_dict[entry])
    return counter