#!/usr/bin/env python3
"""
gnb- bow
"""
import math
from utils import *

class GNB_BOW:
    def __init__(self, pos, neg, occurPerReviewPos, occurPerReviewNeg, numReviews):
        self.pos_count = count(pos)
        self.neg_count = count(neg)
        self.doc_count = self.pos_count + self.neg_count
        self.pos = pos
        self.neg = neg
        self.occurPos = occurPerReviewPos
        self.occurNeg = occurPerReviewNeg
        self.numReviews = numReviews

    def train(self):
        self.features = {}
        self.features['posFeatures'] = {}
        self.features['negFeatures'] = {}

        # Gathering a priori probabilities by class
        self.priorLogPos = math.log(self.pos_count / self.doc_count)
        self.priorLogNeg = math.log(self.neg_count / self.doc_count)

        """
        Each for loop below is calculating probabilities of each feature
        for each class.
        Backslashes in calculations are added for readiblity and serve as 
        line breaks.
        """
        for word, count in self.pos.items():
            mean = (0.5*count) / self.numReviews
            stddev = math.sqrt(sum([pow(x-mean,2) for x in self.occurPos[word]])/float(len(self.occurPos[word]))) # TODO: subtract 1 in denom?
            self.features['posFeatures'][word] = [mean, stddev]

        for word, count in self.neg.items():
            mean = (0.5*count) / self.numReviews
            stddev = math.sqrt(sum([pow(x-mean,2) for x in self.occurNeg[word]])/float(len(self.occurNeg[word]))) # TODO: subtract 1 in denom?
            self.features['negFeatures'][word] = [mean, stddev]

    """
    Takes a given test document and make a classification decision based off
    of a max probability.
    @param document: Test document used to make classification decision.
    return: A two-tuple with the classification decision and its corresponding
    log-space probability.
    """
    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def testHelper(self, validationSet, posTestCount, negTestCount):
        pos_val = self.priorLogPos
        neg_val = self.priorLogNeg

        # Smoothed probabilities are calculated below, these are used when a
        # word in the test document is not found in the given class but is found
        # in another class's feature dict
        smooth_pos = math.log(1 / (self.pos_count + self.doc_count))
        smooth_neg = math.log(1 / (self.neg_count + self.doc_count))

        for line in lines(data_dir + validationSet):
            words = line.strip().split()
            for feature in self.features:
                if feature == 'posFeatures':
                    for word in words:
                        if word in self.features['posFeatures']:
                            mean, stddev = self.features['posFeatures'][word]
                            pos_val += self.calculateProbability(word, mean, stddev)
                        elif word in self.features['negFeatures']:
                            pos_val += smooth_pos
                elif feature == 'negFeatures':
                    for word in words:
                        if word in self.features['negFeatures']:
                            mean, stddev = self.features['negFeatures'][word]
                            neg_val += self.calculateProbability(word, mean, stddev)
                        elif word in self.features['posFeatures']:
                            neg_val += smooth_neg

                # print(pos_val, neg_val)

                if pos_val > neg_val:
                    posTestCount += 1
                    # print('pos', line)
                elif neg_val > pos_val:
                    negTestCount += 1
                    # print('neg', line)
                else: # TODO: Reorg w assert before if
                    assert(pos_val != neg_val)

        return(posTestCount, negTestCount)

    def test(self):
        posTestCount, negTestCount = 0, 0

        posTestCount, negTestCount = self.testHelper(filenames[2], posTestCount, negTestCount)
        print(posTestCount,negTestCount)
        posTestCount, negTestCount = self.testHelper(filenames[3], posTestCount, negTestCount)
        print(posTestCount, negTestCount)

        return(posTestCount, negTestCount)
