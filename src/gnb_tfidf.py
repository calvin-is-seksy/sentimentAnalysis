#!/usr/bin/env python3
"""
gnb-tfidf
"""
import math
from utils import *

class GNB_TFIDF:
    def __init__(self, pos, neg, occurPerReviewPos, occurPerReviewNeg, numReviews, occurPos, occurNeg):
        self.pos_count = count(pos)                         # total number of words in + dataset
        self.neg_count = count(neg)                         # total number of words in - dataset
        self.doc_count = self.pos_count + self.neg_count    # total number of words in both classes
        self.pos = pos                                      # counter object of + dataset
        self.neg = neg                                      # counter object of - dataset
        self.occurPerReviewPos = occurPerReviewPos          # occurrence hashmap of + dataset
        self.occurPerReviewNeg = occurPerReviewNeg          # occurrence hashmap of - dataset
        self.occurPos = occurPos
        self.occurNeg = occurNeg
        self.numReviews = numReviews                        # number of reviews in the whole dataset

    def train(self):
        self.features = {}
        self.features['posFeatures'] = {}
        self.features['negFeatures'] = {}
        self.weights = {}
        self.weights['pos'] = {}
        self.weights['neg'] = {}

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
            TF = count / self.pos_count
            IDF = math.log(self.numReviews / self.occurPos[word])
            self.features['posFeatures'][word] = TF * IDF

        for word, count in self.neg.items():
            TF = count / self.neg_count
            IDF = math.log(self.numReviews / self.occurNeg[word])
            self.features['negFeatures'][word] = TF * IDF

        self.weights['pos']['mean'] = sum(self.features['posFeatures'].values()) / float(len(self.features['posFeatures']))
        self.weights['neg']['mean'] = sum(self.features['negFeatures'].values()) / float(len(self.features['negFeatures']))

        self.weights['pos']['stddev'] = math.sqrt(sum([pow(x-self.weights['pos']['mean'],2) \
                                            for x in self.features['posFeatures'].values()]) \
                                            / float(len(self.features['posFeatures'])))

        self.weights['neg']['stddev'] = math.sqrt(sum([pow(x - self.weights['neg']['mean'], 2) \
                                                       for x in self.features['negFeatures'].values()]) \
                                                  / float(len(self.features['negFeatures'])))

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
        smooth_pos = math.log(1 / (self.pos_count + self.doc_count)) * 2500
        smooth_neg = math.log(1 / (self.neg_count + self.doc_count)) * 2500

        # print(smooth_pos, smooth_neg)

        for line in lines(data_dir + validationSet):
            words = line.strip().split()
            for feature in self.features:
                if feature == 'posFeatures':
                    for word in words:
                        if word in self.features['posFeatures']:
                            tfidf = self.features['posFeatures'][word]
                            mean = self.weights['pos']['mean']
                            stddev = self.weights['pos']['stddev']
                            pos_val += self.calculateProbability(tfidf, mean, stddev)
                        elif word in self.features['negFeatures']:
                            pos_val += smooth_pos

                        # print(pos_val)

                elif feature == 'negFeatures':
                    # print('\n\n\n')
                    for word in words:
                        if word in self.features['negFeatures']:
                            tfidf = self.features['negFeatures'][word]
                            mean = self.weights['neg']['mean']
                            stddev = self.weights['neg']['stddev']
                            neg_val += self.calculateProbability(tfidf, mean, stddev)
                        elif word in self.features['posFeatures']:
                            neg_val += smooth_neg

                #         print(neg_val)
                #
                # print(pos_val, neg_val)

                if pos_val > neg_val:
                    posTestCount += 1
                elif neg_val > pos_val:
                    negTestCount += 1
                else: # TODO: Reorg w assert before if
                    assert(pos_val != neg_val)

            # break

        return(posTestCount, negTestCount)

    def test(self):
        TP, FN = self.testHelper(filenames[2], 0, 0)
        FP, TN = self.testHelper(filenames[3], 0, 0)
        return(TP, FN, FP, TN)
