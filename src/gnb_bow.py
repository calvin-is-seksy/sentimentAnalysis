#!/usr/bin/env python3
"""
gnb- bow
"""
import math
from utils import *

class GNB_BOW:
    def __init__(self, pos, neg, numReviews):
        self.pCount = count(pos)
        self.nCount = count(neg)
        self.totalCount = self.pCount + self.nCount
        self.pos = pos
        self.neg = neg
        self.numReviews = numReviews

    def train(self):
        self.features = {}
        self.features['pos'] = {}
        self.features['neg'] = {}
        self.weights = {}
        self.weights['pos'] = {}
        self.weights['neg'] = {}

        self.priorP = math.log(self.pCount / self.totalCount)
        self.priorN = math.log(self.nCount / self.totalCount)

        # BoW
        for word, count in self.pos.items():
            self.features['pos'][word] = math.log((int(count) + 1) \
                                                          / (self.pCount + self.totalCount))
        for word, count in self.neg.items():
            self.features['neg'][word] = math.log((int(count) + 1) \
                                                          / (self.nCount + self.totalCount))

        # Gaussian Naive Bayes
        self.weights['pos']['mean'] = sum(self.features['pos'].values()) / float(len(self.features['pos']))
        self.weights['neg']['mean'] = sum(self.features['neg'].values()) / float(len(self.features['neg']))

        self.weights['pos']['stddev'] = math.sqrt(sum([pow(x-self.weights['pos']['mean'],2) \
                                            for x in self.features['pos'].values()]) \
                                            / float(len(self.features['pos'])))

        self.weights['neg']['stddev'] = math.sqrt(sum([pow(x - self.weights['neg']['mean'], 2) \
                                                       for x in self.features['neg'].values()]) \
                                                  / float(len(self.features['neg'])))

    def calculateProbability(self, x, mean, stddev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent

    def testHelper(self, validationSet, posTestCount, negTestCount):
        scoreP = self.priorP
        scoreN = self.priorN
        penaltyPos = math.log(1 / (self.pCount + self.totalCount)) * .6
        penaltyNeg = math.log(1 / (self.nCount + self.totalCount)) * .6

        # print(penaltyPos, penaltyNeg)

        for line in lines(validationSet):
            words = line.strip().split()
            for feature in self.features:
                if feature == 'pos':
                    for word in words:
                        # print(scoreP)
                        if word in self.features['pos']:
                            bowCoeff = self.features['pos'][word]
                            mean = self.weights['pos']['mean']
                            stddev = self.weights['pos']['stddev']
                            scoreP += self.calculateProbability(bowCoeff, mean, stddev)
                        elif word in self.features['neg']:
                            scoreP += penaltyPos

                elif feature == 'neg':
                    # print("\n\n\n")
                    for word in words:
                        # print(scoreN)
                        if word in self.features['neg']:
                            bowCoeff = self.features['neg'][word]
                            mean = self.weights['neg']['mean']
                            stddev = self.weights['neg']['stddev']
                            scoreN += self.calculateProbability(bowCoeff, mean, stddev)
                        elif word in self.features['pos']:
                            scoreN += penaltyNeg

            # print(scoreP, scoreN)

            # break

            if scoreP > scoreN:
                posTestCount += 1
            elif scoreN > scoreP:
                negTestCount += 1
            else: # TODO: Reorg w assert before if
                assert(scoreP != scoreN)

            # break

        return(posTestCount, negTestCount)

    def test(self, filenames):
        TP, FN = self.testHelper(filenames[2], 0, 0)
        FP, TN = self.testHelper(filenames[3], 0, 0)
        return(TP, FN, FP, TN)
