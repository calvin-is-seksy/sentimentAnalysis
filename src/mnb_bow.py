#!/usr/bin/env python3
"""
mnb - bow
"""
import math
from utils import *

class MNB_BOW:
    def __init__(self, pos, neg):
        self.pCount = count(pos)
        self.nCount = count(neg)
        self.totalCount = self.pCount + self.nCount
        self.pos = pos
        self.neg = neg

    def train(self):
        self.features = {}
        self.features['pos'] = {}
        self.features['neg'] = {}

        self.priorP = math.log(self.pCount / self.totalCount)
        self.priorN = math.log(self.nCount / self.totalCount)

        # BoW straight into Multinomial Naive Bayes Features found significant increase in accuracy with log probability
        for w, c in self.pos.items():
            self.features['pos'][w] = math.log((int(c) + 1) / (self.pCount + self.totalCount))
        for w, c in self.neg.items():
            self.features['neg'][w] = math.log((int(c) + 1) / (self.nCount + self.totalCount))

    def testHelper(self, validationSet, posTestCount, negTestCount):
        scoreP = self.priorP
        scoreN = self.priorN
        penaltyPos = math.log(1 / (self.pCount + self.totalCount))
        penaltyNeg = math.log(1 / (self.nCount + self.totalCount))

        # print(penaltyPos, penaltyNeg)

        for line in lines(validationSet):
            words = line.strip().split()
            for feature in self.features:
                if feature == 'pos':
                    for word in words:
                        # print(scoreP)
                        if word in self.features['pos']:
                            scoreP += self.features['pos'][word]
                        elif word in self.features['neg']:
                            scoreP += penaltyPos
                elif feature == 'neg':
                    # print('\n\n\n')
                    for word in words:
                        # print(scoreN)
                        if word in self.features['neg']:
                            scoreN += self.features['neg'][word]
                        elif word in self.features['pos']:
                            scoreN += penaltyNeg

            # print(scoreP, scoreN)
            # break

            if scoreP > scoreN:
                posTestCount += 1
            elif scoreN > scoreP:
                negTestCount += 1
            else:
                assert(scoreP != scoreN)

        return(posTestCount, negTestCount)

    def test(self, filenames):
        TP, FN = self.testHelper(filenames[2], 0, 0)
        FP, TN = self.testHelper(filenames[3], 0, 0)
        return(TP, FN, FP, TN)
