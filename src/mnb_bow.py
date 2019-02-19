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

        for word, count in self.pos.items():
            self.features['pos'][word] = math.log((int(count) + 1) / (self.pCount + self.totalCount))
        for word, count in self.neg.items():
            self.features['neg'][word] = math.log((int(count) + 1) / (self.nCount + self.totalCount))

    def testHelper(self, validationSet, posTestCount, negTestCount):
        scoreP = self.priorP
        scoreN = self.priorN

        penaltyPos = math.log(1 / (self.pCount + self.totalCount))
        penaltyNeg = math.log(1 / (self.nCount + self.totalCount))


        for line in lines(validationSet):
            words = line.strip().split()
            for feature in self.features:
                if feature == 'pos':
                    for word in words:
                        if word in self.features['pos']:
                            scoreP += self.features['pos'][word]
                        elif word in self.features['neg']:
                            scoreP += penaltyPos
                elif feature == 'neg':
                    for word in words:
                        if word in self.features['neg']:
                            scoreN += self.features['neg'][word]
                        elif word in self.features['pos']:
                            scoreN += penaltyNeg

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
