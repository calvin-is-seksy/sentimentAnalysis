#!/usr/bin/env python3
"""
Main file to run all 4 classifiers

Running command:
$ python NaiveBayesClassifer.py training_pos.txt training_neg.txt test_pos_private.txt test_neg_private.txt


"""
import sys

from mnb_bow import *
from mnb_tfidf import *
from gnb_bow import *
from gnb_tfidf import *

def main(filenames):
    results = {'ACCURACY MEASURE: TP, FN, FP, TN': [0,0,0,0]}

    pos, neg, occurPerReviewPos, occurPerReviewNeg, numReviews, occurPos, occurNeg = countTraining_gnb_bow(filenames)

    gnb_bow = GNB_BOW(pos, neg, occurPerReviewPos, occurPerReviewNeg, numReviews)
    gnb_bow.train()
    TP, FN, TN, FP = gnb_bow.test(filenames)
    results['gnb_bow'] = [TP, FN, TN, FP]

    gnb_tfidf = GNB_TFIDF(pos, neg, occurPerReviewPos, occurPerReviewNeg, numReviews, occurPos, occurNeg)
    gnb_tfidf.train()
    TP, FN, TN, FP = gnb_tfidf.test(filenames)
    results['gnb_tfidf'] = [TP, FN, TN, FP]

    mnb_bow = MNB_BOW(pos, neg)
    mnb_bow.train()
    TP, FN, TN, FP = mnb_bow.test(filenames)
    results['mnb_bow'] = [TP, FN, TN, FP]

    mnb_tfidf = MNB_TFIDF(pos, neg, occurPos, occurNeg, numReviews)
    mnb_tfidf.train()
    TP, FN, TN, FP = mnb_tfidf.test(filenames)
    results['mnb_tfidf'] = [TP, FN, TN, FP]

    print(results)


if __name__ == "__main__":
    filenames = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    main(filenames)
