# Sentiment Analysis # 

Classifying movie reviews as positive or negative. This implementation is done without any of Python's great NLP packages. Includes feature extractors such as Bag of Words and TF-IDF. Once the features were extracted, the model was trained with both a Gaussian Naive Bayes and Multinomial Naive Bayes. Everything was implemented in pure Python and none of that SKLearn or NLTK stuff. Always fun to dig into the theory a bit more.


Execute the program like this: 
```
$ python NaiveBayesClassifer.py training_pos.txt training_neg.txt test_pos_private.txt test_neg_private.txt
```
