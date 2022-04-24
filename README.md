# SMS Spam Classification

## About

This is a Mini-Project for CZ1115 (Introduction to Data Science and Artificial Intelligence) focusing on classifying Spam and Ham (not Spam) text messages from the UCI ML SMS Spam Collection Dataset.

For detailed walkthrough, please view the source code in order from:

1. [EDA](https://github.com/googlercolin/spam-classification/blob/main/EDA.ipynb)
2. [Model](https://github.com/googlercolin/spam-classification/blob/main/Model.ipynb)

**Known Issue**: If you are not able to view our Plotly graphs on Github code viewer, please download the .ipynb files locally.

## Contributors

1. [@angjustin](https://github.com/angjustin) - RandomizedSearch, Random Forest Classification, Gradient Boosting Trees Classification
2. [@CrazyCyfro](https://github.com/CrazyCyfro) - TF-IDF Vectorization, Multinomial Naive Bayes Classification
3. [@googlercolin](https://github.com/googlercolin) - Exploratory Data Analysis, SMOTE Upsampling, Support Vector Machine Classification

## Problem Definition

To find the ML model which produces the highest precision and accuracy in spam message classification.

## Models Used

1. Multinomial Naive Bayes
2. Random Forest
3. Gradient Boosting Trees
4. Support Vector Machines

## Conclusion

- Using ML to classify spam/ham has been a success, showing generally good precision and accuracy. However, high precision does not always mean high accuracy
- Random Forest seemed to perform the best in terms of producing the second highest test precision of 99.4%, and the third highest test accuracy of 97.6%, even without tuning its hyperparameters.
- TF-IDF was effective in vectorizing words into numerical vectors for model training, where it revealed the important and rare words.
- However, additional features did not improve performance. In fact, in some cases, performance became much worse.
- SMOTE upsampling may not be suitable for TF-IDF vectorized data.
- We can also explore other ways to augment words using the NLPAug library to insert and substitute similar words.

## What did we learn from this project?

- Data cleaning
- Tokenizing words and punctuation, Stemming
- Using TF-IDF (term frequency and inverse document frequency) to convert our stemmed messages into quantitative word vectors
- Handling imbalanced datasets using SMOTE resampling method and imblearn package
- Performing hyperparameter tuning using RandomizedSearch and GridSearch
- Concepts about Precision, Specificity, F1 Score, and AUC

## Requirements

- wordcloud
- nltk
- tables
- imbalanced-learn
- unidecode
- numpy
- requests
- nlpaug
- transformers
- imblearn
- umap-learn

## References

### Tokenization & Stopword Removal

https://www.nltk.org/api/nltk.tokenize.html \
https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/#h2_4

### TF-IDF

https://aiaspirant.com/bag-of-words/ \
https://monkeylearn.com/blog/what-is-tf-idf/

### Model Selection & Scoring Techniques

https://towardsdatascience.com/model-selection-in-text-classification-ac13eedf6146 \
https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124

### Handling Imbalanced Data

https://towardsdatascience.com/how-i-handled-imbalanced-text-data-ba9b757ab1d8 \
https://towardsdatascience.com/upsampling-with-smote-for-classification-projects-e91d7c44e4bf

### Multinomial Naive Bayes

https://stanford.edu/~jurafsky/slp3/slides/7_NB.pdf

### Random Forests

https://dl.acm.org/doi/10.1145/3357384.3357891 \
https://thesai.org/Downloads/Volume11No9/Paper_20-Grid_Search_Tuning_of_Hyperparameters.pdf

### Gradient Boosting Trees

https://www.machinelearningplus.com/machine-learning/an-introduction-to-gradient-boosting-decision-trees/ \
https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full

### Support Vector Machines

https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167 \
https://monkeylearn.com/text-classification-support-vector-machines-svm/

### Spam

https://www.cloudmark.com/en/resources/white-papers/sms-spam-overview-preserving-value-sms-texting \
https://www.cloudmark.com/en/resources/white-papers/true-cost-sms-spam-case-study \
https://www.todayonline.com/singapore/police-recover-s2-million-linked-ocbc-phishing-scam-121-local-bank-accounts-frozen-1817381
