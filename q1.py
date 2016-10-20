# -*- coding: utf-8 -*-
import string
#import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

msia = pd.read_csv('data/MsiaAccidentCases_clean.csv')
osha = pd.read_csv('data/osha_clean.csv')


wnl = nltk.WordNetLemmatizer()
stop = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#stop |= set(['die', 'kill'])
def lemmatize_df_col(df, col):
    res=[]
    for index, row in df.iterrows():
        if type(row[col]) is float:
            print index
            return
        text = row[col].lower()
        sents = sent_tokenize(text)
        pos = [pos_tag(word_tokenize(s)) for s in sents] #map
        pos = reduce(lambda x, y: x + y, pos) #flatten
        pos = filter(lambda (w, t): w.isalpha() and w not in stop, pos)
        text_lem = ' '.join([wnl.lemmatize(w, get_wordnet_pos(t)) for (w, t) in pos])
        res.append(text_lem)
    return res

print 'Distribution of causes for Msia Accident Cases dataset:'
msia_cause_count = msia.groupby('cause').size().sort_values(ascending=False)
msia_cause_count.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print msia_cause_count
print


print 'Training models based on Msia Accident Cases...'
print 'Prediction score based on Title:'
text_lem_list = lemmatize_df_col(msia, 'title')

vectorizer = TfidfVectorizer(max_df=0.9)
X = vectorizer.fit_transform(text_lem_list)
y = msia.cause

seed = 32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

dt = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)
print '\tDecision Tree:\t\t\t%f' % dt.score(X_test, y_test)

knn = KNeighborsClassifier(n_neighbors = 11, weights = 'distance', \
    metric = 'cosine', algorithm = 'brute').fit(X_train, y_train)
print '\tDecision Tree:\t\t\t%f' % knn.score(X, y)

mnb = MultinomialNB().fit(X_train, y_train)
print '\tNaive Bayesian:\t\t\t%f' % mnb.score(X_test, y_test)

svm = SVC(C=1000000.0, gamma='auto', kernel='rbf').fit(X_train, y_train)
print '\tSVM:\t\t\t\t%f' % svm.score(X_test, y_test)

lr = LogisticRegression().fit(X_train, y_train)
print '\tLogistic Regression:\t\t%f' % lr.score(X_test, y_test)

vc = VotingClassifier(estimators=[ \
    ('dt', dt), ('knn', knn), ('mnb', mnb), ('svm', svm), ('lr', lr) \
], voting='hard').fit(X_train, y_train)

print '\tEnsemble (Majority Vote):\t%f' % vc.score(X_test, y_test)

print 'Prediction score based on Summary:'
text_lem_list2 = lemmatize_df_col(msia, 'summary')
vectorizer2 = TfidfVectorizer(max_df=0.9)
X2 = vectorizer2.fit_transform(text_lem_list2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=seed)
vc2 = VotingClassifier(estimators=[ \
    ('dt', dt), ('knn', knn), ('mnb', mnb), ('svm', svm), ('lr', lr) \
], voting='hard').fit(X_train, y_train)
vc2.fit(X2_train, y2_train)

print '\tEnsemble (Majority Vote):\t%f' % vc2.score(X2_test, y2_test)
print

print 'Using Ensemble Model based on Titles of Msia dataset to predice Causes for OSHA dataset...'
print

text_lem_list_osha = lemmatize_df_col(osha, 'title') #title
#vocab = set(reduce(lambda x, y: x + y, [l.split() for l in text_lem_list]))

#vectorizer_osha = TfidfVectorizer(max_df=0.9, vocabulary=vectorizer.get_feature_names())
X_osha = vectorizer.transform(text_lem_list_osha)
osha_pred = vc.predict(X_osha)
osha['cause'] = pd.Series(osha_pred)

print 'Distribution of causes for OSHA Accident Cases dataset (predicted):'
osha_cause_count = osha.groupby('cause').size().sort_values(ascending=False)
osha_cause_count.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print osha_cause_count
print

osha.to_csv('data/osha_clean_predict.csv', index=False)