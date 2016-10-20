# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize#, ne_chunk
from nltk.corpus import stopwords, wordnet

msia = pd.read_csv('data/MsiaAccidentCases_clean.csv')
osha = pd.read_csv('data/osha_clean_predict.csv')
msia['title'] = msia.title.astype('string')
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

def multiline_drop(s):
    return s.split('  ')[0]
#print ne_chunk(pos_tag(word_tokenize('I have a African American pen.')))

#def title_preposition_chunk(df):
#    s = set()
#    for index, row in df.iterrows():
#        pos = pos_tag(word_tokenize(row.title))
#        
#        group = []
#        for w, t in pos:
#            if t == 'IN':
#                group.append(w)
#            elif len(group):
#                s.add(' '.join(group))
#                group = []
#    return s
#print title_preposition_chunk(msia)
#print title_preposition_chunk(osha)

stops = ['height', 'object', 'exposure', 'fall', 'slip', 'abdomen', 'knee', 'head', 'illness', 'heat']
starts = ['in', 'between', 'by', 'with', 'from', 'on', 'off']
def get_objects(chunked):
    res = []
    for n1 in chunked:
        if isinstance(n1, nltk.tree.Tree) and n1.label() == 'NP':
            is_valid = True
            nall = ''
            for n2 in n1:
                if isinstance(n2, nltk.tree.Tree):
                    n2_label = n2.label();
                    if n2_label == 'INALL':
                        if n2.leaves()[-1][0] not in starts: is_valid = False 
                    if n2_label == 'NALL':
                        lem = [wnl.lemmatize(w) for w, t in n2]
                        filtered = [w for w in lem if w not in stops and not (w.endswith('tion') or w.endswith('sion'))]
                        nall = ' '.join(filtered)
            if is_valid and len(nall): res.append(nall)
    return res

pattern = r'''
INALL: {<IN|RP>+}
NALL: {<NN.*>+}
NP: {<INALL><DT>?<NALL>}
'''
chunker = nltk.RegexpParser(pattern)

msia_title_objects = msia.title.apply(multiline_drop).str.lower() \
    .apply(word_tokenize).apply(pos_tag).apply(chunker.parse).apply(get_objects)

msia_title_objects = [x for x in msia_title_objects if len(x)]
msia_title_objects = pd.Series(reduce(lambda x, y: x + y, msia_title_objects))
#msia_title_objects = reduce(lambda x, y: x | set(y), msia_title_objects, set())

print 'Top 10 objects for Msia dataset:'
msia_object_count = msia_title_objects.groupby(msia_title_objects).size().sort_values(ascending=False)
msia_object_count.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print msia_object_count
print

osha_title_objects = osha.title.apply(multiline_drop).str.lower() \
    .apply(word_tokenize).apply(pos_tag).apply(chunker.parse).apply(get_objects)

osha_title_objects = [x for x in osha_title_objects if len(x)]
osha_title_objects = pd.Series(reduce(lambda x, y: x + y, osha_title_objects))

print 'Top 10 objects for Osha dataset:'
osha_object_count = osha_title_objects.groupby(osha_title_objects).size().sort_values(ascending=False)
osha_object_count.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print osha_object_count
print

#print chunker.parse(pos_tag(word_tokenize('Died being caught in between machines')))

            