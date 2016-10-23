# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag, word_tokenize
from wordcloud import WordCloud
wnl = nltk.WordNetLemmatizer()

osha = pd.read_csv('data/osha_clean_predict.csv')

stops = set(['employee', 'victim', 'worker'])
errors = set(['injures', 'sustains', 'explodes', 'suffers', \
    'fails', 'ignites', 'ignite', 'becomes', 'clearing', 'crush', \
    'swings', 'breaks', 'tears', 'commits', 'fractures', 'loading', \
    'amputates', 'punctures', 'kicks', 'smashes', 'strikes']) #pos error
    
def doublespace_multiline_drop(s):
    return s.split('  ')[0] #double space

def fix_pos(pos_list):
    return [(w, 'VB') if w in errors else (w, t) for w, t in pos_list]

def get_occupation(chunked):
    for n1 in chunked:
        if isinstance(n1, nltk.tree.Tree) and n1.label() == 'NP':
            for n2 in n1:
                if isinstance(n2, nltk.tree.Tree):
                    if n2.label() == 'NALL':
                        lem = [wnl.lemmatize(w) for w, t in n2]
                        filtered = [w for w in lem if w not in errors]
                        # reject 'worker' but accept 'farm worker'
                        if len(filtered) and filtered[0] not in stops:
                            return ' '.join(filtered)
    
pattern = r'''
VALL: {<VB.*>+}
NALL: {<NN.*>+}
POSALL: {<POS><NALL>}
NP: {^<DT|CD>?<NALL><POSALL>?<VALL>}
'''
chunker = nltk.RegexpParser(pattern)

osha_title_occupation = osha.title.apply(doublespace_multiline_drop).str.lower() \
    .apply(word_tokenize).apply(pos_tag).apply(fix_pos).apply(chunker.parse).apply(get_occupation)

osha_title_occupation = [x for x in osha_title_occupation if x is not None]
osha_title_occupation = pd.Series(osha_title_occupation)

print 'Top 10 risky occupations for OSHA dataset:'
osha_occupation_count = osha_title_occupation.groupby(osha_title_occupation).size().sort_values(ascending=False)
osha_occupation_count.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print osha_occupation_count.head(10)
print 'OSHA occupations word cloud:'
osha_word_string = ' '.join([w.replace(' ', '_') for w in osha_title_occupation.tolist()])
osha_word_cloud = WordCloud().generate(osha_word_string)
plt.imshow(osha_word_cloud)
plt.axis('off')
plt.show()
print