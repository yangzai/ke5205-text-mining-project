# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag, word_tokenize
wnl = nltk.WordNetLemmatizer()

osha = pd.read_csv('data/osha_clean_predict.csv')

stops = set(['employee', 'victim', 'worker'])
errors = set(['injures', 'sustains', 'explodes', 'suffers', \
    'fails', 'ignites', 'ignite', 'becomes', 'clearing', 'crush', \
    'swings', 'breaks', 'tears', 'commits', 'fractures', 'loading', \
    'amputates', 'punctures', 'kicks', 'smashes', 'strikes']) #pos error
ones = set(['1', 'one'])

def doublespace_multiline_drop(s):
    return s.split('  ')[0] #double space

def fix_pos(pos_list):
    return [(w, 'VB') if w in errors else (w, t) for w, t in pos_list]

def is_single_victim(chunked):
    count = 0
    for n1 in chunked:
        if isinstance(n1, nltk.tree.Tree) and n1.label() == 'NP':
            for n2 in n1:
                if not isinstance(n2, nltk.tree.Tree):
                    n2_tag = n2[1]
                    if n2_tag == 'DT':
                        if count:
                            return False
                        count += 1
                    elif n2_tag == 'CD': #assume no zero
                        if count or n2[0] not in ones:
                            return False
                        else:
                            count += 1
    return True

pattern = r'''
VALL: {<VB.*>+}
NALL: {<NN.*>+}
POSALL: {<POS><NALL>}
NP: {^<DT|CD>?<NALL>(<POSALL><VALL>|<VALL>(<CC><DT|CD><NALL>)?)}
'''
chunker = nltk.RegexpParser(pattern)

osha_title_victim_sm = osha.title.apply(doublespace_multiline_drop).str.lower() \
    .apply(word_tokenize).apply(pos_tag).apply(fix_pos).apply(chunker.parse) \
    .apply(is_single_victim).apply(lambda b: 'single' if b else 'multiple')

osha_title_victim_sm = [x for x in osha_title_victim_sm if x is not None]
osha_title_victim_sm = pd.Series(osha_title_victim_sm)

print 'Single vs multiple victims for OSHA dataset:'
osha_victim_sm_count = osha_title_victim_sm.groupby(osha_title_victim_sm).size().sort_values(ascending=False)
osha_victim_sm_count.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print osha_victim_sm_count
print
print 'single/multiple = %f' % (1.0*osha_victim_sm_count.single/osha_victim_sm_count.multiple)
print