# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from wordcloud import WordCloud
wnl = nltk.WordNetLemmatizer()

msia = pd.read_csv('data/MsiaAccidentCases_clean.csv')
osha = pd.read_csv('data/osha_clean_predict.csv')

pp_errors = set(['clearing', 'loading'])
other_errors = set(['injures', 'sustains', 'explodes', 'suffers', 'fails', \
    'ignites', 'ignite', 'becomes', 'crush', 'swings', 'breaks', 'tears', \
    'commits', 'fractures', 'amputates', 'punctures', 'kicks', 'smashes', 'strikes']) #pos error
wnl_errors = {
    'installing': 'install',
    'installed': 'install',
    'riding': 'ride'
}
past_to_bes = set(['was', 'were'])

def period_multiline_drop(s): # 
    return s.split('  ')[0] #double space
    
def fix_pos_with_pp(pos_list):
    return [
        ((w, 'VBG') if w in pp_errors else
        (w, 'VB') if w in other_errors else (w, t))
        if t.startswith('NN') else (w, t) for w, t in pos_list
    ]

def past_to_be_retag(pos_list):
    return [(w, 'PASTTOBE') if w in past_to_bes and t == 'VBD' else (w, t) for w, t in pos_list]

def get_activities(chunked):
    res = []
    for n1 in chunked:
        if isinstance(n1, nltk.tree.Tree) and n1.label() == 'NP':
            for n2 in n1:
                if isinstance(n2, nltk.tree.Tree) and n2.label() == 'SCOPE':
                    for n3 in n2:
                        if isinstance(n3, nltk.tree.Tree) and n3.label().startswith('ACT'):
                            lem = [wnl_errors[w] if w in wnl_errors else wnl.lemmatize(w, 'v') for w, t in n3.leaves()]
                            res.append(' '.join(lem))
    return res

#ACT1: striking, being struck
pattern = r'''
PERSON: {<PRP>|<NN.*><\#>?<CD>?}
ACT1: {<VBG><VB.*>*<RP>?}
ACT2: {<VB><RP>?}
SCOPE: {<PASTTOBE><.*>*(<ACT1>|<VBN><.*>*<TO><ACT2>)}
NP: {<PERSON><.*>*<SCOPE>}
'''
chunker = nltk.RegexpParser(pattern)

osha_summary_activities = osha.summary.str.lower().apply(period_multiline_drop) \
    .apply(word_tokenize).apply(pos_tag).apply(fix_pos_with_pp) \
    .apply(past_to_be_retag).apply(chunker.parse).apply(get_activities)

osha_summary_activities = [x for x in osha_summary_activities if len(x)]
osha_summary_activities = pd.Series(reduce(lambda x, y: x + y, osha_summary_activities))

print 'Top 10 accident objects for OSHA dataset:'
osha_activity_count = osha_summary_activities.groupby(osha_summary_activities).size().sort_values(ascending=False)
osha_activity_count.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print osha_activity_count.head(10)
print 'OSHA objects word cloud:'
osha_word_string = ' '.join([w.replace(' ', '_') for w in osha_summary_activities.tolist()])
osha_word_cloud = WordCloud().generate(osha_word_string)
plt.imshow(osha_word_cloud)
plt.axis('off')
plt.show()
print