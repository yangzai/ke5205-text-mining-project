# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import inflection as _

import requests
from lxml import html

import time

msia = pd.read_excel('data/MsiaAccidentCases.xlsx')
osha = pd.read_excel('data/osha.xlsx', header=None)

# fix header spacing and standardise
msia.columns = [_.parameterize(name, '_') for name in msia.columns]

# unify other and others
msia.loc[msia.cause == 'Others', 'cause'] = 'Other'

# fix corrupted data by rescraping
osha_title = osha[2]
is_dirty = (osha_title == 'InspectionOpen DateSICEstablishment Name') \
    | (osha_title.apply(lambda s: len(s.split())) < 2)
url = 'https://www.osha.gov/pls/imis/accidentsearch.accident_detail'
length = np.sum(is_dirty)
count = 0
print 'Re-scraping...'
for index, row in osha.loc[is_dirty].iterrows():
    if not(count % 10) or count == length:
        print '%d/%d' % (count, length)
    page = requests.get(url, params={'id': row[0]})
    tree = html.fromstring(page.content)
    
    # tbody may or may not be present
    elem_list = tree.cssselect('table tr > td[colspan="8"]')
    summary = elem_list[1].text
    keywords = elem_list[2].cssselect('div')[0].text_content() \
        .split('\n')[1].split(', ')
    
    osha.loc[index, 2] = summary
    osha.loc[index, 3] = '  '.join(keywords) #double spaces
    count = count + 1
    time.sleep(0.5)

# trim spaces
for i in range(1, 4):
    osha.loc[:, i] = osha[i].astype('string').str.strip()


osha.to_csv('data/osha_clean.csv', index=False, header=False)
msia.to_csv('data/MsiaAccidentCases_clean.csv', index=False)
