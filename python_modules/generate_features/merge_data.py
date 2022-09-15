#!/usr/bin/env python
# coding: utf-8
# Author: Xianglin Wu (xianglin3092@gmail.com)
# Performance 0.11 sec
# This module is to merge the tag from sheet2 into sheet3

import time
import pandas as pd

start = time.time()

sheet3 = pd.read_csv('../../input_data/sheet3.csv')
sheet2 = pd.read_csv('../../input_data/sheet2.csv')

target = sheet2['type']
label_list = []
for num in range(len(target)):
    if target[num] == 'POOL':
        label_list.append('0')
    elif target[num] == 'EXCHANGE':
        label_list.append('1')
    elif target[num] == 'GAMBLING':
        label_list.append('2')

sheet2['label'] = label_list

merge_data = sheet2[['address', 'type', 'label']]

new_sheet3 = sheet3.merge(merge_data, how='left', on='address')
column = new_sheet3.columns.to_list()
column.remove('type')
column.remove('label')
column.insert(2,'type')
column.insert(3,'label')
del column[0]
new_sheet3 = new_sheet3[column]
new_sheet3.to_csv('../../input_data/tag_sheet3.csv')

end = time.time()
print(end - start)


