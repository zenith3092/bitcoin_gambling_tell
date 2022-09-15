# performance 13 min 55 sec
# This module is to classify sheet1 data into two kinds.

import pandas as pd
from tqdm import tqdm
import json

sheet1 = pd.read_csv('../../input_data/sheet1.csv',encoding='utf-8')
sheet2 = pd.read_csv('../../input_data/sheet2.csv',encoding='utf-8')

txid_list = list( set( sheet1['txid'] ))

txid_dict = dict( zip( txid_list, [{'in':[],'out':[]} for item in range(len(txid_list))] ) )

def get_tx(num):
    tx = sheet1.loc[num]
    txid = tx['txid']
    btc = int(tx['btc'])
    if tx['type'] == 'IN':
        txid_dict[txid]['in'].append( btc )
    elif tx['type'] == 'OUT':
        txid_dict[txid]['out'].append( btc )

print('Loop Start')

for num in tqdm(range( len( sheet1 ) )):
    get_tx(num)

with open('../../input_data/tx_classified.json', 'w+') as file:
    json.dump(txid_dict, file)

print('Successfully')