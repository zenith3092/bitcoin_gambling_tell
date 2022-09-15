# Performance: 25 min 51 sec
# This module is to clasify sheet1 data by address.

import pandas as pd
from tqdm import tqdm
import json

sheet1 = pd.read_csv('../../input_data/sheet1.csv')
sheet2 = pd.read_csv('../../input_data/sheet2.csv')
sheet1['date'] = pd.to_datetime(sheet1['date'],unit='s')
sheet1.sort_values(by='date')

tx_data = open('tx_data.json')
tx_data = json.load(tx_data)

address_list = list( set( sheet2['address'] ) )
address_dict = dict( zip( address_list, [{'in_btc':[],'out_btc':[],'total_btc':[],'in_date':[],'out_date':[],'total_date':[],'tx_size':[],'tx_amount':[],'tx_fee':[],'tx_fee_to_in':[]} for item in range(len(address_list))] ) )

def get_add(num):
    add = sheet1.loc[num]
    address = add['address']
    if address in address_list:

        btc = int(add['btc'])
        date = str(add['date'])

        if add['type'] == 'IN':
            address_dict[address]['in_btc'].append(btc)
            address_dict[address]['in_date'].append(date)
        elif add['type'] == 'OUT':
            address_dict[address]['out_btc'].append(btc)
            address_dict[address]['out_date'].append(date)
        
        address_dict[address]['total_btc'].append(btc)
        address_dict[address]['total_date'].append(date)
        
        tx = add['txid']
        tx_info = tx_data[tx]
        tx_size = tx_info['total_num']
        tx_amount = tx_info['total_amount']
        tx_fee = tx_info['fee']
        tx_fee_to_in = tx_info['fee_to_in']
        address_dict[address]['tx_size'].append(tx_size)
        address_dict[address]['tx_amount'].append(tx_amount)
        address_dict[address]['tx_fee'].append(tx_fee)
        address_dict[address]['tx_fee_to_in'].append(tx_fee_to_in)

print('Loop starts!')

for num in tqdm(range(len(sheet1))):
    get_add(num)

with open('../../input_data/address_classified.json', 'w+') as file:
    json.dump(address_dict, file)