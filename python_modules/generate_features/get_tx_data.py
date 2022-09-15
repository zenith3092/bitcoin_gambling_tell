# Performance: 4 sec
# This module is to create features of all the tx ID.

import json
from tqdm import tqdm

tx_sort = open('../../input_data/tx_classified.json', 'r')
tx_sort = json.load(tx_sort)

tx_list = list(tx_sort.keys())
tx_dict = {}

def add_dict(in_data, out_data):
    tmp_dict = {}
    in_num = len(in_data)
    out_num = len(out_data)
    total_num = in_num + out_num

    in_amount = sum(in_data)
    out_amount = sum(out_data)
    total_amount = in_amount + out_amount

    fee = in_amount - out_amount
    fee_to_in = fee / in_amount if in_amount !=0 else 0

    tmp_dict['in_num'] = in_num
    tmp_dict['out_num'] = out_num
    tmp_dict['total_num'] = total_num
    tmp_dict['in_amount'] = in_amount
    tmp_dict['out_amount'] = out_amount
    tmp_dict['total_amount'] = total_amount
    tmp_dict['fee'] = fee
    tmp_dict['fee_to_in'] = fee_to_in 

    tx_dict[txid] = tmp_dict

print('Loop Start')

for num in tqdm(range(len(tx_list))):
    txid = tx_list[num]
    in_data = tx_sort[txid]['in']
    out_data = tx_sort[txid]['out']

    if (in_data==[]) and (out_data==[]):
        continue
    else:
        add_dict(in_data, out_data)

with open('../../input_data/tx_data.json', 'w+') as file:
    json.dump(tx_dict, file)

print('Successfully')