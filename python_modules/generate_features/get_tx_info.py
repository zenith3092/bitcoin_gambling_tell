# Performance: 0.51 sec
# This module is to put tx_data in the same list so that we can get the statistic about tx_data easily.

import json
from tqdm import tqdm
import time

start = time.time()
tx_data = open('../../input_data/tx_data.json', 'r')
tx_data = json.load(tx_data)

tx_list = list(tx_data.keys())
tx_dict = {'total_amount':[], 'fee_to_in':[], 'total_num':[]}

def add_dict(txid):
    tx_dict['total_amount'].append(tx_data[txid]['total_amount'])
    tx_dict['fee_to_in'].append(tx_data[txid]['fee_to_in'])
    tx_dict['total_num'].append(tx_data[txid]['total_num'])

print('Loop Start')
for num in tqdm(range(len(tx_list))):
    txid = tx_list[num]
    add_dict(txid)


with open('../../input_data/tx_info.json', 'w+') as file:
    json.dump(tx_dict, file)

print('Successfully')
end = time.time()
print(end-start)