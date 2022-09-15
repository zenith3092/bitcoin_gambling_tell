#!/usr/bin/env python
# coding: utf-8
# Author: Xianglin Wu (xianglin3092@gmail.com)
# performance 5 sec
# This module is to create features of all the address

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from scipy.stats import skew
from scipy.stats import kurtosis


# get tx percentile data
tx_info = open('../../input_data/tx_info.json', 'r')
tx_info = json.load(tx_info)

ttl_am_arr = np.array(tx_info['total_amount'])
fee_in_arr = np.array(tx_info['fee_to_in'])
ttl_num_arr = np.array(tx_info['total_num'])

ttl_am_pr80 = np.percentile(ttl_am_arr, 80)
fee_in_pr80 = np.percentile(fee_in_arr, 80)
ttl_num_pr80 = np.percentile(ttl_num_arr, 80)

# set address data
address_sort = open('../../input_data/address_classified.json', 'r')
address_sort = json.load(address_sort)

address_list = list(address_sort.keys())

address_data_dict = {'address':[],
             'n_in':[],
             'n_out':[],
             'n_ttl':[],
             'avg_am_in':[],
             'avg_am_out':[],
             'avg_am_ttl':[],
             'ttl_am_in':[],
             'ttl_am_out':[],
             'ttl_am_ttl':[],
             'balance':[],
             'balance_label':[],
             'max_am_in':[],
             'max_am_out':[],
             'max_am_ttl':[],
             'min_am_in':[],
             'min_am_out':[],
             'min_am_ttl':[],
             'med_am_in':[],
             'med_am_out':[],
             'med_am_ttl':[],
             'q1_am_in':[],
             'q1_am_out':[],
             'q1_am_ttl':[],
             'q3_am_in':[],
             'q3_am_out':[],
             'q3_am_ttl':[],
             'range_am_in':[],
             'range_am_out':[],
             'range_am_ttl':[],
             'std_am_in':[],
             'std_am_out':[],
             'std_am_ttl':[],
            #  'n_mout_am_in':[],
            #  'n_mout_am_out':[],
            #  'n_mout_am_ttl':[],
            #  'n_eout_am_in':[],
            #  'n_eout_am_out':[],
            #  'n_eout_am_ttl':[],
             'ske_am_in':[],
             'ske_am_out':[],
             'ske_am_ttl':[],
             'kur_am_in':[],
             'kur_am_out':[],
             'kur_am_ttl':[],
             'ske_am_in_label':[],
             'ske_am_out_label':[],
             'ske_am_ttl_label':[],
             'kur_am_in_label':[],
             'kur_am_out_label':[],
             'kur_am_ttl_label':[],
             'time_range_in':[],   # day
             'time_range_out':[],
             'time_range_ttl':[],
             'n_fre_in':[],
             'n_fre_out':[],
             'n_fre_ttl':[],
             'am_fre_in':[],
             'am_fre_out':[],
             'am_fre_ttl':[],
             'max_fee_to_in':[],
             'n_pr80_fee_to_in':[],
             'n_join_pr80_am':[], 
             'n_in_to_out_ratio':[],
             'n_join_huge':[],
             'n_join_pr80':[]} # >10

def count_fee_ratio_num_pr80(tx_size, tx_fee_to_in):
    count=0
    for num in range(len(tx_size)):
        if tx_size[num] < 10:
            if tx_fee_to_in[num] > fee_in_pr80:
                count += 1
    return count

def sign_label(feature):
    if feature >= 0:
        return 0
    else:
        return 1

def add_data(address):
    address_info = address_sort[address]
    in_btc = np.array(address_info['in_btc'])
    if len(address_info['out_btc']) == 0:
        out_btc = np.array([0])
        n_out = 0
    else:
        out_btc = np.array(address_info['out_btc'])
        n_out = len(out_btc)
    total_btc = np.array(address_info['total_btc'])
    in_date = np.array(address_info['in_date'], dtype='datetime64')
    if len(address_info['out_date']) == 0:
        out_date = np.array(['2022-09-13 09:17:30'], dtype='datetime64')
    else:
        out_date = np.array(address_info['out_date'], dtype='datetime64')
    total_date = np.array(address_info['total_date'], dtype='datetime64')
    tx_size = np.array(address_info['tx_size'])
    tx_amount = np.array(address_info['tx_amount'])
    tx_fee = np.array(address_info['tx_fee'])
    tx_fee_to_in = np.array(address_info['tx_fee_to_in'])
    try:
        n_in = len(in_btc)
        n_ttl = len(total_btc)

        avg_am_in = in_btc.mean()
        avg_am_out = out_btc.mean()
        avg_am_ttl = total_btc.mean()

        ttl_am_in = in_btc.sum()
        ttl_am_out = out_btc.sum()
        ttl_am_ttl = total_btc.sum()

        max_am_in = in_btc.max()
        max_am_out = out_btc.max()
        max_am_ttl = total_btc.max()

        min_am_in = in_btc.min()
        min_am_out = out_btc.min()
        min_am_ttl = total_btc.min()

        med_am_in = np.median(in_btc)
        med_am_out = np.median(out_btc)
        med_am_ttl = np.median(total_btc)

        q1_am_in = np.percentile(in_btc, 25)
        q1_am_out = np.percentile(out_btc, 25)
        q1_am_ttl = np.percentile(total_btc, 25)

        q3_am_in = np.percentile(in_btc, 75)
        q3_am_out = np.percentile(out_btc, 75)
        q3_am_ttl = np.percentile(total_btc, 75)

        range_am_in = max_am_in - min_am_in
        range_am_out = max_am_out - min_am_out
        range_am_ttl = max_am_ttl - min_am_ttl

        std_am_in = np.std(in_btc)
        std_am_out = np.std(out_btc)
        std_am_ttl = np.std(total_btc)

        # outlier ====================
        # iqr_in = q3_am_in - q1_am_in
        # iqr_out = q3_am_out - q1_am_out
        # iqr_ttl = q3_am_ttl - q1_am_ttl

        # in_m_norm = q3_am_in + 1.5 * iqr_in
        # out_m_norm = q3_am_out + 1.5 * iqr_out
        # ttl_m_norm = q3_am_ttl + 1.5 * iqr_ttl

        # in_e_norm = q3_am_in + 3 * iqr_in
        # out_e_norm = q3_am_out + 3 * iqr_out
        # ttl_e_norm = q3_am_ttl + 3 * iqr_ttl

        # n_mout_am_in =  np.count_nonzero( (tx_size > in_m_norm) & (tx_size < in_e_norm) )
        # n_mout_am_out =  np.count_nonzero( (tx_size > out_m_norm) & (tx_size < out_e_norm) )
        # n_mout_am_ttl =  np.count_nonzero( (tx_size > ttl_m_norm) & (tx_size < ttl_e_norm)  )
        # n_eout_am_in =  np.count_nonzero( tx_size > in_e_norm )
        # n_eout_am_out =  np.count_nonzero( tx_size > out_e_norm )
        # n_eout_am_ttl =  np.count_nonzero( tx_size > ttl_e_norm )
        # =============================

        balance = ttl_am_out - ttl_am_in
        balance_label = sign_label(balance)
        balance = abs(balance)

        ske_am_in = skew(in_btc)
        ske_am_in_label = sign_label(ske_am_in)
        ske_am_in = abs(ske_am_in)

        ske_am_out = skew(out_btc)
        ske_am_out_label = sign_label(ske_am_out)
        ske_am_out = abs(ske_am_out)

        ske_am_ttl = skew(total_btc)
        ske_am_ttl_label = sign_label(ske_am_ttl)
        ske_am_ttl = abs(ske_am_ttl)

        kur_am_in = kurtosis(in_btc)
        kur_am_in_label = sign_label(kur_am_in)
        kur_am_in = abs(kur_am_in)

        kur_am_out = kurtosis(out_btc)
        kur_am_out_label = sign_label(kur_am_in)
        kur_am_out = abs(kur_am_out)

        kur_am_ttl = kurtosis(total_btc)
        kur_am_ttl_label = sign_label(kur_am_ttl)
        kur_am_ttl = abs(kur_am_ttl)

        time_range_in = abs(int(str(in_date.max() - in_date.min()).split()[0]))
        time_range_out = abs(int(str(out_date.max() - out_date.min()).split()[0]))
        time_range_ttl = abs(int(str(total_date.max() - total_date.min()).split()[0]))

        n_fre_in = (n_in / time_range_in) if time_range_in != 0 else n_in
        n_fre_out = (n_out / time_range_out) if time_range_out != 0 else n_out
        n_fre_ttl = (n_ttl / time_range_ttl) if time_range_ttl != 0 else n_ttl

        am_fre_in = (ttl_am_in / time_range_in) if time_range_in != 0 else ttl_am_in
        am_fre_out = (ttl_am_out / time_range_out) if time_range_out != 0 else ttl_am_out
        am_fre_ttl = (ttl_am_ttl / time_range_ttl) if time_range_ttl != 0 else ttl_am_ttl

        max_fee_to_in = abs(tx_fee_to_in.max())
        # n_pr80_fee_to_in = count_fee_ratio_num_pr80(tx_size, tx_fee_to_in)
        n_pr80_fee_to_in = np.count_nonzero( tx_fee_to_in < fee_in_pr80)
        n_join_pr80_am = np.count_nonzero( tx_amount < ttl_am_pr80)
        n_in_to_out_ratio = (n_in / n_out) if n_out != 0 else 0
        n_join_huge = np.count_nonzero(tx_size > 10)
        n_join_pr80 = np.count_nonzero(tx_size > ttl_num_pr80)

        address_data_dict['address'].append(address)
        address_data_dict['n_in'].append(n_in)
        address_data_dict['n_out'].append(n_out)
        address_data_dict['n_ttl'].append(n_ttl)
        address_data_dict['avg_am_in'].append(avg_am_in)
        address_data_dict['avg_am_out'].append(avg_am_out)
        address_data_dict['avg_am_ttl'].append(avg_am_ttl)
        address_data_dict['ttl_am_in'].append(ttl_am_in)
        address_data_dict['ttl_am_out'].append(ttl_am_out)
        address_data_dict['ttl_am_ttl'].append(ttl_am_ttl)
        address_data_dict['balance'].append(balance)
        address_data_dict['balance_label'].append(balance_label)
        address_data_dict['max_am_in'].append(max_am_in)
        address_data_dict['max_am_out'].append(max_am_out)
        address_data_dict['max_am_ttl'].append(max_am_ttl)
        address_data_dict['min_am_in'].append(min_am_in)
        address_data_dict['min_am_out'].append(min_am_out)
        address_data_dict['min_am_ttl'].append(min_am_ttl)
        address_data_dict['med_am_in'].append(med_am_in)
        address_data_dict['med_am_out'].append(med_am_out)
        address_data_dict['med_am_ttl'].append(med_am_ttl)
        address_data_dict['q1_am_in'].append(q1_am_in)
        address_data_dict['q1_am_out'].append(q1_am_out)
        address_data_dict['q1_am_ttl'].append(q1_am_ttl)
        address_data_dict['q3_am_in'].append(q3_am_in)
        address_data_dict['q3_am_out'].append(q3_am_out)
        address_data_dict['q3_am_ttl'].append(q3_am_ttl)
        address_data_dict['range_am_in'].append(range_am_in)
        address_data_dict['range_am_out'].append(range_am_out)
        address_data_dict['range_am_ttl'].append(range_am_ttl)
        address_data_dict['std_am_in'].append(std_am_in)
        address_data_dict['std_am_out'].append(std_am_out)
        address_data_dict['std_am_ttl'].append(std_am_ttl)
        # address_data_dict['n_mout_am_in'].append(n_mout_am_in)
        # address_data_dict['n_mout_am_out'].append(n_mout_am_out)
        # address_data_dict['n_mout_am_ttl'].append(n_mout_am_ttl)
        # address_data_dict['n_eout_am_in'].append(n_eout_am_in)
        # address_data_dict['n_eout_am_out'].append(n_eout_am_out)
        # address_data_dict['n_eout_am_ttl'].append(n_eout_am_ttl)
        address_data_dict['ske_am_in'].append(ske_am_in)
        address_data_dict['ske_am_out'].append(ske_am_out)
        address_data_dict['ske_am_ttl'].append(ske_am_ttl)
        address_data_dict['kur_am_in'].append(kur_am_in)
        address_data_dict['kur_am_out'].append(kur_am_out)
        address_data_dict['kur_am_ttl'].append(kur_am_ttl)

        address_data_dict['ske_am_in_label'].append(ske_am_in_label)
        address_data_dict['ske_am_out_label'].append(ske_am_out_label)
        address_data_dict['ske_am_ttl_label'].append(ske_am_ttl_label)
        address_data_dict['kur_am_in_label'].append(kur_am_in_label)
        address_data_dict['kur_am_out_label'].append(kur_am_out_label)
        address_data_dict['kur_am_ttl_label'].append(kur_am_ttl_label)

        address_data_dict['time_range_in'].append(time_range_in)
        address_data_dict['time_range_out'].append(time_range_out)
        address_data_dict['time_range_ttl'].append(time_range_ttl)
        address_data_dict['n_fre_in'].append(n_fre_in)
        address_data_dict['n_fre_out'].append(n_fre_out)
        address_data_dict['n_fre_ttl'].append(n_fre_ttl)
        address_data_dict['am_fre_in'].append(am_fre_in)
        address_data_dict['am_fre_out'].append(am_fre_out)
        address_data_dict['am_fre_ttl'].append(am_fre_ttl)

        address_data_dict['max_fee_to_in'].append(max_fee_to_in)
        address_data_dict['n_pr80_fee_to_in'].append(n_pr80_fee_to_in)
        address_data_dict['n_join_pr80_am'].append(n_join_pr80_am)
        address_data_dict['n_in_to_out_ratio'].append(n_in_to_out_ratio)
        address_data_dict['n_join_huge'].append(n_join_huge)
        address_data_dict['n_join_pr80'].append(n_join_pr80)
    except Exception as error:
        print(address)
        print(out_date.max())
        print('===========================')
        print(error)


for num in tqdm(range(len(address_list))):
    address = address_list[num]
    add_data(address)


address_df = pd.DataFrame(address_data_dict)
address_df.to_csv('../../input_data/sheet3.csv')