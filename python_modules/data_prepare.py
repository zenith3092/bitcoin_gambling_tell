#!/usr/bin/env python
# coding: utf-8
# Author: Xianglin Wu (xianglin3092@gmail.com)

import pandas as pd

def gen_train_data():
    df = pd.read_csv('../input_data/tag_sheet3.csv', encoding='utf-8')

    # 補空值
    df = df.fillna('0')
    # print(df['ske_am_out'][8])

    # 類型轉換
    df = df.astype('str')

    # 確認資料長相
    # print(df.info())
    # print('空值數目\n', df.isnull().sum())
    df = df.reset_index(drop=True)

    # 設定 X data
    df_columns_list = df.columns.to_list()
    df_columns_list.remove('type')
    df_columns_list.remove('label')
    df_columns_list.remove('address')
    df_columns_list.remove('Unnamed: 0')
    # print(df_columns_list)
    Xdata = df[df_columns_list]

    # 設定 Y data
    Ydata = df['label']

    return Xdata, Ydata

def gen_predict_data():
    predictX = None
    predict_data = None
    return predictX, predict_data


if __name__ == '__main__':
    gen_train_data()

