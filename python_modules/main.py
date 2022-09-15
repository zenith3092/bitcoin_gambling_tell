#!/usr/bin/env python
# coding: utf-8
# Author: Xianglin Wu (xianglin3092@gmail.com)

import data_prepare as dp
from ml_tools import *


# Get Data
Xdata, Ydata = dp.gen_train_data()
predictX, predict_data = dp.gen_predict_data()

# Create a ML_SET Object
project = ML_SET(Xdata, Ydata)

flag = 1
while flag != 0:
    # # Initialize algorithms
    input_algorithms = input('\nPlease insert the algorithms you want to use:(hint:Insert 0 to exit)\n')
    input_algorithms = input_algorithms.split(',')

    if input_algorithms == ['0']:
        break
    else:
        input_predict = input('Do you want to predict? (y / other value)\n')
    
    act = project.activate(input_algorithms)
    print(act)

    # Export pkl files
    if act == 'Some elements you insert are not supportable or wrong.':
        continue
    else:
        ex_ml_obj = project.export_ml_obj(['all'])
        print(ex_ml_obj)

    # Merge models
    if ex_ml_obj == 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.':
        continue
    else:
        mg_models = project.merge_models(['all'])
        print(mg_models)

    # Export models
    if mg_models == 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.':
        continue
    else:
        ex_models = project.export_models()
        print(ex_models)

    # Export prediction results
    if ex_models == 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.' :
        continue
    else:
        if input_predict == 'y':
            ex_pred = project.export_predict(['all'])
            print(ex_pred)

            flag+=1
        else:
            continue




