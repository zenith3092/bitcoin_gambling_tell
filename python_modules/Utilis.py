#!/usr/bin/env python
# coding: utf-8
# Author: Xianglin Wu (xianglin3092@gmail.com)

import yaml


def read_yaml(filename):
    with open(filename, 'r') as file:
        try:
            output = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    return output