# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:40:44 2024

@author: peiweike
"""
import json

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data