# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:02:52 2023
Collates pre and post values in prep for sawtooth plots
@author: cognotrend
"""
import numpy as np

def collate(v1, v2):
    if v1.shape == v2.shape:
        collated = np.empty((v1.size * 2,), dtype=v1.dtype)
        collated[::2] = v1
        collated[1::2] = v2
    else:
        collated = -1
    return collated
