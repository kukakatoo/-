# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:17:40 2020

@author: leejongsuk
"""

import numpy as np

def f(x):
    return np.power(x,2)*0.5-x
def df1(x):
    return x-1;
learning_rate=0.9
n=0.01
max_loop=10
x_init=-2
x=x_init
print("Initial value of x is \n",x)
print("Initial value of Df1(x) is \n",df1(x))
for i in range(max_loop):
    x=x-learning_rate*df1(x)
    x1=-learning_rate*df1(x)
    print("Current value of x is \n",x)
    print("Current  value of Df1(x) is \n",df1(x))
    if x1<n:
        break
    
print("min value of x=",x)
