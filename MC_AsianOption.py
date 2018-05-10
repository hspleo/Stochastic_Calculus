# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:06:36 2018

@author: Senpei Hou
"""

import math
import numpy as np
import scipy.stats


T = 1
r = 0.01
sigma = 0.2
K = 100
S0 = 110
m = 12


def MC(n):
    X = []
    Y = []
    for i in range(n):
        z = np.random.normal(0.0, 1.0, m)
        S = (r - sigma*sigma/2)*(T/m) + sigma*np.sqrt(T/m)*z
        S = S.cumsum()
        S = S0 * np.exp(S)
        S_mean = S.mean()
        S_gmean = scipy.stats.gmean(S)
        X.append(max(0, np.exp(-r*T)*(S_gmean - K)))
        Y.append(max(0, np.exp(-r*T)*(S_mean - K)))

    b = np.cov(X, Y)[0,1]/np.var(X)
    corr = np.corrcoef(X, Y)[0,1]

    T_bar = 1/2 * (1 + m) * T / m
    tmp = 0
    for k in range(m):
        tmp += (2*k+1)*(m-k)
        
    sigma_bar = (sigma/m) * np.sqrt((T/T_bar)*(tmp/m))
        
    delta = 1/2 * (sigma*sigma - sigma_bar*sigma_bar)
        
        
    d = (math.log(S0/K) + (r - delta + sigma_bar*sigma_bar/2)*T_bar) / (sigma_bar*np.sqrt(T_bar))

    EX = np.exp(-delta*T_bar-r*(T-T_bar))*S0*scipy.stats.norm(0,1).cdf(d) - K * np.exp(-r*T) * scipy.stats.norm(0,1).cdf(d - sigma_bar*np.sqrt(T_bar))

    X = np.array(X)
    Y = np.array(Y)

    ctrl_var = Y - b*(X-EX)
    
    print("first algorithm: ")
    print("price:  ", Y.mean())
    print("std err:", Y.std()/np.sqrt(n))
    print("")
    
    print("second algorithm: ")
    print("price:  ", ctrl_var.mean())
    print("std err:", ctrl_var.std()/np.sqrt(n))
    print("corrcoef", corr)
    print("")
    print("")

for n in [1000, 10000, 100000, 1000000]:
    print("Number of replications:",n)
    print("")
    MC(n)
    


