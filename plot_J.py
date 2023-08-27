#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 02:07:15 2022

@author: astghik
"""


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize(M_list, avg_cost, std_cost, avg_cost_lqg, std_cost_lqg, avg_cost_true, std_cost_true, avg_cost_true_lqg, std_cost_true_lqg,  dist):
#    t = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    
    t = np.array(M_list)
    
    J_mean = np.array(avg_cost)
    
    J_lqr_mean = np.array(avg_cost_lqg)
    
    J_std = np.array(std_cost)
    
    J_lqr_std = np.array(std_cost_lqg)
    
    J_true_mean = np.array(avg_cost_true*len(t))
    J_true_lqr_mean = np.array(avg_cost_true_lqg*len(t))
    J_true_std = np.array(std_cost_true*len(t))
    J_true_lqr_std = np.array(std_cost_true_lqg*len(t))
    #J_mean = np.array([3309.9349518, 516.74179, 426.051039, 417.74422917, 404.0349027, 424.0342728, 408.02106830, 409.9274766, 409.32559079, 409.7475016108369])
    #J_lqr_mean = np.array([4123.0091387, 1242.0582986, 591.39737055, 572.3771644118, 579.02452919, 582.4813689, 525.0072175033, 560.04638249, 529.90407465, 560.1624879091927])
    #
    #J_std = np.array([413.94808166, 35.54102, 34.9152509406, 33.688702068, 34.267471, 34.148926, 34.082606, 33.866055, 33.266204, 34.793488398])
    #J_lqr_std = np.array([810.8185097, 252.8705, 71.2370977, 72.499760949, 89.5029926, 71.22324, 63.24045, 74.782877, 70.26421, 78.3153360072])
    
    #J_true_mean = np.array([397.55246816499454]*len(t))
    #J_true_lqr_mean = np.array([505.9412059942671]*len(t))
    #J_true_std = np.array([34.566372]*len(t))
    #J_true_lqr_std = np.array([65.526383]*len(t))
    
    fig = plt.figure(figsize=(6,4), dpi=300)
    
    plt.plot(t, J_true_lqr_mean, '#851617', linestyle='dashed', label='LQG (true)')
    plt.fill_between(t, J_true_lqr_mean + 0.25*J_true_lqr_std, J_true_lqr_mean - 0.25*J_true_lqr_std, facecolor='#851617', alpha=0.3)
    plt.plot(t, J_true_mean, '#103E5E', linestyle='dashed', label='WDRC (true)')
    plt.fill_between(t, J_true_mean + 0.25*J_true_std, J_true_mean - 0.25*J_true_std, facecolor='#103E5E', alpha=0.3)
    plt.plot(t, J_lqr_mean, 'tab:red', label='LQG (sample)')
    plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
    plt.plot(t, J_mean, 'tab:blue', label='WDRC (sample)')
    plt.fill_between(t, J_mean + J_std, J_mean - J_std, facecolor='tab:blue', alpha=0.3)
    plt.xlabel(r'Sample Size', fontsize=16)
    plt.ylabel(r'Total Cost', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([t[0], t[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('J_comp.pdf', dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    args = parser.parse_args()
    
    
    print('\n-------Summary-------')
    

#        #Gaussian
#        theta = [0.001, 0.01, 0.05]
#        avg_os_cost = [890.2478712178818, 890.3715524499407 , 898.3418323524306]
#        std_os_cost = [10.71974563244171, 10.724868055933449, 10.95037509875837]
#        avg_os_cost_lqg = [1139.968546479601, 1139.968546479601, 1139.968546479601]
#        std_os_cost_lqg = [51.26707978394956, 51.26707978394956, 51.26707978394956]
    avg_cost = []
    std_cost = []
    avg_cost_lqg = []
    std_cost_lqg = []
    M_list = []
    
    list_ = list(range(10,55,5))
    for M in list_:
        try:
            path = "./results/{}/infinite/M/M={}".format(args.dist, M)
            wdrc_file = open(path + '/wdrc.pkl', 'rb')
            wdrc_data = pickle.load(wdrc_file)
            wdrc_file.close()
            lqg_file = open(path + '/lqg.pkl', 'rb')
            lqg_data = pickle.load(lqg_file)
            lqg_file.close()
            J_list = []
            for out in wdrc_data:
                J_list.append(out['cost'][0])
            J_list_lqg = []
            for out in lqg_data:
                J_list_lqg.append(out['cost'][0])
            avg_cost.append(np.mean(J_list, axis=0))
            std_cost.append(np.std(J_list, axis=0))
            avg_cost_lqg.append(np.mean(J_list_lqg, axis=0))
            std_cost_lqg.append(np.std(J_list_lqg, axis=0))
            M_list.append(M)
        except:
            pass
        
        path = "./results/{}/infinite/M/known".format(args.dist)
        wdrc_file = open(path + '/wdrc.pkl', 'rb')
        wdrc_data = pickle.load(wdrc_file)
        wdrc_file.close()
        lqg_file = open(path + '/lqg.pkl', 'rb')
        lqg_data = pickle.load(lqg_file)
        lqg_file.close()
        J_list = []
        for out in wdrc_data:
            J_list.append(out['cost'][0])
        J_list_lqg = []
        for out in lqg_data:
            J_list_lqg.append(out['cost'][0])
        avg_cost_true = [np.mean(J_list, axis=0)]
        std_cost_true = [np.std(J_list, axis=0)]
        avg_cost_true_lqg = [np.mean(J_list_lqg, axis=0)]
        std_cost_true_lqg = [np.std(J_list_lqg, axis=0)]
        
    summarize(M_list, avg_cost, std_cost, avg_cost_lqg, std_cost_lqg, avg_cost_true, std_cost_true, avg_cost_true_lqg, std_cost_true_lqg, args.dist)
