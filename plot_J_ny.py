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
#impact of number of observable generators
def summarize(M_list, avg_cost, std_cost, avg_cost_lqg, std_cost_lqg, dist):
#    t = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    
    t = np.array(M_list)
    
    J_mean = np.array(avg_cost)
    
    J_lqr_mean = np.array(avg_cost_lqg)
    
    J_std = np.array(std_cost)
    
    J_lqr_std = np.array(std_cost_lqg)
    
    
    fig = plt.figure(figsize=(6,4), dpi=300)
    
    plt.plot(t, J_lqr_mean, 'tab:red', label='LQG (sample)')
    plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
    plt.plot(t, J_mean, 'tab:blue', label='WDRC (sample)')
    plt.fill_between(t, J_mean + J_std, J_mean - J_std, facecolor='tab:blue', alpha=0.3)
    plt.xlabel(r'# of Observable Generators', fontsize=16)
    plt.ylabel(r'Total Cost', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([t[0], t[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('J_ny_{}.pdf'.format(dist), dpi=300, bbox_inches="tight")
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
    
    list_ = list(range(8,25,2))
    for ny in list_:
        try:
            path = "./results/{}/infinite/ny={}".format(args.dist, ny)
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
            M_list.append(int(ny/2))
        except:
            pass
        
#        path = "./results/{}/infinite/multiple/known".format(args.dist)
#        wdrc_file = open(path + '/wdrc.pkl', 'rb')
#        wdrc_data = pickle.load(wdrc_file)
#        wdrc_file.close()
#        lqg_file = open(path + '/lqg.pkl', 'rb')
#        lqg_data = pickle.load(lqg_file)
#        lqg_file.close()
#        J_list = []
#        for out in wdrc_data:
#            J_list.append(out['cost'][0])
#        J_list_lqg = []
#        for out in lqg_data:
#            J_list_lqg.append(out['cost'][0])
#        avg_cost_true = [np.mean(J_list, axis=0)]
#        std_cost_true = [np.std(J_list, axis=0)]
#        avg_cost_true_lqg = [np.mean(J_list_lqg, axis=0)]
#        std_cost_true_lqg = [np.std(J_list_lqg, axis=0)]
        
    summarize(M_list, avg_cost, std_cost, avg_cost_lqg, std_cost_lqg, args.dist)
