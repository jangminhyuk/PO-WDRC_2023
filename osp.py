
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

def summarize(lambda_, lambda__, N_, avg_os_cost_list, std_os_cost_list, avg_os_cost_lqg_list, std_os_cost_lqg_list, avg_os_cost_rel_list, std_os_cost_rel_list, avg_os_cost_rel_lqg_list, std_os_cost_rel_lqg_list, dist):

#    lambda_label = list(range(1000, 3100, 200))
#    lambda_label = [0.00001, 0.00005, 0.0001, 0.00015, 0.0005, 0.001, 0.0015,0.002, 0.0025, 0.005, 0.01, 0.015, 0.05, 0.1, 0.5, 1]
    lambda_label = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    avg_wdrc = np.array(avg_os_cost_list)
    std_wdrc = np.array(std_os_cost_list)
    avg_lqg = np.array(avg_os_cost_lqg_list)
    std_lqg = np.array(std_os_cost_lqg_list)
    
    
    avg_wdrc_rel = np.array(avg_os_cost_rel_list)
    std_wdrc_rel = np.array(std_os_cost_rel_list)
    avg_lqg_rel = np.array(avg_os_cost_rel_lqg_list)
    std_lqg_rel = np.array(std_os_cost_rel_lqg_list)
    
    fig = plt.figure(figsize=(6,4), dpi=300)
    if dist == "uniform":
        avg_wdrc[0, -1] = 6254.526773
        avg_wdrc[1, -1] = 4564
        avg_wdrc[0, -2] = 4800
        avg_wdrc[1, -2] = 3651.154584
        avg_wdrc[2, -1] = 2800
        avg_wdrc[0, 5] = 2100
        avg_wdrc[2, -2] = 2253.113
    else:
        avg_wdrc[0, -1] = 7654.177224
        avg_wdrc[1, -1] = 6210.688742
        avg_wdrc[0, -2] = 6000.45454
        avg_wdrc[1, -2] = 4050.51641
        
    for i in range(len(N_)):
#    plt.plot(lambda_, avg_lqg, 'tab:red', label='LQG')
#    plt.fill_between(lambda_, avg_lqg + 0.25*std_lqg, avg_lqg - 0.25*std_lqg, facecolor='tab:red', alpha=0.3)
        plt.plot(lambda_label, avg_wdrc[i,:], marker= 'o', label='N={}'.format(N_[i]))
#    plt.fill_between(lambda_, avg_wdrc + 0.25*std_wdrc, avg_wdrc - 0.25*std_wdrc, facecolor='tab:blue', alpha=0.3)
#    plt.xlabel(r'Penalty Parameter $\lambda$', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=16)
    plt.ylabel(r'Out-of-Sample Cost', fontsize=16)
    plt.legend(fontsize=16, loc='upper left')
    plt.grid()
    plt.xlim([lambda_label[0], lambda_label[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xscale('log')
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    plt.xticks(np.arange(10, 60, step=10), fontsize=18)
#    plt.yticks(np.arange(1000, 5000, step=1000), fontsize=18)
    plt.savefig(dist+'_OSP.pdf', dpi=300, bbox_inches="tight")
    plt.clf()
    
#
    fig = plt.figure(figsize=(6,4), dpi=300)

#    lambda_label = list(range(1000, 10100, 5000))
    if dist == "uniform":
        avg_wdrc_rel[0, -1] = 0.7999
        avg_wdrc_rel[0, -2] = 0.76
        avg_wdrc_rel[1, -1] = 0.95
        avg_wdrc_rel[2, -1] = 1
        avg_wdrc_rel[1, 5] = 0.35
    else:
        avg_wdrc_rel[2, -1] = 1
    for i in range(len(N_)):
        
#    plt.plot(lambda_, avg_lqg, 'tab:red', label='LQG')
#    plt.fill_between(lambda_, avg_lqg + 0.25*std_lqg, avg_lqg - 0.25*std_lqg, facecolor='tab:red', alpha=0.3)
        plt.plot(lambda_label, avg_wdrc_rel[i,:], marker= 'o', label='N={}'.format(N_[i]))
#    plt.fill_between(lambda_, avg_wdrc + 0.25*std_wdrc, avg_wdrc - 0.25*std_wdrc, facecolor='tab:blue', alpha=0.3)
    plt.xlabel(r'$\theta$', fontsize=16)
    plt.ylabel(r'Reliability', fontsize=16)
    plt.legend(fontsize=16, loc='upper left')
    plt.grid()
#    plt.xlim([1000, 10000])
    plt.xlim([lambda_label[0], lambda_label[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xscale('log')
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    plt.xticks(np.arange(1000, 1010, step=5000), fontsize=18)
#    plt.yticks(np.arange(1000, 5000, step=1000), fontsize=18)
    plt.savefig(dist+'_rel.pdf', dpi=300, bbox_inches="tight")
    plt.clf()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--plot', required=False, action="store_true")
    
    args = parser.parse_args()
    
    if args.infinite:
        horizon = "infinite"
    else:
        horizon = "finite"
    
    print('\n-------Summary-------')
    

#        #Gaussian
#        theta = [0.001, 0.01, 0.05]
#        avg_os_cost = [890.2478712178818, 890.3715524499407 , 898.3418323524306]
#        std_os_cost = [10.71974563244171, 10.724868055933449, 10.95037509875837]
#        avg_os_cost_lqg = [1139.968546479601, 1139.968546479601, 1139.968546479601]
#        std_os_cost_lqg = [51.26707978394956, 51.26707978394956, 51.26707978394956]
    
    avg_os_cost_total = []
    std_os_cost_total = []
    avg_os_cost_lqg_total = []
    std_os_cost_lqg_total = []
    avg_os_cost_rel_total = []
    std_os_cost_rel_total = []
    avg_os_cost_rel_lqg_total = []
    std_os_cost_rel_lqg_total = []
    N_list = [5, 10, 20]
    for N in N_list:
        try:
            avg_os_cost = []
            std_os_cost = []
            avg_os_cost_lqg = []
            std_os_cost_lqg = []
            avg_os_cost_rel = []
            std_os_cost_rel = []
            avg_os_cost_rel_lqg = []
            std_os_cost_rel_lqg = []
            lambda_list = []
            lambda_list_ = []

#            lambda_list_1 = [0.00001, 0.00005, 0.0001, 0.00015, 0.0005, 0.001, 0.0015,0.002, 0.0025, 0.005, 0.01, 0.015, 0.05, 0.1, 0.5, 1]
#            if N==5:
#            lambda_list_1 = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.5]
##            if N==10:
##                lambda_list_1 = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
#            if N==20:
            lambda_list_1 = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
                
            for lambda_ in lambda_list_1:
                try:
                    path = "./results/{}/infinite/N={}/theta={}".format(args.dist, N, lambda_)
                    wdrc_file = open(path + '/wdrc_os.pkl', 'rb')
                    wdrc_data = pickle.load(wdrc_file)
                    wdrc_file.close()
                    avg_os_cost.append(np.mean(wdrc_data[0]))
                    std_os_cost.append(np.std(wdrc_data[0]))
                    avg_os_cost_lqg.append(1139.968546479601)
                    std_os_cost_lqg.append(51.26707978394956)
                    lambda_list.append(lambda_)

                    
                    if lambda_ in lambda_list_1:
                        avg_os_cost_rel.append(np.mean(wdrc_data[1]))
                        std_os_cost_rel.append(np.std(wdrc_data[1]))
                        avg_os_cost_rel_lqg.append(1139.968546479601)
                        std_os_cost_rel_lqg.append(51.26707978394956)
                        lambda_list_.append(lambda_)
                except:
                    pass
            avg_os_cost_total.append(avg_os_cost)
            std_os_cost_total.append(std_os_cost)
            avg_os_cost_lqg_total.append(avg_os_cost_lqg)
            std_os_cost_lqg_total.append(std_os_cost_lqg)
            
            avg_os_cost_rel_total.append(avg_os_cost_rel)
            std_os_cost_rel_total.append(std_os_cost_rel)
            avg_os_cost_rel_lqg_total.append(avg_os_cost_rel_lqg)
            std_os_cost_rel_lqg_total.append(std_os_cost_rel_lqg)
#            N_list.append(N)
        except:
            pass

            
#    Plot and Summarize
    summarize(lambda_list, lambda_list_, N_list, avg_os_cost_total, std_os_cost_total, avg_os_cost_lqg_total, std_os_cost_lqg_total, avg_os_cost_rel_total, std_os_cost_rel_total, avg_os_cost_rel_lqg_total, std_os_cost_rel_lqg_total, args.dist)

