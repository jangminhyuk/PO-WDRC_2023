#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 02:07:15 2022
Updated on Thu AUG 31 2023 by Minhyuk
@author: astghik
"""


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize_noise(num_noise_list, avg_cost_lqg, std_cost_lqg, avg_cost_wdrc, std_cost_wdrc, avg_cost_drkf_wdrc, std_cost_drkf_wdrc, avg_cost_mmse_wdrc, std_cost_mmse_wdrc, dist, noise_dist, path, application):
#    t = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    
    t = np.array(num_noise_list)
    

    J_lqr_mean = np.array(avg_cost_lqg)
    J_wdrc_mean = np.array(avg_cost_wdrc)
    J_drkf_wdrc_mean = np.array(avg_cost_drkf_wdrc)
    J_mmse_wdrc_mean = np.array(avg_cost_mmse_wdrc)
    
    J_lqr_std = np.array(std_cost_lqg)
    J_wdrc_std = np.array(std_cost_wdrc)
    J_drkf_wdrc_std = np.array(std_cost_drkf_wdrc)
    J_mmse_wdrc_std = np.array(std_cost_mmse_wdrc)
    
    # J_true_mean = np.array(avg_cost_true*len(t))
    # J_true_lqr_mean = np.array(avg_cost_true_lqg*len(t))
    # J_true_std = np.array(std_cost_true*len(t))
    # J_true_lqr_std = np.array(std_cost_true_lqg*len(t))
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
    
    # plt.plot(t, J_true_lqr_mean, '#851617', linestyle='dashed', label='LQG (true)')
    # plt.fill_between(t, J_true_lqr_mean + 0.25*J_true_lqr_std, J_true_lqr_mean - 0.25*J_true_lqr_std, facecolor='#851617', alpha=0.3)
    
    # plt.plot(t, J_true_mean, '#103E5E', linestyle='dashed', label='WDRC (true)')
    # plt.fill_between(t, J_true_mean + 0.25*J_true_std, J_true_mean - 0.25*J_true_std, facecolor='#103E5E', alpha=0.3)
    
    plt.title('{} system disturbance, {} observation noise'.format(dist, noise_dist))
    #----------------------------------------------
    plt.plot(t, J_lqr_mean, 'tab:red', label='LQG (sample)')
    plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
    
    plt.plot(t, J_wdrc_mean, 'tab:blue', label='WDRC (sample)')
    plt.fill_between(t, J_wdrc_mean + 0.25*J_wdrc_std, J_wdrc_mean - 0.25*J_wdrc_std, facecolor='tab:blue', alpha=0.3)
    
    plt.plot(t, J_drkf_wdrc_mean, 'tab:green', label='DRKF-WDRC (sample)')
    plt.fill_between(t, J_drkf_wdrc_mean + 0.25*J_drkf_wdrc_std, J_drkf_wdrc_mean - 0.25*J_drkf_wdrc_std, facecolor='tab:green', alpha=0.3)
    
    #plt.plot(t, J_mmse_wdrc_mean, 'tab:purple', label='MMSE-WDRC (sample)')
    #plt.fill_between(t, J_mmse_wdrc_mean + 0.25*J_mmse_wdrc_std, J_mmse_wdrc_mean - 0.25*J_mmse_wdrc_std, facecolor='tab:purple', alpha=0.3)
    
    
    plt.xlabel(r'Noise Sample Size', fontsize=16)
    plt.ylabel(r'Total Cost', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([t[0], t[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if application:
        plt.savefig(path +'/UAV_J_comp_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight")
    else:
        plt.savefig(path +'/J_comp_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight")
    plt.clf()
    print("hi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform)
    parser.add_argument('--application', required=False, action="store_true")
    parser.add_argument('--theta', required=False, default="1")
    args = parser.parse_args()
    
    
    print('\n-------Summary-------')
    

#        #Gaussian
#        theta = [0.001, 0.01, 0.05]
#        avg_os_cost = [890.2478712178818, 890.3715524499407 , 898.3418323524306]
#        std_os_cost = [10.71974563244171, 10.724868055933449, 10.95037509875837]
#        avg_os_cost_lqg = [1139.968546479601, 1139.968546479601, 1139.968546479601]
#        std_os_cost_lqg = [51.26707978394956, 51.26707978394956, 51.26707978394956]
    # avg_cost = []
    # std_cost = []
    # avg_cost_lqg = []
    # std_cost_lqg = []
    # M_list = []
    if args.application:
        path = "./results/quad_{}_{}/finite/multiple/num_noise_plot".format(args.dist, args.noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/num_noise_plot/{}".format(args.dist, args.noise_dist, args.theta)
    num_noise_list = [5, 10, 15, 20, 25, 30]
    # avg_cost_lqg = [25576.348, 5473.018, 3682.205, 3588.105, 3514.703, 3399.839]
    # avg_cost_wdrc = [4703.05, 1749.400, 1757.495, 1835.393, 1895.03, 1870.068]
    # avg_cost_drkf_wdrc = [2856.819, 2484.1349, 2304.143, 2633.690, 2655.365, 2523.628]
    # avg_cost_mmse_wdrc = [2426.47, 2357.985, 2336.873, 2381.514, 2430.105, 2423.822]
    # std_cost_lqg = [7481.319, 1439.306, 829.679, 901.372, 846.776, 825.623]
    # std_cost_wdrc = [906.994, 270.930, 279.029, 317.802, 336.705, 327.673]
    # std_cost_drkf_wdrc = [607.073, 451.054, 390.515, 476.987, 518.819, 468.139]
    # std_cost_mmse_wdrc = [828.041, 797.635, 768.956, 791.828, 848.611, 813.465]
    avg_cost_lqg_file = open(path + '/lqg_mean.pkl', 'rb' )
    avg_cost_lqg = pickle.load(avg_cost_lqg_file)
    avg_cost_lqg_file.close()
    std_cost_lqg_file = open(path + '/lqg_std.pkl', 'rb' )
    std_cost_lqg = pickle.load(std_cost_lqg_file)
    std_cost_lqg_file.close()
    
    avg_cost_wdrc_file = open(path + '/wdrc_mean.pkl', 'rb' )
    avg_cost_wdrc = pickle.load(avg_cost_wdrc_file)
    #print(avg_cost_wdrc)
    avg_cost_wdrc_file.close()
    std_cost_wdrc_file = open(path + '/wdrc_std.pkl', 'rb' )
    std_cost_wdrc = pickle.load(std_cost_wdrc_file)
    std_cost_wdrc_file.close()
    
    avg_cost_drkf_wdrc_file = open(path + '/drkf_wdrc_mean.pkl', 'rb' )
    avg_cost_drkf_wdrc = pickle.load(avg_cost_drkf_wdrc_file)
    #print(avg_cost_drkf_wdrc)
    avg_cost_drkf_wdrc_file.close()
    std_cost_drkf_wdrc_file = open(path + '/drkf_wdrc_std.pkl', 'rb' )
    std_cost_drkf_wdrc = pickle.load(std_cost_drkf_wdrc_file)
    std_cost_drkf_wdrc_file.close()
    
    avg_cost_mmse_wdrc_file = open(path + '/mmse_wdrc_mean.pkl', 'rb' )
    avg_cost_mmse_wdrc = pickle.load(avg_cost_mmse_wdrc_file)
    #print(avg_cost_mmse_wdrc)
    avg_cost_mmse_wdrc_file.close()
    std_cost_mmse_wdrc_file = open(path + '/mmse_wdrc_std.pkl', 'rb' )
    std_cost_mmse_wdrc = pickle.load(std_cost_mmse_wdrc_file)
    #print(std_cost_mmse_wdrc)
    std_cost_mmse_wdrc_file.close()
    summarize_noise(num_noise_list, avg_cost_lqg, std_cost_lqg, avg_cost_wdrc, std_cost_wdrc, avg_cost_drkf_wdrc, std_cost_drkf_wdrc, avg_cost_mmse_wdrc, std_cost_mmse_wdrc, args.dist, args.noise_dist, path, args.application)
