#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize(out_lq_list, out_dr_list, out_drkf_list, out_mmse_list, dist, noise_dist, path, num, plot_results=True, wdrc = True, drkf=False, mmse=False, application="Nothing"):
    x_lqr_list, J_lqr_list, y_lqr_list, u_lqr_list = [], [], [], []
    x_list, J_list, y_list, u_list = [], [], [], [] # original wdrc with ordinary Kalman Filter
    x_drkf_list, J_drkf_list, y_drkf_list, u_drkf_list = [], [], [], [] # wdrc with Distributionally Robust kalman Filter (neurips)
    x_mmse_list, J_mmse_list, y_mmse_list, u_mmse_list = [], [], [], [] # wdrc with Distributionally Robust MMSE estimation problem (Adversial Anayltics)
    time_list, time_lqr_list, time_drkf_list, time_mmse_list = [], [], [], []
#    SettlingTime_list, SettlingTime_lqr_list = [], []

    if wdrc:
        for out in out_dr_list:
            x_list.append(out['state_traj'])
            J_list.append(out['cost'])
            y_list.append(out['output_traj'])
            u_list.append(out['control_traj'])
            time_list.append(out['comp_time'])
    #         SettlingTime_list.append(out['SettlingTime'])
        x_mean, J_mean, y_mean, u_mean = np.mean(x_list, axis=0), np.mean(J_list, axis=0), np.mean(y_list, axis=0), np.mean(u_list, axis=0)
        x_std, J_std, y_std, u_std = np.std(x_list, axis=0), np.std(J_list, axis=0), np.std(y_list, axis=0), np.std(u_list, axis=0)
        time_ar = np.array(time_list)
        print("WDRC cost : ", J_mean[0])
        print("WDRC cost std : ", J_std[0])
    #    SettlingTime_ar = np.array(SettlingTime_list)
        J_ar = np.array(J_list)


    for out in out_lq_list:
         x_lqr_list.append(out['state_traj'])
         J_lqr_list.append(out['cost'])
         y_lqr_list.append(out['output_traj'])
         u_lqr_list.append(out['control_traj'])
         time_lqr_list.append(out['comp_time'])
#         SettlingTime_lqr_list.append(out['SettlingTime'])
    x_lqr_mean, J_lqr_mean, y_lqr_mean, u_lqr_mean = np.mean(x_lqr_list, axis=0), np.mean(J_lqr_list, axis=0), np.mean(y_lqr_list, axis=0), np.mean(u_lqr_list, axis=0)
    x_lqr_std, J_lqr_std, y_lqr_std, u_lqr_std = np.std(x_lqr_list, axis=0), np.std(J_lqr_list, axis=0), np.std(y_lqr_list, axis=0), np.std(u_lqr_list, axis=0)
    time_lqr_ar = np.array(time_lqr_list)
    print("LQG cost : ", J_lqr_mean[0])
    print("LQG cost std : ", J_lqr_std[0])
#    SettlingTime_lqr_ar = np.array(SettlingTime_lqr_list)
    J_lqr_ar = np.array(J_lqr_list)


    if drkf:
        for out in out_drkf_list:
             x_drkf_list.append(out['state_traj'])
             J_drkf_list.append(out['cost'])
             y_drkf_list.append(out['output_traj'])
             u_drkf_list.append(out['control_traj'])
             time_drkf_list.append(out['comp_time'])
        x_drkf_mean, J_drkf_mean, y_drkf_mean, u_drkf_mean = np.mean(x_drkf_list, axis=0), np.mean(J_drkf_list, axis=0), np.mean(y_drkf_list, axis=0), np.mean(u_drkf_list, axis=0)
        x_drkf_std, J_drkf_std, y_drkf_std, u_drkf_std = np.std(x_drkf_list, axis=0), np.std(J_drkf_list, axis=0), np.std(y_drkf_list, axis=0), np.std(u_drkf_list, axis=0)
        time_drkf_ar = np.array(time_drkf_list)
        print("DRKF cost : ", J_drkf_mean[0])
        print("DRKF cost std : ", J_drkf_std[0])
        J_drkf_ar = np.array(J_drkf_list)
    if mmse:
        for out in out_mmse_list:
             x_mmse_list.append(out['state_traj'])
             J_mmse_list.append(out['cost'])
             y_mmse_list.append(out['output_traj'])
             u_mmse_list.append(out['control_traj'])
             time_mmse_list.append(out['comp_time'])
        x_mmse_mean, J_mmse_mean, y_mmse_mean, u_mmse_mean = np.mean(x_mmse_list, axis=0), np.mean(J_mmse_list, axis=0), np.mean(y_mmse_list, axis=0), np.mean(u_mmse_list, axis=0)
        x_mmse_std, J_mmse_std, y_mmse_std, u_mmse_std = np.std(x_mmse_list, axis=0), np.std(J_mmse_list, axis=0), np.std(y_mmse_list, axis=0), np.std(u_mmse_list, axis=0)
        time_mmse_ar = np.array(time_mmse_list)
        print("MMSE cost : ", J_mmse_mean[0])
        print("MMSE cost std : ", J_mmse_std[0])
        J_mmse_ar = np.array(J_mmse_list)         
        
    #nx = x_mean.shape[1]
    #T = u_mean.shape[0]
    nx = x_drkf_mean.shape[1]
    T = u_drkf_mean.shape[0]
    
    #WDRC
    if wdrc:
        avg = np.mean(x_mean[T-500:T,:], axis=0)
        err_bound_max = avg + 0.05*np.abs(avg)
        err_bound_min = avg - 0.05*np.abs(avg)
        SettlingTime = np.zeros(nx)
        for j in range(nx):
            for i in reversed(range(T)):
                if((x_mean[i,j] <= err_bound_min[j]) | (x_mean[i,j] >= err_bound_max[j])):
                    SettlingTime[j] = (i+1)*0.1
                    break
    #LQR            
    avg = np.mean(x_lqr_mean[T-200:T,:], axis=0)        
    err_bound_max_lqr = avg + 0.1*np.abs(avg)
    err_bound_min_lqr = avg - 0.1*np.abs(avg)
    SettlingTime_lqr = np.zeros(nx)
    for j in range(nx):
        for i in reversed(range(T)):
            if((x_lqr_mean[i,j] <= err_bound_min_lqr[j]) | (x_lqr_mean[i,j] >= err_bound_max_lqr[j])):
                SettlingTime_lqr[j] = (i+1)*0.1
                break
    #DRKF_WDRC            
    if drkf:
        avg = np.mean(x_drkf_mean[T-200:T,:], axis=0)        
        err_bound_max_drkf = avg + 0.1*np.abs(avg)
        err_bound_min_drkf = avg - 0.1*np.abs(avg)
        SettlingTime_drkf = np.zeros(nx)
        for j in range(nx):
            for i in reversed(range(T)):
                if((x_drkf_mean[i,j] <= err_bound_min_drkf[j]) | (x_drkf_mean[i,j] >= err_bound_max_drkf[j])):
                    SettlingTime_drkf[j] = (i+1)*0.1
                    break
    #MMSE_WDRC            
    if mmse:
        avg = np.mean(x_mmse_mean[T-200:T,:], axis=0)        
        err_bound_max_mmse = avg + 0.1*np.abs(avg)
        err_bound_min_mmse = avg - 0.1*np.abs(avg)
        SettlingTime_mmse = np.zeros(nx)
        for j in range(nx):
            for i in reversed(range(T)):
                if((x_mmse_mean[i,j] <= err_bound_min_mmse[j]) | (x_mmse_mean[i,j] >= err_bound_max_mmse[j])):
                    SettlingTime_mmse[j] = (i+1)*0.1
                    break
                        
    if plot_results:
        nx = x_drkf_mean.shape[1]
        T = u_drkf_mean.shape[0]
        nu = u_drkf_mean.shape[1]
        ny= y_drkf_mean.shape[1]

        fig = plt.figure(figsize=(6,4), dpi=300)

        t = np.arange(T+1)
        for i in range(nx):


            if x_lqr_list != []:
                plt.plot(t, x_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, x_lqr_mean[:,i, 0] + 0.3*x_lqr_std[:,i,0],
                               x_lqr_mean[:,i,0] - 0.3*x_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if wdrc:
                plt.plot(t, x_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, x_mean[:,i,0] + 0.3*x_std[:,i,0],
                                x_mean[:,i,0] - 0.3*x_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if drkf:
                plt.plot(t, x_drkf_mean[:,i,0], 'tab:green', label='DRKF-WDRC')
                plt.fill_between(t, x_drkf_mean[:,i, 0] + 0.3*x_drkf_std[:,i,0],
                               x_drkf_mean[:,i,0] - 0.3*x_drkf_std[:,i,0], facecolor='tab:green', alpha=0.3)
            if mmse:
                plt.plot(t, x_mmse_mean[:,i,0], 'tab:purple', label='MMSE-WDRC')
                plt.fill_between(t, x_mmse_mean[:,i, 0] + 0.3*x_mmse_std[:,i,0],
                               x_mmse_mean[:,i,0] - 0.3*x_mmse_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            
                
            plt.xlabel(r'$t$', fontsize=22)
            # if i<=9:
            #     plt.ylabel(r'$\Delta \delta_{{{}}}$'.format(i+1), fontsize=22)
            # else:
            #     plt.ylabel(r'$\Delta \omega_{{{}}}$'.format(i-9), fontsize=22)
            plt.ylabel(r'$x_{{{}}}$'.format(i+1), fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            if application=="Nothing":
                plt.savefig(path +'states_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist), dpi=300, bbox_inches="tight")
            else:
                plt.savefig(path +'states_{}_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist, application), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T)
        for i in range(nu):

            if u_lqr_list != []:
                plt.plot(t, u_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, u_lqr_mean[:,i,0] + 0.25*u_lqr_std[:,i,0],
                             u_lqr_mean[:,i,0] - 0.25*u_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if wdrc:
                plt.plot(t, u_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, u_mean[:,i,0] + 0.25*u_std[:,i,0],
                                u_mean[:,i,0] - 0.25*u_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if drkf:
                plt.plot(t, u_drkf_mean[:,i,0], 'tab:green', label='DRKF-WDRC')
                plt.fill_between(t, u_drkf_mean[:,i,0] + 0.25*u_drkf_std[:,i,0],
                             u_drkf_mean[:,i,0] - 0.25*u_drkf_std[:,i,0], facecolor='tab:green', alpha=0.3)                
            if mmse:
                plt.plot(t, u_mmse_mean[:,i,0], 'tab:purple', label='MMSE-WDRC')
                plt.fill_between(t, u_mmse_mean[:,i,0] + 0.25*u_mmse_std[:,i,0],
                             u_mmse_mean[:,i,0] - 0.25*u_mmse_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$u_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlim([t[0], t[-1]])
            if application=="Nothing":
                plt.savefig(path +'controls_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist), dpi=300, bbox_inches="tight")
            else:
                plt.savefig(path +'controls_{}_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist, application), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T+1)
        for i in range(ny):
            if y_lqr_list != []:
                plt.plot(t, y_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, y_lqr_mean[:,i,0] + 0.25*y_lqr_std[:,i,0],
                             y_lqr_mean[:,i, 0] - 0.25*y_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if wdrc:
                plt.plot(t, y_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, y_mean[:,i,0] + 0.25*y_std[:,i,0],
                                y_mean[:,i, 0] - 0.25*y_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if drkf:
                plt.plot(t, y_drkf_mean[:,i,0], 'tab:green', label='DRKF-WDRC')
                plt.fill_between(t, y_drkf_mean[:,i,0] + 0.25*y_drkf_std[:,i,0],
                             y_drkf_mean[:,i, 0] - 0.25*y_drkf_std[:,i,0], facecolor='tab:green', alpha=0.3)
            if mmse:
                plt.plot(t, y_mmse_mean[:,i,0], 'tab:purple', label='MMSE-WDRC')
                plt.fill_between(t, y_mmse_mean[:,i,0] + 0.25*y_mmse_std[:,i,0],
                             y_mmse_mean[:,i, 0] - 0.25*y_mmse_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$y_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlim([t[0], t[-1]])
            if application=="Nothing":
                plt.savefig(path +'outputs_{}_{}_{}_{}.pdf'.format(i+1,num, dist, noise_dist), dpi=300, bbox_inches="tight")
            else:
                plt.savefig(path +'outputs_{}_{}_{}_{}_{}.pdf'.format(i+1,num, dist, noise_dist, application), dpi=300, bbox_inches="tight")
            plt.clf()


        plt.title('Optimal Value')
        t = np.arange(T+1)

        if J_lqr_list != []:
            plt.plot(t, J_lqr_mean, 'tab:red', label='LQG')
            plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
        if wdrc:
            plt.plot(t, J_mean, 'tab:blue', label='WDRC')
            plt.fill_between(t, J_mean + 0.25*J_std, J_mean - 0.25*J_std, facecolor='tab:blue', alpha=0.3)
        
        if drkf:
            plt.plot(t, J_drkf_mean, 'tab:green', label='DRKF-WDRC')
            plt.fill_between(t, J_drkf_mean + 0.25*J_drkf_std, J_drkf_mean - 0.25*J_drkf_std, facecolor='tab:green', alpha=0.3)
        if mmse:
            plt.plot(t, J_mmse_mean, 'tab:purple', label='MMSE-WDRC')
            plt.fill_between(t, J_mmse_mean + 0.25*J_mmse_std, J_mmse_mean - 0.25*J_mmse_std, facecolor='tab:purple', alpha=0.3)
        
        plt.xlabel(r'$t$', fontsize=16)
        plt.ylabel(r'$V_t(x_t)$', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.xlim([t[0], t[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if application=="Nothing":
            plt.savefig(path +'J_{}_{}_{}.pdf'.format(num, dist, noise_dist), dpi=300, bbox_inches="tight")
        else:
            plt.savefig(path +'J_{}_{}_{}_{}.pdf'.format(num, dist, noise_dist, application), dpi=300, bbox_inches="tight")
        plt.clf()


        ax = fig.gca()
        t = np.arange(T+1)
        #print("Below should be REMOVEd!!!") #below first if line need to be remove
        # max_bin = np.max([J_drkf_ar[:,0], J_lqr_ar[:,0]])
        # min_bin = np.min([J_drkf_ar[:,0], J_lqr_ar[:,0]])
        # max_bin = np.max([ J_lqr_ar[:,0], J_drkf_ar[:,0], J_mmse_ar[:,0]])
        # min_bin = np.min([ J_lqr_ar[:,0], J_drkf_ar[:,0], J_mmse_ar[:,0]])
        
        if drkf and mmse and wdrc:
            max_bin = np.max([J_ar[:,0], J_lqr_ar[:,0], J_drkf_ar[:,0], J_mmse_ar[:,0]])
            min_bin = np.min([J_ar[:,0], J_lqr_ar[:,0], J_drkf_ar[:,0], J_mmse_ar[:,0]])
        elif drkf and wdrc:
            max_bin = np.max([J_ar[:,0], J_lqr_ar[:,0], J_drkf_ar[:,0]])
            min_bin = np.min([J_ar[:,0], J_lqr_ar[:,0], J_drkf_ar[:,0]])
        elif mmse and wdrc:
            max_bin = np.max([J_ar[:,0], J_lqr_ar[:,0], J_mmse_ar[:,0]])
            min_bin = np.min([J_ar[:,0], J_lqr_ar[:,0], J_mmse_ar[:,0]])    
        else: 
            max_bin = np.max([J_ar[:,0], J_lqr_ar[:,0]])
            min_bin = np.min([J_ar[:,0], J_lqr_ar[:,0]])

        
        ax.hist(J_lqr_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label='LQG', alpha=0.5, linewidth=0.5, edgecolor='tab:red')
        if wdrc:
            ax.hist(J_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='WDRC', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
        if drkf:
            ax.hist(J_drkf_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:green', label='DRKF-WDRC', alpha=0.5, linewidth=0.5, edgecolor='tab:green')
        if mmse:
            ax.hist(J_mmse_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:purple', label='MMSE-WDRC', alpha=0.5, linewidth=0.5, edgecolor='tab:purple')
        
        
        if wdrc:
            ax.axvline(J_ar[:,0].mean(), color='navy', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_lqr_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
        if drkf:
            ax.axvline(J_drkf_ar[:,0].mean(), color='green', linestyle='dashed', linewidth=1.5)
        if mmse:
            ax.axvline(J_mmse_ar[:,0].mean(), color='purple', linestyle='dashed', linewidth=1.5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        handles, labels = plt.gca().get_legend_handles_labels()
        # print("BELOw shoudl be removed2")
        # order = [1, 0, 2]
        # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
        
        if drkf and mmse:
            order = [1, 0, 2, 3]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
        elif drkf:
            order = [0, 1, 2]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
        elif mmse:
            order = [1, 0, 2]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
        else:
            order = [1, 0]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

        ax.grid()
        ax.set_axisbelow(True)
        plt.title('{} system disturbance, {} observation noise'.format(dist, noise_dist))
        plt.xlabel(r'Total Cost', fontsize=16)
        plt.ylabel(r'Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if application =="Nothing":
            plt.savefig(path +'J_hist_{}_{}_{}.pdf'.format(num, dist, noise_dist), dpi=300, bbox_inches="tight")
        else:
            plt.savefig(path +'J_hist_{}_{}_{}_{}.pdf'.format(num, dist, noise_dist, application), dpi=300, bbox_inches="tight")
        plt.clf()


        plt.close('all')
    if drkf and mmse:
        print( 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]),'cost_WDRC: {} ({})'.format(J_mean[0], J_std[0]) , 'cost_drkf_WDRC:{} ({})'.format(J_drkf_mean[0],J_drkf_std[0]), 'cost_mmse_WDRC:{} ({})'.format(J_mmse_mean[0],J_mmse_std[0]))
        print( 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()),'time_WDRC: {} ({})'.format(time_ar.mean(), time_ar.std()), 'time_drkf_WDRC: {} ({})'.format(time_drkf_ar.mean(), time_drkf_ar.std()), 'time_mmse_WDRC: {} ({})'.format(time_mmse_ar.mean(), time_mmse_ar.std()))
    #    print('Settling time: {} ({})'.format(SettlingTime_ar.mean(axis=0), SettlingTime_ar.std(axis=0)), 'Settling time_lqr: {} ({})'.format(SettlingTime_lqr_ar.mean(axis=0), SettlingTime_lqr_ar.std(axis=0)))
        print( 'Settling time_lqr: {}'.format(SettlingTime_lqr),'Settling time_WDRC: {} '.format(SettlingTime), 'Settling time_drkf_WDRC: {}'.format(SettlingTime_drkf), 'Settling time_mmse_WDRC: {}'.format(SettlingTime_mmse)) 
    elif drkf:
        print( 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]),'cost_WDRC: {} ({})'.format(J_mean[0], J_std[0]) , 'cost_drkf_WDRC:{} ({})'.format(J_drkf_mean[0],J_drkf_std[0]))
        print( 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()),'time_WDRC: {} ({})'.format(time_ar.mean(), time_ar.std()), 'time_drkf_WDRC: {} ({})'.format(time_drkf_ar.mean(), time_drkf_ar.std()))
    #    print('Settling time: {} ({})'.format(SettlingTime_ar.mean(axis=0), SettlingTime_ar.std(axis=0)), 'Settling time_lqr: {} ({})'.format(SettlingTime_lqr_ar.mean(axis=0), SettlingTime_lqr_ar.std(axis=0)))
        print( 'Settling time_lqr: {}'.format(SettlingTime_lqr),'Settling time_WDRC: {} '.format(SettlingTime), 'Settling time_drkf_WDRC: {}'.format(SettlingTime_drkf))
    elif mmse:
        print('cost_WDRC: {} ({})'.format(J_mean[0], J_std[0]) , 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]), 'cost_mmse_WDRC:{} ({})'.format(J_mmse_mean[0],J_mmse_std[0]))
        print('time_WDRC: {} ({})'.format(time_ar.mean(), time_ar.std()), 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()), 'time_mmse_WDRC: {} ({})'.format(time_mmse_ar.mean(), time_mmse_ar.std()))
    #    print('Settling time: {} ({})'.format(SettlingTime_ar.mean(axis=0), SettlingTime_ar.std(axis=0)), 'Settling time_lqr: {} ({})'.format(SettlingTime_lqr_ar.mean(axis=0), SettlingTime_lqr_ar.std(axis=0)))
        print('Settling time_WDRC: {} '.format(SettlingTime), 'Settling time_lqr: {}'.format(SettlingTime_lqr), 'Settling time_mmse_WDRC: {}'.format(SettlingTime_mmse))
    else:
        print('cost_WDRC: {} ({})'.format(J_mean[0], J_std[0]) , 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]))
        print('time_WDRC: {} ({})'.format(time_ar.mean(), time_ar.std()), 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()))
    #    print('Settling time: {} ({})'.format(SettlingTime_ar.mean(axis=0), SettlingTime_ar.std(axis=0)), 'Settling time_lqr: {} ({})'.format(SettlingTime_lqr_ar.mean(axis=0), SettlingTime_lqr_ar.std(axis=0)))
        print('Settling time_WDRC: {} '.format(SettlingTime), 'Settling time_lqr: {}'.format(SettlingTime_lqr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quad)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quad)
    parser.add_argument('--sim_type', required=False, default="multiple", type=str) #type of simulation runs (single or multiple)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs to plot
    parser.add_argument('--application', required=False, default="Quad3DOF", type=str)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--plot', required=False, action="store_true")
    parser.add_argument('--h_inf', required=False, action="store_true")

    args = parser.parse_args()

    if args.infinite:
        horizon = "infinite"
    else:
        horizon = "finite"
        
        
    print('\n-------Summary-------')
    if args.sim_type == "multiple":
        path = "./results/{}/{}_{}/finite/multiple/".format(args.application, args.dist, args.noise_dist)
        
        #Load data
        lqg_file = open(path + 'lqg.pkl', 'rb')
        wdrc_file = open(path + 'wdrc.pkl', 'rb')
        
        lqg_data = pickle.load(lqg_file)
        wdrc_data = pickle.load(wdrc_file)
        
        lqg_file.close()
        wdrc_file.close()
        
        drkf_wdrc_file = open(path + 'drkf_wdrc.pkl', 'rb')
        drkf_wdrc_data = pickle.load(drkf_wdrc_file)
        drkf_wdrc_file.close()
        
        mmse_wdrc_file = open(path + 'mmse_wdrc.pkl', 'rb')
        mmse_wdrc_data = pickle.load(mmse_wdrc_file)
        mmse_wdrc_file.close()
        
        summarize(lqg_data, wdrc_data, drkf_wdrc_data, mmse_wdrc_data, args.dist, args.noise_dist, path, args.num_sim, plot_results=True, wdrc = True, drkf=True, mmse=False, application=args.application)
    else:
        path = "./results/{}/{}/single/".format(args.dist, horizon)

        for i in range(args.num_sim):
            print('i: ', i)
            #Load data
            lqg_file = open(path + 'lqg.pkl', 'rb')
            wdrc_file = open(path + 'wdrc.pkl', 'rb')
            lqg_data = pickle.load(lqg_file)
            wdrc_data = pickle.load(wdrc_file)
            lqg_file.close()
            wdrc_file.close()
            if args.drkf:
                drkf_wdrc_file = open(path + 'drkf_wdrc.pkl', 'rb')
                drkf_wdrc_data = pickle.load(drkf_wdrc_file)
                drkf_wdrc_file.close()


            #Plot and Summarize
            if args.drkf:
                summarize(lqg_data, wdrc_data, drkf_wdrc_data, args.dist, path, i, args.plot, args.drkf)
            else:
                summarize(lqg_data, wdrc_data, None, args.dist, path, i, args.plot, args.drkf)
            print('---------------------')

