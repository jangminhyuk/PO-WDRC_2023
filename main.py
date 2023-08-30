#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRKF_WDRC import DRKF_WDRC
from controllers.MMSE_WDRC import MMSE_WDRC
from controllers.inf_LQG import inf_LQG
from controllers.inf_WDRC import inf_WDRC
from controllers.inf_DRKF_WDRC import inf_DRKF_WDRC
from controllers.inf_H_infty import inf_H_infty

from plot import summarize
import os
import pickle

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b-a)*np.random.rand(N,n)
    return x.T

def normal(mu, Sigma, N=1):
#    n = mu.shape[0]
#    w = np.random.normal(size=(N,n))
#    if (Sigma == 0).all():
#        x = mu
#    else:
#        x = mu + np.linalg.cholesky(Sigma) @ w.T
    x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
    return x

def multimodal(mu, Sigma, N=1):
    modes = 2
    n = mu[0].shape[0]
    x = np.zeros((n,N,modes))
    for i in range(modes):
        w = np.random.normal(size=(N,n))
        if (Sigma[i] == 0).all():
            x[:,:,i] = mu[i]
        else:
            x[:,:,i] = mu[i] + np.linalg.cholesky(Sigma[i]) @ w.T

    #w = np.random.choice([0, 1], size=(n,N))
    w = 0.5
    y = x[:,:,0]*w + x[:,:,1]*(1-w)
    return y

def gen_sample_dist(dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)

    mean_ = np.average(w, axis = 1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average( (diff @ np.transpose(diff, (0,2,1))) , axis = 0)
    return np.tile(mean_[...,np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

def gen_sample_dist_inf(dist, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)

    mean_ = np.average(w, axis = 1)[...,np.newaxis]
    var_ = np.cov(w)
#    var_ = np.diag(np.diag(var_))
    return mean_, var_

def create_matrices(nx, ny, nu):
    A = np.load("./inputs/A.npy") # (n x n) matrix
    B = np.load("./inputs/B.npy")
    C = np.hstack([np.eye(ny, int(ny/2)), np.zeros((ny, int((nx-ny)/2))), np.eye(ny, int(ny/2), k=-int(ny/2)), np.zeros((ny, int((nx-ny)/2)))])
#    C = np.hstack([np.zeros((ny, nx-ny)), np.eye(ny, ny)])
#    C = np.eye(ny)
    return A, B, C

def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, sim_type, num_sim, num_samples, num_noise_samples, T, plot_results, infinite, out_of_sample, wc, h_inf):
    lambda_ = 1000
    for theta_ind, theta in enumerate([0.5]):
#    enumerate([0.5, 1])            
#    enumerate([0.00001, 0.0001, 0.0005, 0.0015, 0.001, 0.00015, 0.002, 0.0025, 0.005, 0.01, 0.015, 0.05, 0.1, 1]):
                    #:
        #Path for saving the results
        if infinite:
            if sim_type == "multiple":
                if out_of_sample:
                    path = "./results/{}/infinite/N={}/theta={}".format(dist,  num_samples, theta)
                else:
                    path = "./results/{}/infinite/multiple/".format(dist)
            else:
                path = "./results/{}/infinite/single/".format(dist)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            if sim_type == "multiple":
                path = "./results/{}/finite/multiple/".format(dist)
            else:
                path = "./results/{}/finite/single/".format(dist)
            if not os.path.exists(path):
                os.makedirs(path)
    
        #-------Initialization-------
        nx = 20 #state dimension
        nu = 10 #control input dimension
        ny = 12#output dimension
        A, B, C = create_matrices(nx, ny, nu) #system matrices generation
        #cost weights
        Q = np.load("./inputs/Q.npy")
        #np.asarray(np.bmat([[0.5*(np.eye(int(nx/2)) - 0.1*np.ones((int(nx/2), int(nx/2)))), np.zeros((int(nx/2), int(nx/2)))],
        #                    [np.zeros((int(nx/2), int(nx/2))), 0.5
        Qf = np.load("./inputs/Q_f.npy")    
        R = np.load("./inputs/R.npy")
    
        if dist =="uniform":
    #        M = 0.05*np.eye(ny) #observation noise covariance
            v_min = -0.4*np.ones(ny)
            v_max = 0.4*np.ones(ny)
            mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
            M = 1/12*np.diag((v_max - v_min)**2)
            #_, M_hat = gen_sample_dist_inf('uniform', 20, w_min = v_min, w_max = v_max)
            #M_hat = M_hat + 1e-6*np.eye(ny)
#            M_hat = M
#            theta = 0.001 #Wasserstein ball radius
            #disturbance distribution parameters
            w_max = 0.15*np.ones(nx)
            w_min = -0.15*np.ones(nx)
            mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
            Sigma_w = 1/12*np.diag((w_max - w_min)**2)
            #initial state distribution parameters
            x0_max = 0.05*np.ones(nx)
            x0_min = -0.05*np.ones(nx)
    #        x0_max = 0*np.ones(nx)
    #        x0_min = -0*np.ones(nx)
            x0_max[-1] = 1.05
            x0_min[-1] = 0.95
            x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
            x0_cov = 1/12*np.diag((x0_max - x0_min)**2)
#             if num_samples == 5:
# #                lambda_list = [850, 820]
# #                lambda_list = [780.2396483109309, 765.5991469006884]
#                 lambda_list = [780.2396483109309]
# #                lambda_list = [1107.653293607882, 1107.653293607882, 1290.4178108555284, 1290.4178108555284, 1290.4178108555284, 1290.4178108555284, 1290.4178108555284, 1290.4178108555284, 1290.4178108555284, 1279.1174641524258, 1270.3111655056887, 1236.4068466992562, 1052.0625048556794, 912.4077111597371, 780.2396483109309, 765.5991469006884]
#             if num_samples == 10:
#                 lambda_list = [776.0998011937304]
# #                lambda_list = [776.0998011937304, 763.514339455696]
# #                lambda_list = [830, 810]
# #                lambda_list = [1242.240707241318, 1257.5366782405856, 1257.5366782405856, 1257.5366782405856, 1257.5366782405856, 1257.5366782405856, 1257.5366782405856, 1257.5366782405856, 1226.3495924878498, 1226.3495924878498, 1199.345818602059, 1026.146847412157, 894.6657627413663, 776.0998011937304, 763.514339455696]
#             if num_samples == 20:
#                 lambda_list = [781.123992343782]
# #                lambda_list = [781.123992343782, 765.9331733879504]
# #                lambda_list = [825, 800]
# #                lambda_list = [1268.0815729111753, 1268.0815729111753, 1289.6912301014315, 1289.6912301014315, 1289.6912301014315, 1289.6912301014315, 1289.6912301014315, 1289.6912301014315, 1289.6912301014315, 1282.8160606455085, 1282.8160606455085, 1252.8446224026345, 1056.7448705096183, 914.1346001244447, 781.123992343782, 765.9331733879504]
#             lambda_ = lambda_list[theta_ind]
            
        elif dist == "normal":
            M = 0.01*np.eye(ny) #observation noise covariance
            #_, M_hat = gen_sample_dist_inf('normal', 20, mu_w=0*np.ones((ny, 1)), Sigma_w=M)
            #M_hat = M_hat + 1e-6*np.eye(ny)
#            M_hat = M
#            theta = 0.001 #Wasserstein ball radius
            #disturbance distribution parameters
            w_max = None
            w_min = None
            v_max = None
            v_min = None
    #        mu_w = 0.01*np.ones((nx, 1))
    #        Sigma_w= 0.05*np.eye(nx)
            mu_v = 0*np.ones((ny, 1))
            mu_w = 0*np.ones((nx, 1))
            Sigma_w= 0.01*np.eye(nx)
            #initial state distribution parameters
            x0_max = None
            x0_min = None
            x0_mean = np.zeros((nx,1))
            x0_mean[-1] = 1
            x0_cov = 0.01*np.eye(nx)
#             if num_samples == 5:
#                 lambda_list = [785]
# #                lambda_list = [785, 778]
# #                lambda_list = [1385.555267778142, 1385.555267778142, 1358.3935503970677, 1358.3935503970677, 1358.3935503970677, 1358.3935503970677, 1358.3935503970677, 1358.3935503970677, 1358.3935503970677, 1358.3935503970677, 1296.168089557581, 1296.168089557581, 1070.7656103909276,  901.6821492696181, 773.9841316251334, 762.330690162348]
#             if num_samples == 10:
#                 lambda_list = [791]
# #                lambda_list = [791, 780]
# #                lambda_list = [1425.4939614816144, 1425.4939614816144, 1322.0789570185766, 1322.0789570185766, 1322.0789570185766, 1322.0789570185766, 1322.0789570185766, 1322.0789570185766, 1322.0789570185766, 1322.0789570185766, 1314.6500249250216, 1258.672145646804, 1068.38175215323, 906.3784197953951, 777.9021911985495, 764.5118813117117]
#             if num_samples == 20:
#                 lambda_list = [790]
# #                lambda_list = [790, 779]
# #                lambda_list = [1489.399258789312, 1489.399258789312, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1291.5109623563137, 1256.5724282476685, 1052.0678820527962, 897.8983301276867, 776.2229407903919, 763.4769507266147]

#             lambda_ = lambda_list[theta_ind]
                            
        elif dist == "multimodal":
            M = 0.2*np.eye(ny) #observation noise covariance
            theta = 0.8 #Wasserstein ball radius
            #disturbance distribution parameters
            w_max = None
            w_min = None
            mu_w = [np.array([[-0.03],[-0.1], [0.1], [0.2]]), np.array([[0.02],[0.05], [0.01], [0.01]])]
            Sigma_w= [np.array( [[0.01,  0.005, 0.01, 0.02],
                        [0.005, 0.01,  0.001, 0.01],
                        [0.01,  0.001, 0.02, 0.01],
                        [0.001, 0.01, 0.001, 0.03]]),
                      np.array([[0.04,  0.003, 0.002, 0.02],
                        [0.002, 0.003,  0.01, 0.001],
                        [0.001,  0.001, 0.02, 0.001],
                        [0.002, 0.001, 0.001, 0.004]])]
            #initial state distribution parameters
            x0_max = None
            x0_min = None
            x0_mean = np.array([[-1],[-1],[1],[1]])
            x0_cov = 0.001*np.eye(nx)
    
        #-------Estimate the nominal distribution-------
        if out_of_sample:
            mu_hat = []
            Sigma_hat = []
            for i in range(num_sim):
                if infinite:
                    mu_hat_, Sigma_hat_ = gen_sample_dist_inf(dist, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
#                    if dist=="normal":
                    mu_hat_ = 0*np.ones((nx, 1))
                else:
                    mu_hat_, Sigma_hat_ = gen_sample_dist(dist, T, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                mu_hat.append(mu_hat_)
                Sigma_hat.append(Sigma_hat_)
        else:
            if infinite:
                mu_hat, Sigma_hat = gen_sample_dist_inf(dist, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
#                if dist=="normal":
                mu_hat = 0*np.ones((nx, 1))
                _, M_hat = gen_sample_dist(dist, T, num_noise_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate M hat!
            else:
                mu_hat, Sigma_hat = gen_sample_dist(dist, T, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
#                if dist=="normal":
                mu_hat = 0*np.ones((T, nx, 1))
                _, M_hat = gen_sample_dist(dist, T, num_noise_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate M hat!
        
        M_hat = M_hat + 1e-6*np.eye(ny)

        #-------Create a random system-------
        system_data = (A, B, C, Q, Qf, R, M)
    #    print('Sys Data:', system_data)
        #-------Perform n  independent simulations and summarize the results-------
        output_lqg_list = []
        output_wdrc_list = []
        output_drkf_wdrc_list = []
        output_h_infty_list = []
        #Initialize WDRC and LQG controllers
        if out_of_sample:
            drkf_wdrc = []
            wdrc = []
            lqg = []
            for i in range(num_sim):
#                print(i)
                if infinite:
                    #drkf_wdrc_ = inf_DRKF_WDRC(lambda_, theta, T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                    wdrc_ = inf_WDRC(lambda_, theta, T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat[0])
                    lqg_ = inf_LQG(T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat[0])
                
                else:
                    wdrc_ = WDRC(theta, T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                    drkf_wdrc_ = DRKF_WDRC(theta, T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                    
                drkf_wdrc_.backward()
                wdrc_.backward()
                lqg_.backward()

                while not wdrc_.flag or not lqg_.flag or not drkf_wdrc_.flag:
                    print("HERE!")
                    mu_hat[i], Sigma_hat[i] = gen_sample_dist(dist, T, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                    drkf_wdrc_ = DRKF_WDRC(theta, T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                    wdrc_ = WDRC(theta, T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                    lqg_ = LQG(T, dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)

                    drkf_wdrc_.backward()
                    wdrc_.backward()
                    lqg_.backward()

#                print('Success')
#                print(i)
                drkf_wdrc.append(drkf_wdrc_)
                wdrc.append(wdrc_)
                lqg.append(lqg_)
                
        else:
            if infinite:
                    wdrc = inf_WDRC(lambda_, theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat[0])
                    drkf_wdrc = inf_DRKF_WDRC(lambda_, theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat[0])
                    lqg = inf_LQG(T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat[0])
                    
            else:
                drkf_wdrc = DRKF_WDRC(theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                wdrc = WDRC(theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                lqg = LQG(T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)

            drkf_wdrc.backward() 
            wdrc.backward()       
            lqg.backward()
            
        if h_inf:
            h_infty = inf_H_infty(T, dist, system_data, mu_w, Sigma_w, x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_min, v_max, v_max, v_min, M_hat)
            h_infty.backward()
            
        print('---------------------')
        if out_of_sample:
            output_drkf_wdrc = np.empty(num_sim, dtype=object)
            output_wdrc = np.empty(num_sim, dtype=object)
            output_lqg = np.empty(num_sim, dtype=object)
            obj = np.zeros(num_sim)
            os_sample_size = 100 #100
    
    
    
    #    if not infinite:
    #            #Recursively compute the value function and control matrices
    #        wdrc.backward()
    #        lqg.backward()
    		
    #    for i in range(num_sim):
    #        print('i: ', i)
    #
    #
    #        #Perform state estimation and apply the controller
    #        if out_of_sample:
    #           output_wdrc_ = wdrc[i].forward()
    #           output_wdrc.append(output_wdrc_)
    #        else:
    #           if wc:
    #               output_wdrc = wdrc.forward(use_wc=True)
    #               output_lqg = lqg.forward(mu_wc = output_wdrc['mu_wc'], Sigma_wc = output_wdrc['Sigma_wc'])
    #           else:
    #               output_wdrc = wdrc.forward(use_wc=False)
    #               np.random.seed(1337)
    #               output_lqg = lqg.forward()
    #           output_wdrc_list.append(output_wdrc)
    #           output_lqg_list.append(output_lqg)
    #
    #           print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
    #           print('cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
    #
    
        if h_inf:
            np.random.seed(1337)
            for i in range(num_sim):
#                print('i: ', i)
        
        
                if wc:
                    output_h_infty = h_infty.forward(mu_wc = output_wdrc['mu_wc'], Sigma_wc = output_wdrc['Sigma_wc'])
                else:
                    output_h_infty = h_infty.forward()
        
                output_h_infty_list.append(output_h_infty)
        
                print('cost (H_infty):', output_h_infty['cost'][0], 'time (H_infty):', output_h_infty['comp_time'])
    
    
    
        np.random.seed(1337)
        print("Running DRKF_WDRC Forward step ...")
        for i in range(num_sim):
#            print('i: ', i)
            
    
            #Perform state estimation and apply the controller
            if out_of_sample:
                output_drkf_wdrc_sample  = []
                for j in range(os_sample_size):
                   output_drkf_wdrc_ = drkf_wdrc[i].forward()
                   output_drkf_wdrc_sample.append(output_drkf_wdrc_)
                output_drkf_wdrc[i] = output_drkf_wdrc_sample
                obj[i] = drkf_wdrc[i].objective(drkf_wdrc[i].lambda_)
            else:
               if wc:
                   output_drkf_wdrc = drkf_wdrc.forward()
               else:
                   output_drkf_wdrc = drkf_wdrc.forward()
    
               output_drkf_wdrc_list.append(output_drkf_wdrc)
            
               print('cost (DRKF_WDRC):', output_drkf_wdrc['cost'][0], 'time (DRKF_WDRC):', output_drkf_wdrc['comp_time'])
          
        np.random.seed(1337)     
        print("Running WDRC Forward step ...")  
        for i in range(num_sim):
#            print('i: ', i)
    
    
            #Perform state estimation and apply the controller
            if out_of_sample:
                output_wdrc_sample  = []
                for j in range(os_sample_size):
                   output_wdrc_ = wdrc[i].forward()
                   output_wdrc_sample.append(output_wdrc_)
                output_wdrc[i] = output_wdrc_sample
                obj[i] = wdrc[i].objective(wdrc[i].lambda_)
            else:
               if wc:
                   output_wdrc = wdrc.forward()
               else:
                   output_wdrc = wdrc.forward()
    
               output_wdrc_list.append(output_wdrc)
    
               print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
    
        np.random.seed(1337)
        print("Running LQG Forward step ...")
        for i in range(num_sim):
#            print('i: ', i)
    
            if out_of_sample:
                output_lqg_sample  = []
    #            for j in range(os_sample_size):
    #               output_lqg_ = lqg[i].forward()
    #               output_lqg_sample.append(output_lqg_)
    #            output_lqg[i] = output_lqg_sample
            else:
                if wc:
                    output_lqg = lqg.forward(mu_wc = output_wdrc['mu_wc'], Sigma_wc = output_wdrc['Sigma_wc'])
                else:
                    output_lqg = lqg.forward()
        
                output_lqg_list.append(output_lqg)
    
                print('cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
    
    
        #TODO !!!! DRKF needed to be added    
        if out_of_sample:
    
            cost = np.zeros(output_wdrc.shape[0])
            rel = np.zeros(output_wdrc.shape[0])
            for i in range(output_wdrc.shape[0]):
                for j in range(len(output_wdrc[0])):
                    cost[i] = cost[i] + output_wdrc[i][j]['cost'][0]
                    rel[i] = rel[i] + (output_wdrc[i][j]['cost'][0]/T <= obj[i])
                cost[i] = cost[i]/len(output_wdrc[0])
                rel[i] = rel[i]/len(output_wdrc[0])
            avg_os_cost = np.mean(cost)
            std_os_cost = np.std(cost)
            avg_rel = np.mean(rel)
            std_rel = np.std(rel)
            
    #        cost_lqg = np.zeros(output_lqg.shape[0])
    #        for i in range(output_lqg.shape[0]):
    #            for j in range(len(output_lqg[0])):
    #                cost_lqg[i] = cost_lqg[i] + output_lqg[i][j]['cost'][0]
    #            
    #            cost_lqg[i] = cost_lqg[i]/len(output_lqg[0])
    #        
    #        avg_os_cost_lqg = np.mean(cost_lqg)
    #        std_os_cost_lqg = np.std(cost_lqg)
            
            
            print('lambda:', lambda_, )
            print('Average OS cost (WDRC):', avg_os_cost, 'Std_os_cost (WDRC):', std_os_cost)
            print('Average reliability (WDRC):', avg_rel, 'Std reliability (WDRC):', std_rel)
    #        print('Average OS cost: (LQG)', avg_os_cost_lqg, 'Std_os_cost (LQG):', std_os_cost_lqg)
            save_data(path + '/wdrc_os.pkl', [cost, rel])
    #        save_data(path + 'lqg_os.pkl', cost_lqg)
        else:
            #Save results
            save_data(path + 'drkf_wdrc.pkl', output_drkf_wdrc_list)
            save_data(path + 'wdrc.pkl', output_wdrc_list)
            save_data(path + 'lqg.pkl', output_lqg_list)
            if h_inf:
                save_data(path + 'h_infty.pkl', output_h_infty_list)
    
            #Summarize and plot the results
            print('\n-------Summary-------')
#             if h_inf: # TODO !! DRKF NEEDED to be added! 
#                 if sim_type == "multiple":
#                     summarize(output_lqg_list, output_wdrc_list, output_h_infty_list, dist, path, num_sim, plot_results, h_inf)
#                 else:
#                    for i in range(num_sim):
# #                       print('i: ', i)
#                        summarize([output_lqg_list[i]], [output_wdrc_list[i]], [output_h_infty_list[i]], dist, path, i, plot_results, h_inf)
#                        print('---------------------')
#             else:
            print("num_samples : ", num_samples, " num_noise_samples : ", num_noise_samples)
            if sim_type == "multiple":
                summarize(output_lqg_list, output_wdrc_list, output_drkf_wdrc_list , dist, path, num_sim, plot_results, True)
            else:
                for i in range(num_sim):
                    print('i: ', i)
                    summarize([output_lqg_list[i]], [output_wdrc_list[i]], [output_drkf_wdrc_list[i]], dist, path, i, plot_results, True)
                    print('---------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--sim_type', required=False, default="multiple", type=str) #simulation type (single or multiple)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=5, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=10, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=100, type=int) #horizon length
    parser.add_argument('--plot', required=False, action="store_true") #plot results+
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--os', required=False, action="store_true")
    parser.add_argument('--wc', required=False, action="store_true")
    parser.add_argument('--h_inf', required=False, action="store_true")

    args = parser.parse_args()
    np.random.seed(100)
    main(args.dist, args.sim_type, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.plot, args.infinite, args.os, args.wc, args.h_inf)
