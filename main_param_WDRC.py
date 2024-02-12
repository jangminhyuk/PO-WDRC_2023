#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRKF_WDRC import DRKF_WDRC
from controllers.DRKF_WDRC_combined import DRKF_WDRC_0
from controllers.DRKF_WDRC_combined_test import DRKF_WDRC_test
from controllers.MMSE_WDRC import MMSE_WDRC
from controllers.MMSE_WDRC_onlyM import MMSE_WDRC_2
from controllers.inf_LQG import inf_LQG
from controllers.inf_WDRC import inf_WDRC
from controllers.inf_DRKF_WDRC import inf_DRKF_WDRC
from controllers.inf_MMSE_WDRC_onlyM import inf_MMSE_WDRC
from controllers.inf_MMSE_WDRC_both import inf_MMSE_WDRC_both
from controllers.inf_H_infty import inf_H_infty

from plot import summarize
from plot_J import summarize_noise

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
def quad_inverse(x, b, a):
    # row = x.shape[0]
    # col = x.shape[1]
    # beta = (a[0]+b[0])/2.0
    # alpha = 12.0/((b[0]-a[0])**3)
    # for i in range(row):
    #     for j in range(col):
    #         tmp = 3*x[i][j]/alpha - (beta - a[0])**3
    #         if 0<=tmp:
    #             x[i][j] = beta + ( tmp)**(1./3.)
    #         else:
    #             x[i][j] = beta -(-tmp)**(1./3.)
    # return x
    row = x.shape[0]
    col = x.shape[1]
    for i in range(row):
        for j in range(col):
            beta = (a[j]+b[j])/2.0
            alpha = 12.0/((b[j]-a[j])**3)
            tmp = 3*x[i][j]/alpha - (beta - a[j])**3
            if 0<=tmp:
                x[i][j] = beta + ( tmp)**(1./3.)
            else:
                x[i][j] = beta -(-tmp)**(1./3.)
    return x

# quadratic U-shape distrubituon in [wmin , wmax]
def quadratic(wmax, wmin, N=1):
    n = wmin.shape[0]
    x = np.random.rand(N, n)
    #print("wmax : " , wmax)
    x = quad_inverse(x, wmax, wmin)
    return x.T

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
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)

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
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)
        
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

def main(dist, noise_dist1, sim_type, num_sim, num_samples, num_noise_samples, T, method, plot_results, noise_plot_results, infinite, out_of_sample, wc, h_inf):
    application = "Nothing"
    #lambda_ = 1000
    seed = 2024 # any value
    if noise_plot_results: # if you need to draw ploy_J
        num_noise_list = [5, 10, 15, 20, 25, 30]
    else:
        num_noise_list = [num_noise_samples]
    
    # for the noise_plot_results!!
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRKF_WDRC_mean, output_J_MMSE_WDRC_mean =[], [], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRKF_WDRC_std, output_J_MMSE_WDRC_std =[], [], [], [] 
    #-------Initialization-------
    nx = 20 #state dimension
    nu = 10 #control input dimension
    ny = 12#output dimension
    A, B, C = create_matrices(nx, ny, nu) #system matrices generation
    #cost weights
    Q = np.load("./inputs/Q.npy")
    Qf = np.load("./inputs/Q_f.npy")    
    R = np.load("./inputs/R.npy")
    
    #theta_list = [1, 2]
    #theta_list = [0.1, 0.5, 1, 2, 5]
    #theta_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] # for param plot
    #theta_w_list = [0.00001, 0.0001, 0.0005, 0.0015, 0.001, 0.00015, 0.002, 0.0025, 0.005, 0.01, 0.015, 0.05, 0.1, 1]
    theta_w_list =[0.1]
    theta_list = [1]
    noisedist = [noise_dist1] # if you want to test only one distribution
    #noisedist = ["normal", "uniform","quadratic"] # if you want to test 3 distribution at once
    #lambda_list = [1634]
    lambda_list = [750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    for noise_dist in noisedist:
        for lambda_ in lambda_list:
            for theta_w in theta_w_list:
                for theta in theta_list:
                    for num_noise in num_noise_list:
                        print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise)
                        print("lambda : ", lambda_, "/ theta_v : ", theta)
                        np.random.seed(seed) # fix Random seed!
                        print("--------------------------------------------")
                        print("number of noise sample : ", num_noise)
                        print("number of disturbance sample : ", num_samples)
                        #theta = 0.05
                        #Path for saving the results
                        #file_name = f"file_{str(number).replace('.', '_')}.txt"
                        if infinite:
                            if sim_type == "multiple":
                                if out_of_sample:
                                    path = "./results/{}_{}/infinite/N={}/theta={}/".format(noise_dist,  num_samples, theta)
                                else:
                                    path = "./results/{}_{}/infinite/multiple/".format(dist, noise_dist)
                            else:
                                path = "./results/{}_{}/infinite/single/".format(dist, noise_dist)
                            if not os.path.exists(path):
                                os.makedirs(path)
                        else:
                            if sim_type == "multiple":
                                path = "./results/{}_{}/finite/multiple/params/".format(dist, noise_dist)
                            else:
                                path = "./results/{}_{}/finite/single/params/".format(dist, noise_dist)
                            if not os.path.exists(path):
                                os.makedirs(path)
                    
                        
                        #-------Initialization-------
                        if dist =="uniform":
                #            theta = 0.001 #Wasserstein ball radius
                            #disturbance distribution parameters
                            w_max = 0.25*np.ones(nx)
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
                            
                        elif dist == "normal":
                            
                            #_, M_hat = gen_sample_dist_inf('normal', 20, mu_w=0*np.ones((ny, 1)), Sigma_w=M)
                            #M_hat = M_hat + 1e-6*np.eye(ny)
                #            M_hat = M
                #            theta = 0.001 #Wasserstein ball radius
                            #disturbance distribution parameters
                            w_max = None
                            w_min = None

                            mu_w = 0.03*np.ones((nx, 1))
                            Sigma_w= 0.03*np.eye(nx)
                            #initial state distribution parameters
                            x0_max = None
                            x0_min = None
                            x0_mean = np.zeros((nx,1))
                            x0_mean[-1] = 1
                            x0_cov = 0.01*np.eye(nx)
                        elif dist == "quadratic":
                            w_max = 0.25*np.ones(nx)
                            w_min = -0.15*np.ones(nx)
                            mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                            Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                            #initial state distribution parameters
                            x0_max = 0.1*np.ones(nx)
                            x0_min = -0.1*np.ones(nx)
                            x0_max[-1] = 1.05
                            x0_min[-1] = 0.95
                            x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                            x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)               
                        elif dist == "multimodal":
                            M = 0.2*np.eye(ny) #observation noise covariance
                            #theta = 0.8 #Wasserstein ball radius
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
                            
                        #-------Noise distribution ---------#
                        if noise_dist == "uniform":
                            #theta = 0.1 # 0.1!!
                            v_min = -0.3*np.ones(ny)
                            v_max = 0.5*np.ones(ny)
                            mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                            M = 1/12*np.diag((v_max - v_min)**2)
                        elif noise_dist =="normal":
                            #theta = 0.05 # 0.05!!
                            v_max = None
                            v_min = None
                            M = 0.03*np.eye(ny) #observation noise covariance
                            mu_v = 0.03*np.zeros((ny, 1))
                        elif noise_dist =="quadratic":
                            v_min = -0.3*np.ones(ny)
                            v_max = 0.5*np.ones(ny)
                            mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                            M = 3.0/20.0 *np.diag((v_max-v_min)**2)
                            #theta = 0.1 # 0.1!!
                            
                            
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
                                #Sigma_hat = Sigma_hat+ 1e-6*np.eye(nx)
                                mu_hat = 0*np.ones((nx, 1))
                                _, M_hat = gen_sample_dist_inf(noise_dist, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate M hat!
                            else:
                                mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                #                if dist=="normal":
                                #mu_hat = 0*np.ones((T, nx, 1))
                                v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate M hat!(noise)
                        
                        
                        #v_mean_hat = 0*np.ones((T+1, ny, 1))
                        #print(v_mean_hat[0])
                        M_hat = M_hat + 1e-6*np.eye(ny) # to prevent numerical error (if matrix have less than ny samples, it is singular)
                        #print("rank of M : ", np.linalg.matrix_rank(M_hat[0]))
                        #Sigma_hat = Sigma_hat + 1e-7*np.eye(nx)
                        #print(M_hat[0].)
                        
                        #-------Create a random system-------
                        system_data = (A, B, C, Q, Qf, R, M)
                    #    print('Sys Data:', system_data)
                        #-------Perform n  independent simulations and summarize the results-------
                        output_lqg_list = []
                        output_wdrc_list = []
                        output_drkf_wdrc_list = []
                        output_mmse_wdrc_list = []
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
                                    wdrc = inf_WDRC(theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat, -1)
                                    drkf_wdrc = inf_DRKF_WDRC(theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat, -1)
                                    #drkf_wdrc = inf_DRKF_WDRC(lambda_, theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat[0])
                                    mmse_wdrc = inf_MMSE_WDRC_both(theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat, -1)
                                    lqg = inf_LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat)
                                    
                            else: # HERE!!
                                #DRKF method
                                #drkf_wdrc = DRKF_WDRC_test(lambda_, theta_w, theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, -1)
                                #MMSE estimation problem method(MMSE_WDRC: directly from Adversial Analytics(ambiguity for both x and v) /MMSE_WDRC_2: modified to handle the ambiguity only from observation noise v) 
                                #mmse_wdrc = MMSE_WDRC(theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, num_noise_samples, -1)
                                wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, -1)
                                #wdrc = WDRC(theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, -1)
                                #lqg = LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat)
                            
                            
                            wdrc.backward() 
                            #drkf_wdrc.backward()
                            #mmse_wdrc.backward()
                            #lqg.backward()
                            
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
                    
                    
                        if h_inf:
                            for i in range(num_sim):
                #                print('i: ', i)
                        
                        
                                if wc:
                                    output_h_infty = h_infty.forward(mu_wc = output_wdrc['mu_wc'], Sigma_wc = output_wdrc['Sigma_wc'])
                                else:
                                    output_h_infty = h_infty.forward()
                        
                                output_h_infty_list.append(output_h_infty)
                        
                                print('cost (H_infty):', output_h_infty['cost'][0], 'time (H_infty):', output_h_infty['comp_time'])
                    
                
                        np.random.seed(seed) # fix Random seed!
                        #----------------------------
                        print("Running WDRC Forward step ...")
                        for i in range(num_sim):
                #            print('i: ', i)
                            
                            output_wdrc = wdrc.forward()
                
                            output_wdrc_list.append(output_wdrc)
                        
                            print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
                        
                        
                        # print("dist : ", dist, "noise_dist : ",noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
                        #drkf-wdrc print!@!@ just for test!
                        J_WDRC_list = []
                        for out in output_wdrc_list:
                            J_WDRC_list.append(out['cost'])
                        J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
                        J_WDRC_std = np.std(J_WDRC_list, axis=0)
                        output_J_WDRC_mean.append(J_WDRC_mean[0])
                        output_J_WDRC_std.append(J_WDRC_std[0])
                        print(" Average cost (WDRC) : ", J_WDRC_mean[0])
                        print(" std (WDRC) : ", J_WDRC_std[0])
                        
                        #----------------------------             
                
                    
                    
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
                        elif noise_plot_results:
                            J_LQG_list, J_WDRC_list, J_DRKF_WDRC_list, J_MMSE_WDRC_list = [], [], [], []
                            
                            # #lqg
                            # for out in output_lqg_list:
                            #     J_LQG_list.append(out['cost'])
                                
                            # J_LQG_mean= np.mean(J_LQG_list, axis=0)
                            # J_LQG_std = np.std(J_LQG_list, axis=0)
                            # output_J_LQG_mean.append(J_LQG_mean[0])
                            # output_J_LQG_std.append(J_LQG_std[0])
                            
                            # #wdrc
                            # for out in output_wdrc_list:
                            #     J_WDRC_list.append(out['cost'])
                                
                            # J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
                            # J_WDRC_std = np.std(J_WDRC_list, axis=0)
                            # output_J_WDRC_mean.append(J_WDRC_mean[0])
                            # output_J_WDRC_std.append(J_WDRC_std[0])

                            #drkf-wdrc
                            for out in output_drkf_wdrc_list:
                                J_DRKF_WDRC_list.append(out['cost'])
                                
                            J_DRKF_WDRC_mean= np.mean(J_DRKF_WDRC_list, axis=0)
                            J_DRKF_WDRC_std = np.std(J_DRKF_WDRC_list, axis=0)
                            output_J_DRKF_WDRC_mean.append(J_DRKF_WDRC_mean[0])
                            output_J_DRKF_WDRC_std.append(J_DRKF_WDRC_std[0])
                            
                            
                            print("num_noise_sample : ", num_noise, " / finished with dist : ", dist, "/ noise_dist : ", noise_dist, "/ seed : ", seed)
                            print(" Average cost (DRKF-WDRC) : ", J_DRKF_WDRC_mean[0])
                            print(" std (DRKF-WDRC) : ", J_DRKF_WDRC_std[0])
                        else:
                            #Save results
                            theta_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                            
                            #J_DRKF_WDRC_mean= np.mean(J_DRKF_WDRC_list, axis=0)
                            save_data(path + 'wdrc_' + str(lambda_) + '.pkl', J_WDRC_mean)
                            # save_data(path + 'mmse_wdrc.pkl', output_mmse_wdrc_list)
                            # save_data(path + 'wdrc.pkl', output_wdrc_list)
                            # save_data(path + 'lqg.pkl', output_lqg_list)
                            # if h_inf:
                            #     save_data(path + 'h_infty.pkl', output_h_infty_list)
                    
                            #Summarize and plot the results
                            print('\n-------Summary-------')
                            #print("DRKF cost : ", J_DRKF_WDRC_mean)
                            print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
                            print("lambda : ", lambda_)

                    
                    # after running noise_samples lists!
                    if noise_plot_results:
                        if infinite:
                            path = "./results/{}_{}/infinite/multiple/num_noise_plot/{}/".format(dist, noise_dist, theta)
                        else:
                            path = "./results/{}_{}/finite/multiple/num_noise_plot/{}/".format(dist, noise_dist, theta)
                        if not os.path.exists(path):
                            os.makedirs(path)
                            

                        save_data(path + 'drkf_wdrc_mean.pkl', output_J_DRKF_WDRC_mean)
                        save_data(path + 'drkf_wdrc_std.pkl', output_J_DRKF_WDRC_std)  

                        
                        
                        #Summarize and plot the results
                        print('\n-------Summary-------')
                        print("dist : ", dist, "noise_dist : ", noise_dist, "/ num_disturbance_samples : ", num_samples, "/ theta : ", theta, " / noise sample effect PLOT / Seed : ",seed)
                        #summarize_noise(num_noise_list, output_J_LQG_mean, output_J_LQG_std, output_J_WDRC_mean, output_J_WDRC_std, output_J_DRKF_WDRC_mean, output_J_DRKF_WDRC_std, output_J_MMSE_WDRC_mean, output_J_MMSE_WDRC_std, dist, noise_dist, path, False)
                        output_J_MMSE_WDRC_mean, output_J_DRKF_WDRC_mean, output_J_LQG_mean, output_J_WDRC_mean = [], [], [], []
                        output_J_MMSE_WDRC_std, output_J_DRKF_WDRC_std, output_J_LQG_std, output_J_WDRC_std = [], [], [], []
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--sim_type', required=False, default="multiple", type=str) #simulation type (single or multiple)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=5, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=5, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=100, type=int) #horizon length
    parser.add_argument('--method', required=False, default=1, type=int) #method 1 means DRKF(NeurIPS), method 2 means MMSE Estimation, # NOT USED!!
    parser.add_argument('--plot', required=False, action="store_true") #plot results+
    parser.add_argument('--noise_plot', required=False, action="store_true") #plot plotJ_results, (effect of num_noise_samples)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--os', required=False, action="store_true")
    parser.add_argument('--wc', required=False, action="store_true")
    parser.add_argument('--h_inf', required=False, action="store_true")

    args = parser.parse_args()
    np.random.seed(100)
    main(args.dist, args.noise_dist, args.sim_type, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.method, args.plot, args.noise_plot, args.infinite, args.os, args.wc, args.h_inf)
