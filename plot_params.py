#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.ticker as ticker

def summarize(lqg_lambda_values, lqg_theta_values, lqg_cost_values ,wdrc_lambda_values, wdrc_theta_values, wdrc_cost_values , drkf_lambda_values, drkf_theta_values, drkf_cost_values , dist, noise_dist, path):
    # Create 3D plot
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # -------------------
    # LQG
    # Interpolate cost values for smooth surface - LQG
    lambda_grid_lqg, theta_grid_lqg = np.meshgrid(
        np.linspace(min(lqg_lambda_values), max(lqg_lambda_values), 100),
        np.linspace(min(lqg_theta_values), max(lqg_theta_values), 100)
    )
    cost_grid_lqg = griddata(
        (lqg_lambda_values, lqg_theta_values), lqg_cost_values,
        (lambda_grid_lqg, theta_grid_lqg), method='cubic'
    )

    # Plot data points - LQG
    #ax.scatter(lqg_lambda_values, lqg_theta_values, lqg_cost_values, label='LQG')

    # Plot smooth surface - LQG
    ax.plot_surface(lambda_grid_lqg, theta_grid_lqg, cost_grid_lqg, alpha=0.4, color='red', label='LQG')
    
    #-------------------------
    
    # Repeat the process for WDRC
    # Interpolate cost values for smooth surface - WDRC
    lambda_grid_wdrc, theta_grid_wdrc = np.meshgrid(
        np.linspace(min(wdrc_lambda_values), max(wdrc_lambda_values), 100),
        np.linspace(min(wdrc_theta_values), max(wdrc_theta_values), 100)
    )
    cost_grid_wdrc = griddata(
        (wdrc_lambda_values, wdrc_theta_values), wdrc_cost_values,
        (lambda_grid_wdrc, theta_grid_wdrc), method='cubic'
    )

    # Plot data points - WDRC
    #ax.scatter(wdrc_lambda_values, wdrc_theta_values, wdrc_cost_values, label='WDRC')

    # Plot smooth surface - WDRC
    ax.plot_surface(lambda_grid_wdrc, theta_grid_wdrc, cost_grid_wdrc, alpha=0.3, color='blue', label='WDRC')
    #--------------
    # Plot DRKF data points
    #ax.scatter(drkf_lambda_values, drkf_theta_values, drkf_cost_values, label='DRKF')

    # Interpolate cost values for smooth surface - DRKF
    lambda_grid_drkf, theta_grid_drkf = np.meshgrid(
        np.linspace(min(drkf_lambda_values), max(drkf_lambda_values), 100),
        np.linspace(min(drkf_theta_values), max(drkf_theta_values), 100)
    )
    cost_grid_drkf = griddata(
        (drkf_lambda_values, drkf_theta_values), drkf_cost_values,
        (lambda_grid_drkf, theta_grid_drkf), method='cubic'
    )

    # Plot smooth surface - DRKF
    ax.plot_surface(lambda_grid_drkf, theta_grid_drkf, cost_grid_drkf, alpha=0.5, color='green', label='DRKF-WDRC')
    ##
    desired_ticks = 5
    
    # Get the lambda values corresponding to the ticks
    lambda_min = min(min(lqg_lambda_values), min(wdrc_lambda_values), min(drkf_lambda_values))
    lambda_max = max(max(lqg_lambda_values), max(wdrc_lambda_values), max(drkf_lambda_values))
    lambda_values = np.linspace(lambda_min, lambda_max, desired_ticks)

    # Set the locator and formatter for the lambda axis
    ax.xaxis.set_major_locator(ticker.FixedLocator(lambda_values))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
    ##
    
    ax.legend()
    # Set labels
    ax.set_xlabel(r'$\lambda$', fontsize=12)
    ax.set_ylabel(r'$\theta_v$', fontsize=12)
    ax.set_zlabel(r'Total Cost', fontsize=12, labelpad=1)
    #ax.set_title('Normal Disturbance and Noise Distributions', fontsize=14)
    
    # ax.set_xlim(min(drkf_lambda_values), max(drkf_lambda_values))
    # ax.set_ylim(min(drkf_theta_values), max(drkf_theta_values))
    # ax.set_zlim(min(drkf_cost_values), max(drkf_cost_values))
    
    ax.view_init(elev=20, azim=30)
    #plt.show()
    fig.savefig(path + 'params_{}_{}.pdf'.format(dist, noise_dist), dpi=150, bbox_inches="tight", pad_inches=0.2)
    #plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    args = parser.parse_args()
    
    path = "./results/{}_{}/finite/multiple/params/".format(args.dist, args.noise_dist)

    #Load data
    lqg_data =[]
    wdrc_data = []
    # lqg_file = open(path + 'lqg.pkl', 'rb')
    # wdrc_file = open(path + 'wdrc.pkl', 'rb')
    
    # lqg_data = pickle.load(lqg_file)
    # wdrc_data = pickle.load(wdrc_file)
    
    # lqg_file.close()
    # wdrc_file.close()
    
    ##DRKF datas
    # Initialize lists to store lambda, theta, and cost values
    drkf_lambda_values = []
    drkf_theta_values = []
    drkf_cost_values = []
    
    wdrc_lambda_values = []
    wdrc_theta_values = []
    wdrc_cost_values = []
    
    lqg_lambda_values = []
    lqg_theta_values = []
    lqg_cost_values = []
    
    theta_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lambda_list = [750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    # Regular expression pattern to extract numbers from file names
    #pattern = r"drkf_wdrc_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)"
    pattern = r"drkf_wdrc_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
    pattern_fixed_theta = r"wdrc_(\d+(?:\.\d+)?)"
    pattern_lqg = r"lqg"
    # Iterate over each file in the directory
    for filename in os.listdir(path):
        match = re.search(pattern, filename)
        if match:
            lambda_value = float(match.group(1))  # Extract lambda value
            theta_value = float(match.group(2))   # Extract theta value
            
            theta_str = match.group(3)
            theta_value += float(theta_str)/10
            #changed _1_5_ to 1.5!
            #print(theta_value)
            # Store lambda and theta values
            drkf_lambda_values.append(lambda_value)
            drkf_theta_values.append(theta_value)
            
            drkf_file = open(path + filename, 'rb')
            drkf_cost = pickle.load(drkf_file)
            drkf_file.close()

            drkf_cost_values.append(drkf_cost[0])  # Store cost value
        else:
            match_fixed_theta = re.search(pattern_fixed_theta, filename)
            if match_fixed_theta: # wdrc
                lambda_value = float(match_fixed_theta.group(1))  # Extract lambda value
                wdrc_file = open(path + filename, 'rb')
                wdrc_cost = pickle.load(wdrc_file)
                wdrc_file.close()
                for aux_theta in theta_list:
                    wdrc_lambda_values.append(lambda_value)
                    wdrc_theta_values.append(aux_theta) # since wdrc not affected by theta v, just add auxilary theta for plot
                    wdrc_cost_values.append(wdrc_cost[0])
                    #print(wdrc_cost[0])
            else:
                match_lqg = re.search(pattern_lqg, filename)
                if match_lqg:
                    lqg_file = open(path + filename, 'rb')
                    lqg_cost = pickle.load(lqg_file)
                    lqg_file.close()
                    for aux_theta in theta_list:
                        for aux_lambda in lambda_list:
                            lqg_lambda_values.append(aux_lambda)
                            lqg_theta_values.append(aux_theta)
                            lqg_cost_values.append(lqg_cost[0])
                            #print(lqg_cost[0])
                
                    

    # Convert lists to numpy arrays
    drkf_lambda_values = np.array(drkf_lambda_values)
    drkf_theta_values = np.array(drkf_theta_values)
    drkf_cost_values = np.array(drkf_cost_values)
    
    wdrc_lambda_values = np.array(wdrc_lambda_values)
    wdrc_theta_values = np.array(wdrc_theta_values)
    wdrc_cost_values = np.array(wdrc_cost_values)
    
    lqg_lambda_values = np.array(lqg_lambda_values)
    lqg_theta_values = np.array(lqg_theta_values)
    lqg_cost_values = np.array(lqg_cost_values)
    summarize(lqg_lambda_values, lqg_theta_values, lqg_cost_values ,wdrc_lambda_values, wdrc_theta_values, wdrc_cost_values , drkf_lambda_values, drkf_theta_values, drkf_cost_values , args.dist, args.noise_dist, path)


