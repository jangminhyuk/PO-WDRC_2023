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
from scipy.interpolate import interp1d

def summarize(lqg_lambda_values, lqg_theta_v_values, lqg_cost_values ,wdrc_lambda_values, wdrc_theta_v_values, wdrc_cost_values , drkf_lambda_values, drkf_theta_v_values, drkf_cost_values, drlqc_theta_w_values, drlqc_theta_v_values, drlqc_cost_value, dist, noise_dist, path):
    surfaces = []
    labels = []
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
    theta_w_grid_lqg, theta_v_grid_lqg = np.meshgrid(
        np.linspace(min(lqg_lambda_values), max(lqg_lambda_values), 100),
        np.linspace(min(lqg_theta_v_values), max(lqg_theta_v_values), 100)
    )
    cost_grid_lqg = griddata(
        (lqg_lambda_values, lqg_theta_v_values), lqg_cost_values,
        (theta_w_grid_lqg, theta_v_grid_lqg), method='cubic'
    )

    # Plot data points - LQG
    #ax.scatter(lqg_lambda_values, lqg_theta_values, lqg_cost_values, label='LQG')

    # Plot smooth surface - LQG
    surface_lqg =ax.plot_surface(theta_w_grid_lqg, theta_v_grid_lqg, cost_grid_lqg, alpha=0.4, color='red', label='LQG')
    surfaces.append(surface_lqg)
    labels.append('LQG')
    #-------------------------
    
    # Repeat the process for WDRC
    # Interpolate cost values for smooth surface - WDRC
    theta_w_grid_wdrc, theta_v_grid_wdrc = np.meshgrid(
    np.linspace(min(wdrc_lambda_values), max(wdrc_lambda_values), 100),
    np.linspace(min(wdrc_theta_v_values), max(wdrc_theta_v_values), 100)
    )
    cost_grid_wdrc = griddata(
        (wdrc_lambda_values, wdrc_theta_v_values), wdrc_cost_values,
        (theta_w_grid_wdrc, theta_v_grid_wdrc), method='linear'  # Use linear interpolation
    )

    # Plot data points - WDRC
    #ax.scatter(wdrc_lambda_values, wdrc_theta_values, wdrc_cost_values, label='WDRC')

    # Plot smooth surface - WDRC
    surface_wdrc =ax.plot_surface(theta_w_grid_wdrc, theta_v_grid_wdrc, cost_grid_wdrc, alpha=0.5, color='blue', label='WDRC')
    surfaces.append(surface_wdrc)
    labels.append('WDRC')
    #--------------
    # Plot DRKF data points
    #ax.scatter(drkf_lambda_values, drkf_theta_values, drkf_cost_values, label='DRKF')

    # Interpolate cost values for smooth surface - DRKF
    theta_w_grid_drkf, theta_v_grid_drkf = np.meshgrid(
        np.linspace(min(drkf_lambda_values), max(drkf_lambda_values), 100),
        np.linspace(min(drkf_theta_v_values), max(drkf_theta_v_values), 100)
    )
    cost_grid_drkf = griddata(
        (drkf_lambda_values, drkf_theta_v_values), drkf_cost_values,
        (theta_w_grid_drkf, theta_v_grid_drkf), method='cubic'
    )
    
    # Plot smooth surface - DRKF
    surface_drkf = ax.plot_surface(theta_w_grid_drkf, theta_v_grid_drkf, cost_grid_drkf, alpha=0.6, color='green', label='DRCE')
    surfaces.append(surface_drkf)
    labels.append('DRCE')
    
    #--------------
    # Plot DRLQC data points

    # # Interpolate cost values for smooth surface - DRLQC
    # theta_w_grid_drlqc, theta_v_grid_drlqc = np.meshgrid(
    #     np.linspace(min(drlqc_theta_w_values), max(drlqc_theta_w_values), 100),
    #     np.linspace(min(drlqc_theta_v_values), max(drlqc_theta_v_values), 100)
    # )
    # cost_grid_drlqc = griddata(
    #     (drlqc_theta_w_values, drlqc_theta_v_values), drlqc_cost_values,
    #     (theta_w_grid_drlqc, theta_v_grid_drlqc), method='cubic'
    # )
    
    # # Plot smooth surface 
    # surface_drlqc = ax.plot_surface(theta_w_grid_drlqc, theta_v_grid_drlqc, cost_grid_drlqc, alpha=0.5, color='purple', label='DRLQC')
    # surfaces.append(surface_drlqc)
    # labels.append('DRLQC')
    ##
    #desired_ticks = 5
    
    
    
    # Get the lambda values corresponding to the ticks
    # theta_w_min = min(min(lqg_theta_w_values), min(wdrc_theta_w_values), min(drkf_theta_w_values))
    # theta_w_max = max(max(lqg_theta_w_values), max(wdrc_theta_w_values), max(drkf_theta_w_values))
    # lambda_values = np.linspace(lambda_min, lambda_max, desired_ticks)

    # Set the locator and formatter for the lambda axis
    # ax.xaxis.set_major_locator(ticker.FixedLocator(lambda_values))
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
    ##
    
    ax.legend(handles=surfaces, labels=labels)
    
    # Set labels
    ax.set_xlabel(r'$\lambda$', fontsize=16)
    ax.set_ylabel(r'$\theta_v$', fontsize=16)
    ax.set_zlabel(r'Total Cost', fontsize=16, rotation=90, labelpad=3)
    #ax.set_title('{} system disturbance, {} observation noise'.format(dist, noise_dist), fontsize=16)
    # ax.set_xlim(min(drkf_lambda_values), max(drkf_lambda_values))
    # ax.set_ylim(min(drkf_theta_values), max(drkf_theta_values))
    # ax.set_zlim(min(drkf_cost_values), max(drkf_cost_values))
    
    ax.view_init(elev=20, azim=-65)
    
    plt.show()
    fig.savefig(path + 'params_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight", pad_inches=0.3)
    #plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    args = parser.parse_args()
    
    path = "./results/{}_{}/finite/multiple/params_lambda/".format(args.dist, args.noise_dist)

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
    # Initialize lists to store theta_w, theta_v, and cost values
    drlqc_theta_w_values = []
    drlqc_theta_v_values = []
    drlqc_cost_values = []
    
    drkf_lambda_values = []
    drkf_theta_v_values = []
    drkf_cost_values = []
    
    wdrc_lambda_values = []
    wdrc_theta_v_values = []
    wdrc_cost_values = []
    
    lqg_lambda_values = []
    lqg_theta_v_values = []
    lqg_cost_values = []
    theta_v_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    #lambda_list = [6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    lambda_list = [6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    # Regular expression pattern to extract numbers from file names
    #pattern = r"drkf_wdrc_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)"
    pattern_drkf = r"drkf_wdrc_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
    pattern_drlqc = r"drlqc__(\d+(?:\.\d+)?)_?(\d+(?:\.\d+)?)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
    pattern_wdrc = r"wdrc_(\d+)"
    pattern_lqg = r"lqg"
    # Iterate over each file in the directory
    for filename in os.listdir(path):
        match = re.search(pattern_drkf, filename)
        if match:
            lambda_value = float(match.group(1))  # Extract lambda
            
            theta_v_value = float(match.group(2))   # Extract theta_v value
            theta_v_str = match.group(3)
            theta_v_value += float(theta_v_str)/10
        
            #changed _1_5_ to 1.5!
            #print('theta w : ', theta_w_value, ' / theta v :', theta_v_value)
            
            # Store theta_w and theta values
            drkf_lambda_values.append(lambda_value)
            drkf_theta_v_values.append(theta_v_value)
            
            drkf_file = open(path + filename, 'rb')
            drkf_cost = pickle.load(drkf_file)
            drkf_file.close()

            drkf_cost_values.append(drkf_cost[0])  # Store cost value
        else:
            match_drlqc = re.search(pattern_drlqc, filename)
            if match_drlqc:
                theta_w_value = float(match_drlqc.group(1))  # Extract theta_w value
                theta_w_str = match_drlqc.group(2)
                theta_w_value += float(theta_w_str)/10
                
                theta_v_value = float(match_drlqc.group(3))   # Extract theta_v value
                theta_v_str = match_drlqc.group(4)
                theta_v_value += float(theta_v_str)/10
            
                #changed _1_5_ to 1.5!
                #print(theta_w_value)
                # Store theta_w and theta values
                drlqc_theta_w_values.append(theta_w_value)
                drlqc_theta_v_values.append(theta_v_value)
                
                drlqc_file = open(path + filename, 'rb')
                drlqc_cost = pickle.load(drlqc_file)
                drlqc_file.close()

                drlqc_cost_values.append(drlqc_cost[0])  # Store cost value
            else:
                match_wdrc = re.search(pattern_wdrc, filename)
                if match_wdrc: # wdrc
                    lambda_value = float(match_wdrc.group(1))  # Extract lambda
                    #print('theta w : ', theta_w_value)
                    wdrc_file = open(path + filename, 'rb')
                    wdrc_cost = pickle.load(wdrc_file)
                    wdrc_file.close()
                    #print(wdrc_cost[0])
                    for aux_theta_v in theta_v_list:
                        wdrc_lambda_values.append(lambda_value)
                        #print(wdrc_cost[0])
                        wdrc_theta_v_values.append(aux_theta_v) # since wdrc not affected by theta v, just add auxilary theta for plot
                        wdrc_cost_values.append(wdrc_cost[0])
                        #print(wdrc_cost[0])
                else:
                    match_lqg = re.search(pattern_lqg, filename)
                    if match_lqg:
                        lqg_file = open(path + filename, 'rb')
                        lqg_cost = pickle.load(lqg_file)
                        lqg_file.close()
                        for aux_lambda in lambda_list:
                            for aux_theta_v in theta_v_list:
                                lqg_lambda_values.append(aux_lambda)
                                lqg_theta_v_values.append(aux_theta_v)
                                lqg_cost_values.append(lqg_cost[0])
                                #print(lqg_cost[0])
                
                    

    # Convert lists to numpy arrays
    drkf_lambda_values = np.array(drkf_lambda_values)
    drkf_theta_v_values = np.array(drkf_theta_v_values)
    drkf_cost_values = np.array(drkf_cost_values)
    
    drlqc_theta_w_values = np.array(drlqc_theta_w_values)
    drlqc_theta_v_values = np.array(drlqc_theta_v_values)
    drlqc_cost_values = np.array(drlqc_cost_values)
    
    wdrc_lambda_values = np.array(wdrc_lambda_values)
    wdrc_theta_v_values = np.array(wdrc_theta_v_values)
    wdrc_cost_values = np.array(wdrc_cost_values)
    
    lqg_lambda_values = np.array(lqg_lambda_values)
    lqg_theta_v_values = np.array(lqg_theta_v_values)
    lqg_cost_values = np.array(lqg_cost_values)
    summarize(lqg_lambda_values, lqg_theta_v_values, lqg_cost_values ,wdrc_lambda_values, wdrc_theta_v_values, wdrc_cost_values , drkf_lambda_values, drkf_theta_v_values, drkf_cost_values ,drlqc_theta_w_values, drlqc_theta_v_values, drlqc_cost_values , args.dist, args.noise_dist, path)


