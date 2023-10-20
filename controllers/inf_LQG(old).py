#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import scipy 
import control

class inf_LQG:
    def __init__(self, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat):
        self.dist = dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data
        self.M_hat = M_hat
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0_mean = x0_mean
        self.x0_cov = x0_cov
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat
#        self.mu_hat = 0*np.ones((self.nx, 1))
#        self.Sigma_hat= 0.01*np.eye(self.nx)
        self.mu_w = mu_w
        self.Sigma_w = Sigma_w
        if self.dist=="uniform":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
            self.v_max = v_max
            self.v_min = v_min


        self.error_bound = 1e-6
        self.max_iteration = 1000

        self.J = 0
        self.flag = True
        ctrb = control.ctrb(self.A, self.B)
        obs = control.obsv(self.A, scipy.linalg.sqrtm(self.Q))
        obs_ = control.obsv(self.A, self.C)
        ctrb_ = control.ctrb(self.A, scipy.linalg.sqrtm(self.Sigma_hat))
#        if np.linalg.matrix_rank(ctrb) == self.nx:
#            print('Rank is {} - Controllable'.format(self.nx))
#        else:
#            print('Rank is {} - Not Controllable'.format(np.linalg.matrix_rank(ctrb)))
#        if np.linalg.matrix_rank(obs) == self.nx:
#            print('Rank is {} - Observable'.format(self.nx))
#        else:
#            print('Rank is {} - Not Observable'.format(np.linalg.matrix_rank(obs)))
#        if np.linalg.matrix_rank(ctrb_) == self.nx:
#            print('Rank is {} - Controllable'.format(self.nx))
#        else:
#            print('Rank is {} - Not Controllable'.format(np.linalg.matrix_rank(ctrb_)))
#        if np.linalg.matrix_rank(obs_) == self.nx:
#            print('Rank is {} - Observable'.format(self.nx))
#        else:
#            print('Rank is {} - Not Observable'.format(np.linalg.matrix_rank(obs_)))


    def uniform(self, a, b, N=1):
        n = a.shape[0]
        x = a + (b-a)*np.random.rand(N,n)
        return x.T

    def normal(self, mu, Sigma, N=1):
        n = mu.shape[0]
#        w = np.random.normal(size=(n,N)).T
        w = scipy.stats.norm.ppf(np.random.rand(n,N)).T
        
        if (Sigma == 0).all():
            x = mu
        else:
            x = mu + np.linalg.cholesky(Sigma) @ w.T
#        x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
        return x

    def multimodal(self, mu, Sigma, N=1):
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

    def kalman_filter(self, x, y, mu_w, t):
        if t==0:
            x_new = x + self.P_post @ self.C.T @ np.linalg.inv(self.M_hat) @ (y - self.C @ x)
        else:
            #Performs state estimation based on the current state estimate, control input and new observation
            x_new = (self.A + self.B @ self.K_ss) @ x + self.B @ self.L_ss + mu_w + self.P_post @ self.C.T @ np.linalg.inv(self.M_hat) @ (y - self.C @ (self.A + self.B @ self.K_ss) @ x - self.C @ self.B @ self.L_ss)
        return x_new

    def KF_riccati(self, P_ss=None, P_w=None):
#        P_ss = np.zeros((self.nx,self.nx))
        for t in range(self.max_iteration):
            P_ss_temp = self.A @ (P_ss - P_ss @ self.C.T @ np.linalg.solve(self.C @ P_ss @ self.C.T + self.M_hat, self.C @ P_ss)) @ self.A.T + P_w
            max_diff = 0
            for row in range(len(P_ss)):
                for col in range(len(P_ss[0])):
                    if abs(P_ss[row, col] - P_ss_temp[row, col]) > max_diff:
                        max_diff = abs(P_ss[row, col] - P_ss_temp[row, col])
            P_ss = P_ss_temp
            if max_diff < self.error_bound:
                P_post = P_ss - P_ss @ self.C.T @ np.linalg.solve(self.C @ P_ss @ self.C.T + self.M_hat, self.C @ P_ss)
                self.P_post = P_post
                return
        print("Minimax Riccati iteration did not converge")
        P_post = P_ss - P_ss @ self.C.T @ np.linalg.solve(self.C @ P_ss @ self.C.T + self.M_hat, self.C @ P_ss)
        self.P_post = P_post

    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat):
        #Riccati equation for standard LQG

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ (P + S) @ self.A - P_
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        z_ = z + np.trace((S + P) @ Sigma_hat) \
            + (2*mu_hat - Phi @ r).T @ temp @ r + mu_hat.T @ temp @ P @ mu_hat
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        L = - temp2 @ temp @ (r + P @ mu_hat)
        return P_, S_, r_, z_, K, L

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        #Compute P, S, r, z, K and L backward in time
        all_P = np.zeros((self.nx, self.nx, self.max_iteration))
        P = np.zeros((self.nx, self.nx))
        P_ = np.zeros((self.nx, self.nx))
        S = np.zeros((self.nx, self.nx))
        r = np.zeros((self.nx, 1))
        z = 0
        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T
        for t in range(self.max_iteration):
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp  = self.riccati(Phi, P, S, r, z, self.Sigma_hat, self.mu_hat)
            
            all_P[:,:,t] = P
            P_temp_ = self.A.T @ (P_ - P_@ self.B @ np.linalg.inv(self.R + self.B.T @ P_ @ self.B) @ self.B.T @ P_) @ self.A + self.Q
            
            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P[0])):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            r = r_temp
            P_ = P_temp_
            if max_diff < self.error_bound:
                self.P_ss = P
                self.S_ss = S
                self.r_ss = r
                temp2 = np.linalg.solve(self.R, self.B.T)
                temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
                self.K_ss = - temp2 @ temp @ P @ self.A
                self.L_ss = - temp2 @ temp @ (r + P @ self.mu_hat)
                self.KF_riccati(P_ss=self.x0_cov, P_w=self.Sigma_hat)
                return
        print("Minimax Riccati iteration did not converge")
        self.flag = False
        self.P_ss = P
        self.S_ss = S
        self.r_ss = r
        self.K_ss = None
        self.L_ss = None



    def forward(self, mu_wc=None, Sigma_wc=None):
        #Apply the controller forward in time.
        
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        x_cov = np.zeros((self.T+1, self.nx, self.nx))
        J = np.zeros(self.T+1)

        #Initial state
        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
#            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        elif self.dist=="multimodal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        y[0] = self.get_obs(x[0], true_v) #initial observation
        x_mean[0] = self.kalman_filter(self.x0_mean, y[0], self.mu_hat, 0) #initial state estimation
        x_cov[0] = self.P_post

        for t in range(self.T):
            #disturbance sampling
            if self.dist=="normal":
                if mu_wc is not None:
                    true_w = self.normal(mu_wc[t], Sigma_wc)
                else:
                    true_w = self.normal(self.mu_w, self.Sigma_w)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise

            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
                true_v = self.uniform(self.v_max, self.v_min) #observation noise

            elif self.dist=="multimodal":
                true_w = self.multimodal(self.mu_w, self.Sigma_w)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise

            #Apply the control input to the system
            u[t] = self.K_ss @ x_mean[t] + self.L_ss
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the nominal mean and covariance)
            x_mean[t+1] = self.kalman_filter(x_mean[t], y[t+1], self.mu_hat, t+1)
            x_cov[t+1] = self.P_post

        #Compute the total cost
        J[self.T] = x[self.T].T @ self.Q @ x[self.T]
        for t in range(self.T-1, -1, -1):
            J[t] = J[t+1] + x[t].T @ self.Q @ x[t] + u[t].T @ self.R @ u[t]

        end = time.time()
        time_ = end-start
        
#        err_bound_max = x[-1] + 0.03*np.abs(x[-1])
#        err_bound_min = x[-1] - 0.03*np.abs(x[-1])
#        
#        SettlingTime = np.zeros(self.nx)
#        for j in range(self.nx):
#            for i in reversed(range(self.T)):
#                if((x[i,j] <= err_bound_min[j]) | (x[i,j] >= err_bound_max[j])):
#                    SettlingTime[j] = (i+1)*0.1
#                    break
            
        return {'comp_time': time_,
                'state_traj': x,
                'output_traj': y,
                'control_traj': u,
                'cost': J}