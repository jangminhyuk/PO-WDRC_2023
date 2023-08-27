#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import scipy 
import control

class inf_H_infty:
    def __init__(self, T, dist, system_data, mu_w, Sigma_w, x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_min, v_max, M_hat):
        self.dist = dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data
        self.M_hat = M_hat
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0_mean = x0_mean
        self.x0_cov = x0_cov
#        self.x0_mean = x0_mean
#        self.x0_cov = x0_cov
#        self.mu_hat = mu_hat
#        self.Sigma_hat = Sigma_hat
#        self.mu_hat = 0*np.ones((self.nx, 1))
#        self.Sigma_hat= 0.01*np.eye(self.nx)
        self.v_min = v_min
        self.v_max = v_max
        self.mu_w = mu_w
        self.Sigma_w = Sigma_w
        if self.dist=="uniform":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min

        self.error_bound = 1e-6
        self.max_iteration = 1000
        if dist=="normal":
            self.lambda_ = 20000
        elif dist=="uniform":
            self.lambda_ = 10000
        elif dist=="multimodal":
            self.lambda_ = 10000
            
        self.J = 0
        ctrb = control.ctrb(self.A, self.B)
        obs = control.obsv(self.A, scipy.linalg.sqrtm(self.Q))
        obs_ = control.obsv(self.A, self.C)
#        ctrb_ = control.ctrb(self.A, scipy.linalg.sqrtm(self.Sigma_hat))
        if np.linalg.matrix_rank(ctrb) == self.nx:
            print('Rank is {} - Controllable'.format(self.nx))
        else:
            print('Rank is {} - Not Controllable'.format(np.linalg.matrix_rank(ctrb)))
        if np.linalg.matrix_rank(obs) == self.nx:
            print('Rank is {} - Observable'.format(self.nx))
        else:
            print('Rank is {} - Not Observable'.format(np.linalg.matrix_rank(obs)))
#        if np.linalg.matrix_rank(ctrb_) == self.nx:
#            print('Rank is {} - Controllable'.format(self.nx))
#        else:
#            print('Rank is {} - Not Controllable'.format(np.linalg.matrix_rank(ctrb_)))
        if np.linalg.matrix_rank(obs_) == self.nx:
            print('Rank is {} - Observable'.format(self.nx))
        else:
            print('Rank is {} - Not Observable'.format(np.linalg.matrix_rank(obs_)))


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

    def filter_(self, x_cap, y, u, t):
        if t==0:
            x_new = x_cap
        else:
            x_new =  self.A @ x_cap + self.B @ u + self.A @ np.linalg.solve(np.eye(self.nx) + self.P_post @ (self.C.T @ self.C - 1/self.lambda_* self.Q), self.P_post) @ (1/self.lambda_ * self.Q @ x_cap + self.C.T @(y - self.C @ x_cap))
            #Performs state estimation based on the current state estimate, control input and new observation
        x_hat = np.linalg.solve(np.eye(self.nx) - self.P_post @ (1/self.lambda_*np.eye(self.nx) - self.C.T @ self.C), x_new + self.P_post @ self.C.T @ y)
        return x_new, x_hat

    def filter_riccati(self):
        P_ss = np.eye(self.nx)
        for t in range(self.max_iteration):
            P_ss_temp = self.A @ np.linalg.solve(np.eye(self.nx) + P_ss @ (self.C.T @ self.C - 1/self.lambda_* self.Q), P_ss) @ self.A.T + np.eye(self.nx)
            max_diff = 0
            for row in range(len(P_ss)):
                for col in range(len(P_ss[0])):
                    if abs(P_ss[row, col] - P_ss_temp[row, col]) > max_diff:
                        max_diff = abs(P_ss[row, col] - P_ss_temp[row, col])
            P_ss = P_ss_temp
            if max_diff < self.error_bound:
                self.P_post = P_ss
                return
        print("Minimax Riccati iteration did not converge")
        self.P_post = P_ss_temp

    def riccati(self, Phi, P):
        #Riccati equation for standard LQG

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        return P_, K

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        #Compute P, S, r, z, K and L backward in time
        all_P = np.zeros((self.nx, self.nx, self.max_iteration))
        P = np.zeros((self.nx, self.nx))

        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T
        for t in range(self.max_iteration):
            P_temp, K_temp  = self.riccati(Phi, P)
            
            all_P[:,:,t] = P
            
            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P[0])):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp

            if max_diff < self.error_bound:
                self.P_ss = P
                self.K_ss = K_temp
                self.filter_riccati()
                return
        print("Minimax Riccati iteration did not converge")
        self.P_ss = P
        self.K_ss = K_temp



    def forward(self, mu_wc=None, Sigma_wc=None):
        #Apply the controller forward in time.
        
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_hat = np.zeros((self.T+1, self.nx, 1))
        x_cap = np.zeros((self.T+1, self.nx, 1))

        J = np.zeros(self.T+1)

#        x[0,-1] = 1
#        x_cap[0,-1] = 1
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

        x_cap[0], x_hat[0] =  self.filter_(x_cap[0], y[0], 0, 0) #initial state estimation

        for t in range(self.T):
            #disturbance sampling
            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
#                true_v = self.uniform(self.v_max, self.v_min)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
            elif self.dist=="multimodal":
                true_w = self.multimodal(self.mu_w, self.Sigma_w)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise

            #Apply the control input to the system
            u[t] = self.K_ss @ x_hat[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the nominal mean and covariance)
            x_cap[t+1], x_hat[t+1] =  self.filter_(x_cap[t], y[t+1], u[t], t+1) 

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