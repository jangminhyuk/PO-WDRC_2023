#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy
import control

class inf_MMSE_WDRC:
    def __init__(self, theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat, app_lambda):
        self.dist = dist
        self.noise_dist = noise_dist
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
        self.mu_w = mu_w
        self.Sigma_w = Sigma_w
        self.error_bound = 1e-5
        self.max_iteration = 1000
        self.theta = theta
        self.rho_v = 0.05
        
        if self.dist=="uniform" or self.dist=="quadratic":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
            
        if self.noise_dist =="uniform" or self.noise_dist =="quadratic":
            self.v_max = v_max
            self.v_min = v_min
        #Initial state
        if self.dist=="normal":
            self.lambda_ = 1400
        elif self.dist=="uniform":
            self.lambda_ = 1400
        elif self.dist=="quadratic":
            self.lambda_ = 1400
        
        if app_lambda>0:
            self.lambda_ = app_lambda #use for lambda modification in application
            
        print("inf_MMSE_WDRConly M / ", self.dist, " / ", self.noise_dist)
        #self.lambda_ = self.optimize_penalty() 
        self.Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/self.lambda_ * np.eye(self.nx)
        self.flag= True
        
        #self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
#        self.lambda_ = 3.5
        #self.binarysearch_infimum_penalty_finite()
        # self.P = np.zeros((self.T+1, self.nx, self.nx))
        # self.S = np.zeros((self.T+1, self.nx, self.nx))
        # self.r = np.zeros((self.T+1, self.nx, 1))
        # self.z = np.zeros(self.T+1)
        # self.K = np.zeros(( self.T, self.nu, self.nx))
        # self.L = np.zeros(( self.T, self.nu, 1))
        # self.H = np.zeros(( self.T, self.nx, self.nx))
        # self.h = np.zeros(( self.T, self.nx, 1))
        # self.g = np.zeros(( self.T, self.nx, self.nx))
        
        #self.sdp_prob = self.gen_sdp(self.lambda_)
    
    def optimize_penalty(self):
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption 1)
        self.infimum_penalty = self.binarysearch_infimum_penalty_finite()
        print("Infimum penalty:", self.infimum_penalty)
        #Optimize penalty using nelder-mead method
        #optimal_penalty = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='nelder-mead', options={'xatol': 1e-6, 'disp': False}).x[0]
        #self.infimum_penalty = 1.5
        #np.max(np.linalg.eigvals(self.Qf)) + 1e-6
        output = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='Nelder-Mead', options={'disp': False, 'maxiter': 100, 'ftol': 1e-7})
        print(output.message)
        optimal_penalty = output.x[0]
        #optimal_penalty = 2* self.infimum_penalty
        print("WDRC Optimal penalty (lambda_star):", optimal_penalty)
        return optimal_penalty

    def objective(self, penalty):
        #Compute the upper bound in Proposition 1
        P = np.zeros((self.nx,self.nx))        
        S = np.zeros((self.nx,self.nx))
        r = np.zeros((self.nx,1))
        z = 0

        if np.max(np.linalg.eigvals(P)) > penalty:
                return np.inf

        for t in range(0, self.max_iteration):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/penalty * np.eye(self.nx)
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp, H_temp, h_temp, g_temp = self.riccati(Phi, P, S, r, z, self.Sigma_hat, self.mu_hat, penalty, t)
            if np.max(np.linalg.eigvals(P_temp)) > penalty or np.max(np.linalg.eigvals(P_temp + S_temp)) > penalty:
                return np.inf

            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P)):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            r = r_temp
            if max_diff < self.error_bound:
                P_ss = P
                S_ss = S
                r_ss = r
                temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
#                sdp_prob = self.gen_sdp(penalty)
                P_post_ss, sigma_wc_ss, z_tilde_ss, status = self.KF_riccati(self.x0_cov, P_ss, S_ss, penalty)
                if status in ["infeasible", "unbounded", "unknown"]:
                    print(status)
                    return np.inf
                if np.max(sigma_wc_ss) >= 1e2:
                    return np.inf
#                Sigma_hat_chol = np.linalg.cholesky(self.Sigma_hat)
#                temp1 = np.linalg.cholesky(Sigma_hat_chol @ sigma_wc_ss @ Sigma_hat_chol)
#                z_tilde_ss = penalty**2 * np.trace( np.linalg.inv(penalty*np.eye(self.nx) - P_ss) @ self.Sigma_hat )
                rho = (2*self.mu_hat - Phi @ r_ss).T @ temp @ r_ss - penalty* np.trace(self.Sigma_hat) + self.mu_hat.T @ temp @ P_ss @ self.mu_hat + z_tilde_ss
#                print('Lambda: ', penalty, 'Objective: ', penalty*self.theta**2 + rho[0])
                return penalty*self.theta**2 + rho[0]
            
    def binarysearch_infimum_penalty_finite(self):
        left = 0
        right = 100000

        while right - left > 1e-6:
            mid = (left + right) / 2.0
#            print("lambda: ", mid)
            if self.check_assumption(mid):
                right = mid
            else:
                left = mid
        lam_hat = right
        return lam_hat

    def check_assumption(self, penalty):
        #Check Assumption 1
        P = np.zeros((self.nx,self.nx))
        S = np.zeros((self.nx,self.nx))
        r = np.zeros((self.nx,1))
        z = np.zeros((1,1))
        if penalty < 0:
            return False
        if np.max(np.linalg.eigvals(P+S)) >= penalty:
        #or np.max(np.linalg.eigvals(P + S)) >= penalty:
                return False
        for t in range(0, self.max_iteration):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/penalty * np.eye(self.nx)
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp, H_temp, h_temp, g_temp = self.riccati(Phi, P, S, r, z, self.Sigma_hat, self.mu_hat, penalty, t)
            if np.max(np.linalg.eigvals(P_temp+S_temp)) > penalty or np.max(np.linalg.eigvals(P_temp)) > penalty:
                return False
            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P[0])):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            r = r_temp
            z = z_temp
            if max_diff < self.error_bound:
#                sdp_prob = self.gen_sdp(penalty)
                #print(np.max(np.linalg.eigvals(self.x0_cov)))
                P_post_ss, sigma_wc_ss, z_tilde_ss, status = self.KF_riccati(self.x0_cov, P, S, penalty)
                if status in ["infeasible", "unbounded"]:
                    #print(status)
                    return False
                if np.max(sigma_wc_ss) >= 1e2:
                    return False
                return True
        print("Minimax Riccati iteration did not converge")

    def uniform(self, a, b, N=1):
        n = a.shape[0]
        x = a + (b-a)*np.random.rand(N,n)
        return x.T

    def normal(self, mu, Sigma, N=1):
        n = mu.shape[0]
        w = scipy.stats.norm.ppf(np.random.rand(n,N)).T
        
        if (Sigma == 0).all():
            x = mu
        else:
            x = mu + np.linalg.cholesky(Sigma) @ w.T
#        x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
        return x
    
    def quad_inverse(self, x, b, a):
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
    def quadratic(self, wmax, wmin, N=1):
        n = wmin.shape[0]
        x = np.random.rand(N, n)
        x = self.quad_inverse(x, wmax, wmin)
        return x.T
    
    def gen_sdp(self, lambda_):
        Sigma = cp.Variable((self.nx,self.nx), symmetric=True)
        Y = cp.Variable((self.nx,self.nx), symmetric=True)
        X = cp.Variable((self.nx,self.nx), symmetric=True)
        X_pred = cp.Variable((self.nx,self.nx), symmetric=True)
    
        P_var = cp.Parameter((self.nx,self.nx))
        S_var = cp.Parameter((self.nx,self.nx))
        Sigma_hat_12_var = cp.Parameter((self.nx,self.nx))
        
        obj = cp.Maximize(cp.trace((P_var - lambda_*np.eye(self.nx)) @ Sigma) + 2*lambda_*cp.trace(Y) + cp.trace(S_var @ X))
        
        constraints = [
                cp.bmat([[Sigma_hat_12_var @ Sigma @ Sigma_hat_12_var, Y],
                            [Y, np.eye(self.nx)]
                            ]) >> 0,
                Sigma >> 0,
                X_pred >> 0,
                cp.bmat([[X_pred - X, X_pred @ self.C.T],
                        [self.C @ X_pred, self.C @ X_pred @ self.C.T + self.M_hat]
                        ]) >> 0,        
                X_pred == self.A @ X @ self.A.T + Sigma,
                self.C @ X_pred @ self.C.T + self.M_hat >> 0,
                Y >> 0,
                X >> 0
                ]
        prob = cp.Problem(obj, constraints)
        return prob
#         Sigma = cp.Variable((self.nx,self.nx), symmetric=True)
#         U = cp.Variable((self.nx,self.nx), symmetric=True)
#         V = cp.Variable((self.nx,self.nx), symmetric=True)
#         P_bar = cp.Variable((self.nx,self.nx), symmetric=True)
    
#         P_var = cp.Parameter((self.nx,self.nx))
#         S_var = cp.Parameter((self.nx,self.nx))
#         Sigma_hat_12_var = cp.Parameter((self.nx,self.nx))
        
#         obj = cp.Maximize(cp.trace((P_var - lambda_*np.eye(self.nx)) @ Sigma) + 2*lambda_*cp.trace(U) + cp.trace(S_var @ V))
        
#         constraints = [
#                 cp.bmat([[Sigma_hat_12_var @ Sigma @ Sigma_hat_12_var, U],
#                             [U, np.eye(self.nx)]
#                             ]) >> 0,
#                 Sigma >> 0,
#                 P_bar >> 0,
#                 cp.bmat([[P_bar - V, P_bar @ self.C.T],
#                             [self.C @ P_bar, self.C @ P_bar @ self.C.T + np.linalg.inv(self.M_hat)]
#                         ]) >> 0,        
# #                    P_bar >= self.A @ V @ self.A.T + Sigma,
#                 cp.bmat([[self.A @ P_bar @ self.A.T - P_bar, self.A @ P_bar @ self.C.T],
#                             [self.C @ P_bar @ self.A.T, self.C @ P_bar @ self.C.T + np.linalg.inv(self.M_hat)]
#                         ]) >> 0,        
#                 U >> 0,
#                 V >> 0
#                 ]
#         prob = cp.Problem(obj, constraints)
#         return prob
        
        
    def solve_sdp(self, sdp_prob, P, S, Sigma_hat):
        params = sdp_prob.parameters()
        params[0].value = P
        params[1].value = S
        params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat))
        
        sdp_prob.solve(solver=cp.MOSEK)
        Sigma = sdp_prob.variables()[0].value
        cost = sdp_prob.value
        status = sdp_prob.status
        return Sigma, cost, status
    
    #The Gelbrich MMSE Estimation problem!! Ambiguity only with observation noise v (M)
    def solve_DR_sdp(self, X_cov): 
        #construct problem
        #Variables
        Alpha = cp.Variable((self.nx, self.ny))
        gamma_v = cp.Variable()
        U = cp.Variable((self.nx, self.nx), symmetric = True)
        U_v = cp.Variable((self.ny, self.ny), symmetric = True)
        V_v = cp.Variable((self.ny, self.ny), symmetric = True)
        
        #Parameters
        Sigma_x = cp.Parameter((self.nx, self.nx))
        Sigma_x_root = cp.Parameter((self.nx, self.nx))
        Sigma_v = cp.Parameter((self.ny, self.ny))
        Sigma_v_root = cp.Parameter((self.ny, self.ny))
        rho_v = cp.Parameter(nonneg = True)
        
        Sigma_x.value = X_cov
        Sigma_x_root.value = np.real(scipy.linalg.sqrtm(X_cov))
        Sigma_v.value = self.M_hat
        Sigma_v_root.value = np.real(scipy.linalg.sqrtm(self.M_hat))
        
        rho_v.value = self.rho_v # can be modified!
        
        obj = cp.Minimize( cp.trace(U) + gamma_v*(rho_v**2 - cp.trace(Sigma_v)) + cp.trace(U_v) )
        constraints = [
                gamma_v>=0,
                #U >>0,
                #UU_v >>0,
                V_v >>0,
                cp.bmat([[U, Sigma_x_root @ (np.eye(self.nx)- Alpha @ self.C).T],
                        [(np.eye(self.nx)- Alpha @ self.C) @ Sigma_x_root, np.eye(self.nx)]
                        ]) >> 0,
                cp.bmat([[U_v, gamma_v*Sigma_v_root],
                        [gamma_v*Sigma_v_root, V_v]
                        ]) >> 0,
                cp.bmat([[gamma_v*np.eye(self.ny)-V_v, Alpha.T],
                        [Alpha, np.eye(self.nx)]
                        ]) >> 0
                ]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'False in DRKF method 2!!!!!!!!!!!!!')
        
        gamma_v_opt = gamma_v.value
        Alpha_opt = Alpha.value
        temp = np.linalg.inv(gamma_v_opt * np.eye(self.ny) - Alpha_opt.T @ Alpha_opt)
        M_opt = (gamma_v_opt**2)*temp @ self.M_hat @ temp
        
        return Alpha_opt, M_opt
    
    def kalman_filter(self, x, y, mu_w, t):
        if t==0:
            x_new = x + self.X_pred @ self.C.T @ np.linalg.inv(self.M_hat) @ (y - self.C @ x)
        else:
            #Performs state estimation based on the current state estimate, control input and new observation
            x_new = (self.A + self.B @ self.K_ss) @ x + self.B @ self.L_ss + mu_w + self.X_pred @ self.C.T @ np.linalg.inv(self.M_hat) @ (y - self.C @ (self.A + self.B @ self.K_ss) @ x - self.C @ self.B @ self.L_ss)
            
        return x_new
    
    def DR_Estimation(self, x, y, v, mu_w=None, u=None):
        if u is None:
            x_ = x
        else:
            #prediction
            x_ = self.A @ x + self.B @ u + mu_w
        
        b_opt = x_ - self.Alpha_opt @ (self.C @ x_ + v)
        x_new = self.Alpha_opt @ y + b_opt        
        return x_new
    
    def KF_riccati(self, X_pred, P_ss, S_ss, lambda_):
        sdp_prob = self.gen_sdp(lambda_)
        sigma_wc_1, z_tilde__, stat_ = self.solve_sdp(sdp_prob, P_ss, S_ss, self.Sigma_hat)
        
        X_pred_ = X_pred
        for t in range(self.max_iteration):
            X_pred_temp_ = self.A @ (X_pred_ - X_pred_ @ self.C.T @ np.linalg.solve(self.C @ X_pred_ @ self.C.T + self.M_hat, self.C @ X_pred_)) @ self.A.T + sigma_wc_1
            if min(np.linalg.eigvals(self.C @ X_pred_temp_ @ self.C.T + self.M_hat)) <= 0:
                    return np.inf
    
            max_diff = 0
            for row in range(len(X_pred_)):
               for col in range(len(X_pred_[0])):
                   if abs(X_pred_[row, col] - X_pred_temp_[row, col]) > max_diff:
                       max_diff = X_pred_[row, col] - X_pred_temp_[row, col]
            X_pred_ = X_pred_temp_
            X_post_ = X_pred_ - X_pred_ @ self.C.T @ np.linalg.solve(self.C @ X_pred_ @ self.C.T + self.M_hat, self.C @ X_pred_)
            
            #print("X_post_", np.linalg.norm(X_post_), "/ max_diff : ", max_diff)
            if abs(max_diff) < self.error_bound:
                if np.linalg.matrix_rank(control.ctrb(self.A, scipy.linalg.sqrtm(sigma_wc_1))) < self.nx:
                        print('(A, sqrt(Sigma)) is not controllable!!!!!!')
                        stat_ = "infeasible"
                return X_post_, sigma_wc_1, z_tilde__, stat_
            
        return X_post_, sigma_wc_1, z_tilde__, stat_
    
    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat, lambda_, t):
        #Riccati equation corresponding to the Theorem 1

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ P @ self.A - P_

        # Double check Assumption 1
#        if lambda_ <= np.max(np.linalg.eigvals(P)) or lambda_ <= np.max(np.linalg.eigvals(P+S)):
#            print("t={}: False!!!!!!!!!".format(t))
#            return None
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        z_ = z + - lambda_* np.trace(Sigma_hat) \
                + (2*mu_hat - Phi @ r).T @ temp @ r + mu_hat.T @ temp @ P @ mu_hat
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        L = - temp2 @ temp @ (r + P @ mu_hat)
        h = np.linalg.inv(lambda_ * np.eye(self.nx) - P) @ (r + P @ self.B @ L + lambda_*mu_hat)
        H = np.linalg.inv(lambda_* np.eye(self.nx)  - P) @ P @ (self.A + self.B @ K)
        g = lambda_**2 * np.linalg.inv(lambda_*np.eye(self.nx) - P - S) @ Sigma_hat @ np.linalg.inv(lambda_*np.eye(self.nx) - P - S)
        return P_, S_, r_, z_, K, L, H, h, g
    
    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        #Compute P, S, r, z, K and L, as well as the worst-case distribution parameters H, h and g backward in time
        #\bar{w}_t^* = H[t] \bar{x}_t + h[t], \Sigma_t^* = g[t]
        self.P_list = np.zeros((self.max_iteration+1, self.nx, self.nx))
        self.S_list = np.zeros((self.max_iteration+1, self.nx, self.nx))
        P = np.zeros((self.nx, self.nx))
        S = np.zeros((self.nx, self.nx))
        self.P_list[0,:,:] = P
        self.S_list[0,:,:] = S
        
        r = np.zeros((self.nx, 1))
        z = 0
#        if self.lambda_ <= np.max(np.linalg.eigvals(P)) or self.lambda_<= np.max(np.linalg.eigvals(P + S)):
#            print("t={}: False!".format(0))

        for t in range(self.max_iteration):
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp, H_temp, h_temp, g_temp = self.riccati(self.Phi, P, S, r, z, self.Sigma_hat, self.mu_hat, self.lambda_, t)
            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P[0])):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            self.P_list[t+1,:,:] = P
            self.S_list[t+1,:,:] = S
            r = r_temp
            z = z_temp
            if max_diff < self.error_bound:
                self.P_ss = P
                self.S_ss = S
                self.r_ss = r
                temp2 = np.linalg.solve(self.R, self.B.T)
                temp = np.linalg.inv(np.eye(self.nx) + P @ self.Phi)
                self.K_ss = - temp2 @ temp @ P @ self.A
                self.L_ss = - temp2 @ temp @ (r + P @ self.mu_hat)
                self.h_ss = np.linalg.inv(self.lambda_ * np.eye(self.nx) - P) @ (r + P @ self.B @ self.L_ss + self.lambda_*self.mu_hat)
                self.H_ss = np.linalg.inv(self.lambda_* np.eye(self.nx)  - P) @ P @ (self.A + self.B @ self.K_ss)
                self.g_ss = self.lambda_**2 * np.linalg.inv(self.lambda_*np.eye(self.nx) - P - S) @ self.Sigma_hat @ np.linalg.inv(self.lambda_*np.eye(self.nx) - P - S)
                self.max_it_P = t
                self.P_list[t+1:,:,:] = P
                self.S_list[t+1:,:,:] = S
                P_post, sigma_wc, z_tilde_, status = self.KF_riccati(self.x0_cov, self.P_ss, self.S_ss, self.lambda_)
                self.X_pred = P_post
                self.Alpha_opt, self.M_opt = self.solve_DR_sdp(self.X_pred)
                self.sigma_wc = sigma_wc
                self.z_tilde = z_tilde_
                return
            
        print("Minimax Riccati iteration did not converge")
        self.P_ss = P
        self.S_ss = S
        self.r_ss = r
        self.K_ss = None
        self.L_ss = None
        self.h_ss = None
        self.H_ss = None
        self.g_ss = None
        self.flag = False

    def forward(self):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        x_cov = np.zeros((self.T+1, self.nx, self.nx))
        J = np.zeros(self.T+1)
        mu_wc = np.zeros((self.T, self.nx, 1))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))

        #---system----
        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
        elif self.dist=="quadratic":
            x[0] = self.quadratic(self.x0_max, self.x0_min)
        #---noise----
        if self.noise_dist=="normal":
            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        elif self.noise_dist=="uniform":
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.noise_dist=="quadratic":
            true_v = self.quadratic(self.v_max, self.v_min) #observation noise
            
        y[0] = self.get_obs(x[0], true_v) #initial observation
        v = np.zeros((self.ny,1)) # can be changed!!
        #x_mean[0] = self.kalman_filter(self.x0_mean, y[0], self.mu_hat, 0) #initial state estimation
        x_mean[0] = self.DR_Estimation(self.x0_mean, y[0], v)
        x_cov[0] = self.X_pred
        
        for t in range(self.T):
            mu_wc[t] = self.H_ss @ x_mean[t] + self.h_ss #worst-case mean
            
            #disturbance sampling
            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
            elif self.dist=="quadratic":
                true_w = self.quadratic(self.w_max, self.w_min)
            #noise sampling
            if self.noise_dist=="normal":
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
            elif self.noise_dist=="uniform":
                true_v = self.uniform(self.v_max, self.v_min) #observation noise
            elif self.noise_dist=="quadratic":
                true_v = self.quadratic(self.v_max, self.v_min) #observation noise

            #Apply the control input to the system
            u[t] = self.K_ss @ x_mean[t] + self.L_ss
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the worst-case mean and covariance)
            #x_mean[t+1] = self.kalman_filter(x_mean[t], y[t+1], mu_wc[t], t+1)
            x_mean[t+1] = self.DR_Estimation(x_mean[t], y[t+1], v, mu_wc[t], u[t])
            x_cov[t+1] = self.X_pred

        #Compute the total cost
        J[self.T] = x[self.T].T @ self.Q @ x[self.T]
        for t in range(self.T-1, -1, -1):
            J[t] = J[t+1] + x[t].T @ self.Q @ x[t] + u[t].T @ self.R @ u[t]

        
        #print("DRKF Optimal penalty (lambda_star):", self.lambda_)
        
        end = time.time()
        time_ = end-start
        return {'comp_time': time_,
                'state_traj': x,
                'output_traj': y,
                'control_traj': u,
                'cost': J}


