#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import scipy
import control
import cvxpy as cp

class inf_DRKF_WDRC: # TODO NOT IMPLEMENTED YET!!!!!!!!!
    def __init__(self, lambda_, theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat):
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
        self.mu_w = mu_w
        self.Sigma_w = Sigma_w
        if self.dist=="uniform":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
            self.v_min = v_min
            self.v_max = v_max

        self.error_bound = 1e-5
        self.max_iteration = 1000

        self.theta = theta
        if dist=="normal":
#            self.lambda_ = 1446.7981611649352
            self.lambda_ = lambda_
        elif dist=="uniform":
#            self.lambda_= 1107.653293607882
            self.lambda_ = lambda_
#        elif dist=="multimodal":
#            self.lambda_ = 10000
#        ass = self.check_assumption(self.lambda_)
#        self.lambda_ = self.optimize_penalty() 
#        self.optimize_penalty() #optimize penalty parameter for theta
        self.Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/self.lambda_ * np.eye(self.nx)
        ctrb = control.ctrb(self.A, scipy.linalg.sqrtm(self.Phi))
        obs = control.obsv(self.A, scipy.linalg.sqrtm(self.Q))
        obs_ = control.obsv(self.A, self.C)
        ctrb_ = control.ctrb(self.A, scipy.linalg.sqrtm(self.Sigma_hat))
        self.flag= True
#        self.sdp_prob = self.gen_sdp(self.lambda_)
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


    def optimize_penalty(self):
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption 1)
        self.infimum_penalty = self.binarysearch_infimum_penalty_finite()
        print("Infimum penalty:", self.infimum_penalty)
        #Optimize penalty using nelder-mead method
        #optimal_penalty = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='nelder-mead', options={'xatol': 1e-6, 'disp': False}).x[0]
#        self.infimum_penalty = 751.3355114497244
        #np.max(np.linalg.eigvals(self.Qf)) + 1e-6
        output = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='Nelder-Mead', options={'disp': False, 'maxiter': 100, 'ftol': 1e-7})
        print(output.message)
        optimal_penalty = output.x[0]
        print("Optimal penalty (lambda_star):", optimal_penalty)
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
#                    print(status)
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
                P_post_ss, sigma_wc_ss, z_tilde_ss, status = self.KF_riccati(self.x0_cov, P, S, penalty)
                if status in ["infeasible", "unbounded"]:
#                    print(status)
                    return False
                if np.max(sigma_wc_ss) >= 1e2:
                    return False
                return True
#        print("Minimax Riccati iteration did not converge")

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

    def gen_sdp(self, lambda_):
            Sigma = cp.Variable((self.nx,self.nx), symmetric=True)
            U = cp.Variable((self.nx,self.nx), symmetric=True)
            V = cp.Variable((self.nx,self.nx), symmetric=True)
            P_bar_1 = cp.Variable((self.nx,self.nx), symmetric=True)
        
            P_var = cp.Parameter((self.nx,self.nx))
            S_var = cp.Parameter((self.nx,self.nx))
            Sigma_hat_12_var = cp.Parameter((self.nx,self.nx))
            P_bar = cp.Parameter((self.nx,self.nx))
            
            obj = cp.Maximize(cp.trace((P_var - lambda_*np.eye(self.nx)) @ Sigma) + 2*lambda_*cp.trace(U) + cp.trace(S_var @ V))
            
            constraints = [
                    cp.bmat([[Sigma_hat_12_var @ Sigma @ Sigma_hat_12_var, U],
                             [U, np.eye(self.nx)]
                             ]) >> 0,
                    Sigma >> 0,
                    P_bar_1 >> 0,
                    cp.bmat([[P_bar_1 - V, P_bar_1 @ self.C.T],
                             [self.C @ P_bar_1, self.C @ P_bar_1 @ self.C.T + self.M_hat]
                            ]) >> 0,        
                    P_bar_1 == self.A @ P_bar @ self.A.T + Sigma,
                    self.C @ P_bar_1 @ self.C.T + self.M_hat >> 0,
                    U >> 0,
                    V >> 0
                    ]
            prob = cp.Problem(obj, constraints)
            return prob

    def gen_sdp_1(self, lambda_):
            Sigma = cp.Variable((self.nx,self.nx), symmetric=True)
            U = cp.Variable((self.nx,self.nx), symmetric=True)
            V = cp.Variable((self.nx,self.nx), symmetric=True)
            P_bar = cp.Variable((self.nx,self.nx), symmetric=True)
        
            P_var = cp.Parameter((self.nx,self.nx))
            S_var = cp.Parameter((self.nx,self.nx))
            Sigma_hat_12_var = cp.Parameter((self.nx,self.nx))
            
            obj = cp.Maximize(cp.trace((P_var - lambda_*np.eye(self.nx)) @ Sigma) + 2*lambda_*cp.trace(U) + cp.trace(S_var @ V))
            
            constraints = [
                    cp.bmat([[Sigma_hat_12_var @ Sigma @ Sigma_hat_12_var, U],
                             [U, np.eye(self.nx)]
                             ]) >> 0,
                    Sigma >> 0,
                    P_bar >> 0,
                    cp.bmat([[P_bar - V, P_bar @ self.C.T],
                             [self.C @ P_bar, self.C @ P_bar @ self.C.T + np.linalg.inv(self.M_hat)]
                            ]) >> 0,        
#                    P_bar >= self.A @ V @ self.A.T + Sigma,
                    cp.bmat([[self.A @ P_bar @ self.A.T - P_bar, self.A @ P_bar @ self.C.T],
                             [self.C @ P_bar @ self.A.T, self.C @ P_bar @ self.C.T + np.linalg.inv(self.M_hat)]
                            ]) >> 0,        
                    U >> 0,
                    V >> 0
                    ]
            prob = cp.Problem(obj, constraints)
            return prob
        
    def solve_sdp(self, sdp_prob, x_cov, P, S, Sigma_hat):
        params = sdp_prob.parameters()
        params[0].value = P
        params[1].value = S
        params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat + 1e-4*np.eye(self.nx)))
#        params[2].value = np.linalg.cholesky(Sigma_hat + 1e-4*np.eye(self.nx))
        params[3].value = x_cov
        
        try:
            sdp_prob.solve(solver=cp.MOSEK)
            Sigma = sdp_prob.variables()[0].value
            U = sdp_prob.variables()[1].value
            V = sdp_prob.variables()[2].value
            P_post = sdp_prob.variables()[3].value
            cost = sdp_prob.value
            status = sdp_prob.status
        except cp.error.SolverError:
#            sdp_prob.solve(solver=cp.SCS)
            Sigma = sdp_prob.variables()[0].value
            U = sdp_prob.variables()[1].value
            V = sdp_prob.variables()[2].value
            P_post = sdp_prob.variables()[3].value
            cost = sdp_prob.value
            status = "unknown"

        return Sigma, P_post, U, V, cost, status
	
    def solve_sdp_1(self, sdp_prob, P, S, Sigma_hat):
        params = sdp_prob.parameters()
        params[0].value = P
        params[1].value = S
        params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat + 1e-4*np.eye(self.nx)))
#        params[3].value = x_cov
        
        sdp_prob.solve(solver=cp.MOSEK)
        Sigma = sdp_prob.variables()[0].value
        P_post = sdp_prob.variables()[3].value
        cost = sdp_prob.value
        status = sdp_prob.status
        return Sigma, cost, status
    
    def kalman_filter(self, x, y, mu_w, t):
        if t==0:
            x_new = x + self.P_post @ self.C.T @ np.linalg.inv(self.M_hat) @ (y - self.C @ x)
        else:
            #Performs state estimation based on the current state estimate, control input and new observation
            x_new = (self.A + self.B @ self.K_ss) @ x + self.B @ self.L_ss + mu_w + self.P_post @ self.C.T @ np.linalg.inv(self.M_hat) @ (y - self.C @ (self.A + self.B @ self.K_ss) @ x - self.C @ self.B @ self.L_ss)
        return x_new
    
    def KF_riccati(self, P_ss_, P_ss, S_ss, lambda_):
#        P_post = P_ss_ - P_ss_ @ self.C.T @ np.linalg.solve(self.C @ P_ss_ @ self.C.T + self.M_hat, self.C @ P_ss_)
#        self.sigma_wc_all = np.zeros((self.max_iteration, self.nx, self.nx))
#        self.z_tilde_all = np.zeros((self.max_iteration, 1))
        sdp_prob_1 = self.gen_sdp_1(lambda_)
        sigma_wc_1, z_tilde__, stat_ = self.solve_sdp_1(sdp_prob_1, P_ss, S_ss, self.Sigma_hat)
        
        P_ss__ = P_ss_
        for t in range(self.max_iteration):
            P_ss_temp_ = self.A @ (P_ss__ - P_ss__ @ self.C.T @ np.linalg.solve(self.C @ P_ss__ @ self.C.T + self.M_hat, self.C @ P_ss__)) @ self.A.T + sigma_wc_1
            if min(np.linalg.eigvals(self.C @ P_ss_temp_ @ self.C.T + self.M_hat)) <= 0:
                    return np.inf
    
            max_diff = 0
            for row in range(len(P_ss__)):
               for col in range(len(P_ss__[0])):
                   if abs(P_ss__[row, col] - P_ss_temp_[row, col]) > max_diff:
                       max_diff = P_ss__[row, col] - P_ss_temp_[row, col]
            P_ss__ = P_ss_temp_
            P_post_ = P_ss__ - P_ss__ @ self.C.T @ np.linalg.solve(self.C @ P_ss__ @ self.C.T + self.M_hat, self.C @ P_ss__)

            if abs(max_diff) < self.error_bound:
                if np.linalg.matrix_rank(control.ctrb(self.A, scipy.linalg.sqrtm(sigma_wc_1))) < self.nx:
                        print('(A, sqrt(Sigma)) is not controllable!!!!!!')
                        stat_ = "infeasible"
                return P_post_, sigma_wc_1, z_tilde__, stat_

#        sigma_wc = np.zeros((self.nx, self.nx))
#        z_tilde_ = 0
#        for t in range(self.max_iteration):
#            P_ss = self.P_list[t+1,:,:]
#            S_ss = self.S_list[t+1, :,:]
#            sigma_wc_temp, p_post, U, V, z_tilde_temp, status = self.solve_sdp(sdp_prob, P_post, P_ss, S_ss, self.Sigma_hat)
#            if status in ["infeasible", "unbounded", "unknown"]:
#                print(status)
#                return 0, 0, 0, status
#            if np.max(sigma_wc_temp) >= 1e2:
#                return 0, 0, 0, "infeasible"
#
#            P_ss_temp = self.A @ (P_ss_ - P_ss_ @ self.C.T @ np.linalg.solve(self.C @ P_ss_ @ self.C.T + self.M_hat, self.C @ P_ss_)) @ self.A.T + sigma_wc_temp
#            
#            if min(np.linalg.eigvals(self.C @ P_ss_temp @ self.C.T + self.M_hat)) <= 0:
#                return np.inf
#
#            max_diff = 0
#            for row in range(len(P_ss_)):
#               for col in range(len(P_ss_[0])):
#                   if abs(P_ss_[row, col] - P_ss_temp[row, col]) > max_diff:
#                       max_diff = P_ss_[row, col] - P_ss_temp[row, col]
#            max_diff_sigma = 0
#            for row in range(len(sigma_wc)):
#                for col in range(len(sigma_wc[0])):
#                   if abs(sigma_wc[row, col] - sigma_wc_temp[row, col]) > max_diff_sigma:
#                       max_diff_sigma = sigma_wc[row, col] - sigma_wc_temp[row, col]
#            max_diff_tilde = z_tilde_ - z_tilde_temp
#            print('KF it: {}, Sigma: {}, z_tilde: {}, P_bar: {}'.format(t, max_diff_sigma, max_diff_tilde, max_diff))
#            
#            
#            P_ss_ = P_ss_temp
#            z_tilde_ = z_tilde_temp
#            sigma_wc = sigma_wc_temp
#            self.sigma_wc_all[t,:,:] = sigma_wc
#            self.z_tilde_all[t,:] = z_tilde_
#            P_post = P_ss_ - P_ss_ @ self.C.T @ np.linalg.solve(self.C @ P_ss_ @ self.C.T + self.M_hat, self.C @ P_ss_)
#            
#            
#            if np.max([abs(max_diff),abs(max_diff_sigma)]) < self.error_bound:
#                if np.linalg.matrix_rank(control.ctrb(self.A, scipy.linalg.sqrtm(sigma_wc))) < self.nx:
#                    print('(A, sqrt(Sigma)) is not controllable!!!!!!')
#                    status = "infeasible"
#                return P_post, sigma_wc, z_tilde_, status
#        print("Minimax Riccati iteration did not converge")
#        P_post = P_ss - P_ss @ self.C.T @ np.linalg.solve(self.C @ P_ss @ self.C.T + self.M_hat, self.C @ P_ss)
        return P_post_, sigma_wc_1, z_tilde__, stat_


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
                self.P_post = P_post
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

    def forward(self, use_wc=False):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        x_cov = np.zeros((self.T+1, self.nx, self.nx))
        J = np.zeros(self.T+1)
        mu_wc = np.zeros((self.T, self.nx, 1))
        if self.lambda_ <= np.max(self.P_ss):
            print("t={}: False!".format(0))

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
            mu_wc[t] = self.H_ss @ x_mean[t] + self.h_ss #worst-case mean

            #disturbance sampling
            if self.dist=="normal":
                if use_wc:
                    true_w = self.normal(mu_wc[t], self.g_ss)
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

            #Update the state estimation (using the worst-case mean and covariance)
            x_mean[t+1] = self.kalman_filter(x_mean[t], y[t+1], mu_wc[t], t+1)
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
                'cost': J,
                'mu_wc': mu_wc,
                'Sigma_wc': self.sigma_wc.copy(),
                'lambda': self.lambda_}


