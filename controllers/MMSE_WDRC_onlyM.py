#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy

#This method is similar to the SDP formualation of MMSE estimation problem from "Adversial Analytics" by Viet Anh NGUYEN. https://doi.org/10.5075/epfl-thesis-9731
#However, estimation step in this method only handles ambiguity of observation noise(M), because ambiguity of state variable is already handled while solving SDP
# The Gelbrich MMSE Estimation Problem

class MMSE_WDRC_2:
    def __init__(self, theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat, num_noise_samples):
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
        if self.dist=="uniform" or self.dist=="quadratic":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
            self.v_max = v_max
            self.v_min = v_min

        self.theta = theta
        self.rho_v = 0.05
        #as you have more sample data, the nominal values becomes more accurate
        #so, you don't have to keep the radius of ambiguity set large if you have more sample data
        #rho_v is the radius of ambiguity set used to handle observation noise. Values below were selected arbitarily, and can be modified 
        # if num_noise_samples==5:
        #     self.rho_v = 0.05
        # elif num_noise_samples ==10:
        #     self.rho_v = 0.05
        # elif num_noise_samples ==15:
        #     self.rho_v = 0.03
        # elif num_noise_samples ==20:
        #     self.rho_v = 0.02
        # elif num_noise_samples ==25:
        #     self.rho_v = 0.01
        # elif num_noise_samples ==30:
        #     self.rho_v = 0.01
        #Initial state
        if self.dist=="normal":
            self.lambda_ = 780
        elif self.dist=="uniform":
            self.lambda_ = 780
        elif self.dist=="quadratic":
            self.lambda_ = 780
            
        print("MMSE-WDRC only M")
#        self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
#        self.lambda_ = 3.5
        #self.binarysearch_infimum_penalty_finite()
        self.P = np.zeros((self.T+1, self.nx, self.nx))
        self.S = np.zeros((self.T+1, self.nx, self.nx))
        self.r = np.zeros((self.T+1, self.nx, 1))
        self.z = np.zeros(self.T+1)
        self.K = np.zeros(( self.T, self.nu, self.nx))
        self.L = np.zeros(( self.T, self.nu, 1))
        self.H = np.zeros(( self.T, self.nx, self.nx))
        self.h = np.zeros(( self.T, self.nx, 1))
        self.g = np.zeros(( self.T, self.nx, self.nx))
        #self.sdp_prob = self.gen_sdp(self.lambda_)
        

    def optimize_penalty(self):
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption 1)
        self.infimum_penalty = self.binarysearch_infimum_penalty_finite()
        print("Infimum penalty:", self.infimum_penalty)
        #Optimize penalty using nelder-mead method
        #optimal_penalty = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='nelder-mead', options={'xatol': 1e-6, 'disp': False}).x[0]
        #self.infimum_penalty = 1.5
        #np.max(np.linalg.eigvals(self.Qf)) + 1e-6
        output = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='L-BFGS-B', options={'maxfun': 100000, 'disp': False, 'maxiter': 100000})
        print(output.message)
        optimal_penalty = output.x[0]
        print("DRKF Optimal penalty (lambda_star):", optimal_penalty)
        return optimal_penalty

    def objective(self, penalty):
        #Compute the upper bound in Proposition 1
        P = np.zeros((self.T+1, self.nx,self.nx))        
        S = np.zeros((self.T+1, self.nx,self.nx))
        r = np.zeros((self.T+1, self.nx,1))
        z = np.zeros((self.T+1, 1))
        z_tilde = np.zeros((self.T+1, 1))

        if np.max(np.linalg.eigvals(P)) > penalty:
        #or np.max(np.linalg.eigvals(P + S)) > penalty:
                return np.inf
        if penalty < 0:
            return np.inf
        
        P[self.T] = self.Qf
        if np.max(np.linalg.eigvals(P[self.T])) > penalty:
                return np.inf
        for t in range(self.T-1, -1, -1):

            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/penalty * np.eye(self.nx)
            P[t], S[t], r[t], z[t], K, L, H, h, g = self.riccati(Phi, P[t+1], S[t+1], r[t+1], z[t+1], self.Sigma_hat[t], self.mu_hat[t], penalty, t)
            if np.max(np.linalg.eigvals(P[t])) > penalty:
                return np.inf
        
        #sdp_prob = self.gen_sdp(penalty)
        x_cov = np.zeros((self.T, self.nx, self.nx))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))
        y = self.get_obs(self.x0_init, self.true_v_init)
        x0_mean, x_cov[0] = self.kalman_filter(self.M_hat[0], self.x0_mean, self.x0_cov, y) #initial state estimation
        
        for t in range(0, self.T-1):
            x_cov[t+1] = self.kalman_filter_cov(self.M_hat[t], x_cov[t], sigma_wc[t])
            sdp_prob = self.gen_sdp(penalty, self.M_hat[t])
            sigma_wc[t], z_tilde[t], status = self.solve_sdp(sdp_prob, x_cov[t], P[t+1], S[t+1], self.Sigma_hat[t])
            if status in ["infeasible", "unbounded"]:
                print(status)
                return np.inf
                

        
        return penalty*self.T*self.theta**2 + (x0_mean.T @ P[0] @ x0_mean)[0][0] + 2*(r[0].T @ x0_mean)[0][0] + z[0][0] + np.trace((P[0] + S[0]) @ x_cov[0]) + z_tilde.sum()

    def binarysearch_infimum_penalty_finite(self):
        left = 0
        right = 100000
        while right - left > 1e-6:
            mid = (left + right) / 2.0
            if self.check_assumption(mid):
                right = mid
            else:
                left = mid
        lam_hat = right
        return lam_hat

    def check_assumption(self, penalty):
        #Check Assumption 1
        P = self.Qf
        S = np.zeros((self.nx,self.nx))
        r = np.zeros((self.nx,1))
        z = np.zeros((1,1))
        if penalty < 0:
            return False
        if np.max(np.linalg.eigvals(P)) >= penalty:
        #or np.max(np.linalg.eigvals(P + S)) >= penalty:
                return False
        for t in range(self.T-1, -1, -1):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/penalty * np.eye(self.nx)
            P, S, r, z, K, L, H, h, g = self.riccati(Phi, P, S, r, z, self.Sigma_hat[t], self.mu_hat[t], penalty, t)
            if np.max(np.linalg.eigvals(P)) >= penalty:
                return False
        return True

    def uniform(self, a, b, N=1):
        n = a.shape[0]
        x = a + (b-a)*np.random.rand(N,n)
        return x.T

    def normal(self, mu, Sigma, N=1):
        n = mu.shape[0]
        w = np.random.normal(size=(N,n))
        if (Sigma == 0).all():
            x = mu
        else:
            x = mu + np.linalg.cholesky(Sigma) @ w.T
        return x
    def quad_inverse(self, x, b, a):
        row = x.shape[0]
        col = x.shape[1]
        beta = (a[0]+b[0])/2.0
        alpha = 12.0/((b[0]-a[0])**3)
        for i in range(row):
            for j in range(col):
                tmp = 3*x[i][j]/alpha - (beta - a[0])**3
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
    
    def gen_sdp(self, lambda_, M_hat):
            Sigma = cp.Variable((self.nx,self.nx), symmetric=True)
            Y = cp.Variable((self.nx,self.nx), symmetric=True)
            X = cp.Variable((self.nx,self.nx), symmetric=True)
            X_pred = cp.Variable((self.nx,self.nx), symmetric=True)
        
            P_var = cp.Parameter((self.nx,self.nx))
            S_var = cp.Parameter((self.nx,self.nx))
            Sigma_hat_12_var = cp.Parameter((self.nx,self.nx))
            X_bar = cp.Parameter((self.nx,self.nx))
            
            obj = cp.Maximize(cp.trace((P_var - lambda_*np.eye(self.nx)) @ Sigma) + 2*lambda_*cp.trace(Y) + cp.trace(S_var @ X))
            
            constraints = [
                    cp.bmat([[Sigma_hat_12_var @ Sigma @ Sigma_hat_12_var, Y],
                             [Y, np.eye(self.nx)]
                             ]) >> 0,
                    Sigma >> 0,
                    X_pred >> 0,
                    cp.bmat([[X_pred - X, X_pred @ self.C.T],
                             [self.C @ X_pred, self.C @ X_pred @ self.C.T + M_hat]
                            ]) >> 0,        
                    X_pred == self.A @ X_bar @ self.A.T + Sigma,
                    self.C @ X_pred @ self.C.T + M_hat >> 0,
                    Y >> 0,
                    X >> 0
                    ]
            prob = cp.Problem(obj, constraints)
            return prob
        
        
    def solve_sdp(self, sdp_prob, x_cov, P, S, Sigma_hat):
        params = sdp_prob.parameters()
        params[0].value = P
        params[1].value = S
#        params[2].value = np.linalg.cholesky(Sigma_hat)
        params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat + 1e-4*np.eye(self.nx)))
        #params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat)) # need to be erased!! 
        params[3].value = x_cov
        
        sdp_prob.solve(solver=cp.MOSEK)
        Sigma_w = sdp_prob.variables()[0].value
        X = sdp_prob.variables()[2].value
        status = sdp_prob.status
        return Sigma_w, X, status

    def kalman_filter_cov(self, M_hat, P, P_w=None): # not used!!
        #Performs state estimation based on the current state estimate, control input and new observation
        if P_w is None:
            #Initial state estimate
            P_ = P
        else:
            #Prediction update
            P_ = self.A @ P @ self.A.T + P_w

        #Measurement update
        temp = np.linalg.solve(self.C @ P_ @ self.C.T + M_hat, self.C @ P_)
        P_new = P_ - P_ @ self.C.T @ temp
        return P_new
    
    def kalman_filter(self, M_hat, x, P, y, mu_w=None, P_w=None, u = None): # not used!!
        #Performs state estimation based on the current state estimate, control input and new observation
        if u is None:
            #Initial state estimate
            x_ = x
#            P_ = P
        else:
            #Prediction update
            x_ = self.A @ x + self.B @ u + mu_w
#            P_ = self.A @ P @ self.A.T + P_w

        #Measurement update
        resid = y - self.C @ x_

#        temp = np.linalg.solve(self.C @ P_ @ self.C.T + self.M, self.C @ P_)
#        P_new = P_ - P_ @ self.C.T @ temp
        x_new = x_ + P @ self.C.T @ np.linalg.inv(M_hat) @ resid
        return x_new

    #The Gelbrich MMSE Estimation problem!! Ambiguity only with observation noise v
    def solve_DR_sdp(self, M_hat, X_cov):
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
        Sigma_v.value = M_hat
        #Sigma_v_root.value = np.real(scipy.linalg.sqrtm(M_hat + 1e-4*np.eye(self.ny)))
        Sigma_v_root.value = np.real(scipy.linalg.sqrtm(M_hat))
        
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
        M_opt = (gamma_v_opt**2)*temp @ M_hat @ temp
        
        return Alpha_opt, M_opt
    
    #DR MMSE estimation !
    def DR_Estimation(self, x, Alpha, v, y, mu_w = None, u = None):
        if u is None:
            #Initial state estimate
            x_ = x
        else:
            #Prediction step
            x_ = self.A @ x + self.B @ u + mu_w
        
        b_opt = x_ - Alpha @ (self.C @ x_ + v)
        x_estim = Alpha @ y + b_opt # Linear estimator
        
        return x_estim
    
    def DR_Estimation_cov(self, M_hat, X_cov, Cov_w=None):
        #Performs state estimation based on the current state estimate, control input and new observation
        if Cov_w is None:
            #Initial state estimate
            X_cov_ = X_cov
        else:
            #Prediction update
            X_cov_ = self.A @ X_cov @ self.A.T + Cov_w # using worst case

        X_cov_ = X_cov# need to be erased!!
        Alpha, M_opt= self.solve_DR_sdp(M_hat, X_cov_)
        #X_cov_ = X_cov_ -  1e-6*np.eye(self.nx)# need to be erased!!
        #Alpha= self.solve_DR_sdp(M_hat, X_cov) #need to be erased
        return X_cov_, Alpha, M_opt
    
    def DR_Estimation_cov_initial(self, M_hat, X_cov): # handle both x and v ! 
        
        #X_cov_ = X_cov # need to be erased! just for test
        Alpha, gamma_x = self.solve_DR_sdp_initial(M_hat, X_cov)
        
        K = np.eye(self.nx)-Alpha @ self.C
        temp = np.linalg.inv(gamma_x*np.eye(self.nx)- K.T @ K)
        X_cov = (gamma_x**2) * temp @ X_cov @ temp # worst Cov
        return X_cov , Alpha
    def solve_DR_sdp_initial(self, M_hat, X_cov): # handle both x and v ! 
        #construct problem
        #Variables
        Alpha = cp.Variable((self.nx, self.ny))
        gamma_x = cp.Variable()
        gamma_v = cp.Variable()
        U_x = cp.Variable((self.nx, self.nx), symmetric = True)
        V_x = cp.Variable((self.nx, self.nx), symmetric = True)
        U_v = cp.Variable((self.ny, self.ny), symmetric = True)
        V_v = cp.Variable((self.ny, self.ny), symmetric = True)
        
        #Parameters
        Sigma_x = cp.Parameter((self.nx, self.nx))
        Sigma_x_root = cp.Parameter((self.nx, self.nx))
        Sigma_v = cp.Parameter((self.ny, self.ny))
        Sigma_v_root = cp.Parameter((self.ny, self.ny))
        rho_x = cp.Parameter(nonneg = True)
        rho_v = cp.Parameter(nonneg = True)
        
        Sigma_x.value = X_cov
        #Sigma_x_root.value = np.real(scipy.linalg.sqrtm(X_cov + 1e-4*np.eye(self.nx)))
        Sigma_x_root.value = np.real(scipy.linalg.sqrtm(X_cov))
        
        Sigma_v.value = M_hat
        #Sigma_v_root.value = np.real(scipy.linalg.sqrtm(M_hat + 1e-4*np.eye(self.ny)))
        Sigma_v_root.value = np.real(scipy.linalg.sqrtm(M_hat))
        
        rho_x.value = self.theta
        rho_v.value = self.theta # can be modified!
        
        obj = cp.Minimize( gamma_x*(rho_x**2 - cp.trace(Sigma_x)) + cp.trace(U_x) + gamma_v*(rho_v**2 - cp.trace(Sigma_v)) + cp.trace(U_v) )
        constraints = [
                gamma_x>=0,
                gamma_v>=0,
                U_x >>0,
                V_x >>0,
                U_v >>0,
                V_v >>0,
                cp.bmat([[U_x, gamma_x*Sigma_x_root],
                        [gamma_x*Sigma_x_root, V_x]
                        ]) >> 0,
                cp.bmat([[gamma_x*np.eye(self.nx)-V_x, np.eye(self.nx)-self.C.T @ Alpha.T],
                        [np.eye(self.nx)-Alpha @ self.C , np.eye(self.nx)]
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
        
        Alpha_opt = Alpha.value
        gamma_x_opt = gamma_x.value
        
        return Alpha_opt, gamma_x_opt
    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat, lambda_, t):
        #Riccati equation corresponding to the Theorem 1

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ P @ self.A - P_

        # Double check Assumption 1
        if lambda_ <= np.max(np.linalg.eigvals(P)):
        #or lambda_ <= np.max(np.linalg.eigvals(P+S)):
            print("t={}: False!!!!!!!!!".format(t))
            return None
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        z_ = z + - lambda_* np.trace(Sigma_hat) \
                + (2*mu_hat - Phi @ r).T @ temp @ r + mu_hat.T @ temp @ P @ mu_hat
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        L = - temp2 @ temp @ (r + P @ mu_hat)
        h = np.linalg.inv(lambda_ * np.eye(self.nx) - P) @ (r + P @ self.B @ L + lambda_*mu_hat)
        H = np.linalg.inv(lambda_* np.eye(self.nx)  - P) @ P @ (self.A + self.B @ K)
        g = lambda_**2 * np.linalg.inv(lambda_*np.eye(self.nx) - P) @ Sigma_hat @ np.linalg.inv(lambda_*np.eye(self.nx) - P)
        return P_, S_, r_, z_, K, L, H, h, g

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        #Compute P, S, r, z, K and L, as well as the worst-case distribution parameters H, h and g backward in time
        #\bar{w}_t^* = H[t] \bar{x}_t + h[t], \Sigma_t^* = g[t]

        self.P[self.T] = self.Qf
        if self.lambda_ <= np.max(np.linalg.eigvals(self.P[self.T])) or self.lambda_<= np.max(np.linalg.eigvals(self.P[self.T] + self.S[self.T])):
            print("t={}: False!".format(self.T))

        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/self.lambda_ * np.eye(self.nx)
        for t in range(self.T-1, -1, -1):
            self.P[t], self.S[t], self.r[t], self.z[t], self.K[t], self.L[t], self.H[t], self.h[t], self.g[t] = self.riccati(Phi, self.P[t+1], self.S[t+1], self.r[t+1], self.z[t+1], self.Sigma_hat[t], self.mu_hat[t], self.lambda_, t)

        self.x_cov = np.zeros((self.T+1, self.nx, self.nx))
        self.Alpha = np.zeros((self.T+1, self.nx, self.ny))
        self.M_opt = np.zeros((self.T+1, self.ny, self.ny))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))

        #self.x_cov[0], self.Alpha[0]  = self.DR_Estimation_cov_initial(self.M_hat[0], self.x0_cov) # modified!
        #_, self.Alpha[1]  = self.DR_Estimation_cov_initial(self.M_hat[0], self.x_cov[0]) #self.x_cov[] contains worst case covariance
        self.x_cov[0], self.Alpha[0] = self.DR_Estimation_cov_initial(self.M_hat[0], self.x0_cov)
        for t in range(self.T):
            print("MMSE WDRC Offline step : ",t,"/",self.T)
            sdp_prob = self.gen_sdp(self.lambda_, self.M_hat[t])
            sigma_wc[t], X , status = self.solve_sdp(sdp_prob, self.x_cov[t], self.P[t+1], self.S[t+1], self.Sigma_hat[t]) # X is the worst cov in (t+1)
            if status in ["infeasible", "unbounded"]:
                print(status, 'False!!!!!!!!!!!!!')
                
            #choice MM-7  # below 3 lines need to be erased except MM-7
            # X_ = self.A @ self.x_cov[t] @ self.A.T + sigma_wc[t]
            # temp = np.linalg.solve(self.C @ X_ @ self.C.T + self.M_hat[t], self.C @ X_)
            # X_wc = X_ - X_ @ self.C.T @ temp #worst covariance X
            # #self.x_cov[t+1], self.Alpha[t+1] = self.DR_Estimation_cov(self.M_hat[t+1], X_wc, sigma_wc[t]) #choice MM-7
            # self.x_cov[t+1], self.Alpha[t+1] = self.DR_Estimation_cov(self.M_hat[t], X_wc, sigma_wc[t]) #choice MM-8
            # print("true max e.v: ", np.max(np.linalg.eigvals(self.M)))
            # print("Mhat max e.v: ", np.max(np.linalg.eigvals(self.M_hat[t])))
            # print("true M norm: ", np.linalg.norm(self.M))
            #print("Sigma hat norm: ", np.linalg.norm(self.Sigma_hat[t]))
            self.x_cov[t+1], self.Alpha[t+1], self.M_opt[t] = self.DR_Estimation_cov(self.M_hat[t], X, sigma_wc[t]) #choice MM-9 #
            #print(np.max(np.linalg.eigvals(self.M_opt[t])))
            for i in range(1): # repeated 20 times!!
                sdp_prob = self.gen_sdp(self.lambda_, self.M_opt[t])
                sigma_wc[t], X , status = self.solve_sdp(sdp_prob, self.x_cov[t], self.P[t+1], self.S[t+1], self.Sigma_hat[t]) # changed!!
                self.x_cov[t+1], self.Alpha[t+1], self.M_opt[t] = self.DR_Estimation_cov(self.M_opt[t], X, sigma_wc[t]) # modified!@!!!!!!!!!!!
                #print("M_opt max e.v : ", np.max(np.linalg.eigvals(self.M_opt[t])))
                print("x_cov[t+1] norm : ", np.linalg.norm(self.x_cov[t+1]))

            #self.x_cov[t+1], self.Alpha[t+1] = self.DR_Estimation_cov(self.M_hat[t], self.x_cov[t], sigma_wc[t]) #choice MM-2 # Mosek error sometimes happens
            #self.x_cov[t+1] = X # need to be erased!
            #self.x_cov[t+1], self.Alpha[t+1], self.M_opt[t] = self.DR_Estimation_cov(self.M_hat[t], X, sigma_wc[t]) #choice MM-3 # Best option for MMSE but not good for theory
            #self.x_cov[t+1], self.Alpha[t+1] = self.DR_Estimation_cov(self.M_hat[t], X, self.Sigma_hat[t+1]) #choice MM-4
            #self.x_cov[t+1], self.Alpha[t+1] = self.DR_Estimation_cov(self.M_hat[t+1], X, self.Sigma_hat[t+1]) #choice MM-5
            #self.x_cov[t+2], self.Alpha[t+2] = self.DR_Estimation_cov(self.M_hat[t], X, self.Sigma_hat[t+1]) #choice MM-6
#            if np.min(self.C @ (self.A @ x_cov[t] @ self.A + sigma_wc[t]) @ self.C.T + self.M) < 0:
#                print('False!!!!!!!!!!!!!')
#                break
            #print('old:', self.g[t], 'new:', sigma_wc[t])

    def forward(self):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        J = np.zeros(self.T+1)
        mu_wc = np.zeros((self.T, self.nx, 1))

        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.dist=="quadratic":
            x[0] = self.quadratic(self.x0_max, self.x0_min)
            true_v = self.quadratic(self.v_max, self.v_min) #observation noise    
        y[0] = self.get_obs(x[0], true_v) #initial observation
        v = np.zeros((self.ny,1)) # can be changed!!
        x_mean[0] = self.DR_Estimation(self.x0_mean, self.Alpha[0], v, y[0]) #initial state estimation

        for t in range(self.T):
            #disturbance sampling
            mu_wc[t] = self.H[t] @ x_mean[t] + self.h[t] #worst-case mean

            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise

            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
                true_v = self.uniform(self.v_max, self.v_min) #observation noise
            elif self.dist=="quadratic":
                true_w = self.quadratic(self.w_max, self.w_min)
                true_v = self.quadratic(self.v_max, self.v_min) #observation noise

            #Apply the control input to the system
            u[t] = self.K[t] @ x_mean[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the worst-case mean and covariance)
            x_mean[t+1] = self.DR_Estimation(x_mean[t], self.Alpha[t+1], v, y[t+1], mu_wc[t], u = u[t])

        #Compute the total cost
        J[self.T] = x[self.T].T @ self.Qf @ x[self.T]
        for t in range(self.T-1, -1, -1):
            J[t] = J[t+1] + x[t].T @ self.Q @ x[t] + u[t].T @ self.R @ u[t]

        end = time.time()
        time_ = end-start
        return {'comp_time': time_,
                'state_traj': x,
                'output_traj': y,
                'control_traj': u,
                'cost': J}


