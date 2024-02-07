#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy

#This method implements the SDP formualation of MMSE estimation problem from "Shafieezadeh Abadeh, Soroosh, et al. "Wasserstein distributionally robust Kalman filtering." Advances in Neural Information Processing Systems 31 (2018)."
#For Kalman filtering, this method makes the ambiguity set of dimension [nx + ny] 
class DRKF_WDRC_test:
    def __init__(self, theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, app_lambda):
        self.dist = dist
        self.noise_dist = noise_dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data
        self.v_mean_hat = v_mean_hat
        self.M_hat = M_hat
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0_mean = x0_mean
        self.x0_cov = x0_cov
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat
        self.mu_w = mu_w
        self.mu_v = mu_v
        self.Sigma_w = Sigma_w
        if self.dist=="uniform" or self.dist=="quadratic":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min

        if self.noise_dist =="uniform" or self.noise_dist =="quadratic":
            self.v_max = v_max
            self.v_min = v_min
            
        self.theta = theta
        #Initial state
        # if self.dist=="normal":
        #     self.lambda_ = 785
        # elif self.dist=="uniform":
        #     self.lambda_ = 780.2396483109309
        # elif self.dist=="quadratic":
        #     self.lambda_ = 780
        self.lambda_ = 1400
        
        if app_lambda>0:
            self.lambda_ = app_lambda #use for lambda modification in application
            
        print("DRKF-WDRC combined")
        #self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
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
        #output = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='L-BFGS-B', options={'maxfun': 100000, 'disp': False, 'maxiter': 100000})
        #print(output.message)
        optimal_penalty = 2*self.infimum_penalty
        print("DRKF WDRC Optimal penalty (lambda_star):", optimal_penalty)
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
        params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat))
        params[3].value = x_cov
        
        sdp_prob.solve(solver=cp.MOSEK)
        Sigma = sdp_prob.variables()[0].value
        X = sdp_prob.variables()[2].value
        status = sdp_prob.status
        return Sigma, X, status

    def kalman_filter_cov(self, M_hat, P, P_w=None):
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
    
    def kalman_filter(self, v_mean_hat, M_hat, x, P, y, mu_w=None, P_w=None, u = None):
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
        resid = y - (self.C @ x_ + v_mean_hat)

#        temp = np.linalg.solve(self.C @ P_ @ self.C.T + self.M, self.C @ P_)
#        P_new = P_ - P_ @ self.C.T @ temp
        x_new = x_ + P @ self.C.T @ np.linalg.inv(M_hat) @ resid
        return x_new

    def solve_DR_sdp(self, P_t1, S_t1, M, X_cov, Sigma_hat, theta):
        #construct problem
        #Variables
        V = cp.Variable((self.nx, self.nx))
        Y = cp.Variable((self.nx,self.nx))
        Sigma_wc = cp.Variable((self.nx, self.nx), symmetric=True)
        Sigma_z = cp.Variable((self.nx+self.ny, self.nx+self.ny), symmetric=True)
        N = cp.Variable((self.ny, self.ny), symmetric=True)
        X_pred = cp.Variable((self.nx,self.nx), symmetric=True)
        M_test = cp.Variable((self.ny, self.ny), symmetric=True)
        
        #Parameters
        #Sigma_hat_12_var = cp.Parameter((self.nx, self.nx)) # nominal Sigma_w root
        Sigma_w = cp.Parameter((self.nx, self.nx)) # nominal Sigma_w
        radi = cp.Parameter(nonneg=True)
        x_cov = cp.Parameter((self.nx, self.nx)) # x_cov from before time step
        M_hat = cp.Parameter((self.ny, self.ny))
        M_hat_12_var = cp.Parameter((self.ny, self.ny)) # nominal M_hat root
        P_var = cp.Parameter((self.nx,self.nx))
        S_var = cp.Parameter((self.nx,self.nx))
        
        #Sigma_hat_12_var.value = np.real(scipy.linalg.sqrtm(Sigma_hat))
        Sigma_w.value = Sigma_hat
        radi.value = theta
        x_cov.value = X_cov
        M_hat.value = M
        M_hat_12_var.value = np.real(scipy.linalg.sqrtm(M))
        P_var.value = P_t1 # P[t+1]
        S_var.value = S_t1 # S[t+1]
        
        
        #use Schur Complements
        #obj function
        obj = cp.Maximize(cp.trace(S_var @ V) + cp.trace((P_var - self.lambda_ * np.eye(self.nx)) @ Sigma_wc) + 2*self.lambda_*cp.trace(Y)) 
        
        #constraints
        constraints = [
                cp.bmat([[Sigma_w, Y],
                         [Y.T, Sigma_wc]
                         ]) >> 0,
                # cp.bmat([[Sigma_hat_12_var @ Sigma_wc @ Sigma_hat_12_var, Y],
                #         [Y, np.eye(self.nx)]
                #         ]) >> 0,
                X_pred == self.A @ x_cov @ self.A.T + Sigma_wc,
                cp.bmat([[X_pred-V, X_pred @ self.C.T],
                        [self.C @ X_pred, self.C @ X_pred @ self.C.T + M_test]
                        ]) >> 0 ,
                self.C @ X_pred @ self.C.T + M_test >>0,
                #S >> cp.lambda_min(Sigma_z) * np.eye(self.nx+self.ny),
                cp.trace(M_hat + M_test - 2*N ) <= radi**2,
                # cp.bmat([[M_hat, N],
                #          [N.T, M_test]
                #          ]) >> 0,
                cp.bmat([[M_hat_12_var @ M_test @ M_hat_12_var, N],
                        [N, np.eye(self.ny)]
                        ]) >> 0,
                N>>0
                ]
        
        prob = cp.Problem(obj, constraints)
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'False in DRKF combined!!!!!!!!!!!!!')
        
        
        S_xx_opt = X_pred.value
        S_xy_opt = X_pred.value @ self.C.T
        S_yy_opt = self.C @ X_pred.value @ self.C.T + M_test.value 
        #S_opt = S.value
        Sigma_wc_opt = Sigma_wc.value
        return  S_xx_opt, S_xy_opt, S_yy_opt, Sigma_wc_opt
    
    #DR Kalman FILTER !!!!!!!!!!!!!!!!!!!!!!!!
    def DR_kalman_filter(self, v_mean_hat, M_hat, x, y, S_xx, S_xy, S_yy, mu_w=None, u = None):
        if u is None:
            #Initial state estimate
            x_ = x
            y_ = self.C @ x 
            #y_ = self.C @ x
        else:
            #Prediction step
            x_ = self.A @ x + self.B @ u + mu_w
            y_ = self.C @ (self.A @ x + self.B @ u + mu_w) + v_mean_hat
            #y_ = self.C @ (self.A @ x + self.B @ u + mu_w)
        
        x_new = S_xy @ np.linalg.inv(S_yy) @ (y - y_) + x_
        return x_new
    
    def DR_kalman_filter_cov(self, P, S, M_hat, X_cov, Sigma_hat):
        #Performs state estimation based on the current state estimate, control input and new observation
        theta = self.theta
        S_xx, S_xy, S_yy, Sigma_wc = self.solve_DR_sdp(P, S, M_hat, X_cov, Sigma_hat, theta) # used theta as a radius!!! (can be changed on purpose)
        
        X_cov_new = S_xx - S_xy @ np.linalg.inv(S_yy) @ S_xy.T
        
        return X_cov_new, S_xx, S_xy, S_yy, Sigma_wc
    
    def DR_kalman_filter_cov_repeat(self, Sigma): # for repeating steps!!
        
        S, S_xx, S_xy, S_yy = self.solve_DR_sdp(Sigma, self.theta) # used theta as a radius!!! (can be changed on purpose)
        
        X_cov_new = S_xx - S_xy @ np.linalg.inv(S_yy) @ S_xy.T
        return X_cov_new, S, S_xx, S_xy, S_yy
    
    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat, lambda_, t):
        #Riccati equation corresponding to the Theorem 1

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ P @ self.A - P_

        # Double check Assumption 1
        if lambda_ <= np.max(np.linalg.eigvals(P)):
        #or lambda_ <= np.max(np.linalg.eigvals(P+S)):
            print("t={}: lambda check False!!!!!!!!!".format(t))
            print("np.max(np.linalg.eigvals(P) : ", np.max(np.linalg.eigvals(P)))
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
        self.S_opt = np.zeros((self.T+1, self.nx + self.ny, self.nx + self.ny))
        self.S_xx = np.zeros((self.T+1, self.nx, self.nx))
        self.S_xy = np.zeros((self.T+1, self.nx, self.ny))
        self.S_yy = np.zeros((self.T+1, self.ny, self.ny))
        self.sigma_wc = np.zeros((self.T, self.nx, self.nx))
        #print(self.v_mean_hat[0])
        self.x_cov[0], self.S_xx[0], self.S_xy[0], self.S_yy[0], self.sigma_wc[0]= self.DR_kalman_filter_cov(self.P[0], self.S[0], self.M_hat[0], self.x0_cov, self.Sigma_hat[0])
        for t in range(self.T):
            print("DRKF WDRC Offline step : ",t,"/",self.T)
            self.x_cov[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1], self.sigma_wc[t] = self.DR_kalman_filter_cov(self.P[t+1], self.S[t+1], self.M_hat[t], self.x_cov[t], self.Sigma_hat[t])
            # sdp_prob = self.gen_sdp(self.lambda_, self.M_hat[t])
            # sigma_wc[t], X_wc, status = self.solve_sdp(sdp_prob, self.x_cov[t], self.P[t+1], self.S[t+1], self.Sigma_hat[t])
            # if status in ["infeasible", "unbounded"]:
            #     print(status, 'False!!!!!!!!!!!!!')
            # #self.x_cov[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1] = self.DR_kalman_filter_cov(self.M_hat[t], self.x_cov[t], self.Sigma_hat[t]) #choice 1
            # self.x_cov[t+1], self.S_opt[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1] = self.DR_kalman_filter_cov(self.M_hat[t], self.x_cov[t], sigma_wc[t]) #choice 2 !!!
            # #self.x_cov[t+1], self.S_opt[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1] = self.DR_kalman_filter_cov(self.M_hat[t], X_wc, sigma_wc[t]) #choice 3
            
            # for i in range(0): # 0 now!!! no repeat!!!
            #     self.x_cov[t+1], self.S_opt[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1] = self.DR_kalman_filter_cov_repeat(self.S_opt[t+1]) #choice 4 !! repeat using S[t+1]
            
            #print("x_cov[] norm : ", np.linalg.norm(self.x_cov[t+1]))
            print("sigma_wc[t] norm : ", np.linalg.norm(self.sigma_wc[t]))
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

        #---system----
        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
        elif self.dist=="quadratic":
            x[0] = self.quadratic(self.x0_max, self.x0_min)
        #---noise----
        if self.noise_dist=="normal":
            true_v = self.normal(self.mu_v, self.M) #observation noise
        elif self.noise_dist=="uniform":
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.noise_dist=="quadratic":
            true_v = self.quadratic(self.v_max, self.v_min) #observation noise
            
                
        y[0] = self.get_obs(x[0], true_v) #initial observation
        
        x_mean[0] = self.DR_kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean, y[0], self.S_xx[0], self.S_xy[0], self.S_yy[0]) #initial state estimation

        for t in range(self.T):
            mu_wc[t] = self.H[t] @ x_mean[t] + self.h[t] #worst-case mean
            
            #disturbance sampling
            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
            elif self.dist=="quadratic":
                true_w = self.quadratic(self.w_max, self.w_min)
            #noise sampling
            if self.noise_dist=="normal":
                true_v = self.normal(self.mu_v, self.M) #observation noise
            elif self.noise_dist=="uniform":
                true_v = self.uniform(self.v_max, self.v_min) #observation noise
            elif self.noise_dist=="quadratic":
                true_v = self.quadratic(self.v_max, self.v_min) #observation noise

            #Apply the control input to the system
            u[t] = self.K[t] @ x_mean[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w 
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the worst-case mean and covariance)
            x_mean[t+1] = self.DR_kalman_filter(self.v_mean_hat[t], self.M_hat[t], x_mean[t], y[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1], mu_wc[t], u=u[t])

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


