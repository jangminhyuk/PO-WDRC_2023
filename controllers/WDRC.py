#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
<<<<<<< HEAD
import scipy
import control
=======
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e

class WDRC:
    def __init__(self, theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, M_hat):
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
            self.v_max = v_max
            self.v_min = v_min

        
        self.theta = theta
<<<<<<< HEAD
        #Initial state
        if self.dist=="normal":
            self.lambda_ = 785
        elif self.dist=="uniform":
            self.lambda_ = 780.2396483109309
            
        #Initial state
        if self.dist=="normal":
            self.x0_init = self.normal(self.x0_mean, self.x0_cov)
            self.true_v_init = self.normal(np.zeros((self.ny,1)), self.M)
        elif self.dist=="uniform":
            self.x0_init = self.uniform(self.x0_max, self.x0_min)
            self.true_v_init = self.uniform(v_max, v_min)
        
        
        print("WDRC ", self.dist, )
#        self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
#        self.lambda_ = 3.5
=======
        self.true_v_init = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        #Initial state
        if self.dist=="normal":
            self.x0_init = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            self.x0_init = self.uniform(self.x0_max, self.x0_min)
            
        self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
        #self.lambda_ = 3.5
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
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
<<<<<<< HEAD
        
        #self.sdp_prob = self.gen_sdp(self.lambda_)
    
=======
        self.sdp_prob = self.gen_sdp(self.lambda_)
        

>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
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
<<<<<<< HEAD
        print("DRKF Optimal penalty (lambda_star):", optimal_penalty)
=======
        print("Optimal penalty (lambda_star):", optimal_penalty)
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
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
        
<<<<<<< HEAD
        #sdp_prob = self.gen_sdp(penalty)
        x_cov = np.zeros((self.T, self.nx, self.nx))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))
        y = self.get_obs(self.x0_init, self.true_v_init)
        x0_mean, x_cov[0] = self.kalman_filter(self.M_hat[0], self.x0_mean, self.x0_cov, y) #initial state estimation
        
        for t in range(0, self.T-1):
            x_cov[t+1] = self.kalman_filter_cov(self.M_hat[t], x_cov[t], sigma_wc[t])
            sdp_prob = self.gen_sdp(penalty, self.M_hat[t])
=======
        sdp_prob = self.gen_sdp(penalty)
        x_cov = np.zeros((self.T, self.nx, self.nx))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))
        y = self.get_obs(self.x0_init, self.true_v_init)
        x0_mean, x_cov[0] = self.kalman_filter(self.x0_mean, self.x0_cov, y) #initial state estimation
        
        for t in range(0, self.T-1):
            x_cov[t+1] = self.kalman_filter_cov(x_cov[t], sigma_wc[t])
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
            sigma_wc[t], z_tilde[t], status = self.solve_sdp(sdp_prob, x_cov[t], P[t+1], S[t+1], self.Sigma_hat[t])
            if status in ["infeasible", "unbounded"]:
                print(status)
                return np.inf
                
<<<<<<< HEAD
        return penalty*self.T*self.theta**2 + (x0_mean.T @ P[0] @ x0_mean)[0][0] + 2*(r[0].T @ x0_mean)[0][0] + z[0][0] + np.trace((P[0] + S[0]) @ x_cov[0]) + z_tilde.sum()
    
=======

        
        return penalty*self.T*self.theta**2 + (x0_mean.T @ P[0] @ x0_mean)[0][0] + 2*(r[0].T @ x0_mean)[0][0] + z[0][0] + np.trace((P[0] + S[0]) @ x_cov[0]) + z_tilde.sum()

>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
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
    
<<<<<<< HEAD
    def gen_sdp(self, lambda_, M_hat):
=======
    def gen_sdp(self, lambda_):
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
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
<<<<<<< HEAD
                             [self.C @ P_bar_1, self.C @ P_bar_1 @ self.C.T + M_hat]
                            ]) >> 0,        
                    P_bar_1 == self.A @ P_bar @ self.A.T + Sigma,
                    self.C @ P_bar_1 @ self.C.T + M_hat >> 0,
                    U >> 0,
                    V >> 0
=======
                             [self.C @ P_bar_1, self.C @ P_bar_1 @ self.C.T + np.linalg.inv(self.M)]
                            ]) >> 0,        
                    P_bar_1 == self.A @ P_bar @ self.A.T + Sigma,
                    #U >> 0,
                    #V >> 0
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
                    ]
            prob = cp.Problem(obj, constraints)
            return prob
        
        
    def solve_sdp(self, sdp_prob, x_cov, P, S, Sigma_hat):
        params = sdp_prob.parameters()
        params[0].value = P
        params[1].value = S
<<<<<<< HEAD
#        params[2].value = np.linalg.cholesky(Sigma_hat)
        params[2].value = np.real(scipy.linalg.sqrtm(Sigma_hat + 1e-5*np.eye(self.nx)))
=======
        params[2].value = np.linalg.cholesky(Sigma_hat)
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
        params[3].value = x_cov
        
        sdp_prob.solve(solver=cp.MOSEK)
        Sigma = sdp_prob.variables()[0].value
        cost = sdp_prob.value
        status = sdp_prob.status
        return Sigma, cost, status

<<<<<<< HEAD
    def kalman_filter_cov(self, M_hat, P, P_w=None):
=======
    def kalman_filter_cov(self, P, P_w=None):
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
        #Performs state estimation based on the current state estimate, control input and new observation
        if P_w is None:
            #Initial state estimate
            P_ = P
        else:
            #Prediction update
            P_ = self.A @ P @ self.A.T + P_w

        #Measurement update
<<<<<<< HEAD
        temp = np.linalg.solve(self.C @ P_ @ self.C.T + M_hat, self.C @ P_)
        P_new = P_ - P_ @ self.C.T @ temp
        return P_new
    
    def kalman_filter(self, M_hat, x, P, y, mu_w=None, P_w=None, u = None):
=======
        temp = np.linalg.solve(self.C @ P_ @ self.C.T + self.M, self.C @ P_)
        P_new = P_ - P_ @ self.C.T @ temp
        return P_new
    
    def kalman_filter(self, x, P, y, mu_w=None, P_w=None, u = None):
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
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
        sigma_wc = np.zeros((self.T, self.nx, self.nx))

        self.x_cov[0] = self.kalman_filter_cov(self.M_hat[0], self.x0_cov)
        for t in range(self.T):
            print("WDRC backward step(Offline) : ",t,"/",self.T)
            sdp_prob = self.gen_sdp(self.lambda_, self.M_hat[t])
            sigma_wc[t], _, status = self.solve_sdp(sdp_prob, self.x_cov[t], self.P[t+1], self.S[t+1], self.Sigma_hat[t])
            if status in ["infeasible", "unbounded"]:
                print(status, 'False!!!!!!!!!!!!!')
            self.x_cov[t+1] = self.kalman_filter_cov(self.M_hat[t], self.x_cov[t], sigma_wc[t])


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
        sigma_wc = np.zeros((self.T, self.nx, self.nx))

<<<<<<< HEAD
        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
#            true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
            
        y[0] = self.get_obs(x[0], true_v) #initial observation
        x_mean[0] = self.kalman_filter(self.M_hat[0], self.x0_mean, self.x_cov[0], y[0]) #initial state estimation
=======

        x[0] = self.x0_init
        y[0] = self.get_obs(x[0], self.true_v_init) #initial observation
        x_mean[0], x_cov[0] = self.kalman_filter(self.x0_mean, self.x0_cov, y[0]) #initial state estimation
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e

        for t in range(self.T):
            #disturbance sampling
            mu_wc[t] = self.H[t] @ x_mean[t] + self.h[t] #worst-case mean
<<<<<<< HEAD

            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
                true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise

=======
            sigma_wc[t], _, status = self.solve_sdp(self.sdp_prob, x_cov[t], self.P[t+1], self.S[t+1], self.Sigma_hat[t])
            if status in ["infeasible", "unbounded"]:
                print(status, 'False!!!!!!!!!!!!!')
                
            if np.min(self.C @ (self.A @ x_cov[t] @ self.A + sigma_wc[t]) @ self.C.T + self.M) < 0:
                print('False!!!!!!!!!!!!!')
                break
            #print('old:', self.g[t], 'new:', sigma_wc[t])
            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e
            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
                true_v = self.uniform(self.v_max, self.v_min) #observation noise


            #Apply the control input to the system
            u[t] = self.K[t] @ x_mean[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the worst-case mean and covariance)
<<<<<<< HEAD
            x_mean[t+1] = self.kalman_filter(self.M_hat[t], x_mean[t], self.x_cov[t], y[t+1], mu_wc[t], u=u[t])
=======
            x_mean[t+1], x_cov[t+1] = self.kalman_filter(x_mean[t], x_cov[t], y[t+1], mu_wc[t], sigma_wc[t], u=u[t])

>>>>>>> 0f51a8e36db57fc7fda9eadc72231480dedaac1e

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


