#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
from function_utils import *
import cvxpy as cp
import scipy
import control

# Controller from Taşkesen, Bahar, et al. "Distributionally Robust Linear Quadratic Control." arXiv preprint arXiv:2305.17037 (2023).
# https://github.com/RAO-EPFL/DR-Control 
class DRLQC:
    def __init__(self, lambda_, theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, app_lambda):
        self.dist = dist
        self.noise_dist = noise_dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data
        self.v_mean_hat = v_mean_hat
        self.M_hat = M_hat
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        m = self.nu
        n = self.nx
        p = self.ny
        self.m = m
        self.n = n
        self.p = p
        self.x0_mean = x0_mean
        self.x0_cov = x0_cov
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat
        self.mu_w = mu_w
        self.mu_v = mu_v
        self.Sigma_w = Sigma_w
        self.rho = theta
        if self.dist=="uniform" or self.dist=="quadratic":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
            
        if self.noise_dist =="uniform" or self.noise_dist =="quadratic":
            self.v_max = v_max
            self.v_min = v_min
            
        #---system----
        if self.dist=="normal":
            self.x0_init = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            self.x0_init = self.uniform(self.x0_max, self.x0_min)
        elif self.dist=="quadratic":
            self.x0_init = self.quadratic(self.x0_max, self.x0_min)
        #---noise----
        if self.noise_dist=="normal":
            self.true_v_init = self.normal(self.mu_v, self.M) #observation noise
        elif self.noise_dist=="uniform":
            self.true_v_init = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.noise_dist=="quadratic":
            self.true_v_init = self.quadratic(self.v_max, self.v_min) #observation noise
            
        
        self.lambda_ = lambda_
        
        print("DRLQC ", self.dist, " / ", self.noise_dist, " / theta_w : ", self.theta_w)

        #### Creating Block Matrices for SDP ####
        self.R_block = np.zeros([T, T, m, m])
        self.C_block = np.zeros([T, T + 1, p, n])
        for t in range(T):
            self.R_block[t, t] = self.R[:, :, t]
            self.C_block[t, t] = self.C[:, :, t]
        self.Q_block = np.zeros([n * (T + 1), n * (T + 1)])
        for t in range(T + 1):
            self.Q_block[t * n : t * n + n, t * n : t * n + n] = self.Q[:, :, t]

        self.R_block = np.reshape(self.R_block.transpose(0, 2, 1, 3), (m * T, m * T))
        # Q_block = np.reshape(Q_block.transpose(0, 2, 1, 3), (n * (T + 1), n * (T + 1)))
        self.C_block = np.reshape(self.C_block.transpose(0, 2, 1, 3), (p * T, n * (T + 1)))

        # initialize H and G as zero matrices
        self.G = np.zeros((n * (T + 1), n * (T + 1)))
        self.H = np.zeros((n * (T + 1), m * T))
        for t in range(T + 1):
            for s in range(t + 1):
                # breakpoint()
                # print(GG[t * n : t * n + n, s * n : s * n + n])
                self.G[t * n : t * n + n, s * n : s * n + n] = cumulative_product(self.A, s, t)
                if t != s:
                    self.H[t * n : t * n + n, s * m : s * m + m] = (
                        cumulative_product(self.A, s + 1, t) @ self.B[:, :, s]
                    )
        self.D = np.matmul(self.C_block, self.G)
        self.inv_cons = np.linalg.inv(self.R_block + self.H.T @ self.Q_block @ self.H)
        

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
    
    def solve_sdp(self):
        ### OPTIMIZATION MODEL ###
        E = cp.Variable((self.m * self.T, self.m * self.T), symmetric=True)
        E_x0 = cp.Variable((self.n, self.n), symmetric=True)
        W_var = cp.Variable((self.n * (self.T + 1), self.n * (self.T + 1)))
        V_var = cp.Variable((self.p * self.T, self.p * self.T))
        E_w = []
        E_v = []
        W_var_sep = []  # cp.Variable((n*(T+1),n*(T+1)), symmetric=True)
        V_var_sep = []  # cp.Variable((p*T, p*T), symmetric=True)
        for t in range(self.T):
            E_w.append(cp.Variable((self.n, self.n), symmetric=True))
            E_v.append(cp.Variable((self.p, self.p), symmetric=True))
            W_var_sep.append(cp.Variable((self.n, self.n), symmetric=True))
            V_var_sep.append(cp.Variable((self.p, self.p), symmetric=True))
        W_var_sep.append(cp.Variable((self.n, self.n), symmetric=True))
        M_var = cp.Variable((self.m * self.T, self.p * self.T))
        M_var_sep = []
        num_lower_tri = num_lower_triangular_elements(self.T, self.T)
        for k in range(num_lower_tri):
            M_var_sep.append(cp.Variable((self.m, self.p)))
        k = 0
        cons = []
        for t in range(self.T):
            for s in range(t + 1):
                cons.append(M_var[t * self.m : t * self.m + self.m, self.p * s : self.p * s + self.p] == M_var_sep[k])
                cons.append(M_var_sep[k] == np.zeros((self.m, self.p)))
                k = k + 1

        for t in range(self.T + 1):
            cons.append(W_var[self.n * t : self.n * t + self.n, self.n * t : self.n * t + self.n] == W_var_sep[t])
            cons.append(W_var_sep[t] >> 0)

        # Setting the rest of the elements of the matrix to zero
        for i in range(W_var.shape[0]):
            for j in range(W_var.shape[1]):
                # If the element is not in one of the blocks
                if not any(
                    self.n * t <= i < self.n * (t + 1) and self.n * t <= j < self.n * (t + 1)
                    for t in range(self.T + 1)
                ):
                    cons.append(W_var[i, j] == 0)

        for t in range(self.T):
            cons.append(V_var[self.p * t : self.p * t + self.p, self.p * t : self.p * t + self.p] == V_var_sep[t])
            cons.append(V_var_sep[t] >> 0)
            cons.append(E_v[t] >> 0)
            cons.append(E_w[t] >> 0)
        # Setting the rest of the elements of the matrix to zero
        for i in range(V_var.shape[0]):
            for j in range(V_var.shape[1]):
                # If the element is not in one of the blocks
                if not any(
                    self.p * t <= i < self.p * (t + 1) and self.p * t <= j < self.p * (t + 1)
                    for t in range(self.T + 1)
                ):
                    cons.append(V_var[i, j] == 0)

        cons.append(E >> 0)
        cons.append(E_x0 >> 0)

        cons.append(cp.trace(W_var_sep[0] + self.x0_cov - 2 * E_x0) <= self.rho**2)
        cons.append(W_var_sep[0] >> np.min(np.linalg.eigvals(self.x0_cov)) * np.eye(self.n))
        for t in range(T):
            cons.append(
                cp.trace(W_var_sep[t + 1] + W_hat[:, :, t] - 2 * E_w[t]) <= rho**2
            )
            cons.append(cp.trace(V_var_sep[t] + V_hat[:, :, t] - 2 * E_v[t]) <= rho**2)
            cons.append(
                W_var_sep[t + 1] >> np.min(np.linalg.eigvals(W_hat[:, :, t])) * np.eye(n)
            )
            cons.append(
                V_var_sep[t] >> np.min(np.linalg.eigvals(V_hat[:, :, t])) * np.eye(p)
            )
        X0_hat_sqrt = sqrtm(X0_hat)
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(X0_hat_sqrt, W_var_sep[0]), X0_hat_sqrt), E_x0],
                    [E_x0, np.eye(n)],
                ]
            )
            >> 0
        )
        for t in range(T):
            temp = sqrtm(W_hat[:, :, t])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, W_var_sep[t + 1]), temp), E_w[t]],
                        [E_w[t], np.eye(n)],
                    ]
                )
                >> 0
            )
            temp = sqrtm(V_hat[:, :, t])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, V_var_sep[t]), temp), E_v[t]],
                        [E_v[t], np.eye(p)],
                    ]
                )
                >> 0
            )

        cons.append(
            cp.bmat(
                [
                    [
                        E,
                        cp.matmul(
                            cp.matmul(cp.matmul(cp.matmul(H.T, Q_block), G), W_var), D.T
                        )
                        + M_var / 2,
                    ],
                    [
                        (
                            cp.matmul(
                                cp.matmul(cp.matmul(cp.matmul(H.T, Q_block), G), W_var),
                                D.T,
                            )
                            + M_var / 2
                        ).T,
                        cp.matmul(cp.matmul(D, W_var), D.T) + V_var,
                    ],
                ]
            )
            >> 0
        )
        obj = -cp.trace(cp.matmul(E, inv_cons)) + cp.trace(
            cp.matmul(cp.matmul(cp.matmul(G.T, Q_block), G), W_var)
        )

        prob = cp.Problem(cp.Maximize(obj), cons)
        # breakpoint()
        prob.solve(
            solver="MOSEK",
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol},
            verbose=False,
        )
        # breakpoint()
        E_check = (
            (H.T @ Q_block @ G @ W_var.value @ D.T + M_var.value / 2)
            @ np.linalg.inv(D @ W_var.value @ D.T + V_var.value)
            @ (M_var.value / 2 + H.T @ Q_block @ G @ W_var.value @ D.T).T
        )
        M = M_var.value
        M[np.abs(M) <= 1e-11] = 0
        W_var_clean = W_var.value
        V_var_clean = V_var.value
        W_var_clean[W_var_clean <= 1e-11] = 0
        V_var_clean[V_var_clean <= 1e-11] = 0

        E_new = (
            (H.T @ Q_block @ G @ W_var_clean @ D.T + M / 2)
            @ np.linalg.inv(D @ W_var_clean @ D.T + V_var_clean)
            @ (M / 2 + H.T @ Q_block @ G @ W_var_clean @ D.T).T
        )
        obj_clean = -np.trace(np.matmul(E_new, inv_cons)) + np.trace(
            np.matmul(np.matmul(np.matmul(G.T, Q_block), G), W_var_clean)
        )

        # breakpoint()

        return obj.value, obj_clean, W_var.value, V_var.value


    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

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
        x_mean[0] = self.kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean, self.x_cov[0], y[0]) #initial state estimation

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
            x_mean[t+1] = self.kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], self.x_cov[t+1], y[t+1], mu_wc[t], u=u[t])

        #Compute the total cost
        J[self.T] = x[self.T].T @ self.Qf @ x[self.T]
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


