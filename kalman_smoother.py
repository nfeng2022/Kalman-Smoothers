# imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

class ut_kalman_smoother:
    # construct a unscented kalman smoother class for furture use
    def __init__(self, Nt, dt, r, x0_true, P0, ny):
        self.Nt = Nt          # Number of observed data points
        self.r = r             # Variance of noise in the measurement model
        self.x0_true = x0_true  # Initial values of state variables
        self.mu0 = x0_true  # Predicted mean value for x0
        self.P0 = P0       # Predicted covariance matrix for x0
        self.dt = dt       # time step size
        self.ny = ny       # dimension of measure quantity

    def ut_cubature_rule(self, n, m, P, alpha=0.5, beta=2, k0=3):
        '''
            :param alpha: from (0,1]
            :param beta: optimal 2 for gaussian distribution
            :param k0: parameter for controlling kappa
            :param n: the input dimension
            :param m: mean of the inputs
            :param P: covariance of the inputs
            :return:
            sig_pt: integration points
            wm: weights for mean
            wc: weights for covariance
            '''
        kappa = k0 - n
        lam = alpha ** 2 * (n + kappa) - n
        sig_pt = [m]
        wm = [lam / (n + lam)]
        wc = [lam / (n + lam) + 1 - alpha ** 2 + beta]
        L = np.linalg.cholesky(P)
        for i in range(n):
            sig1 = m + np.sqrt(n + lam) * L[:, i]
            w1 = 1 / (2 * (n + lam))
            wm.append(w1)
            wc.append(w1)
            sig_pt.append(sig1)
            sig2 = m - np.sqrt(n + lam) * L[:, i]
            w2 = 1 / (2 * (n + lam))
            wm.append(w2)
            wc.append(w2)
            sig_pt.append(sig2)
        return (sig_pt, wm, wc)

    def forward_model(self, x):
        # State space form for the single-degree-of-freedom dynamics system
        dt = self.dt
        m = 13.5
        c = 6.8
        E = 2131.8
        g = (x[3] - E * x[0] - (c + E * dt) * x[1] - (0.5 * c * dt + 0.25 * E * dt ** 2) * x[2]) \
            / (m + 0.5 * dt * c + 0.25 * E * dt ** 2)
        y = np.zeros(shape=3)
        y[0] = x[0] + dt * x[1] + 0.25 * dt ** 2 * (x[2] + g)
        y[1] = x[1] + 0.5 * dt * (x[2] + g)
        y[2] = g
        return y

    def measure_model(self, x):
        return x[2]

    def ut_mu_cov(self, mu0, P0, y_fun, ny):
        # For computing the mean and covariance matrix for measurement variable y0
        n = mu0.shape[0]
        mu0_vec = np.reshape(mu0, (mu0.shape[0], 1))
        (sig_pt, wm, wc) = self.ut_cubature_rule(n, mu0[:, 0], P0, alpha=0.5, beta=2, k0=3)
        # compute mean from ut
        mu_ut = np.zeros(ny)
        for i in range(2 * n + 1):
            yi = y_fun(sig_pt[i])
            mu_ut = mu_ut + wm[i] * yi
        mu_ut_vec = np.reshape(mu_ut, (mu_ut.shape[0], 1))
        # compute covariance from ut
        cov1_ut = np.zeros((ny, ny))
        cov2_ut = np.zeros((mu0.shape[0], ny))
        for i in range(2 * n + 1):
            yi = y_fun(sig_pt[i])
            sig_pt_vec = np.reshape(sig_pt[i], (sig_pt[i].shape[0], 1))
            if ny != 1:
                yi_vec = np.reshape(yi, (yi.shape[0], 1))
                cov1_ut = cov1_ut + wc[i] * ((yi_vec - mu_ut_vec) @ (yi_vec - mu_ut_vec).T)
                cov2_ut = cov2_ut + wc[i] * ((sig_pt_vec - mu0_vec) @ (yi_vec - mu_ut_vec).T)
            else:
                cov1_ut = cov1_ut + wc[i] * ((yi - mu_ut) * (yi - mu_ut))
                cov2_ut = cov2_ut + wc[i] * ((sig_pt_vec - mu0_vec) * (yi - mu_ut))
        return (mu_ut_vec, cov1_ut, cov2_ut)

    def ut_kalman_filter(self, truth_data):
        Nt = self.Nt  # total observation time steps
        dt = self.dt
        r = self.r
        x0_true = self.x0_true
        mu0 = self.mu0  # mean for x0
        P0 = self.P0
        measure_ydata = np.zeros(Nt + 1)
        y0_measure = x0_true[2] + np.random.normal(loc=0, scale=r)
        measure_ydata[0] = y0_measure
        for i in range(Nt):
            measure_ydata[i + 1] = truth_data[2, i + 1] + np.random.normal(loc=0, scale=r)
        # perform kalman filtering
        ny = self.ny
        mu0 = np.reshape(mu0, (mu0.shape[0], 1))
        (mu_ut, cov1_ut, cov2_ut) = self.ut_mu_cov(mu0, P0, self.measure_model, ny)
        cov1_ut = cov1_ut + r
        m_k0 = mu0 + cov2_ut / cov1_ut * (measure_ydata[0] - mu_ut)  # initialize m_k-1
        cov2_ut_vec = np.reshape(cov2_ut, (cov2_ut.shape[0], 1))
        P_k0 = P0 - (cov2_ut_vec @ cov2_ut_vec.T) / cov1_ut  # initialize P_k-1
        mu_Nt = [m_k0[:, 0]]  # store posterior mean history of state variables
        P_Nt = [P_k0]  # store posterior variance history
        for i in range(Nt):
            # prediction step
            m_k01 = np.concatenate((m_k0, np.array([[10 * np.sin(10 * (i + 1) * dt + 0.25 * np.pi)]])), axis=0)
            P_k01 = np.concatenate((P_k0, np.zeros((1, P_k0.shape[1]))), axis=0)
            P_k01 = np.concatenate((P_k01, np.zeros((P_k01.shape[0], 1))), axis=1)
            P_k01[-1, -1] = 0.04 * np.sin(10 * (i + 1) * dt + 0.25 * np.pi) ** 2
            (mu_ut, cov1_ut, cov2_ut) = self.ut_mu_cov(m_k01, P_k01, self.forward_model, m_k0.shape[0])
            m_km = mu_ut
            P_km = cov1_ut
            # update step
            (mu_ut, cov1_ut, cov2_ut) = self.ut_mu_cov(m_km, P_km, self.measure_model, ny)
            cov1_ut = cov1_ut + r
            kalman_gain_mat = cov2_ut / cov1_ut
            m_k1 = m_km + kalman_gain_mat * (measure_ydata[i + 1] - mu_ut)
            P_k1 = P_km - (kalman_gain_mat @ cov1_ut) @ kalman_gain_mat.T
            mu_Nt.append(m_k1[:, 0])
            P_Nt.append(P_k1)
            m_k0 = m_k1
            P_k0 = P_k1
        kalman_update = np.array(mu_Nt).T
        return (measure_ydata, kalman_update, mu_Nt, P_Nt)

    def ut_kalman_smoother(self, mu_Nt, P_Nt):
        Nt = self.Nt  # total observation time steps
        dt = self.dt
        # perform kalman smoothing
        mu_Nt_s = [mu_Nt[-1]]  # store smoothed posterior mean history of state variables
        P_Nt_s = [P_Nt[-1]]  # store smoothed posterior variance history
        for i in range(Nt, 0, -1):
            # prediction step
            mu0 = np.reshape(mu_Nt[i - 1], (mu_Nt[i - 1].shape[0], 1))
            m_k01 = np.concatenate((mu0, np.array([[10 * np.sin(10 * i * dt + 0.25 * np.pi)]])), axis=0)
            P_k01 = np.concatenate((P_Nt[i - 1], np.zeros((1, P_Nt[i - 1].shape[1]))), axis=0)
            P_k01 = np.concatenate((P_k01, np.zeros((P_k01.shape[0], 1))), axis=1)
            P_k01[-1, -1] = 0.04 * np.sin(10 * i * dt + 0.25 * np.pi) ** 2
            (mu_ut, cov1_ut, cov2_ut) = self.ut_mu_cov(m_k01, P_k01, self.forward_model, mu_Nt[i - 1].shape[0])
            m_km = mu_ut[:, 0]
            P_km = cov1_ut
            # update step
            C_k = cov2_ut[0:-1, :]
            G_k = C_k @ np.linalg.inv(P_km)
            m_ks = mu_Nt[i - 1] + G_k @ (mu_Nt_s[-1] - m_km)
            P_ks = P_Nt[i - 1] + G_k @ (P_Nt_s[-1] - P_km) @ G_k.T
            mu_Nt_s.append(m_ks)
            P_Nt_s.append(P_ks)
        mu_Nt_s.reverse()
        P_Nt_s.reverse()
        kalman_smooth_update = np.array(mu_Nt_s).T
        return (kalman_smooth_update, mu_Nt_s, P_Nt_s)

    def filtering_plotting(self, t_axis, kalman_update, truth_data):
        plt.figure()
        plt.plot(t_axis, truth_data[0, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_update[0, :], linewidth=2, label='Filtered')
        plt.grid(True)
        plt.legend()
        plt.title('$u_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[1, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_update[1, :], linewidth=2, label='Filtered')
        plt.grid(True)
        plt.legend()
        plt.title('$v_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[2, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_update[2, :], linewidth=2, label='Filtered')
        plt.grid(True)
        plt.legend()
        plt.title('$a_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

    def smoothing_plotting(self, t_axis, kalman_update, kalman_smooth_update, truth_data):
        plt.figure()
        plt.plot(t_axis, truth_data[0, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_smooth_update[0, :], linewidth=2, label='Smoothed')
        plt.grid(True)
        plt.legend()
        plt.title('$u_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[1, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_smooth_update[1, :], linewidth=2, label='Smoothed')
        plt.grid(True)
        plt.legend()
        plt.title('$v_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[2, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_smooth_update[2, :], linewidth=2, label='Smoothed')
        plt.grid(True)
        plt.legend()
        plt.title('$a_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

