# imports
import numpy as np
import matplotlib.pyplot as plt
import kalman_smoother as ks


r = 3     # possible values of variance for noise in forward model
ny = 1    # dimension of measurement quantity
Nt = 250  # total observation time steps
dt = 0.02  # time step size
t_axis = np.arange(Nt+1)*dt           # time axis
x0_true = np.array([0, 0, 5*1.414/13.5])   # true value for initial state variables
mu0 = np.zeros(3)  # mean for q_k-1
P0_diag = np.array([3, 3, 3])
P0 = np.diag(P0_diag)            # Covaraince matrix for initial state variables
truth_data = np.zeros((mu0.shape[0], Nt + 1))
truth_data[:, 0] = x0_true

# Generate some observation data points
ks_ins = ks.ut_kalman_smoother(Nt, dt, r, x0_true, P0, ny)
for i in range(Nt):
    mu_fk = 10*np.sin(10*(i+1)*dt+0.25*np.pi)
    std_fk = 0.2*np.abs(np.sin(10*(i+1)*dt+0.25*np.pi))
    fk = np.random.normal(loc=mu_fk, scale=std_fk)
    temp_input = np.concatenate((truth_data[:, i], np.array([fk])), axis=0)
    truth_data[:, i + 1] = ks_ins.forward_model(temp_input)

# Predict the state variables using uncented Kalman filtering and plot the results
(measure_ydata, kalman_update, mu_Nt, P_Nt) = ks_ins.ut_kalman_filter(truth_data)
ks_ins.filtering_plotting(t_axis, kalman_update, truth_data)

# Predict the state variables using uncented Kalman smoothing and plot the results
(kalman_smooth_update, mu_Nt_s, P_Nt_s) = ks_ins.ut_kalman_smoother(mu_Nt, P_Nt)
ks_ins.smoothing_plotting(t_axis, kalman_update, kalman_smooth_update, truth_data)
plt.show()