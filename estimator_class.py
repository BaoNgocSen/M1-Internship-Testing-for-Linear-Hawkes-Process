import numpy as np
import math
from scipy.optimize import minimize
from scipy.stats import norm
import numdifftools as nd  # pip install numdifftools

def estimate_hawkes_params_(timestamps, T):
    def neg_log_likelihood(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf

        I, t_prev, ll_sum = 0.0, 0.0, 0.0
        for ti in timestamps:
            I *= math.exp(-beta * (ti - t_prev))
            lam = mu + I
            if lam <= 0:
                return np.inf
            ll_sum += math.log(lam)
            I += alpha
            t_prev = ti

        arr = np.array(timestamps)
        compensator = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - arr)))
        return -(ll_sum - compensator)

    res = minimize(
        fun=neg_log_likelihood,
        x0=[0.1, 0.1, 0.1],
        bounds=[(1e-5, None)] * 3,
        method="L-BFGS-B"
    )
    mu_hat, alpha_hat, beta_hat = res.x


    hessian = nd.Hessian(neg_log_likelihood)([mu_hat, alpha_hat, beta_hat])
    fisher_info = hessian / T
    cov_matrix = np.linalg.inv(fisher_info)

    A = 1 - alpha_hat / beta_hat
    grad = np.array([
        1 / A,
        mu_hat / (beta_hat * A**2),
        -mu_hat * alpha_hat / (beta_hat**2 * A**2)
    ])

    variance = np.abs(grad @ cov_matrix @ grad)

    return -res.fun, mu_hat, alpha_hat, beta_hat, variance


def hawkes_test_(N_T, T, mu, alpha, beta, alpha_level=0.05):
   
    eta = alpha / beta
    z_q =  norm.ppf(1 - alpha_level / 2)

    estimate_rate = mu / (1 - eta)
    empirical_rate = N_T / T
    stat = np.sqrt(T) * np.abs(empirical_rate - estimate_rate)/np.sqrt(mu / (1 - alpha/beta)**3)

    passed = stat - z_q
    if passed < 0:
        return 0 # print("On peut pas rejetter H0")
    else:
        return 1 #print("On peut rejette H0")

def hawkes_test2_(T,Realite, Estimateur,variance, alpha_level=0.05):

    z_q =  norm.ppf(1 - alpha_level / 2)

    stat = np.sqrt(T) * np.abs( np.array(Estimateur)- Realite)/np.sqrt(variance)

    passed = stat - z_q
    if passed < 0:
        return 0 # print("On peut pas rejetter H0")
    else:
        return 1 #print("On peut rejette H0")

def hawkes_test3_(N_T, T, mu, alpha, beta, variance, alpha_level=0.05):
   
    eta = alpha / beta
    z_q =  norm.ppf(1 - alpha_level / 2)

    estimate_rate = mu / (1 - eta)
    empirical_rate = N_T / T
    stat = np.sqrt(T) * np.abs(empirical_rate - estimate_rate)/np.sqrt(mu / (1 - alpha/beta)**3+variance)

    passed = stat - z_q
    if passed < 0:
        return 0 # print("On peut pas rejetter H0")
    else:
        return 1 #print("On peut rejette H0")
