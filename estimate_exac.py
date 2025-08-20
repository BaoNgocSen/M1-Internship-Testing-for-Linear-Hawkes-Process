import numpy as np
import math
from scipy.optimize import minimize
from scipy.stats import norm

def _observed_hessian(process, T, mu, alpha, beta):
    t_sorted = np.sort(process)
    n = len(t_sorted)
    tn = t_sorted[-1] if n > 0 else 0.0

    A = [0.0] * n
    B = [0.0] * n
    C = [0.0] * n

    for i in range(1, n):
        dt = t_sorted[i] - t_sorted[i-1]
        exp_term = np.exp(-beta * dt)

        A[i] = exp_term * A[i-1] + exp_term
        B[i] = exp_term * B[i-1] + dt * exp_term
        C[i] = exp_term * C[i-1] + (dt**2) * exp_term
    
    H_mu_mu = 0.0
    H_alpha_alpha = 0.0
    H_beta_beta = 0.0
    H_mu_alpha = 0.0
    H_mu_beta = 0.0
    H_alpha_beta = 0.0

    for i in range(n):
        
        ti = t_sorted[i]
        Ai = A[i]
        Bi = B[i]
        Ci = C[i]       
        denom_mu_alpha_Ai = (mu + alpha * Ai)
        

        H_mu_mu += 1 / (denom_mu_alpha_Ai)**2
        H_mu_alpha += Ai / (denom_mu_alpha_Ai)**2
        H_mu_beta += -alpha * Bi / (denom_mu_alpha_Ai)**2
        
        
        H_alpha_alpha += (Ai / denom_mu_alpha_Ai)**2

        term1_alpha_beta= (1/beta) * (T - ti) * np.exp(-beta * (T - ti))
        term2_alpha_beta= (1/beta**2) * (np.exp(-beta * (T - ti)) - 1)

        H_alpha_beta_intensity_part = Bi / denom_mu_alpha_Ai -(alpha * Ai * Bi) / (denom_mu_alpha_Ai)**2
        
        H_alpha_beta += H_alpha_beta_intensity_part + term1_alpha_beta+term2_alpha_beta

        term1_beta_beta = (1/beta) * (T - ti)**2 * np.exp(-beta * (T - ti))
        term2_beta_beta = (2/beta**2) * (T - ti) * np.exp(-beta * (T - ti))
        term3_beta_beta = (2/beta**3) * (np.exp(-beta * (T - ti)) - 1)
        H_beta_beta_compensator_part = -alpha * (term1_beta_beta + term2_beta_beta + term3_beta_beta)
        
        H_beta_beta_intensity_part = (alpha * Ci) / denom_mu_alpha_Ai -(alpha * Bi / denom_mu_alpha_Ai)**2
        
        H_beta_beta += H_beta_beta_compensator_part - H_beta_beta_intensity_part

    Hessian = np.zeros((3, 3))
    
    Hessian[0, 0] = H_mu_mu
    Hessian[0, 1] = H_mu_alpha
    Hessian[0, 2] = H_mu_beta

    Hessian[1, 0] = H_mu_alpha
    Hessian[1, 1] = H_alpha_alpha
    Hessian[1, 2] = H_alpha_beta

    Hessian[2, 0] = H_mu_beta
    Hessian[2, 1] = H_alpha_beta
    Hessian[2, 2] = H_beta_beta
    
    return -Hessian

def estimate_hawkes_params(process, T):

    process = sorted(process)
    n = len(process)
    timestamps = process  # list sorted
    
    def neg_log_likelihood(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf
        I = 0.0
        t_prev = 0.0
        ll_sum = 0.0
        for ti in timestamps:

            I *= math.exp(-beta * (ti - t_prev))
            lam = mu + I
            if lam <= 0:
                return np.inf
            ll_sum += math.log(lam)
            I += alpha
            t_prev = ti
        # compensator
        arr = np.array(timestamps)
        comp = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - arr)))
        return -(ll_sum - comp)

    res = minimize(
        fun=neg_log_likelihood,
        x0=[0.1, 0.1, 0.1],
        bounds=[(1e-5, None), (1e-5, None), (1e-5, None)],
        method="L-BFGS-B")

    mu_hat, alpha_hat, beta_hat = res.x
    
    fisher_information_matrix =  1/T*_observed_hessian(process, T,mu_hat, alpha_hat, beta_hat)
    covariance_matrix = np.linalg.inv(fisher_information_matrix)

    A = (1 - alpha_hat / beta_hat)
            
    grad_mu = 1 / A
    grad_alpha = mu_hat / (beta_hat * (A)**2)
    grad_beta = - (mu_hat * alpha_hat) / (beta_hat**2 * (A)**2)

    gradient_vector = np.array([grad_mu, grad_alpha, grad_beta])

    long_term_rate_variance = np.abs(gradient_vector.T @ covariance_matrix @ gradient_vector)
    # -res.fun là log-likelihood tại ước lượng
    return -res.fun, mu_hat, alpha_hat, beta_hat, long_term_rate_variance


def hawkes_test(N_T, T, mu, alpha, beta, alpha_level=0.05):
   
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

def hawkes_test2(T,Realite, Estimateur,variance, alpha_level=0.05):

    z_q =  norm.ppf(1 - alpha_level / 2)

    stat = np.sqrt(T) * np.abs( np.array(Estimateur)- Realite)/np.sqrt(variance)

    passed = stat - z_q
    if passed < 0:
        return 0 # print("On peut pas rejetter H0")
    else:
        return 1 #print("On peut rejette H0")

def hawkes_test3(N_T, T, mu, alpha, beta, variance, alpha_level=0.05):
   
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