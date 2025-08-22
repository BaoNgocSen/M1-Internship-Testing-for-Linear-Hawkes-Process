import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

import numpy as np
import math

class thinning_hawkes(object):
    """
    Univariate Hawkes process with exponential, gaussian, or box kernel.
    No events or initial condition before initial time.
    """

    def __init__(self, mu, alpha, beta, T, last_jumps=None, Process=None, kernel_type='exponential'):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.lambda_max = mu
        self.simulated = False
        self.kernel_type = kernel_type
        self.Process = []

    def simulate(self):
        if self.simulated:
            print("We've already simulated this object")
            return self.Process

        self.Process = []
        self.simulated = True
        current_t = 0.0
        I_decayed_sum = 0.0

        while current_t < self.T:
            if self.kernel_type == 'exponential':
                upper_intensity = self.mu + I_decayed_sum
                u1 = np.random.uniform(0, 1)
                dt = -math.log(u1) / upper_intensity
                t_candidate = current_t + dt
                if t_candidate >= self.T:
                    break

                I_decayed_sum *= math.exp(-self.beta * (t_candidate - current_t))
                lam_at_t_candidate = self.mu + I_decayed_sum
                u2 = np.random.uniform(0, 1)

                if u2 <= lam_at_t_candidate / upper_intensity:
                    self.Process.append(t_candidate)
                    I_decayed_sum += self.alpha
                    current_t = t_candidate
                else:
                    current_t = t_candidate

            if self.kernel_type == 'gaussien':
                lam_current = self.mu + sum(
                    self.alpha * np.exp(-((current_t - tk) ** 2) / (2 * self.beta ** 2))
                    for tk in self.Process
                )
                
            
                upper_intensity = lam_current + self.alpha
                
                if upper_intensity <= 0:
                    current_t = self.T
                    break

                u = np.random.uniform(0, 1)
                w = -np.log(u) / upper_intensity
                t_candidate = current_t + w
                if t_candidate >= self.T:
                    break

                lam_candidate = self.mu + sum(
                    self.alpha * np.exp(-((t_candidate - tk) ** 2) / (2 * self.beta ** 2))
                    for tk in self.Process
                )

                D = np.random.uniform(0, 1)
                if D <= lam_candidate / upper_intensity:
                    self.Process.append(t_candidate)
                    current_t = t_candidate
                else:
                    current_t = t_candidate
                    
            if self.kernel_type == 'box':
                # Thêm kernel hình hộp
                num_active_events = sum(1 for tk in self.Process if (current_t - tk) < self.beta)
                upper_intensity = self.mu + num_active_events * self.alpha
                
                if upper_intensity <= 0:
                    current_t = self.T
                    break
                
                w = -np.log(np.random.uniform(0, 1)) / upper_intensity
                t_candidate = current_t + w
                
                if t_candidate >= self.T:
                    break
                

                self.Process.append(t_candidate)
                current_t = t_candidate
        
        return self.Process

    def intensity_at_t(self, t):
        if not self.simulated:
            raise RuntimeError("Simulate first before calling intensity_at_t()")

        A = self.mu
        if self.kernel_type == 'exponential':
            for k in self.Process:
                if k < t:
                    A += self.alpha * np.exp(-self.beta * (t - k))
        elif self.kernel_type == 'gaussien':
            for k in self.Process:
                if k < t:
                    A += self.alpha * np.exp(-((t - k) ** 2) / (2 * self.beta ** 2))
        
        elif self.kernel_type == 'box':
            # λ(t) = μ + α × #{k : 0 < t - k < β}
            A += self.alpha * sum(1 for k in self.Process if 0 < t - k < self.beta)

        else:
            raise ValueError(f"Unknown kernel_type={self.kernel_type}")

        return A

    def Compensator_exp_at_t(self, t):
        if not self.simulated:
            raise RuntimeError("Simulate first before calling Compensator.")

        compensator = self.mu * t
        for tk in self.Process:
            if tk < t:
                compensator += (self.alpha / self.beta) * (1 - np.exp(-self.beta * (t - tk)))
        return compensator

    def plot_intensity(self, steps, t):
        if t > self.T:
            print("t is greater than T")
            return
        if not self.simulated:
            print("Simulate first")
            return

        x = np.linspace(0, t, steps)
        y = [self.intensity_at_t(i) for i in x]

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, lw=1.5)
        plt.xlabel("Time")
        plt.ylabel("Intensity λ(t)")
        plt.title(f"Hawkes intensity — kernel={self.kernel_type}, T={self.T}, "f"muy={self.mu}, alpha={self.alpha}, beta={self.beta}") 
        plt.tight_layout()
        plt.show()

    def log_likelihood(self):
        if not self.simulated:
            raise RuntimeError("Simulate first before calling log_likelihood.")

        ll = 0.0
        I = 0.0
        t_prev = 0.0

        for ti in self.Process:
            I *= math.exp(-self.beta * (ti - t_prev))
            lam = self.mu + I
            if lam <= 0:
                return -np.inf
            ll += np.log(lam)
            I += self.alpha
            t_prev = ti

        ll -= self.Compensator_exp_at_t(self.T)
        return ll
