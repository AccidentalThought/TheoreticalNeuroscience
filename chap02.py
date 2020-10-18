"""
Script for exercises from Chapter 2 of Theoretical Neuroscience book
"""

import numpy as np
import matplotlib.pyplot as plt
import neuro_tools as nt
from scipy import signal


def exercise1(back_rate = 50,  # Hz
              time_step = 0.01,  # seconds
              sigma = 10**0.5,  # variability of white noise
              duration = 10,  # seconds
             ):
    # Kernel, Hz/ms
    ker_func = lambda t: -np.cos(2*np.pi*(t-0.02)/0.14)*np.exp(-t/0.06)
    times = np.arange(0, 0.6, time_step)
    kernel = ker_func(times)
    # Illustration of the decay of the kernel
    plt.plot(times, ker_func(times))
    plt.title("Linear kernel function")
    plt.xlabel("time [s]")
    plt.show()

    # Gaussian white noise
    wnstim = nt.gaussian_white_noise(sigma, duration, time_step)

    # Firing rate
    rate = back_rate + nt.cyclic_conv(wnstim, kernel)*time_step

    # rs correlation
    rs_corr = nt.cyclic_corr(rate, wnstim)/rate.size
    plt.plot(np.linspace(-duration//2, duration//2, rs_corr.size), rs_corr)
    plt.title("Stimulus-response correlation")
    plt.xlabel("time [s]")
    plt.show()

    # compare correlation with the kernel - equation 2.6
    t_zero = rs_corr.size//2  # index of the time = 0
    plt.plot(times, kernel, label=r"Kernel $D(\tau)$")
    relevant_part = rs_corr[t_zero: t_zero-times.size:-1]/sigma**2
    plt.plot(times, relevant_part,
            label=r"rs correlation $Q_{rs}(-\tau)/\sigma^2$")
    plt.legend()
    plt.xlabel("time [s]")
    plt.show()

