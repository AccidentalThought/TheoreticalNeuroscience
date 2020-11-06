"""
Script for exercises from Chapter 2 of Theoretical Neuroscience book
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import neuro_tools as nt
from scipy import signal


def exercise1(back_rate = 50,  # Hz
              time_step = 0.01,  # seconds
              sigma = 10**0.5,  # variability of white noise
              duration = 10,  # seconds
             ):
    # Conversion of sigma to seconds
    sigma = sigma*10**(-3/2)
    # Kernel, Hz/ms
    ker_func = lambda t: -1000*np.cos(2*np.pi*(t-0.02)/0.14)*np.exp(-t/0.06)
    times = np.arange(0, 0.6, time_step)
    kernel = ker_func(times)
    # Illustration of the decay of the kernel
    plt.plot(times, ker_func(times))
    plt.title("Linear kernel function")
    plt.ylabel(r"$D(\tau) [Hz/s]$")
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


def exercise2(duration = 20*60,  # seconds
             sampling = 500,  # Hz
             time_step = 0.002,  # seconds
             max_time = 0.3,  # seconds
             ):
    response, stimulus = nt.load_c1p8()  # Loading data

    stimulus = stimulus - stimulus.mean()
    # Computing spike triggered average -> Kernel
    n = response.sum()
    time_bins = int(max_time/time_step)
    sta = stimulus[np.arange(-time_bins+1, 1)
                    + response.nonzero()[0][:, np.newaxis]].sum(axis=0)/n
    avg_rate = n/duration
    sigma = stimulus.std()*np.sqrt(time_step)
    kernel = sta[::-1]*avg_rate/(sigma**2)
    # Plot Kernel to check
    plt.plot(kernel)
    plt.title("Linear kernel function")
    plt.ylabel(r"$D(\tau) [Hz/s]$")
    plt.xlabel("time [s]")
    plt.show()
    # plt.plot(nt.cyclic_conv(stimulus, kernel))
    # plt.show()

    rate = avg_rate + nt.cyclic_conv(stimulus, kernel)*time_step
    # 40% r<0 !!!!
    (rate<0).sum()
    plt.plot(sta[::-1])
    plt.show()

    plt.plot(np.arange(0, duration, time_step), rate)
    plt.title("Weird resulting firing rate")
    plt.xlabel("time [s]")
    plt.show()


def exercise3(time_bin = 15.6,  # ms
             ):
    counts, stim = nt.load_c2p3()  # load data
    n = counts.sum()  # number of spikes
    sta = np.empty((16, 16, 12))
    idx = counts.nonzero()[0]
    # stimulus trigger averages
    for i in range(12):
        sta[:,:,i] = (counts[idx]*stim[:,:,idx-i]).sum(axis=2)/n

    # plotting STAs one after another
    for i in range(1, 13):
        plt.imshow(sta[:,:,-i], extent = [-1, 1, -1, 1], 
                    vmin = sta.min(), vmax=sta.max(), cmap='jet')
        plt.title(f"time = {(13-i)*time_bin} ms")
        plt.colorbar()
        plt.show()

    # Cut solution
    plt.imshow(sta[:,sta.shape[1]//2,:], cmap='jet')
    plt.title("Cuts in y direction")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()
    plt.show()
    # Sum solution
    plt.imshow(sta.sum(axis=1), cmap='jet')
    plt.title("Values in y direction summed")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()
    plt.show()

    fig = plt.figure()
    ims = []
    for i in range(1, 13):
        im = plt.imshow(sta[:,:,-i], animated=True, cmap='jet', 
                        vmin = sta.min(), vmax=sta.max())
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat=True)
    plt.colorbar()
    plt.show()


def exercise5(lam = 1.2,  # cm, constant for the map
              eps0 = 1.0,  # degs, constant for the map
              k = 8,  # 1/cm, spatial frequency
              theta = np.pi/6,  # rads, rotation of the stim 
              phi = 0,  # rads, phase shift
              stim = None,  # None for default, otherwise a function of x,y
             ):
    """
    Core of the given MATLAB code transformed into Python
    """
    # mapping from cortex into the retina
    x = np.arange(0.05, 3.01, 0.01)
    y = np.arange(-2, 2.01, 0.01)
    xx, yy = np.meshgrid(x,y)  # global coordinates for the cortex

    eps = eps0*(np.exp(xx/lam)-1)  # eccentricity
    a = -(180*(eps0+eps)*yy)/(lam*eps*np.pi)  # azimuth
    ins = np.abs(a) < 90
    a = a*(np.abs(a) < 90) + 90*(a >= 90) - 90*(a <= -90)  # from matlab code
    # First alternative
    # a = np.where(a>=90, 90, np.where(a<=-90, -90, a))
    # Second alternative
    # a[a>90] = 90
    # a[a<-90] = -90

    # Generating stimulus
    if stim is None:  # use default stimulus
        stim = np.cos(k*xx*np.cos(theta) + k*yy*np.sin(theta) - phi)
    else:
        stim = stim(xx, yy)

    # Plots
    plt.subplot(1,2,1)
    plt.contourf(xx, yy, stim*ins)
    plt.colorbar()
    plt.title('visual cortex')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')

    plt.subplot(1,2,2)
    plt.contourf(eps*np.cos(a*2*np.pi/360), eps*np.sin(a*2*np.pi/360), stim)
    plt.title('visual space')
    plt.xlabel('degrees')
    plt.ylabel('degrees')

    plt.tight_layout()
    plt.show()
