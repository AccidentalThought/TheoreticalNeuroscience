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
              plot_animation = True,  # Flag for plot
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
    
    if plot_animation:
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


def exercise6():
    # Implementation of a spatial linear estimate
    def spatial_linest(spatial_phase, orientation, spatial_freq_ratio,
                    ksigma=2, pref_phase=0, amplitude=1):
        expfac = np.exp(-ksigma**2/2*(1+spatial_freq_ratio**2))
        val = ksigma**2*spatial_freq_ratio*np.cos(orientation)
        first_cos = np.cos(pref_phase-spatial_phase)*np.exp(val)
        second_cos = np.cos(pref_phase+spatial_phase)*np.exp(-val)
        return amplitude/2*expfac*(first_cos+second_cos)

    # Implementations of temporal linear estimate

    # Given result in exercises
    def given_res(alpha, omega, t):
        lam = omega/alpha
        frac = lam*np.sqrt(4+lam**2)/((1+lam**2)**4)
        delta = 8*np.arctan(lam) + np.arctan(2/lam) - np.pi
        return frac*np.cos(omega*t-delta)

    # Implementation of purely numerical solution
    def numerical_solution(alpha, omega, t, step = 0.01, max_tau=100):
        lam = omega/alpha
        taus = np. arange(0, max_tau*alpha, step)
        if type(t) == np.ndarray:
            taus = taus[np.newaxis,]
            t = t[:,np.newaxis]
        vals = np.cos(omega*t-lam*taus)*np.exp(-taus)*((taus**5)/factorial(5)-(taus**7)/factorial(7))
        return (vals.sum(axis=1)- (vals[:,0] + vals[:,-1])/2)*step

    # My slightly different result, gives the same numbers nonetheless
    def my_res(alpha, omega, t):
        lam = omega/alpha
        arc = np.arctan(lam)
        t = omega*t
        return (np.cos(t-6*arc) - np.cos(t-8*arc)/(1+lam**2))/((1+lam**2)**(3))

    # Replicating Figure 2.15
    # Figure A
    plt.subplot(131)
    orientations = np.linspace(-np.pi/2, np.pi/2)
    spl_a = spatial_linest(spatial_phase=0, orientation=orientations, spatial_freq_ratio=1)
    plt.plot(orientations, spl_a)
    plt.xlabel(r"$\Theta$")
    # B
    plt.subplot(132)
    spfreq_ratios = np.linspace(0, 3)
    spl_b = spatial_linest(spatial_phase=0, orientation=0, spatial_freq_ratio=spfreq_ratios)
    plt.plot(spfreq_ratios, spl_b)
    plt.xlabel(r"$K/k$")
    # C
    plt.subplot(133)
    sp_phases = np.linspace(-np.pi, np.pi)
    spl_c = spatial_linest(spatial_phase=sp_phases, orientation=0, spatial_freq_ratio=1)
    plt.plot(sp_phases, spl_c)
    plt.xlabel(r"$\Phi$")
    plt.show()


    # Figure 2.16
    freqs = np.linspace(0, 20, 100)
    # Alpha figure 2.14 1/15 ms-1 = 1/0.015 s-1
    alpha = 1/0.015
    t = 0.05 # is the best
    plt.plot(freqs, given_res(alpha, 2*np.pi*freqs, t))
    plt.title(f"t = {t} s")
    plt.xlabel("frequency [Hz]")
    plt.show()

    # Figure delta(omega)
    omegas = np.linspace(0, 10*np.pi, 1000)
    lams = omegas/alpha
    deltas = 8*np.arctan(lams) + np.arctan(2/lams) - np.pi
    # at omega=0 it is not defined, there a jump of size pi
    plt.plot(omegas, deltas, label=r"$\delta(\omega)$")
    plt.plot(omegas[[0,-1]], deltas[[0,-1]], "r", label="linear function")
    plt.legend()
    plt.xlabel(r"$\omega$ [Hz]")
    plt.show()