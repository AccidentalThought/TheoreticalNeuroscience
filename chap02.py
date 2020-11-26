"""
Script for exercises from Chapter 2 of Theoretical Neuroscience book
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import neuro_tools as nt
from scipy import signal
from math import factorial


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
        lam_square = 1+lam**2
        t = omega*t
        return (np.cos(t-6*arc) - np.cos(t-8*arc)/lam_square)/(lam_square**(3))

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

def exercise7():
    """This is very slow!
    
    Might help play a bit with the steps and boundaries... 
    """

    def integrand(x,y, spat_freq=2, phase=0, amp=50, sigma_x=1, sigma_y=1,
                  pref_spat_freq=2, pref_phase=0, orientation=0):
        stim = np.cos(spat_freq*x*np.cos(orientation) 
                        + spat_freq*y*np.sin(orientation) - phase)
        norm = amp/(2*np.pi*sigma_x*sigma_y)
        exponent = np.exp(-x**2/(2*sigma_x**2)-y**2/(2*sigma_y**2))
        cosine = np.cos(pref_spat_freq*x-pref_phase)
        return stim*norm*exponent*cosine

    def integrate(vals, steps):
        for step in steps:
            vals = np.trapz(vals, dx=step)
        return vals

    min_x, max_x, step_x = -10, 10, 0.01
    x = np.arange(min_x, max_x+step_x, step_x)
    min_y, max_y, step_y = -10, 10, 0.01
    y = np.arange(min_y, max_y+step_y, step_y)

    spatial_freqs = np.linspace(0, 6, 100)
    phases = np.linspace(-np.pi, np.pi, 100)

    # Expansion of dimensions
    x = np.expand_dims(x, axis=(1))
    y = np.expand_dims(y, axis=(0))
    # The case to have 3 dim array was limited by memory of my laptop,
    # had to go with looping, which is slower
    # spatial_freqs = np.expand_dims(spatial_freqs, axis=(1,2))
    # phases = np.expand_dims(phases, axis=(1,2))

    first = np.array([integrate(integrand(x, y, spat_freq=spat_freq),
                             [step_x, step_y]) for spat_freq in spatial_freqs])
    plt.plot(spatial_freqs, first)
    plt.title("Numerical spatial linear estimate, dependence on spatial frequance")
    plt.xlabel(r"$K [^\circ]^{-1}$")
    plt.show()
    second = np.array([integrate(integrand(x, y, phase=phase), [step_x, step_y])
                         for phase in phases])
    plt.title("Spatial linear estimate")
    plt.xlabel(r"$\phi [rad]$")
    plt.plot(phases, second)
    plt.show()


def exercise8():
    """To speed up this computation - we use analytical result from exercise 6
    instead of numerical solution from exercise 7
    """
    def spatial_linest(phase=0, orientation=0, spatial_freq_ratio=1,
                    ksigma=2, pref_phase=0, amplitude=5):
        expfac = np.exp(-ksigma**2/2*(1+spatial_freq_ratio**2))
        val = ksigma**2*spatial_freq_ratio*np.cos(orientation)
        first_cos = np.cos(pref_phase-phase)*np.exp(val)
        second_cos = np.cos(pref_phase+phase)*np.exp(-val)
        return amplitude/2*expfac*(first_cos+second_cos)

    def analytic_squared_sum(phase=0, orientation=0, spatial_freq_ratio=1,
                            ksigma=2, pref_phase=0, amplitude=5):
        expfac = amplitude**2*np.exp(-ksigma**2*(1+spatial_freq_ratio**2))
        trig = np.cosh(ksigma**2*spatial_freq_ratio*np.cos(orientation))**2
        trig -= np.sin(phase)**2
        return expfac*trig

    spatial_freqs = np.linspace(0, 6, 100)
    phases = np.linspace(-np.pi, np.pi, 100)

    first = (spatial_linest(spatial_freq_ratio=spat_freqs/2)**2 +
        spatial_linest(spatial_freq_ratio=spat_freqs/2, pref_phase=-np.pi/2)**2)
    plt.plot(spatial_freqs, first)
    plt.title("Numerical spatial linear estimate, dependence on spatial frequance")
    plt.xlabel(r"$K [^\circ]^{-1}$")
    plt.show()

    second = (spatial_linest(phase=phases)**2 +
        spatial_linest(phase=phases, pref_phase=-np.pi/2)**2)
    third = analytic_squared_sum(phase=phases)
    plt.plot(phases, second)
    plt.plot(phases, third, "r")
    plt.title("Spatial linear estimate")
    plt.xlabel(r"$\phi [rad]$")
    plt.show()


def exercise9():
    def temporal_linest(t, alpha=1/0.015, omega=6*np.pi):
        lam = omega/alpha
        arc = np.arctan(lam)
        lam = 1+lam**2
        t = omega*t
        return (np.cos(t-6*arc) - np.cos(t-8*arc)/lam)/(lam**3)

    ts = np.linspace(0, 1, 150)

    temp_linest = temporal_linest(ts)

    simple_cell = np.where(temp_linest > 0, temp_linest, 0)
    complex_cell = temp_linest**2

    plt.plot(ts, simple_cell, label="Simple cell")
    plt.plot(ts, complex_cell, label="Complex cell")
    plt.legend()
    plt.title("Temporal linear estimate of simple and complex cells")
    plt.xlabel(r"time [s]")
    plt.show()


def exercise10(alpha=1/0.015,
               sigma_x = 1,
               sigma_y = 1,
               pref_phase = 0,
               pref_spat_freq = 2,
               omega = 8*np.pi,
               spat_freq = 2,
              ):

    def spatial_linest(spatial_phase=0, orientation=0, 
                    spatial_freq_ratio=spat_freq/pref_spat_freq,
                    ksigma=sigma_x*pref_spat_freq, pref_phase=0, amplitude=1):
        expfac = np.exp(-ksigma**2/2*(1+spatial_freq_ratio**2))
        val = ksigma**2*spatial_freq_ratio*np.cos(orientation)
        first_cos = np.cos(pref_phase-spatial_phase)*np.exp(val)
        second_cos = np.cos(pref_phase+spatial_phase)*np.exp(-val)
        return amplitude/2*expfac*(first_cos+second_cos)

    def temporal_amplitude(alpha=alpha, omega=omega):
        lam = omega/alpha
        return (lam*np.sqrt(lam**2+4))/((1+lam**2)**4)

    def temporal_linest(t, alpha=alpha, omega=omega):
        lam = omega/alpha
        arc = np.arctan(lam)
        lam_square = 1+lam**2
        t = omega*t
        return (np.cos(t-6*arc) - np.cos(t-8*arc)/lam_square)/(lam_square**(3))

    ts = np.linspace(0, 6*np.pi/omega, 100)  # three periods
    response = spatial_linest()*temporal_linest(ts)
    response[response<0] = 0
    plt.plot(ts, response)
    plt.title("Response of the simple cell")
    plt.xlabel("time [s]")
    plt.show()

    freqs = np.linspace(0, 20, 100)
    amplitude = spatial_linest()*temporal_amplitude(omega=omegas*2*np.pi)
    plt.plot(freqs, amplitude)
    plt.title("Response amplitude as a function of temporal frequency")
    plt.xlabel("freqeuncy [Hz]")
    plt.show()

    spatial_freqs = np.linspace(0, 10, 100)
    amplitude = spatial_linest(spatial_freq_ratio=spatial_freqs/pref_spat_freq)*temporal_amplitude()
    plt.plot(freqs, amplitude)
    plt.title("Response amplitude as a function of spatial frequency")
    plt.xlabel(r"K [deg$^{-1}$]")
    plt.show()


def exercise11():
    def response(time=False, alpha=1/0.015, sigma_x=1, k=2, stim_k=2, omega=8*np.pi):
        lam = (omega/alpha)**2
        expfac = np.exp(-(sigma_x**2)*(k**2+stim_k**2))
        factor = lam*(lam+4)/(2*(1+lam)**8)
        coshfac = np.cosh(2*sigma_x**2*k*stim_k)
        if time is False:
            cosfac = 1
        else:
            delta = 8*np.arctan(omega/alpha)+ np.arctan(2*alpha/omega) - np.pi
            cosfac = np.cos(2*omega*time -2*delta)
        return factor*expfac*(coshfac+cosfac)
    # First time dependent plot
    # period is 0.25 s -> 1 second interval shows 8 periods
    ts = np.linspace(0,1, 200)
    res = response(time=ts)
    plt.plot(ts, res)
    plt.xlabel("time [s]")
    plt.title("Response of a complex cell")
    plt.show()

    # Second plot -> function of omega
    omegas = np.linspace(0, 40*np.pi, 200)
    res = response(omega=omegas)
    plt.plot(omegas/(2*np.pi), res)
    plt.xlabel("frequency [Hz]")
    plt.title("Response of a complex cell")
    plt.show()

    # Third plot -> function of stim_k
    stim_ks = np.linspace(0, 10, 200)
    res = response(stim_k=stim_ks)
    plt.plot(stim_ks, res)
    plt.xlabel("Stimulus spatial frequency [$1/^\circ$]")
    plt.title("Response of a complex cell")
    plt.show()


def exercise12():
    def integrand(tau, x, stim_k, stim_shift, omega=8*np.pi,
                alpha=1/0.015, psi=np.pi/9, c=20, sigma_x=1, vf_k=2, phi=0
                ):
        """stim_shift = 0 -> cos, integrand A
        stim_shift = -np.pi/2 -> sin, integrand B"""
        factor = alpha/(np.sqrt(2*np.pi)*sigma_x)
        stim = np.cos(stim_k*x+omega*tau+stim_shift)
        x, tau = x*np.cos(psi) - c*tau*np.sin(psi), tau*np.cos(psi) + x/c*np.sin(psi)
        tau = alpha*tau
        temporal = (tau**5/factorial(5)-tau**7/factorial(7))
        spatial = np.exp(-x**2/(2*sigma_x**2)-tau)*np.cos(vf_k*x-phi)
        return factor*stim*temporal*spatial

    vels = np.linspace(2, 100, 50)
    est_pos = np.empty_like(vels)
    est_neg = np.empty_like(vels)
    for i, vel in enumerate(vels):
        print(i)
        k = 8*np.pi/vel
        a = integrate.dblquad(integrand, -np.inf, np.inf, 0, np.inf, args=(k,0))[0]
        b = integrate.dblquad(integrand, -np.inf, np.inf, 0, np.inf, args=(k,-np.pi/2))[0]
        est_pos[i] = np.sqrt(a**2+b**2)

        a = integrate.dblquad(integrand, -np.inf, np.inf, 0, np.inf, args=(-k,0))[0]
        b = integrate.dblquad(integrand, -np.inf, np.inf, 0, np.inf, args=(-k,-np.pi/2))[0]
        est_neg[i] = np.sqrt(a**2+b**2)

    plt.plot(vels, est_pos, label="right-going stimuli")
    plt.plot(vels, est_neg, label="left-going stimuli")
    plt.legend()
    plt.title("Amplitude of a response")
    plt.xlabel(r"velocity $[^\circ s^{-1}]$")
    plt.show()


def exercise13(amplitude=1,
              )
    disparity = np.linspace(0, 2*np.pi, 200)
    response = amplitude**2*(2+np.cos(disparity))
    plt.plot(disparity, response)
    plt.xlabel("disparity [rad]")
    plt.title("Response of disparity tuned complex cell")
    plt.show()


def exercise14(sigma_c = 0.3,  # deg
               sigma_s = 1.5,  # deg
               b = 5,
               alpha = 1/0.016,  # sec^-1
               beta = 1/0.064,  # sec^-1
              ):
    ks = np.linspace(0,20, 200)
    res_cen = np.exp(-(sigma_c**2)*(ks**2)/2)
    res_sur = np.exp(-(sigma_s**2)*(ks**2)/2)
    response = res_cen - b*res_sur
    plt.plot(ks, response)
    plt.title("Selectivity to spatial frequency")
    plt.xlabel("K [$1/^\circ$]")
    plt.show()

