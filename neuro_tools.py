"""
Collection of useful functions related to the exercises
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import factorial

class SpikeTrain(object):
    """Object representing spike train
    """

    def __init__(self, duration, spikes):
        """
            duration: time of the experiment in seconds

            spikes: array or list with times of occurrance of spikes
                    in seconds
        """
        
        self.duration = duration
        self.spikes = spikes
        self.isi = self.interspike_intervals()

    def interspike_intervals(self):
        """Return interspike intervals
        """
        
        return np.diff(self.spikes)

    def coefficient_of_variation(self):
        """Computes coefficient of variation

        C_v = sigma_tau / mean_tau,
        where tau is interspike interval.

        For Poisson processes the value approaches 1
        """

        return self.isi.std()/self.isi.mean()

    def spike_counts(self, bin_size):
        """Returns the number of spikes in time bins of a given bin size

        Splits the experiment time into bins of the given size (bin_size)
        and counts the number of spikes in each bin.

        Arguments:
            bin_size: size of the counting interval in seconds
        """
        cumulative_counts = []
        for t in np.arange(0, self.duration+bin_size, bin_size):
            cumulative_counts.append(np.count_nonzero(self.spikes <= t))
        return np.diff(cumulative_counts)

    def fano_factor(self, counting_intervals):
        """Return Fano factor for given list of counting intervals

        FanoFactor = sigma^2_n / mean_n

        For homogenous Poisson processes it is 1 regardless of the counting
        interval. It also approaches the squared value of coefficient of 
        variation for point processes.

        Arguments
            counting_interval: list, array of counting intervals in seconds
                               if a number is given it will process as if list
                               with the number was given
        Returns
            array of Fano factors for given counting intervals
        """

        if type(counting_intervals) in [int, float]:
            counting_intervals = [counting_intervals]


        ffs = []
        for interval in counting_intervals:
            counts = self.spike_counts(interval)
            ffs.append(counts.var()/counts.mean())

        # NOTE: it is possible to get fano factor by using numpy histogram
        # function, as counts is the same as histogram of spikes

        # for interval in counting_intervals:
        #     counts = np.histogram(self.spikes, 
        #                 bins=int(self.duration/interval), 
        #                 range=(0,self.duration))[0]
        #     ffs.append(counts.var()/counts.mean())

        return np.array(ffs)

    def interspike_interval_histogram(self, bin_num, bin_max=None):
        """Returns array of the numbers of interspike intervals in bins

        Method divides the range of interspike intervals times into a given 
        number of bins (given by bin_num) and returns an array of counts in
        given bins and an array of used bins

        Arguments
            bin_num: number of bins into which the interspike intervals 
                     are divided

        Returns
            counts: array of the numbers of interspike intervals in a given
                    bin

            bins: array of bin values
        """
        if not bin_max:
            bin_max = self.isi.max()
        return np.histogram(self.isi, bins=bin_num, range=(0, bin_max))

    def plot_interspike_interval_histogram(self, bin_num, bin_max=None,
                        figsize=None, title='Interspike interval histogram'):
        """ Plots and returns histogram of inerspike interval

        Method divides the range of interspike intervals times into a given 
        number of bins (given by bin_num) and returns an array of counts in
        given bins and an array of used bins

        Arguments
            bin_num: number of bins into which the interspike intervals 
                     are divided

        Returns
            counts: array of the numbers of interspike intervals in a given
                    bin

            bins: array of bin values
        """
        isi = self.isi*1000  # Conversion to miliseconds
        if not bin_max:
            bin_max = isi.max()
        plt.figure(figsize=figsize)
        counts, bins, _ = plt.hist(isi, bins=bin_num, range=(0, bin_max))

        # NOTE: equivalent to:
        # counts, bins = np.histogram(isi, bins=bin_num, range=(0, bin_max))
        # bin_size = bin_max/bin_num
        # bins = (bins + bin_size/2)[:-1]  # shift to the center of bin
        # plt.bar(bins, counts, width=bin_size)

        plt.title(title)
        plt.xlabel('time (ms)')
        return counts, bins

    def autocorrelation(self, bin_size, max_interval=None, sym=False):
        """

        Arguments
            bin_size: number representing time bin size in seconds

            max_interval:

            sym: Flag parameter for returning symmetrized histogram
                 or just one side

        """

        # ##### IGNORE #####
        # This way is a good approximation, but has the problem of
        # precision - first it puts spikes into histogram of a
        # given bin size aka translate spike times into some kind of
        # Neural response function (rho) and then does the
        # autocorrelation. But binning puts uncertainty on the times
        # which is even more apparent during autocorrelation.
        # This method is OK if we are OK with the uncertainty
        # approximately about the value bin_size.
        # We want to avoid that, leaving it here just for learning
        # purposes commented...
        #
        # bins = np.arange(-bin_size/2, self.duration+bin_size, bin_size)
        # counts, bins = np.histogram(self.spikes, bins=bins)
        # autocorr = np.correlate(counts, counts, mode='full')
        # autocorr = autocorr.astype(float)
        # # subtraction of uniform distribution
        # autocorr -= (self.spikes.size**2)*bin_size/self.duration
        # autocorr = autocorr/self.duration
        # ##### IGNORE #####

        # Brute force way
        # Nothing particularly witty, just straightforward way to
        # compute the problem...
        if not max_interval:
            max_interval = self.duration
        bin_count = int(max_interval/bin_size) + 1
        vals = np.zeros(bin_count)
        for i, t1 in enumerate(self.spikes):
            for t2 in self.spikes[i+1:]:
                delta = t2 - t1
                if delta <= max_interval:
                    m = int(0.5 + delta/bin_size)
                    # vals[0] is bin with t in [-0.5, 0.5]
                    vals[m] += 1
                else:
                    # delta is going to be only bigger and we can skip to next
                    # t1
                    break
        
        if sym:
            # Only one side was computed, symmetrization 
            bins_total = 2*bin_count - 1
            sym_vals = np.zeros(bins_total)
            sym_vals[:bin_count] += vals[::-1]
            sym_vals[bin_count-1:] += vals
            vals = sym_vals
            vals[bin_count-1] += self.spikes.size
        else:
            vals[0] += self.spikes.size

        hist = vals/self.duration
        # Remove average value
        hist -= (self.spikes.size**2)*bin_size/(self.duration**2)
        return hist/bin_size  

    def plot_autocorrelation_histogram(self, bin_size, max_interval=None,
                             figsize=None, sym=False,
                             title='Autocorrelation histogram'):

        if max_interval is None:
            max_interval = self.duration

        hist = self.autocorrelation(bin_size, sym=sym, 
                                   max_interval=max_interval)
        # convert to ms
        times = np.arange(0, max_interval + bin_size, bin_size) * 1000  

        plt.figure(figsize=figsize)
        # plt.bar(times, hist, width=1, bottom=0)
        plt.plot(times, hist)
        plt.title(title, loc='left')
        plt.xlabel('time (ms)')


def recovering_firing_rate(rate, time, last_spike=None, tau=0):
    """Returns firing rate value based on time elapsed since the last spike

    Returns firing rate which follows equation:
        τ_ref* (dr/dt) = r_0 − r,
    where r is set to zero immediately after spike. Initial value is r_0


    Arguments:
        rate: float representing r_0 in the equation in Hz

        time: float representing time since the begining of the experiment
              in seconds

        last_spike: float representing time of the last occured spike
                    in seconds

        tau: float representing τ_ref in equation in seconds.
             0 value gives constant firing rate in the whole time range

    Returns:
        float of actual rate in Hz
    """
    if last_spike == None or tau == 0:
        return rate
    else:
        return rate*(1-np.exp(-(time-last_spike)/tau))


def poisson_spike_generator(duration, rate_bound, rate=None):
    """Returns array of spike times in a given time interval

    Function generates Poisson distribution for constant firing rate or
    constant firing rate with refractory period using rejection sampling
    (spike thinning) technique.

    Implementation of this poisson spike generator is based on the book
    Theoretical Neuroscience page 30.


    Arguments:
        duration: duration of the experiment in seconds

        rate_bound: float representing upper bound for values of rate 
                    function in Hz
                    rate_bound >= rate(time, last_spike_time) for all times

        rate: function representing firing rate in Hz
              function has to take two float values: time, last_spike_time
              if None skips spike thinning

    Returns:
        spike_times: array of occurrance times of spikes in seconds
    """

    # First, generation of spikes for constant firing rate
    times = [0]
    while times[-1] <= duration:
        times.append(times[-1] - np.log(np.random.rand())/rate_bound)

    # if rate function is not given, spike thinning is skipped
    if rate is None:
        return np.array(times[1:-1])

    # Second, spike thinning
    new_times = [times[1]]
    prev_time = new_times[-1]
    for next_time in times[2:-1]:
        # We are interested in r(t)/r_max, that's why rate_ratio
        rate_ratio = rate(next_time, prev_time)/rate_bound
        if rate_ratio >= np.random.rand():  # We keep these spikes
            new_times.append(next_time)
            prev_time = next_time
    return np.array(new_times)


def poisson_response_generator(rate, time_step):
    """Returns array of neuronal responses in given time bins

    Function compares the values rate*time_step in each bin with random
    values in range [0,1)

    Arguments:
        rate: array of firing rate values in Hz in given time bins

        time_step: float representing the time step in seconds 

    Returns:
        response: array of zeros and ones, where zero = no response
                  one = generated spike
    """
    assert rate.max()*time_step < 1, "Warning! Time step is too small"
    rng = np.random.default_rng()
    random_values = rng.random(rate.size)
    return (rate*time_step > random_values).astype(int)


def approximate_rate(t, spikes, tau, init_rate):
    """Returns firing rate in given time(s) based on decay equation and spikes.

    Rate is approximated by following equation
        τ_approx * (dr_approx/dt) = −r_approx,
    except that r_approx → r_approx + 1/τ_approx every time a spike occurs.

    Bit of math leads to the following solution:
    r_approx(t) = r_0*exp(-(t-t_0)/τ_approx) 
            + sum_{k=1}^{N: t_{N+1}> t} 1/τ_approx*exp(-(t-t_k)/τ_approx)
    
    Arguments:
        t: time in which we want to know the rate.
           can be a single number (float, int) or array of values
           times should be given in seconds

        spikes: list of times of occurred spikes
                times in seconds

        tau: number controling the decay rate
             time in seconds

        init_rate: initial rate for zero time, after enough time this value
                   has no impact

    Returns:
        firing rate: array of approximate firing rate in given times  
    """
    if type(t) != np.ndarray:
        t = np.array([t])[:, np.newaxis]
    elif t.ndim == 1:
        t = t[:,np.newaxis]

    # Decay of the initial part: r_0*exp(-(t-t_0)/τ_approx) 
    init_rates = init_rate*np.exp(-t/tau)
    init_rates = init_rates[:,0]  # transforming to 1D array

    # Decay of the additive parts:
    # sum_{k=1}^{N: t_{N+1}> t} 1/τ_approx*exp(-(t-t_k)/τ_approx)
    times = t - spikes
    rates = np.where(times > 0, np.exp(-times/tau), 0).sum(axis=1)/tau
 
    return init_rates + rates


def load_c1p8():
    """Returns tuple of arrays with response and stimulus in given times

    Returns:
        response: array of neuron's responses Binary values
                  value 0: no response
                  value 1: spike

        stimulus: array of stimulus values
    """
    data = loadmat("data/c1p8.mat")
    response = data['rho'][:,0]  # rho - response, 0 - nothing, 1 - spike
    stimulus = data['stim'][:,0]  # stim - stimulus at a certain point
    return response, stimulus


def load_c2p3():
    """Returns tuple of arrays with response and stimulus in given times

    Returns:
        counts: array of neuron's responses with number of spikes in given
                time bin

        stimulus: array of stimulus values, 
                  shape (16,16,32767) = (x, y, t)
    """
    data = loadmat("data/c2p3.mat")
    counts = data['counts'][:,0]  # number of spikes in given time bin
    stimulus = data['stim']  # stim - stimulus at a certain point
    return counts, stimulus


def gaussian_white_noise(sigma, duration, time_step):
    """Generates Gaussian white noise with given sigma

    Variance of generated noise is: 
            sigma**2/time_step 
   
    Based on the book, page 23
    """
    rng = np.random.default_rng()
    bins = int(duration/time_step)
    noise_std = sigma/np.sqrt(time_step)
    stim = noise_std*rng.standard_normal(bins)
    return stim - stim.mean()


def cyclic_conv(signal, kernel, mode='valid'):
    """Standard convolution with additional assumption that signal is cyclic

    A way to get rid of boundary effects in convolution is the assumption
    that given signal is cyclic. Function replicated the signal so it appears
    cyclic and then proceeds with standard numpy convolution
    """
    signal = np.concatenate((signal[-kernel.size+1:], signal))
    return np.convolve(signal, kernel, mode=mode)


def cyclic_corr(signal, kernel, mode='valid'):
    """Standard correlation with additional assumption that signal is cyclic

    A way to get rid of boundary effects in correlation is the assumption
    that given signal is cyclic. Function replicated the signal so it appears
    cyclic and then proceeds with standard numpy correlation

    Signal and kernel has to be of the same size!

    Takes kernel duplicates by adding copy at the end then call numpy
    correlation function which check sizes of kernel and signal, therefore,
    the assertion of sizes. The correlation result is then shifted so that
    the result at time=0 is in the middle (since this operation is cyclic
    this shift is correct)
    """
    assert signal.size == kernel.size, "Check sizes of the signals!"
    kernel = np.concatenate((kernel, kernel[:signal.size-1]))
    return np.roll(np.correlate(kernel, signal, mode=mode), signal.size//2)

