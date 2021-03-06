"""
Script for exercises from Chapter 3 of Theoretical Neuroscience book
"""

import numpy as np
import matplotlib.pyplot as plt
import neuro_tools as nt


def exercise1(sigma = 1,
              r_neg = 1,
              r_pos = 4,
              num_curves = 5, # number of curves in ROC plot
              simulations = 10000,
             ):

    # Plot ROC for num_curves values of d, where min is zero, and max is given
    # by given r_neg and r_pos

    # random generator for the simulation
    rng = np.random.default_rng()

    print("|  d  |  P[correct] simulation  |  P[correct] integration  |")
    print("|-----|-------------------------|--------------------------|")
    for r in np.linspace(r_neg, r_pos, num_curves):
        d = (r-r_neg)/sigma
        alp, bet = nt.roc_curve(sigma, r_neg, r, points=2000)

        # plot ROC
        plt.plot(alp,bet, label=r"$d'=${}".format(d))

        # Integrating ROC
        # the values are not distributed equidistantly, so the error here 
        # is quite high, the error is proportional to 
        # np.abs(np.diff(alp)).max()
        # the error is still lesser than simulations error
        num_int = -np.trapz(bet, alp)  # minus because of the direction

        # Simulating the associated two-alternative forced choice
        plus = rng.normal(r, sigma, simulations)
        minus = rng.normal(r_neg, sigma, simulations)
        correct = (plus > minus).mean()

        # printing results
        print(f"|{d:4.1f} |{correct*100:23.2f}  |{num_int*100:24.2f}  |")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title("ROC curves")
    plt.legend()
    plt.show()


def exercise2(num_trials = 1000,
              num_d = 100,
              roc_ds = [2, 3, 4, 5],
             ):
    ds = np.linspace(0, 10, num_d)
    correct = []
    for d in ds:
        z = 20 + 5*d
        rng = np.random.default_rng()
        trials = rng.integers(2, size=num_trials)  # 1 => +; 0 => -
        rates = rng.normal(20, 10, num_trials) + 10*d*trials 
        rates[rates<0] = 0
        correct.append(((rates > z) == trials).mean())
    plt.plot(ds, correct)
    plt.xlabel("discriminability")
    plt.ylabel("correct")
    plt.show()

    # ROC
    # z range: min 0 is clear, max: we cut at 3*sigma
    # i.e. z_max = 20 + 3*10 + 10*d, so for d == 10 => z_max = 150
    # suggested cut is clearly ok for d ~< 10

    for d in roc_ds:
        z = np.linspace(0, 140, 1000)[np.newaxis]
        trials = rng.integers(2, size=num_trials)[:,np.newaxis]
        rates = rng.normal(20, 10, num_trials)[:,np.newaxis] + 10*d*trials 
        rates[rates<0] = 0
        pos_trials = trials.sum()
        beta = ((rates > z)*trials).sum(axis=0)/pos_trials
        alpha = ((rates > z)*(1-trials)).sum(axis=0)/(num_trials-pos_trials)
        plt.plot(alpha, beta, label=r"$d'=${}".format(d))
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title("ROC curves")
    plt.legend()
    plt.show()


def exercise3(trials = 1000,
              num_thetas = 100,
              theta_min = -np.pi/2,
              theta_max = np.pi/2,
             ):
    rng = np.random.default_rng()
    # (trial, theta, selectivity)
    thetas = np.linspace(theta_min, theta_max, num_thetas)
    thetas = thetas.reshape(1, num_thetas, 1)

    selectivity = (np.arange(1,8,2)*np.pi/4)
    selectivity = selectivity.reshape(1,1,4)

    noise = rng.normal(0, 5, (trials, num_thetas, 4))

    rates = 50*np.cos(thetas-selectivity) 
    rates[rates<0] = 0
    rates = rates + noise
    rates[rates<0] = 0

    x = (rates*np.cos(selectivity)).sum(axis=-1)
    y = (rates*np.sin(selectivity)).sum(axis=-1)

    # Computing estimation
    # Spliting into quadrants
    first = (x >= 0)*(np.abs(y) <= x)  # angles (-pi/4; pi/4)
    second = (y > 0)*(np.abs(x) < y)  # angles(pi/4;3pi/4)
    third = (y < 0)*(np.abs(x) < -y)  # angles (-3pi/4; -pi/4)
    fourth = (x < 0)*(y > 0)*(y <= -x)  # angles (3pi/4;pi)
    fifth = (x < 0)*(y < 0)*(-y <= -x)  # angles(-pi;-3pi/4)

    # Computing the estimations
    estimation = np.zeros_like(x)
    estimation[first] = np.arctan((y/x)[first])
    estimation[second] = np.arctan((-x/y)[second]) + np.pi/2
    estimation[third] = np.arctan((-x/y)[third]) - np.pi/2
    estimation[fourth] = np.arctan((y/x)[fourth]) + np.pi
    estimation[fifth] = np.arctan((y/x)[fifth]) - np.pi

    # Plotting the error
    error = ((estimation-thetas[...,0])**2).mean(axis=0)
    plt.plot(thetas[0,:,0]*180/np.pi, np.sqrt(error))
    plt.title(r"Error of the estimated angle $\theta_{est}$")
    plt.xlabel(r"$\theta$")
    plt.show()
