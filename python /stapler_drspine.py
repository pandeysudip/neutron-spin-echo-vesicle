#!/usr/bin/env python

from __future__ import print_function

import sys
import os.path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit, OptimizeWarning

from echodet_reader import read_wbfile, write_csv

warnings.filterwarnings('ignore', '', RuntimeWarning)
warnings.filterwarnings('error' , '', OptimizeWarning)

A2ns = 1e-7 # A^2/ns = 1e-7 cm^2/s - diffusion is coefficient often reported in cm^2/s

# ============================================================================================================================
# ============================================================================================================================
def fit_sqt(q, data, model='Zilman-Granek', **kwargs):
    """
    fit S(q,t) = I(q,t)/I(q,0) according to a model

    returns a tuple:
        parameter, parameter_error, fit_function
    """
    A    = kwargs.pop('A',  1.0)
    G    = kwargs.pop('G',  1.0)
    t0   = kwargs.pop('t0', 1.0)
    t1   = kwargs.pop('t1', 1.0)
    beta = kwargs.pop('beta', 2.0/3.0)
    R    = kwargs.pop('R', 1.0)
    bounds  = kwargs.pop('bounds', (-np.inf, np.inf))
    init_par = kwargs.pop('init_par', None)
    max_tau = kwargs.pop('max_tau', np.inf)

    function_library = {
        'KWW'            : lambda t, t0, beta:         np.exp(-(t/t0)**beta),   # simple KWW
        'KWW-Fix'        : lambda t, t0:               np.exp(-(t/t0)**beta),   # KWW with beta fixed
        'KWW-Norm'       : lambda t, t0, beta, A:    A*np.exp(-(t/t0)**beta),   # KWW with prefactor
        'KWW-Norm-Fix'   : lambda t, t0, A:          A*np.exp(-(t/t0)**beta),   # KWW with prefactor and beta fixed
        'Zilman-Granek'  : lambda t, G:                np.exp(-(G*t)**beta),    # Zilman-Granek, KWW with b=2/3
        'Zilman-Granek-Norm'  : lambda t, G, A:      A*np.exp(-(G*t)**beta),    # Zilman-Granek, KWW with b=2/3
        'Exp'            : lambda t, t0, A:          A*np.exp(-(t/t0)),         # simple exponential function
        'Diffusion'      : lambda t, G:                np.exp(-G*q*q*t),        # simple diffusion
        'Diffusion-Norm' : lambda t, G,  A:          A*np.exp(-G*q*q*t),        # simple diffusion with prefactor
        'Diffusion+Const': lambda x, G,  A:      (1-A)*np.exp(-G*q*q*x)+A,      # simple diffusion with constant "background"
        'Power'          : lambda x, b,  A:      A*x**b,                        # power law
        'Linear'         : lambda x, x0, A:      A*(x-x0),                      # linear function
        'Constant'       : lambda x, A:          A*np.ones_like(x),             # constant
        'Cumulant'       : lambda x, c1, c2:     np.exp(-c1*x+c2*x**2/2.0),     # 2nd cumulant
        'Diffusion+2ND'  : lambda t, G,  c2:     np.exp(-G*q*q*t+c2*t*t) ,      # diffusion + 2nd cumuland
        'Richter-Rouse-Norm': lambda t, Ws, A: A*np.exp(-0.25*(q*R)**2) + A*(1-np.exp(-0.25*(q*R)**2))*np.exp(-q**2*np.sqrt(Ws*t/9/np.pi)), # RR
        'Richter-Rouse'  : lambda t, Ws, R:      np.exp(-0.25*(q*R)**2) +   (1-np.exp(-0.25*(q*R)**2))*np.exp(-q**2*np.sqrt(Ws*t/9/np.pi)), # RR
    }

    tau  = data[0]
    sqt  = data[1]
    serr = data[2]

    func = function_library.get(model, None)
    if func is None: raise RuntimeError("model '%s' is invalid" % model)

    popt, pcov = curve_fit(func, tau[tau<max_tau], sqt[tau<max_tau], sigma=serr[tau<max_tau], p0=init_par,
                             maxfev=10000, ftol=1e-5)
    par, epar  = popt[0], np.sqrt(pcov[0,0])
    fit_func = lambda x : func(x, *popt)
    chisq = np.sum(((fit_func(tau[tau<max_tau]) - sqt[tau<max_tau])/serr[tau<max_tau])**2)
    chisq = chisq/(len(sqt[tau<max_tau]) - len(popt))
    return (par,epar, fit_func, chisq, popt)

# ============================================================================================================================
# ============================================================================================================================
def prune_data(data, min_points=3, min_taus=3):
    """
    Prune/cleanup DrSPINE reduced data

    """
    if not len(data):
        return None
    data =  data.T
    data = data[data[:,5]>min_points] # at least min_points contributing points
    data = data[data[:,4]>0] # var(tau)
    data = data[data[:,3]>0] # var(sqt)
    data = data[data[:,2]>0] # err(sqt)
    data = data[data[:,1]>0] # require positive resultant sqt
    if len(data[:,4])<min_taus: # at least min_taus "good" taus
        return None
    return data.T

# ============================================================================================================================
# ============================================================================================================================
def main(*args, **kwargs):
    """
    main function of the stapler
    DOCS: to do
    """
    model = kwargs.pop('model', 'KWW-Norm') # fit model (see fit_sqt)
    mpars = kwargs.pop('model_pars', {} )   # model parameters
    rings = kwargs.pop('rings', (0,)   )    # which q-rings to use
    plot_summary = kwargs.pop('plot_summary', False)
    plot_fits    = kwargs.pop('plot_fits', True)
    data_dir     = kwargs.pop('data_dir', '.')
    shift        = kwargs.pop('shift', 0.0)
    other_data   = kwargs.pop('other_data', None)


    results = []           # list of fit results
    print("# ", model)
    for sample in args:
        label  = sample[0]
        print("# ", label)
        if model.startswith('Diffusion'):
            print("# q     dq      D          dD         chi2  extra_pars")


        elif model.startswith('Zilman'):
            print("# q     dq      Gamma      dGamma     chi2  extra_pars")
        elif model.startswith('Richter'):
            print("# q     dq      Wsigma4    dWsigma4   chi2  extra_pars")
        else:
            print("# q     dq      tau0       dtau0      chi2  extra_pars")
        if plot_fits:
            plt.figure(figsize=(8,8))
        fitres = []
        outfile = label.replace('/','_').replace(' ','_').replace('=','').lower() # nice file name

        d_sqt = 0.0 # data shift for presentation
        for filename, slabel in sample[1:]:
            filename = os.path.join(data_dir, filename)
            if not os.path.exists(filename): continue
            dset = read_wbfile(filename)
            for m, d in reversed(dset):
                d     = prune_data(d, min_points=4)
                if d is None: continue
                q  = m['q']
                if q == 'NaN': continue

                dq   = m['q_var'] or 0.0
                tsam = m.get('temp_act') or m.get('temp')
                tau  = d[0]
                sqt  = d[1]
                dsqt = d[2]

                alabel = r"Q=%.3f $\AA^{-1}$ %s" % (q, slabel)
                if plot_fits:
                    p = plt.errorbar(tau, sqt+d_sqt, yerr=dsqt,fmt='s', label=alabel)
                    acolor = p[0].get_color()
                par0, epar0, fit_func, chi2, popt = fit_sqt(q, d, model=model, **mpars)
                if plot_fits:
                    #ftau = np.logspace(np.log10(min(tau))-0.2, np.log10(max(tau))+0.2)
                    ftau = np.logspace(-2,1.99)
                    fsqt = fit_func(ftau)
                    plt.plot(ftau, fsqt+d_sqt, '--', color=acolor) # fit
                fitres.append((q, dq, par0, epar0,tsam))
                print("%.4f, %.4f, %.3e, %.3e, %.3g,\t" % (q, dq, par0, epar0, chi2), end="")
                if len(popt)>1:
                    print(", ".join(["%.3g" % _x for _x in popt[1:]]), end=" ")
                print()
                d_sqt = d_sqt + shift
                data = np.vstack((tau,sqt,dsqt))
#                write_csv("%s-q-%.3f.csv" % (outfile,q), data, label=label, q=q, dq=dq)

        results.append((label, fitres))


        if plot_fits:
            if callable(other_data):
                other_data()
            plt.xscale('log')
            #plt.yscale('log')
            plt.xlim(left=0.05,right=150.0)
            plt.ylim(bottom=0.1, top=1.1)
            plt.xlabel(r'$\tau$ [ns]')
            plt.ylabel(r'$S(Q,\tau)/S(Q,0)$')
            plt.title("%s (%s)" % (label, model))
            plt.grid(which='both')
            plt.legend(loc=0)
            plt.savefig("%s-sqt.pdf" % str(outfile))

    if plot_summary:
        plt.figure(figsize=(8,8))
        print("\n\n# SUMMARY")
        for label, fitres in results:
            fitres = np.asarray(fitres)
            fitres = fitres[fitres[:,0].argsort()].T
            q     = fitres[0]
            dq    = fitres[1]
            par0  = fitres[2]
            dpar0 = fitres[3]
            tsam  = fitres[4]
            plbl  = ''
 #           par0  = par0/par0[0]
#            dpar0 = dpar0/dpar0[0]
            p = plt.errorbar(q, par0, yerr=dpar0, xerr=dq,
                fmt='o', label='%s%s' % (label,plbl))
            acolor = p[0].get_color()

#            if model.lower().startswith('constant'):
#                def zm(x,t0,t1):
#                    return t0/x**2
#            else:
#                def zm(x,a,t0):
#                    return t0*x**a

#            a, acov = curve_fit(zm, q, par0, sigma=dpar0)
#            chisq = np.sum(((zm(q,*a)-par0)/dpar0)**2)/(len(q)-len(a))
#            if model.lower().startswith('richter'):
#                flabel=r'$W\sigma^4 = %.3g\,Q^{%.3g}$' % (a[1],a[0])
#            else:
#                flabel=r'$ A = %g Q^{-2}$' % a[0]
#            pflabel = flabel.translate({ord(c): ' ' for c in '$\\,'}) # remove latex chars
#            print("%-48.48s %s %5.2f" % (label,pflabel, chisq))
#            xq = np.logspace(np.log10(min(q))-0.1, np.log10(max(q))+0.1)
#            yp = zm(xq, *a)
#            plt.plot(xq,yp, '--', color=acolor, label=flabel)

        plt.title("Model: %s" % model)
        plt.xlabel(r'$Q \; (\AA^{-1})$')

        if model.lower().startswith('diffusion'):
            plt.ylabel(r'D$\ \; (\AA^{2} ns^{-1}) $')
        elif model.lower().startswith('zilman'):
            plt.ylabel(r'$\Gamma \; (ns^{-1}) $')
#        elif model.lower().startswith('richter'):
#            plt.ylabel(r'$W\sigma^4\; (\AA^4\,ns^{-1}) $')
#            plt.xscale('log')
#            plt.yscale('log')
#        elif model.lower().startswith('constant'):
#            plt.ylabel(r'$A$')
#            #plt.xscale('log')
#            #plt.yscale('log')
        else:
            plt.ylabel(r'$\tau_0 \ \; (ns) $')
 #           plt.xscale('log')
 #           plt.yscale('log')

        plt.legend(loc='best')
        plt.grid(which='both')
#        plt.savefig(model.lower()+'-summary.pdf')

    plt.show()


if __name__ == "__main__":
    #                           sample name         reduced data filename           extra label (if any)
    #sam  =      [ r'Sample 1 and 2', ("Sample1_sqt.dat",""), ("Sample2_sqt.dat",""),]
    sam1  =      [ r'VES1', ("VES1_sqt.dat",""),]
    sam2  =      [ r'MIC1', ("MIC1_sqt.dat",""),]   
    sam3  =      [ r'GEL1', ("GEL1_sqt.dat",""),]   
    
    main(
        sam1,
	sam2,
	sam3,
        

        model='Zilman-Granek-Norm',
        #model_pars=dict(bounds=([0.0,0.00],[np.inf,1.01]), init_par=[1000.0,1.0], R=32),

        plot_summary=True,  ###true or false to plot the fit model
        plot_fits=True,
        data_dir='.',
    )


