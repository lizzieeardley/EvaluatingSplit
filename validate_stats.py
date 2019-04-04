import pandas as pd 
import numpy as np
import math

from scipy.stats import binom, norm, t
from scipy.stats import ttest_ind_from_stats as ttest
from scipy.stats import chi2_contingency as chi2

from statsmodels.stats.power import NormalIndPower, tt_ind_solve_power
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic


def getMLDE_Lizzie(base_converted, base_sample, verbose=False, sig=0.05,power=0.2, vverbose=False):  
# Minimum Likely Detectable DELTA -- POST-HOC adaptation
# Inputs: P1 is control conversion, n is sample size, sig / power should be self-explanatory. 
# NB: n = denominator of metric
# Outputs: expected threshold percentage change for significance
    n=base_sample
    p1=base_converted/float(base_sample)
    z = norm.isf(sig/2) #two-sided t test
    zp = -1 * norm.isf(power)
    x = pow(zp+z,2)/(2*n)
    a = (1+x)
    b = 2*(p1*(x-1)-x)
    c = pow(p1,2)*(1+x) - 2*x*p1
    p2 = max((-b-pow((pow(b,2) - 4*a*c),0.5)) / (2*a),(-b+pow((pow(b,2) - 4*a*c),0.5)) / (2*a))
    delta = (p2-p1)
    delta_relper = (100.0*(p2-p1))#/p1)
    return delta_relper, delta#.round(2)    


def getSignificance(p, p_t = 0.05):
    if p <= p_t:
        return True
    else:
        return False 
    
def getSignificanceText(p, p_t = 0.05):
    if p <= p_t:
        return "Significant"
    else:
        return "Not Significant"     
    
def getSignificanceFromCI(upper, lower):
    if upper < 0.0 or lower >0.0:
        return True
    else:
        return False 
    
### Error Margins for Binomial Metrics
def propci_wilson_cc(count, nobs, alpha=0.05):
    # get confidence limits for proportion
    # using wilson score method w/ cont correction
    # i.e. Method 4 in Newcombe [1]; 
    n = nobs
    p = count/n
    q = 1.-p
    z = norm.isf(alpha / 2.)
    z2 = z**2   
    denom = 2*(n+z2)
    num = 2.*n*p+z2-1.-z*np.sqrt(z2-2-1./n+4*p*(n*q+1))    
    ci_l = num/denom
    num = 2.*n*p+z2+1.+z*np.sqrt(z2+2-1./n+4*p*(n*q-1))
    ci_u = num/denom
    if p == 0:
        ci_l = 0.
    elif p == 1:
        ci_u = 1.
    return ci_l, ci_u

def dpropci_wilson_cc(a,m,b,n,alpha=0.05):
    # get confidence limits for difference in proportions
    #   a/m - b/n
    # using wilson score method w/ cont correction
    # i.e. Method 11 in Newcombe [2]    
    # verified via Table II  
    theta = a/m - b/n    
    l1, u1 = propci_wilson_cc(count=a, nobs=m, alpha=alpha)
    l2, u2 = propci_wilson_cc(count=b, nobs=n, alpha=alpha)    
    ci_u = theta + np.sqrt((a/m-u1)**2+(b/n-l2)**2)
    ci_l = theta - np.sqrt((a/m-l1)**2+(b/n-u2)**2)   

    sym_margin = 100.0*(max(abs(np.sqrt((a/m-u1)**2+(b/n-l2)**2)), abs(np.sqrt((a/m-l1)**2+(b/n-u2)**2)))/(b/n))
    return sym_margin, theta, ci_l, ci_u

#power_t=tt_ind_solve_power
def getMLDE_t(nobs1, nobs2, alpha=0.05, default_typ2 = 0.2, alternative='two-sided'):
    ratio = nobs2/float(nobs1)
    return tt_ind_solve_power( effect_size=None, nobs1=nobs1, alpha=alpha, power=1.0 - default_typ2, ratio=ratio, alternative=alternative)


def getTConfInt( b_n,b_std, c_n, c_std, alpha = 0.05):
    df = (b_n + c_n - 2)    
    std_pooled = math.sqrt( ((b_n - 1)*(b_std)**2 + (c_n - 1)*(c_std)**2) / df) 
    MOE = t.ppf(1.0 - (alpha/2.0), df) * std_pooled * math.sqrt(1.0/b_n + 1.0/c_n)
    return MOE

power_z = NormalIndPower()
def getMLDE_z(nobs1, nobs2, alpha=0.05, default_typ2 = 0.2, alternative='two-sided'):
    power=1.0 - default_typ2
    ratio = nobs2/float(nobs1)
    return power_z.solve_power( effect_size=None, nobs1=nobs1, alpha=alpha, power=power, ratio=ratio, alternative=alternative)

