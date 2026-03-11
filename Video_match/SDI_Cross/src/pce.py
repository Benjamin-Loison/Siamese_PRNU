"""
Please read the copyright notice located on the readme file (README.md).
"""
import SDI_Cross.src.Functions as Fu
import numpy as np
from scipy import special


def PCE(C, shift_range=[0,0], squaresize=11):

    if any(np.greater_equal(shift_range,C.shape)):
        shift_range = min(shift_range,C.shape-1)   # all possible shift in at least one dimension

    shift_range = np.array(shift_range)
    Out = dict(PCE=[], pvalue=[], PeakLocation=[], peakheight=[], P_FA=[], log10P_FA=[])

    if not C.any():            # the case when cross-correlation C has zero energy (see crosscor2)
        Out['PCE'] = 0
        Out['pvalue'] = 1
        Out['PeakLocation'] = [0,0]
        return

    # Cinrange = C[-1-shift_range[0]:,-1-shift_range[1]:]  	# C[-1,-1] location corresponds to no shift of the first matrix argument of 'crosscor2'
    Cinrange = C[:1+shift_range[0],:1+shift_range[1]]  	# C[-1,-1] location corresponds to no shift of the first matrix argument of 'crosscor2'
    [max_cc, imax] = np.max(Cinrange.flatten()), np.argmax(Cinrange.flatten())
    [ypeak, xpeak] = np.unravel_index(imax,Cinrange.shape)[0], np.unravel_index(imax,Cinrange.shape)[1]
    Out['peakheight'] = Cinrange[ypeak,xpeak]
    del Cinrange
    Out['PeakLocation'] = [ypeak, xpeak]
    Out['PeakValue'] = max_cc

    C_without_peak = _RemoveNeighborhood(C, [ypeak, xpeak], squaresize)
    del C

    # signed PCE, peak-to-correlation energy
    PCE_energy = np.mean(C_without_peak*C_without_peak)
    Out['PCE'] = (Out['peakheight']**2)/PCE_energy * np.sign(Out['peakheight'])

    return Out['PCE']
# ----------------------------------------

def _RemoveNeighborhood(X,x,ssize):
    # Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
    # ssize     square neighborhood has size (ssize x ssize) square
    [M,N] = X.shape
    radius = (ssize-1)/2
    X = np.roll(X,[np.int(radius-x[0]),np.int(radius-x[1])], axis=[0,1])
    Y1 = X[ssize:,:ssize];   Y1 = Y1.flatten()
    Y2 = X[:,ssize:];   Y2 = Y2.flatten()
    # Y = np.concatenate([Y1, X.flatten()[int(M*ssize):]], axis=0)
    Y = np.concatenate([Y1, Y2], axis=0)
    return Y

def _FAfromPCE(pce,search_space):
    # Calculates false alarm probability from having peak-to-cross-correlation (PCE) measure of the peak
    # pce           PCE measure obtained from PCE.m
    # seach_space   number of correlation samples from which the maximum is taken
    #  USAGE:   FA = FAfromPCE(31.5,32*32);

    [p,logp] = Fu.Qfunction(np.sign(pce)*np.sqrt(np.abs(pce)))
    if pce<50:
        FA = np.power(1-(1-p),search_space)
    else:
        FA = search_space*p                # an approximation

    if FA==0:
        FA = search_space*p
        log10FA = np.log10(search_space)+logp*np.log10(np.exp(1))
    else:
        log10FA = np.log10(FA)

    return FA, log10FA
