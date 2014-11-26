#!/usr/bin/env python


def GammaS_to_SpinRelaxationRate(GammaS, W, rho):
    """ transforms spin flip probability GammaS to a spin relaxation rate
        We assume a nanoribbon of width W containing one adatom.
        rho is the adatom density we're calculating the rate for.
        W has to be given in units of a
        """
    t = 2.6
    a = 1
    hbar = 6.58e-16

    prefactor = 4 * W * rho * t / a / hbar
    result = GammaS * prefactor

    return result
