from __future__ import division

from collections import OrderedDict
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table


class polyco(Table):
    def __init__(self, name):
        super(polyco,self).__init__(polyco2table(name))

    def __call__(self, mjd_in, index=None, rphase=None):
        mjd = np.atleast_1d(mjd_in)
        if index is None:
            i = np.searchsorted(self['mjd_mid'], mjd)
            i = np.clip(i, 1, len(self)-1)
            i -= mjd-self['mjd_mid'][i-1] < self['mjd_mid'][i]-mjd
        else:
            i = np.atleast_1d(index)

        if np.any(np.abs(mjd - self['mjd_mid'][i])*1440 > self['span']/2):
            raise ValueError('(some) MJD outside of polyco range')

        phases = np.zeros_like(mjd)
        for j in set(i):
            in_set = i == j
            phasepol = self.phasepol(j, rphase)
            phases[in_set] = phasepol(mjd[in_set])

        return phases

    def phasepol(self, index, rphase=None):
        domain = np.array([-1, 1]) * self['span'][index]/2
        phasepol = Polynomial(self['coeff'][index],
                              domain/1440.+self['mjd_mid'][index], domain)
        phasepol.coef[0] += self['rphase'][index] if rphase is None else rphase
        phasepol.coef[1] += self['f0'][index]*60.
        return phasepol


def polyco2table(name):
    with open(name, 'r') as polyco:
        line = polyco.readline()
        t = None
        while line != '':
            d = OrderedDict(zip(['psr','date','utc_mid','mjd_mid',
                                 'dm','vbyc_earth','lgrms'],
                                line.split()))
            d.update(dict(zip(['rphase','f0','obs','span','ncoeff',
                               'freq','binphase'],
                              polyco.readline().split()[:7])))
            for key in d:
                try:
                    d[key] = int(d[key])
                except ValueError:
                    try:
                        d[key] = float(d[key])
                    except ValueError:
                        pass
            d['coeff'] = []
            while len(d['coeff']) < d['ncoeff']:
                d['coeff'] += polyco.readline().split()

            d['coeff'] = np.array([float(item) for item in d['coeff']])

            if t is None:
                t = Table([[v] for v in d.values()], names=d.keys())
            else:
                t.add_row(d.values())

            line = polyco.readline()

    return t
