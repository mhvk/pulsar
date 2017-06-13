import numpy as np
from numpy.polynomial.polynomial import Polynomial
from astropy.coordinates.angles import Angle
import astropy.units as u


class Ephemeris(dict):
    """Empheris for ELL1 model"""

    def __init__(self, name):
        d, e, f = par2dict(name)
        # make dictionary
        dict.__init__(self, d)
        self.err = e
        self.fix = f

    def evaluate(self, par, mjd, t0par=None, integrate=False):
        """Evaluate 'par' at given MJD(s), with zero point t0par

        Parameters
        ----------
        par : string
            key into dictionary = parameter to evaluate.  Takes into
            account possible par+'DOT' (for 'F': F0, F1, F2, F3)
        mjd : float or Time object
            MJD at which to evaluate (or Time with .tdb.mjd attribute)
        t0par : string or None
            key into dictionary with zero point for par
            default: None -> PEPOCH for 'F', 'TASC' for all others
        integrate : bool
            Whether to integrate the polynomial (e.g., to get mean
            anomaly out of 'FB' or pulse phase out of 'F')
        """
        if par == 'F':
            t0par = t0par or 'PEPOCH'
            parpol = Polynomial((self['F0'], self.get('F1', 0.),
                                 self.get('F2',0.), self.get('F3',0.)))
        else:
            t0par = t0par or 'TASC'
            parpol = Polynomial((self[par], self.get(par+'DOT', 0.)))

        if integrate:
            parpol = parpol.integ()

        # given time can be Time object
        if hasattr(mjd, 'tdb'):
            mjd = mjd.tdb.mjd

        return parpol((mjd - self[t0par]) * 86400.)

    def pos(self, mjd):
        """Position including proper motion (to linear order, bad near pole)"""
        ra = np.deg2rad(self.evaluate('RAJ', mjd, 'POSEPOCH'))
        dec = np.deg2rad(self.evaluate('DECJ', mjd, 'POSEPOCH'))
        ca = np.cos(ra)
        sa = np.sin(ra)
        cd = np.cos(dec)
        sd = np.sin(dec)
        return np.array([ca*cd, sa*cd, sd])


class ELL1Ephemeris(Ephemeris):
    """Ephemeris for ELL1 model"""

    def mean_anomaly(self, mjd):
        return 2.*np.pi*self.evaluate('FB', mjd, integrate=True)

    def orbital_delay(self, mjd):
        """Delay in s.  Includes higher order terms and Shapiro delay."""
        ma = self.mean_anomaly(mjd)
        an = 2.*np.pi*self.evaluate('FB', mjd)
        a1, e1, e2 = self['A1'], self['EPS1'], self['EPS2']
        dre = a1*(np.sin(ma)-0.5*(e1*np.cos(2*ma)-e2*np.sin(2*ma)))
        drep = a1*np.cos(ma)
        drepp = -a1*np.sin(ma)
        d2bar = dre*(1-an*drep+(an*drep)**2+0.5*an**2*dre*drepp)
        if 'M2' in self:
            brace = 1.-self['SINI']*np.sin(ma)
            d2bar += -2.*self['M2']*np.log(brace)
        return d2bar

    def radial_velocity(self, mjd):
        """Radial velocity in lt-s/s.  Higher-order terms ignored."""
        ma = self.mean_anomaly(mjd)
        kcirc = 2.*np.pi*self['A1']*self.evaluate('FB', mjd)
        e1, e2 = self['EPS1'], self['EPS2']
        vrad = kcirc*(np.cos(ma)+e1*np.sin(2*ma)+e2*np.cos(2*ma))
        return vrad


def par2dict(name, substitutions={'DM1': 'DMDOT',
                                  'PMRA': 'RAJDOT', 'PMDEC': 'DECJDOT'}):
    """Read in a TEMPO .par file and convert to a dictionary.

    Parameters
    ----------
    name : str
       filename
    substitutions: dict
       dictionary of name substitutions

    Returns
    -------
    d, e, f: dict
       dictionaries with data, uncertainties, and fixed-flags for
       each of the parameters in the tempo file.

    Notes
    -----
    Where possible, values listed are converted to numbers.  E.g.,
    RAJ, DECJ are converted to degrees, PMRA, PMDEC to degrees/s.
    """

    d = {}
    e = {}
    f = {}
    with open(name, 'r') as parfile:
        for lin in parfile:
            if lin[0] == '#':
                continue
            parts = lin.split()
            item = parts[0].upper()
            item = substitutions.get(item, item)
            assert 2 <= len(parts) <= 4
            try:
                value = float(parts[1].lower().replace('d', 'e'))
                d[item] = value
            except ValueError:
                d[item] = parts[1]
            if len(parts) > 2:
                # for numbers last item is the corresponding uncertainty
                e[item] = float(parts[-1].lower().replace('d', 'e'))
                # for tempo output, middle number is whether parameter was
                # fixed or not; for ATNF PSRCAT, this is absent
                f[item] = int(parts[2]) if len(parts) == 4 else 0

        # convert RA, DEC from strings (hh:mm:ss.sss, ddd:mm:ss.ss) to degrees
        d['RAJ'] = Angle(d['RAJ'], u.hr).degree
        e['RAJ'] = e['RAJ']/15./3600.
        d['DECJ'] = Angle(d['DECJ'], u.deg).degree
        e['DECJ'] = e['DECJ']/3600.
        if 'RAJDOT' in d and 'DECJDOT' in d:
            # convert to degrees/s
            conv = (1.*u.mas/u.yr).to(u.deg/u.s).value
            cosdec = np.cos((d['DECJ']*u.deg).to(u.rad).value)
            d['RAJDOT'] *= conv/cosdec
            e['RAJDOT'] *= conv/cosdec
            d['DECJDOT'] *= conv
            e['DECJDOT'] *= conv

        if 'FB' not in d and 'PB' in d:
            pb = d.pop('PB')
            d['FB'] = 1./(pb*24.*3600.)
            e['FB'] = e.pop('PB')/pb*d['FB']
            f['FB'] = f.pop('PB')
            if 'PBDOT' in d:
                d['FBDOT'] = -d.pop('PBDOT')/pb*d['FB']
                e['FBDOT'] = e.pop('PBDOT')/pb*d['FB']
                f['FBDOT'] = f.pop('PBDOT')

    return d, e, f
