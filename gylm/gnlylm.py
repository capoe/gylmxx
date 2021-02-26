import numpy as np
import multiprocessing as mp
from scipy.special import gamma
from scipy.linalg import sqrtm, inv
from gylm._gylm import *
from . import ptable
from . import connectivity

def eval_single(args):
    return args["calc"].evaluate(**args)

class GylmCalculator(object):
    def __init__(
            self,
            rcut,
            rcut_width,
            nmax,
            lmax,
            rmin=0.0,
            sigma=1.0,
            part_sigma=0.5,
            wconstant=True,
            wcentre=1.,
            wscale=1.,
            ldamp=4.,
            types=None,
            periodic=False,
            normalize=False,
            power=True,
            encoder=lambda s: ptable.lookup[s].z,
            decoder=lambda z: ptable.lookup[int(z)].name):
        self.types = types
        self.types_z = np.array(sorted([ encoder(s) for s in self.types ]))
        self.types_elem = np.array([ decoder(z) for z in self.types_z ])
        self.types_set = set(list(self.types_z))
        self._Nt = len(self.types_z)
        self._eta = 1/(2*sigma**2)
        self._sigma = sigma
        self._gnl_centres, self._gnl_alphas = self.setupBasisGylm(rmin, rcut, sigma, nmax)
        self._rmin = rmin
        self._rcut = rcut
        self._rcut_width = rcut_width
        self._nmax = nmax
        self._lmax = lmax
        self.part_sigma = part_sigma
        self.wconstant = wconstant
        self.wcentre = wcentre
        self.wscale = wscale
        self.ldamp = ldamp
        self.periodic = periodic
        self.normalize = normalize
        self.power = power
        self.epsilon = 1e-10
    def getDim(self, with_power=None):
        if with_power is None: with_power = self.power
        return self.getChannelDim(with_power)*self.getNumberOfChannels(with_power)
    def getChannelDim(self, with_power):
        if with_power:
            return self._nmax*self._nmax*(self._lmax+1)
        else:
            return self._nmax*(self._lmax+1)**2
    def getNumberOfChannels(self, with_power):
        if with_power:
            return self._Nt*(self._Nt+1)/2
        else:
            return self._Nt
    def getNumberofTypes(self):
        return len(self.types_z)
    def evaluate_mp(self, systems, positions=None,
            power=True,
            normalize=True,
            verbose=False,
            procs=1):
        args = [ {
            "calc": self,
            "system": systems[i],
            "positions": positions[i] if positions != None else None,
            "power": power,
            "normalize": normalize,
            "verbose": verbose } \
            for i in range(len(systems)) ]
        pool = mp.Pool(processes=procs)
        X_list = pool.map(eval_single, args)
        pool.close()
        return X_list
    def evaluate(self, system, positions=None,
            verbose=False,
            calc=None):
        if positions is None:
            positions = system.get_positions()
        if system.get_cell() is not None:
            system = connectivity.pad_cell_to_cutoff(system, self._rcut)
        X = self.evaluateGylm(
            system,
            positions,
            self._gnl_centres,
            self._gnl_alphas,
            rcut=self._rcut,
            cutoff_padding=self._rcut_width,
            nmax=self._nmax,
            lmax=self._lmax,
            eta=self._eta,
            atomic_numbers=None,
            power=self.power,
            verbose=verbose)
        if self.normalize:
            z = 1./(np.sum(X**2, axis=1)+self.epsilon)**0.5
            X = (X.T*z).T
        return X
    def evaluateGylm(self, system, centers,
            gnl_centres, gnl_alphas,
            rcut, cutoff_padding,
            nmax, lmax, eta, atomic_numbers=None,
            use_global_types=True, power=True, verbose=False):
        n_tgt = len(system)
        n_src = len(centers)
        positions, Z_sorted, n_types, atomtype_lst = self.flattenPositions(
            system, atomic_numbers)
        if len(set(Z_sorted).union(self.types_set)) > len(self.types_set):
            raise ValueError("Some types not recognized:", set(Z_sorted))
        centers = np.array(centers)
        n_centers = centers.shape[0]
        centers = centers.flatten()
        Z_sorted_global = self.types_z if use_global_types \
            else np.array(list(set(Z_sorted)))
        n_types = len(Z_sorted_global)
        if power:
            coeffs = np.zeros(nmax*nmax*(lmax+1)*int((n_types*(n_types + 1))/2)*n_src, dtype=np.float64)
            shape = (n_centers, nmax*nmax*(lmax+1)*int((n_types*(n_types+1))/2))
        else:
            coeffs = np.zeros(nmax*(lmax+1)*(lmax+1)*n_types*n_src, dtype=np.float64)
            shape = (n_centers, nmax*(lmax+1)*(lmax+1)*n_types)
        evaluate_gylm(coeffs, centers, positions,
            gnl_centres, gnl_alphas, Z_sorted, Z_sorted_global,
            rcut, cutoff_padding,
            n_src, n_tgt, n_types,
            nmax, lmax,
            self.part_sigma,
            self.wconstant,
            self.wscale,
            self.wcentre,
            self.ldamp,
            power, verbose)
        coeffs = coeffs.reshape(shape)
        return coeffs
    def flattenPositions(self, system, atomic_numbers=None):
        Z = system.get_atomic_numbers()
        pos = system.get_positions()
        if atomic_numbers is not None:
            atomtype_set = set(atomic_numbers)
        else:
            atomtype_set = set(Z)
        atomic_numbers_sorted = np.sort(list(atomtype_set))
        pos_lst = []
        z_lst = []
        for atomtype in atomic_numbers_sorted:
            condition = (Z == atomtype)
            pos_onetype = pos[condition]
            z_onetype = Z[condition]
            pos_lst.append(pos_onetype)
            z_lst.append(z_onetype)
        n_types = len(atomic_numbers_sorted)
        positions_sorted = np.concatenate(pos_lst, axis=0)
        atomic_numbers_sorted = np.concatenate(z_lst).ravel()
        return positions_sorted, atomic_numbers_sorted, n_types, atomic_numbers_sorted
    def setupBasisGylm(self, rmin, rcut, sigma, nmax):
        centres = np.linspace(rmin, rcut, nmax);
        alphas = np.ones_like(centres)/(2.*sigma**2)
        return centres, alphas
