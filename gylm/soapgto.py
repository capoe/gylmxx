"""
Adapted and extended from dscribe/descriptors/soap.py:

Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import multiprocessing as mp
from scipy.special import gamma
from scipy.linalg import sqrtm, inv
from gylm._gylm import *
from . import ptable
from . import connectivity

class SoapGtoCalculator(object):
    def __init__(
            self,
            rcut,
            nmax,
            lmax,
            sigma=1.0,
            types=None,
            periodic=False,
            average=False,
            sparse=False,
            normalize=False,
            crossover=True,
            encoder=lambda s: ptable.lookup[s].z,
            decoder=lambda z: ptable.lookup[int(z)].name,
            power=True):
        self.types = types
        self.types_z = np.array(sorted([ encoder(s) for s in self.types ]))
        self.types_elem = np.array([ decoder(z) for z in self.types_z ])
        self._Nt = len(self.types_z)
        self._eta = 1/(2*sigma**2)
        self._sigma = sigma
        self._alphas, self._betas = self.setupBasisGTO(rcut, nmax)
        self._rcut = rcut
        self._nmax = nmax
        self._lmax = lmax
        self.periodic = periodic
        self._average = average
        self._normalize = normalize
        self._crossover = crossover
        self.power = power
    def getDim(self):
        return self.getChannelDim()*self.getNumberOfChannels()
    def getChannelDim(self):
        return self._nmax*(self._nmax+1)/2*(self._lmax+1)
    def getNumberOfChannels(self):
        return self._Nt*(self._Nt+1)/2
    def getNumberofTypes(self):
        return len(self.types_z)
    def evaluate(self, system, positions=None):
        if positions is None:
            positions = system.get_positions()
        threshold = 0.001
        cutoff_padding = self._sigma*np.sqrt(-2*np.log(threshold))
        if np.any(system.pbc) and system.get_cell() is not None:
            system = connectivity.pad_cell_to_cutoff(
                system, self._rcut+cutoff_padding)
        X = self.evaluateGTO(
            system,
            positions,
            self._alphas,
            self._betas,
            rcut=self._rcut,
            cutoff_padding=cutoff_padding,
            nmax=self._nmax,
            lmax=self._lmax,
            eta=self._eta,
            atomic_numbers=None)
        if self._normalize:
            z = 1./np.sum(X**2, axis=1)**0.5
            X = (X.T*z).T
        return X
    def evaluateGTO(self, system, centers, 
            alphas, betas, 
            rcut, cutoff_padding, 
            nmax, lmax, eta, atomic_numbers=None, 
            use_global_types=True):
        n_atoms = len(system)
        positions, Z_sorted, n_types, atomtype_lst = self.flattenPositions(system, atomic_numbers)
        centers = np.array(centers)
        n_centers = centers.shape[0]
        centers = centers.flatten()
        alphas = alphas.flatten()
        betas = betas.flatten()
        Z_sorted_global = self.types_z if use_global_types \
            else np.array(list(set(Z_sorted)))
        n_types = len(Z_sorted_global)

        # >>> dim = int((nmax*(nmax+1))/2)*(lmax+1)*int((n_types*(n_types + 1))/2)
        # >>> c = np.zeros(dim*n_centers, dtype=np.float64)
        # >>> shape = (n_centers, dim)

        if self.power:
            if self._crossover:
                dim = nmax*nmax*(lmax+1)*int((n_types*(n_types + 1))/2)
            else:
                dim = int(nmax*(nmax+1)/2*(lmax+1)*n_types)
        else:
            dim = n_types*nmax*(lmax+1)**2
        c = np.zeros(dim*n_centers, dtype=np.float64)
        shape = (n_centers, dim)

        evaluate_soapgto(c, positions, centers, 
            alphas, betas, Z_sorted, Z_sorted_global,
            rcut, cutoff_padding, 
            n_atoms, n_types, 
            nmax, lmax, n_centers, eta, self._crossover, 
            self.power)
        c = c.reshape(shape)

        # TODO Check rotation invariance
        # >>> if not self.power:
        # >>>     dim = nmax*nmax*(lmax+1)*int((n_types*(n_types + 1))/2)
        # >>>     X = np.zeros((n_centers, dim))
        # >>>     evaluate_power(
        # >>>         X, c, n_centers, n_types, nmax, lmax)
        # >>>     print(X)
        # >>>     print(X.dot(X.T))
        # >>> print(self.power)
        # >>> print(c)

        return c
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
    def setupBasisGTO(self, rcut, nmax):
        # These are the values for where the different basis functions should decay
        # to: evenly space between 1 angstrom and rcut.
        a = np.linspace(1, rcut, nmax)
        threshold = 1e-3  # This is the fixed gaussian decay threshold
        alphas_full = np.zeros((10, nmax))
        betas_full = np.zeros((10, nmax, nmax))
        for l in range(0, 10):
            # The alphas are calculated so that the GTOs will decay to the set
            # threshold value at their respective cutoffs
            alphas = -np.log(threshold/np.power(a, l))/a**2
            # Calculate the overlap matrix
            m = np.zeros((alphas.shape[0], alphas.shape[0]))
            m[:, :] = alphas
            m = m + m.transpose()
            S = 0.5*gamma(l + 3.0/2.0)*m**(-l-3.0/2.0)
            # Get the beta factors that orthonormalize the set with Loewdin
            # orthonormalization
            betas = sqrtm(inv(S))
            # If the result is complex, the calculation is currently halted.
            if (betas.dtype == np.complex128):
                raise ValueError(
                    "Could not calculate normalization factors for the radial "
                    "basis in the domain of real numbers. Lowering the number of "
                    "radial basis functions (nmax) or increasing the radial "
                    "cutoff (rcut) is advised."
                )
            alphas_full[l, :] = alphas
            betas_full[l, :, :] = betas
        return alphas_full, betas_full

