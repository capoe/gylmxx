#! /usr/bin/env python
import numpy as np
import json
import os
from . import ptable
from .logger import log, Mock
try:
    import ase.io
except ImportError:
    ase = Mock()
    ase.io = None

class ExtendedXyz(object):
    def __init__(self):
        self.info = {}
        self.cell = None
        self.pbc = np.array([False, False, False])
        self.atoms = []
        self.positions = []
        self.symbols = []
    def __len__(self):
        return len(self.symbols)
    def set_pbc(self, booleans):
        self.pbc = np.array(booleans)
    def get_positions(self):
        return self.positions
    def get_chemical_symbols(self):
        return self.symbols
    def get_atomic_numbers(self):
        return np.array([ ptable.lookup[s].z for s in self.get_chemical_symbols() ])
    def get_cell(self):
        if self.cell is None and "Lattice" in self.info:
            self.cell = np.array(list(
                map(float, self.info["Lattice"].split()))).reshape((3,3))
            self.set_pbc()
        return self.cell
    def padToCutoff(self, r_cut):
        cell = np.array(self.get_cell())
        if cell is None: return self
        # Calculate # replicates
        u, v, w = cell[0], cell[1], cell[2]
        a = np.cross(v, w, axis=0)
        b = np.cross(w, u, axis=0)
        c = np.cross(u, v, axis=0)
        ua = np.dot(u, a) / np.dot(a,a) * a
        vb = np.dot(v, b) / np.dot(b,b) * b
        wc = np.dot(w, c) / np.dot(c,c) * c
        proj = np.linalg.norm(np.array([ua, vb, wc]), axis=1)
        nkl = np.ceil(r_cut/proj).astype('int')
        # Replicate
        n_atoms = len(self)
        n_images = np.product(2*nkl + 1) 
        positions_padded = np.tile(self.positions, (n_images, 1))
        offset = 0
        for i in np.append(np.arange(0, nkl[0]+1), np.arange(-nkl[0], 0)):
            for j in np.append(np.arange(0, nkl[1]+1), np.arange(-nkl[1], 0)):
                for k in np.append(np.arange(0, nkl[2]+1), np.arange(-nkl[2], 0)):
                    ijk = np.array([i,j,k])
                    positions_padded[offset:offset+n_atoms] += np.sum((cell.T*ijk).T, axis=0)
                    offset += n_atoms
        symbols_padded = np.tile(np.array(self.symbols), n_images)
        config_padded = self.__class__(
            positions=positions_padded,
            symbols=symbols_padded,
            cell=nkl*cell)
        return config_padded
    def create(self, n_atoms, fs):
        # Parse header: key1="str1" key2=123 key3="another value" ...
        header = fs.readline().replace("\n", "")
        tokens = []
        pos0 = 0
        pos1 = 0
        status = "<"
        quotcount = 0
        while pos1 < len(header):
            status_out = status
            # On the lhs of the key-value pair?
            if status == "<":
                if header[pos1] == "=":
                    tokens.append(header[pos0:pos1])
                    pos0 = pos1+1
                    pos1 = pos1+1
                    status_out = ">"
                    quotcount = 0
                else:
                    pos1 += 1
            # On the rhs of the key-value pair?
            elif status == ">":
                if header[pos1-1:pos1] == '"':
                    quotcount += 1
                if quotcount == 0 and header[pos1] == ' ':
                    quotcount = 2
                if quotcount <= 1:
                    pos1 += 1
                elif quotcount == 2:
                    tokens.append(header[pos0:pos1])
                    pos0 = pos1+1
                    pos1 = pos1+1
                    status_out = ""
                    quotcount = 0
                else:
                    assert False
            # In between key-value pairs?
            elif status == "":
                if header[pos1] == ' ':
                    pos0 += 1
                    pos1 += 1
                else:
                    status_out = "<"
            else:
                assert False
            status = status_out
        kvs = []
        for i in range(len(tokens)//2):
            kvs.append([tokens[2*i], tokens[2*i+1]])
        # Process key-value pairs
        for kv in kvs:
            key = kv[0]
            value = '='.join(kv[1:])
            value = value.replace('"','').replace('\'','')
            # Float?
            if '.' in value:
                try:
                    value = float(value)
                except: pass
            else:
                # Int?
                try:
                    value = int(value)
                except: pass
            self.info[kv[0]] = value
        # Read atoms
        self.positions = []
        self.symbols = []
        for i in range(n_atoms):
            new_atom = self.create_atom(fs.readline())
            self.positions.append(new_atom.pos)
            self.symbols.append(new_atom.name)
        self.positions = np.array(self.positions)
        return
    def create_atom(self, ln):
        ln = ln.split()
        name = ln[0]
        pos = list(map(float, ln[1:4]))
        pos = np.array(pos)
        new_atom = ExtendedXyzAtom(name, pos)
        self.atoms.append(new_atom)
        return new_atom

class ExtendedXyzAtom(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

def read(
        config_file,
        index=':'):
    if ase.io is not None: 
        return ase.io.read(config_file, index)
    configs = []
    ifs = open(config_file, 'r')
    while True:
        header = ifs.readline().split()
        if header != []:
            assert len(header) == 1
            n_atoms = int(header[0])
            config = ExtendedXyz()
            config.create(n_atoms, ifs)
            configs.append(config)
        else: break
    return configs

def write(
        config_file,
        configs):
    if type(configs) != list:
        configs = [ configs ]
    if ase.io is not None:
        return ase.io.write(config_file, configs)
    ofs = open(config_file, 'w')
    for c in configs:
        ofs.write('%d\n' % (len(c)))
        for k in sorted(c.info.keys()):
            # int or float?
            if type(c.info[k]) not in { str }:
                ofs.write('%s=%s ' % (k, c.info[k]))
            # String
            else:
                ofs.write('%s="%s" ' % (k, c.info[k]))
        ofs.write('\n')
        for i in range(len(c)):
            ofs.write('%s %+1.4f %+1.4f %+1.4f\n' % (
                c.get_chemical_symbols()[i], 
                c.positions[i][0], 
                c.positions[i][1], 
                c.positions[i][2]))
    ofs.close()
    return

