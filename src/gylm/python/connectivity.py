import numpy as np

def covalent_cutoff(rad1, rad2):
    return 1.15*(rad1+rad2)

def calculate_dmat(R, P):
    return np.sqrt(np.subtract.outer(
        R[:,0], P[:,0])**2 + np.subtract.outer(
        R[:,1], P[:,1])**2 + np.subtract.outer(
        R[:,2], P[:,2])**2)

def calculate_lmat(
        distance_mat, 
        type_vec, 
        cutoff=None, 
        cutoff_scale=1.15):
    dim = distance_mat.shape[0]
    if cutoff != None:
        connectivity_mat = np.zeros((dim,dim), dtype=bool)
        for i in range(dim):
            for j in range(dim):
                if distance_mat[i,j] < constant:
                    connectivity_mat[i,j] = True
                else:
                    connectivity_mat[i,j] = False
    else:
        cr = np.array([ COVRAD_TABLE[t] for t in type_vec ])
        rrcut = cutoff_scale*np.add.outer(cr,cr)
        connectivity_mat = (np.heaviside(-distance_mat+rrcut, 0)).astype(bool)
    return connectivity_mat

def pad_cell_to_cutoff(config, r_cut):
    cell = np.array(config.get_cell())
    if cell is None: return config
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
    n_atoms = len(config)
    n_images = np.product(2*nkl + 1) 
    positions_padded = np.tile(config.positions, (n_images, 1))
    offset = 0
    for i in np.append(np.arange(0, nkl[0]+1), np.arange(-nkl[0], 0)):
        for j in np.append(np.arange(0, nkl[1]+1), np.arange(-nkl[1], 0)):
            for k in np.append(np.arange(0, nkl[2]+1), np.arange(-nkl[2], 0)):
                ijk = np.array([i,j,k])
                positions_padded[offset:offset+n_atoms] += np.sum((cell.T*ijk).T, axis=0)
                offset += n_atoms
    symbols_padded = np.tile(np.array(config.symbols), n_images)
    config_padded = config.__class__(
        positions=positions_padded,
        symbols=symbols_padded)
    return config_padded

# COVALENT RADII (from Cambridge Structural Database
# (see http://en.wikipedia.org/wiki/Covalent_radius)
COVRAD_TABLE = {}
COVRAD_TABLE['H'] = 0.31
COVRAD_TABLE['B'] = 0.85
COVRAD_TABLE['C'] = 0.76
COVRAD_TABLE['N'] = 0.71
COVRAD_TABLE['F'] = 0.57
COVRAD_TABLE['O'] = 0.66
COVRAD_TABLE['S'] = 1.05
COVRAD_TABLE['Br'] = 1.20
COVRAD_TABLE['Cl'] = 1.02
COVRAD_TABLE['I'] = 1.39
COVRAD_TABLE['P'] = 1.07
COVRAD_TABLE['Si'] = 1.11
COVRAD_TABLE['Rh'] = 1.35
COVRAD_TABLE['Fe'] = 1.25

