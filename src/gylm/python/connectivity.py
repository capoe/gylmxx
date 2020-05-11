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

