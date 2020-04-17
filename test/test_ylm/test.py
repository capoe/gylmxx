import time
import numpy as np
import gylm
import scipy.special
np.random.seed(7)
log = gylm.log

def assert_equal(z, target, eps=1e-5):
    if np.abs(z-target) > eps: raise ValueError(z)
    else: log << "+" << log.flush

def test_ylm(lmax=7):
    log << log.mg << "<test_ylm>" << log.endl
    xyz_orig = np.random.normal(0, 1., size=(3,))
    xyz = xyz_orig/np.dot(xyz_orig,xyz_orig)**0.5
    # Reference via scipy
    theta = np.arccos(xyz[2])
    phi = np.arctan2(xyz[1], xyz[0])
    if phi < 0.: phi += 2*np.pi
    ylm_ref = []
    ylm_ref_re = []
    for l in range(lmax+1):
        ylm_ref_l = []
        ylm_ref_re_l = []
        for m in range(-l, l+1):
            ylm_ref_l.append(scipy.special.sph_harm(m, l, phi, theta))
            ylm_ref_re_l.append(0)
        reco = 1./2**0.5
        imco = np.complex(0,1)/2**0.5
        ylm_ref_re_l[l] = ylm_ref_l[l].real
        for m in range(1, l+1):
            qlm = ylm_ref_l[l+m]
            ql_m = ylm_ref_l[l-m]
            s = (imco*(ql_m - (-1)**m*qlm)).real
            r = (reco*(ql_m + (-1)**m*qlm)).real
            ylm_ref_re_l[l-m] = s
            ylm_ref_re_l[l+m] = r
        ylm_ref.extend(ylm_ref_l)
        ylm_ref_re.extend(ylm_ref_re_l)
    ylm_ref = np.array(ylm_ref) 
    ylm_ref_re = np.array(ylm_ref_re)
    # soap::ylm
    S = 1000
    ylm_out = np.zeros((S*(lmax+1)**2,))
    xyz = np.tile(xyz_orig, (S,1))
    x = np.copy(xyz[:,0])
    y = np.copy(xyz[:,1])
    z = np.copy(xyz[:,2])
    t0 = time.time()
    gylm.ylm(x, y, z, x.shape[0], lmax, ylm_out)
    t1 = time.time()
    log << "delta_t =" << t1-t0 << log.endl
    for j in np.random.randint(0, S, size=(3,)):
        ll = (lmax+1)**2
        dy = ylm_out[j*ll:(j+1)*ll] - ylm_ref_re
        for l in range(lmax+1):
            log << "l=%d" % l << log.flush
            for m in range(-l,l+1):
                lm = l**2+l+m
                assert_equal(np.abs(dy[lm]), 0.0, 1e-7)
            log << log.endl
        
if __name__ == "__main__":
    test_ylm()

