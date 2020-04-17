import time
import numpy as np
import gylm
import scipy.special
np.random.seed(7)
log = gylm.log

def assert_equal(z, target, eps=1e-5):
    if np.abs(z-target) > eps: raise ValueError(z)
    else: log << "+" << log.flush

def get_calc(scale):
    return gylm.GylmCalculator(
        rcut=scale*3.5,
        rcut_width=scale*0.5,
        nmax=9,
        lmax=6,
        sigma=scale*0.5,
        part_sigma=scale*0.5,
        wconstant=True,
        wscale=scale*1.0,
        wcentre=1.0,
        ldamp=4.,
        normalize=True,
        types="C,N,O,S,H,F,Cl,Br,I,B,P".split(","),
        periodic=False)

def test_gylm_scaleinv():
    log << log.mg << "<test_scaleinv>" << log.endl
    calc0 = get_calc(scale=1.)
    configs = gylm.io.read('../test_data/structures.xyz')
    for cidx, config in enumerate(configs):
        log << "Struct" << cidx << log.flush
        heavy = np.where(np.array(config.symbols) != "H")[0]
        x0 = calc0.evaluate(system=config, positions=config.positions[heavy])
        pos_orig = np.copy(config.positions)
        for scale in [ 0.5, 1.5, 2.5 ]:
            config.positions = scale*pos_orig
            calc1 = get_calc(scale=scale)
            x1 = calc1.evaluate(
                system=config, 
                positions=config.positions[heavy])
            diff = np.max(np.abs(x0.dot(x0.T) - x1.dot(x1.T)))
            assert_equal(diff, 0.0, 1e-10)
        log << log.endl

def test_gylm_rotinv():
    log << log.mg << "<test_rotinv>" << log.endl
    gylm_calc = gylm.GylmCalculator(
        rcut=4.0,
        rcut_width=0.5,
        nmax=9,
        lmax=6,
        sigma=0.5,
        types="C,N,O,S,H,F,Cl,Br,I,B,P".split(","),
        periodic=False)
    configs = gylm.io.read('../test_data/structures.xyz')
    for cidx, config in enumerate(configs):
        log << "Struct" << cidx << log.flush
        heavy = np.where(np.array(config.symbols) != "H")[0]
        x0 = gylm_calc.evaluate(system=config, positions=config.positions[heavy])
        norm = 1./np.sum(x0**2, axis=1)**0.5
        x0 = (x0.T*norm).T
        for i in range(4):
            R = gylm.tf.get_random_rotation_matrix()
            config.positions = config.positions.dot(R)
            x1 = gylm_calc.evaluate(system=config, positions=config.positions[heavy])
            norm = 1./np.sum(x1**2, axis=1)**0.5
            x1 = (x1.T*norm).T
            diff = np.max(np.abs(x0.dot(x0.T) - x1.dot(x1.T)))
            assert_equal(diff, 0.0, 1e-10)
        log << log.endl

def test_gylm_parity():
    log << log.mg << "<test_parity>" << log.endl
    gylm_calc = gylm.GylmCalculator(
        rcut=4.0,
        rcut_width=0.5,
        nmax=9,
        lmax=6,
        sigma=0.5,
        types="C,N,O,S,H,F,Cl,Br,I,B,P".split(","),
        periodic=False)
    configs = gylm.io.read('../test_data/structures.xyz')
    for cidx, config in enumerate(configs):
        log << "Struct" << cidx << log.flush
        heavy = np.where(np.array(config.symbols) != "H")[0]
        x0 = gylm_calc.evaluate(system=config, positions=config.positions[heavy])
        norm = 1./np.sum(x0**2, axis=1)**0.5
        x0 = (x0.T*norm).T
        pos0 = np.copy(config.positions)
        for i in range(3):
            R = gylm.tf.get_random_rotation_matrix()
            config.positions = pos0.dot(R)
            config.positions[:,i] = - config.positions[:,i]
            x1 = gylm_calc.evaluate(system=config, positions=config.positions[heavy])
            norm = 1./np.sum(x1**2, axis=1)**0.5
            x1 = (x1.T*norm).T
            diff = np.max(np.abs(x0.dot(x0.T) - x1.dot(x1.T)))
            assert_equal(diff, 0.0, 1e-10)
        log << log.endl

if __name__ == "__main__":
    test_gylm_rotinv()
    test_gylm_parity()
    test_gylm_scaleinv()

