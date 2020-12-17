try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

def check_available():
    return jax is not None

def ylm(x, y, z):
    lmax = 7
    dim = (lmax+1)*(lmax+1)
    epsilon = 1e-10
    # ylm_out = jnp.zeros((x.shape[0], dim)) # NOTE Any point preallocating the output?

    radius_eps  = +1.00000000000e-10  # constexpr double radius_eps = 1e-10
    pi          = +3.14159265359e+00  # constexpr double pi = 3.141592653589793
    invs_pi     = +5.64189583548e-01  # constexpr double invs_pi = sqrt(1./pi)
    invs_2      = +7.07106781187e-01  # constexpr double invs_2 = sqrt(0.5)
    s_2         = +1.41421356237e+00  # constexpr double s_2 = sqrt(2.)
    s_3         = +1.73205080757e+00  # constexpr double s_3 = sqrt(3.)
    s_5         = +2.23606797750e+00  # constexpr double s_5 = sqrt(5.)
    s_7         = +2.64575131106e+00  # constexpr double s_7 = sqrt(7.)
    s_11        = +3.31662479036e+00  # constexpr double s_11 = sqrt(11.)
    s_13        = +3.60555127546e+00  # constexpr double s_13 = sqrt(13.)
    pre1        = +3.45494149471e-01  # constexpr double pre1 = invs_2*0.5*s_3*invs_pi
    pre1_a      = +2.44301255951e-01  # constexpr double pre1_a = invs_2*pre1
    pre2        = +2.23015514519e-01  # constexpr double pre2 = invs_2* 0.25*s_5*invs_pi
    pre2_a      = +5.46274215296e-01  # constexpr double pre2_a = 2*pre2*s_3*invs_2
    pre2_b      = +2.73137107648e-01  # constexpr double pre2_b = pre2*s_3*invs_2
    pre3        = +2.63875515353e-01  # constexpr double pre3 = invs_2* 0.25*invs_pi*s_7
    pre3_a      = +2.28522899732e-01  # constexpr double pre3_a = 0.5*pre3*s_3
    pre3_b      = +7.22652860660e-01  # constexpr double pre3_b = pre3*s_3*s_5*invs_2
    pre3_c      = +2.95021794963e-01  # constexpr double pre3_c = 0.5*pre3*s_5
    pre4        = +7.48016775753e-02  # constexpr double pre4 = invs_2* 0.1875*invs_pi
    pre4_a      = +3.34523271779e-01  # constexpr double pre4_a = 2*pre4*s_5
    pre4_b      = +2.36543673939e-01  # constexpr double pre4_b = pre4_a*invs_2
    pre4_c      = +8.85065384890e-01  # constexpr double pre4_c = pre4_a*s_7
    pre4_d      = +3.12917867725e-01  # constexpr double pre4_d = pre4_b*0.5*s_7
    pre5        = +8.26963660688e-02  # constexpr double pre5 = invs_2* 0.0625*invs_pi*s_11
    pre5_a      = +2.26473325598e-01  # constexpr double pre5_a = pre5*s_3*s_5*invs_2
    pre5_b      = +1.19838419624e+00  # constexpr double pre5_b = pre5_a*s_7*2
    pre5_c      = +2.44619149718e-01  # constexpr double pre5_c = pre5*0.5*s_7*s_5
    pre5_d      = +1.03783115744e+00  # constexpr double pre5_d = pre5_c*6*invs_2
    pre5_e      = +3.28191028420e-01  # constexpr double pre5_e = pre5*1.5*s_7
    pre6        = +4.49502139981e-02  # constexpr double pre6 = invs_2* 0.03125*invs_pi*s_13
    pre6_a      = +2.91310681259e-01  # constexpr double pre6_a = pre6*s_7*s_3*invs_2*2
    pre6_b      = +2.30301314879e-01  # constexpr double pre6_b = pre6_a*s_5*0.25*s_2
    pre6_c      = +4.60602629757e-01  # constexpr double pre6_c = pre6_b*2
    pre6_d      = +2.52282450364e-01  # constexpr double pre6_d = pre6*s_7*invs_2*3
    pre6_e      = +1.18330958112e+00  # constexpr double pre6_e = pre6_d*s_11*s_2
    pre6_f      = +3.41592052596e-01  # constexpr double pre6_f = pre6_e*s_3/6.
    pre7        = +4.82842752529e-02  # constexpr double pre7   = invs_2* 1./32*s_5*s_3*invs_pi
    pre7_a      = +4.51658037913e-02  # constexpr double pre7_a = 0.5*pre7*invs_2*s_7
    pre7_b      = +1.10633173111e-01  # constexpr double pre7_b = 3*pre7_a/s_3*s_2
    pre7_c      = +7.82294669311e-02  # constexpr double pre7_c = pre7_b*invs_2
    pre7_d      = +5.18915578720e-01  # constexpr double pre7_d = 2*pre7_c*s_11
    pre7_e      = +2.59457789360e-01  # constexpr double pre7_e = 0.5*pre7_d
    pre7_f      = +1.32298033090e+00  # constexpr double pre7_f = pre7_e*s_13*s_2
    pre7_g      = +3.53581366262e-01  # constexpr double pre7_g = pre7_f/s_2/s_7

    # l = 0
    r = jnp.sqrt(x**2 + y**2 + z**2) + epsilon
    zr = z/r
    xr = x/r
    yr = y/r
    # l = 1
    a1r = xr
    a1i = -yr
    c1r = zr
    b1r = xr
    b1i = yr
    # l = 2
    a2r = a1r*a1r - a1i*a1i
    a2i = 2*a1r*a1i
    c2r = c1r*c1r
    c2r_a = 3*c2r-1.
    b2r = b1r*b1r - b1i*b1i
    b2i = 2*b1r*b1i
    # l = 3
    a3r = a2r*a1r - a2i*a1i
    a3i = a2r*a1i + a2i*a1r
    c3r = c2r*c1r
    c3r_a = 5*c2r-1.
    c3r_b = 5*c3r-3*c1r
    b3r = b2r*b1r - b2i*b1i
    b3i = b2r*b1i + b2i*b1r
    # l = 4
    a4r = a2r*a2r - a2i*a2i
    a4i = 2*a2r*a2i
    c4r = c2r*c2r
    c4r_0 = 35.*c4r - 30*c2r + 3.
    c4r_1 = 7*c3r - 3*c1r
    c4r_2 = 7*c2r - 1.
    b4r = b2r*b2r - b2i*b2i
    b4i = 2*b2r*b2i
    # l = 5
    a5r = a4r*a1r - a4i*a1i
    a5i = a4r*a1i + a4i*a1r
    c5r = c4r*c1r
    c5r_0 = 63*c5r - 70*c3r + 15*c1r
    c5r_1 = 21*c4r - 14*c2r + 1
    c5r_2 = 3*c3r - c1r
    c5r_3 = 9*c2r - 1
    b5r = b4r*b1r - b4i*b1i
    b5i = b4r*b1i + b4i*b1r
    # l = 6
    a6r = a3r*a3r - a3i*a3i
    a6i = 2*a3r*a3i
    c6r = c3r*c3r
    c6r_0 = 231*c6r - 315*c4r + 105*c2r - 5
    c6r_1 = 33*c5r - 30*c3r + 5*c1r
    c6r_2 = 33*c4r - 18*c2r + 1
    c6r_3 = 11*c3r - 3*c1r
    c6r_4 = 11*c2r - 1
    b6r = b3r*b3r - b3i*b3i
    b6i = 2*b3r*b3i
    # l = 7
    a7r = a4r*a3r - a4i*a3i
    a7i = a4r*a3i + a4i*a3r
    c7r = c4r*c3r
    c7r_0 = 429*c7r - 693*c5r + 315*c3r - 35*c1r
    c7r_1 = 429*c6r - 495*c4r + 135*c2r - 5
    c7r_2 = 143*c5r - 110*c3r + 15*c1r
    c7r_3 = 143*c4r - 66*c2r + 3
    c7r_4 = 13*c3r - 3*c1r
    c7r_5 = 13*c2r - 1
    b7r = b4r*b3r - b4i*b3i
    b7i = b4r*b3i + b4i*b3r

    ylm_out = jnp.concatenate([
         jnp.tile(0.5*invs_pi, (x.shape[0],)),
        -pre1_a* (a1i-b1i),
         s_2*pre1*c1r,
         pre1_a* (a1r+b1r),
        -pre2_b*  (a2i-b2i),
        -pre2_a*  (a1i - b1i)*c1r,
         s_2*pre2*c2r_a,
         pre2_a*  (a1r + b1r)*c1r,
         pre2_b*  (a2r+b2r),
        -pre3_c* (a3i-b3i),
        -pre3_b* (a2i-b2i)*c1r,
        -pre3_a* (a1i-b1i)*c3r_a,
         s_2*pre3*c3r_b,
         pre3_a* (a1r+b1r)*c3r_a,
         pre3_b* (a2r+b2r)*c1r,
         pre3_c* (a3r+b3r),
        -pre4_d* (a4i-b4i),
        -pre4_c* (a3i-b3i)*c1r,
        -pre4_b* (a2i-b2i)*c4r_2,
        -pre4_a* (a1i-b1i)*c4r_1,
         s_2*pre4*c4r_0,
         pre4_a* (a1r+b1r)*c4r_1,
         pre4_b* (a2r+b2r)*c4r_2,
         pre4_c* (a3r+b3r)*c1r,
         pre4_d* (a4r+b4r),
        -pre5_e* (a5i-b5i),
        -pre5_d* (a4i-b4i)*c1r,
        -pre5_c* (a3i-b3i)*c5r_3,
        -pre5_b* (a2i-b2i)*c5r_2,
        -pre5_a* (a1i-b1i)*c5r_1,
         s_2*pre5*c5r_0,
         pre5_a* (a1r+b1r)*c5r_1,
         pre5_b* (a2r+b2r)*c5r_2,
         pre5_c* (a3r+b3r)*c5r_3,
         pre5_d* (a4r+b4r)*c1r,
         pre5_e* (a5r+b5r),
        -pre6_f* (a6i-b6i),
        -pre6_e* (a5i-b5i)*c1r,
        -pre6_d* (a4i-b4i)*c6r_4,
        -pre6_c* (a3i-b3i)*c6r_3,
        -pre6_b* (a2i-b2i)*c6r_2,
        -pre6_a* (a1i-b1i)*c6r_1,
         s_2*pre6*c6r_0,
         pre6_a* (a1r+b1r)*c6r_1,
         pre6_b* (a2r+b2r)*c6r_2,
         pre6_c* (a3r+b3r)*c6r_3,
         pre6_d* (a4r+b4r)*c6r_4,
         pre6_e* (a5r+b5r)*c1r,
         pre6_f* (a6r+b6r),
        -pre7_g* (a7i-b7i),
        -pre7_f* (a6i-b6i)*c1r,
        -pre7_e* (a5i-b5i)*c7r_5,
        -pre7_d* (a4i-b4i)*c7r_4,
        -pre7_c* (a3i-b3i)*c7r_3,
        -pre7_b* (a2i-b2i)*c7r_2,
        -pre7_a* (a1i-b1i)*c7r_1,
         s_2*pre7*c7r_0,
         pre7_a* (a1r+b1r)*c7r_1,
         pre7_b* (a2r+b2r)*c7r_2,
         pre7_c* (a3r+b3r)*c7r_3,
         pre7_d* (a4r+b4r)*c7r_4,
         pre7_e* (a5r+b5r)*c7r_5,
         pre7_f* (a6r+b6r)*c1r,
         pre7_g* (a7r+b7r),
      ]).reshape((-1, x.shape[0])).T
    return ylm_out
