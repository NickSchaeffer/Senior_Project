import numpy as np
from scipy.optimize import least_squares
from four_body import EMS4body

# constants (km, kg, s) -- same values as four_body.py
G = 6.6743e-20
m1 = 1.989e30   # Sun
m2 = 5.974e24   # Earth
m3 = 7.348e22   # Moon
A  = 149.6e6    # Sun to Earth-Moon barycenter [km]
a  = 384400.0   # Earth to Moon [km]
Re = 6378.0
Rm = 1737.4

Omega_1 = np.sqrt(G*(m1+m2+m3)/A**3)
Omega_2 = np.sqrt(G*(m2+m3)/a**3)
a_1     = a*m3/(m2+m3)     # Earth to Earth-Moon barycenter (matches EMS4body's a_1)
mu      = m3/(m2+m3)

LU = a              # length unit
TU = 1/Omega_2       # time unit
VU = LU/TU            # velocity unit


def eci_from_rotating(r0, v0):
    """Invert EMS4body's internal ECI(Earth-centered)->rotating(barycentric)
    transform, so that calling EMS4body(R0,V0,TSPAN) starts the integration
    from the desired rotating-frame state (r0,v0) at time TSPAN[0].
    r0, v0: 3-vectors, km and km/s, in EMS4body's rotating barycentric frame.
    """
    R0 = r0 + np.array([a_1, 0.0, 0.0])
    V0 = v0 + np.array([0.0, a_1*Omega_2, 0.0]) + Omega_2*np.array([-r0[1], r0[0], 0.0])
    return R0, V0


def jacobi_energy(x, y, u, v):
    """Eq. 6, nondimensional PCR3BP Jacobi energy (evaluated instantaneously)."""
    r1 = np.sqrt((x+mu)**2 + y**2)
    r2 = np.sqrt((x+mu-1)**2 + y**2)
    return -(u**2+v**2) + (x**2+y**2) + 2*(1-mu)/r1 + 2*mu/r2 + mu*(1-mu)


def psi_i(x, y, u, v, ri):
    """Eq. 7: departure-point constraint (nondim), centered on Earth."""
    r_err = (x+mu)**2 + y**2 - ri**2
    tang  = (x+mu)*(u-y) + y*(v+x+mu)
    return np.array([r_err, tang])


def Cf_min(rf, direct=True):
    """Theorem III.2 lower bound on Jacobi energy for ballistic capture."""
    sign = 1.0 if direct else -1.0
    return 3*(1-mu) - (1-mu)*rf**2 + sign*2*np.sqrt(2*mu*rf)


def insertion_state(alpha_f, Cf, rf, direct=True):
    """Eqs. 16-17 (direct) / 30-31 (retrograde): nondim insertion state."""
    xf = rf*np.cos(alpha_f) + 1 - mu
    yf = rf*np.sin(alpha_f)
    r1f = np.sqrt((xf+mu)**2 + yf**2)
    W = (xf**2+yf**2) + 2*(1-mu)/r1f + 2*mu/rf + mu*(1-mu)
    Vf2 = -Cf + W
    if Vf2 < 0:
        return None
    Vf = np.sqrt(Vf2)
    if direct:
        uf, vf = -Vf*np.sin(alpha_f),  Vf*np.cos(alpha_f)
    else:
        uf, vf =  Vf*np.sin(alpha_f), -Vf*np.cos(alpha_f)
    return xf, yf, uf, vf


def propagate_backward(alpha_f, Cf, theta_Sf, TOF, direct=True):
    """Builds the insertion state, sets the epoch from the Sun phase angle,
    and propagates backward by TOF (seconds) with EMS4body (unmodified).
    Returns the departure-point rotating-frame state (nondim) or None.
    """
    state = insertion_state(alpha_f, Cf, rf, direct)
    if state is None:
        return None
    xf, yf, uf, vf = state

    r0 = np.array([xf, yf, 0.0]) * LU
    v0 = np.array([uf, vf, 0.0]) * VU
    R0, V0 = eci_from_rotating(r0, v0)

    t_f = theta_Sf/(Omega_2 - Omega_1)         # aligns Sun phase angle at insertion (Sec. II-A)
    sol = EMS4body(R0, V0, [t_f, t_f - TOF])
    if not sol.success:
        return None
    r_dep, v_dep = sol.y[:3, -1], sol.y[3:, -1]
    xi, yi, ui, vi = r_dep[0]/LU, r_dep[1]/LU, v_dep[0]/VU, v_dep[1]/VU
    return xi, yi, ui, vi


# ---------------------------------------------------------------------------
# 1) Grid search: scan (alpha_f, Cf, theta_Sf), flag points where the
#    departure state is close to satisfying psi_i = 0 (Eq. 7).
# ---------------------------------------------------------------------------
hi, hf = 167.0, 100.0
ri = (Re + hi)/LU
rf = (Rm + hf)/LU

# NOTE: the paper's actual grid is alpha_f in steps of pi/360 (720 pts),
# Cf in steps of 1e-4 (~2500 pts), theta_Sf in steps of pi/360 (720 pts),
# each requiring a 200-day backward integration (~4 s here per call) --
# that is not runnable interactively. The grid below is a small, fast
# illustration of the same pipeline; widen it to the paper's resolution
# and TOF_max=200*86400 for production use (run offline/in parallel).
TOF_grid = 30*86400.0

alpha_grid    = np.linspace(0, 2*np.pi, 5, endpoint=False)
Cf_grid       = np.linspace(Cf_min(rf, direct=True), 3.2003, 3)
theta_Sf_grid = np.linspace(0, 2*np.pi, 5, endpoint=False)

initial_guesses = []
for alpha_f in alpha_grid:
    for Cf in Cf_grid:
        for theta_Sf in theta_Sf_grid:
            dep = propagate_backward(alpha_f, Cf, theta_Sf, TOF_grid, direct=True)
            if dep is None:
                continue
            xi, yi, ui, vi = dep
            if abs(psi_i(xi, yi, ui, vi, ri)[0]) < 5e-2:   # coarse pre-screen on radius only
                initial_guesses.append([alpha_f, Cf, theta_Sf, TOF_grid])

print(f"{len(initial_guesses)} coarse candidate(s) found")


# ---------------------------------------------------------------------------
# 2) Differential correction: drive psi_i -> 0 by adjusting
#    y = [alpha_f, Cf, theta_Sf, TOF] (Eq. 44), holding alpha_f and theta_Sf
#    near their grid values and letting TOF/Cf absorb most of the correction.
# ---------------------------------------------------------------------------
def residual(y):
    alpha_f, Cf, theta_Sf, TOF = y
    dep = propagate_backward(alpha_f, Cf, theta_Sf, TOF, direct=True)
    if dep is None:
        return np.array([1e3, 1e3])
    xi, yi, ui, vi = dep
    return psi_i(xi, yi, ui, vi, ri)

solutions = []
for y0 in initial_guesses[:5]:      # cap the number corrected, for runtime
    lb = [y0[0]-0.05, Cf_min(rf, True), y0[2]-0.05, 1*86400]
    ub = [y0[0]+0.05, 3.2003,           y0[2]+0.05, 60*86400]
    res = least_squares(residual, y0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12)
    if np.linalg.norm(res.fun) < 1e-7:
        solutions.append(res.x)

print(f"{len(solutions)} corrected ballistic lunar transfer(s)")
for s in solutions:
    alpha_f, Cf, theta_Sf, TOF = s
    dep = propagate_backward(alpha_f, Cf, theta_Sf, TOF, direct=True)
    E_f = None
    state = insertion_state(alpha_f, Cf, rf, direct=True)
    if state is not None:
        xf, yf, uf, vf = state
        r1f = np.sqrt((xf+mu-1)**2 + yf**2)
        E_f = 0.5*((uf-yf)**2 + (vf+xf+mu-1)**2) - mu/rf
    print(f"alpha_f={alpha_f:.4f} Cf={Cf:.5f} theta_Sf={theta_Sf:.4f} "
          f"TOF={TOF/86400:.2f} d  Ef={E_f}  psi_i={residual(s)}")