import numpy as np

def fourbody():
    xi_1 = -A_1*np.cos(omega_1*t)
    xi_2 = A_2*np.cos(omega_1*t)-a_1*np.cos(omega_2*t)
    eta_2 = A_2*np.sin(omega_1)
    eta_1 = -A_1*np.sin(omega_1*t)

    xi_ddot = -G*m1*((xi-xi_1)/r_1**3) -G*m2*((xi-xi_2)/r_2**3)-G*m3*((xi-xi_3)/r_3**3)
    eta_ddot = -G*m1*((eta-eta_1)/r_1**3) - G*m2*((eta-eta_2)/r_2**3)-G*m3*((eta-eta_3)/r_3**3)
    zeta_ddot = -G*m1*((zeta-zeta_1)/r_1**3) - G*m2*((zeta-zeta_2)/r_2**3)-G*m3*((zeta-zeta_3)/r_3**3)