import numpy as np
from scipy.integrate import solve_ivp

def EMS4body(R0,V0,TSPAN): # ECI Vector inputs 
    #constants (km, kg, s)
    G = 6.6743e-20 # [km3/kg/s2]
    m1 = 1.989e30 # Sun
    m2 = 5.974e24 # Earth
    m3 = 7.348e22 # Moon
    A = 149.6e6 # Sun to Earth-Moon barycenter [km]
    a = 384400 # Earth to Moon [km]

    A_1 = A*(m2+m3)/(m1+m2+m3) # Sun to system barycenter
    A_2 = A - A_1              # Earth-Moon barycenter to system barycenter
    a_1 = a*m3/(m2+m3)         # Earth to Earth-Moon barycenter
    a_2 = a - a_1              # Moon to Earth-Moon barycenter

    Omega_1 = np.sqrt(G*(m1+m2+m3)/A**3)
    Omega_2 = np.sqrt(G*(m2+m3)/a**3)

    # Paper's frame: origin at the Earth-Moon barycenter, x-axis rotating with the
    # Earth-Moon line (Earth at x=-a_1, Moon at x=+a_2). Convert the Earth-centered
    # inertial initial state into that frame (at t=0 the axes are aligned).
    r0 = np.array(R0) + np.array([-a_1,0,0])
    v0 = np.array(V0) + np.array([0,-a_1*Omega_2,0]) - Omega_2*np.array([-r0[1],r0[0],0])
    Y0 = np.concatenate((r0, v0))

    def fourbody(t,y,G,m1,m2,m3,A,A_2,a_1,a_2,Omega_1,Omega_2):
        """
    Very restricted four-body function (Huang, NASA TN D-501, Eqs. 16-18).
    Returns the derivative of the state space variables in the rotating
    Earth-Moon barycenter frame.
    """
        # Sun position in the rotating frame (Eq. 13)
        x_1 = -A*np.cos((Omega_2-Omega_1)*t)
        y_1 =  A*np.sin((Omega_2-Omega_1)*t)

        # distances to the Sun, Earth, Moon (Eq. 19)
        r_1 = np.sqrt((y[0]-x_1)**2 + (y[1]-y_1)**2 + y[2]**2)
        r_2 = np.sqrt((y[0]+a_1)**2 + y[1]**2 + y[2]**2)
        r_3 = np.sqrt((y[0]-a_2)**2 + y[1]**2 + y[2]**2)

        k = (A_2/A)*Omega_1**2 

        y_dot = [0]*6 
        y_dot[0] = y[3]
        y_dot[1] = y[4]
        y_dot[2] = y[5]
        y_dot[3] = Omega_2**2*y[0] - k*x_1 - G*m1*(y[0]-x_1)/r_1**3 - G*m2*(y[0]+a_1)/r_2**3 - G*m3*(y[0]-a_2)/r_3**3 + 2*Omega_2*y[4]
        y_dot[4] = Omega_2**2*y[1] - k*y_1 - G*m1*(y[1]-y_1)/r_1**3 - G*m2*y[1]/r_2**3 - G*m3*y[1]/r_3**3 - 2*Omega_2*y[3]
        y_dot[5] = -G*(m1/r_1**3 + m2/r_2**3 + m3/r_3**3)*y[2]

        return y_dot

    solution = solve_ivp(fourbody,TSPAN,Y0, rtol=1e-8, atol=1e-8,args=(G,m1,m2,m3,A,A_2,a_1,a_2,Omega_1,Omega_2))
    return solution
