#!/usr/bin/python

# Import all necessary modules
import numpy as np
import h5py as hp

############################################## USER PARAMETERS #######################################

# Choose the grid sizes from sLst so as to allow for Multigrid operations
# [2, 4, 6, 10, 18, 34, 66, 130, 258, 514, 1026, 2050]
#  0  1  2  3   4   5   6    7    8    9    10    11
sInd = np.array([5, 5, 5])

# Flag for setting periodicity along X and Y directions of the domain
xyPeriodic = False

# Toggle fwMode between ASCII and HDF5  to write output data in corresponding format
fwMode = "HDF5"

# Time-step
dt = 0.01

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 1.0

# Final time
tMax = 0.1

# Reynolds number
Re = 1000

# Tangent-hyperbolic grid stretching factor
beta = 1.0

# Tolerance value in Jacobi iterations
tolerance = 0.00001

# Depth of each V-cycle in multigrid
VDepth = 3

# Number of V-cycles to be computed
vcCnt = 10

# Number of iterations during pre-smoothing
preSm = 10

# Number of iterations during post-smoothing
pstSm = 50

# Number of iterations during smoothing in between prolongation operators
proSm = 60

########################################### END OF USER PARAMETERS ####################################

# Redefine frequently used numpy object
npax = np.newaxis

def euler(U, V, W, P, hx, hy, hz):
    global dt, Re
    global N, M, L
    global xi_xColl, et_yColl, zt_zColl
    global xi_xStag, et_yStag, zt_zStag
    global xixxColl, xix2Coll, etyyColl, ety2Coll, ztzzColl, ztz2Coll
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    Hx = np.zeros([L, M+1, N+1])
    Hx[1:L-1, 1:M, 1:N] = ((xixxColl[1:L-1, npax, npax]*D_Xi(U, L-1, M, N, hx) + etyyStag[0:M-1, npax]*D_Et(U, L-1, M, N, hy) + ztzzStag[0:N-1]*D_Zt(U, L-1, M, N, hz))/Re +
                           (xix2Coll[1:L-1, npax, npax]*DDXi(U, L-1, M, N, hx) + ety2Stag[0:M-1, npax]*DDEt(U, L-1, M, N, hy) + ztz2Stag[0:N-1]*DDZt(U, L-1, M, N, hz))*0.5/Re -
                            xi_xColl[1:L-1, npax, npax]*D_Xi(U, L-1, M, N, hx)*U[1:L-1, 1:M, 1:N] -
                      0.25*(V[1:L-1, 0:M-1, 1:N] + V[1:L-1, 1:M, 1:N] + V[2:L, 1:M, 1:N] + V[2:L, 0:M-1, 1:N])*et_yStag[0:M-1, npax]*D_Et(U, L-1, M, N, hy) - 
                      0.25*(W[1:L-1, 1:M, 0:N-1] + W[1:L-1, 1:M, 1:N] + W[2:L, 1:M, 1:N] + W[2:L, 1:M, 0:N-1])*zt_zStag[0:N-1]*D_Zt(U, L-1, M, N, hz))

    Hy = np.zeros([L+1, M, N+1])
    Hy[1:L, 1:M-1, 1:N] = ((xixxStag[0:L-1, npax, npax]*D_Xi(V, L, M-1, N, hx) + etyyColl[1:M-1, npax]*D_Et(V, L, M-1, N, hy) + ztzzStag[0:N-1]*D_Zt(V, L, M-1, N, hz))/Re +
                           (xix2Stag[0:L-1, npax, npax]*DDXi(V, L, M-1, N, hx) + ety2Coll[1:M-1, npax]*DDEt(V, L, M-1, N, hy) + ztz2Stag[0:N-1]*DDZt(V, L, M-1, N, hz))*0.5/Re -
                                                                     et_yColl[1:M-1, npax]*D_Et(V, L, M-1, N, hy)*V[1:L, 1:M-1, 1:N] -
                      0.25*(U[0:L-1, 1:M-1, 1:N] + U[1:L, 1:M-1, 1:N] + U[1:L, 2:M, 1:N] + U[0:L-1, 2:M, 1:N])*xi_xStag[0:L-1, npax, npax]*D_Xi(V, L, M-1, N, hx) -
                      0.25*(W[1:L, 1:M-1, 0:N-1] + W[1:L, 1:M-1, 1:N] + W[1:L, 2:M, 1:N] + W[1:L, 2:M, 0:N-1])*zt_zStag[0:N-1]*D_Zt(V, L, M-1, N, hz))

    Hz = np.zeros([L+1, M+1, N])
    Hz[1:L, 1:M, 1:N-1] = ((xixxStag[0:L-1, npax, npax]*D_Xi(W, L, M, N-1, hx) + etyyStag[0:M-1, npax]*D_Et(W, L, M, N-1, hy) + ztzzColl[1:N-1]*D_Zt(W, L, M, N-1, hz))/Re +
                           (xix2Stag[0:L-1, npax, npax]*DDXi(W, L, M, N-1, hx) + ety2Stag[0:M-1, npax]*DDEt(W, L, M, N-1, hy) + ztz2Coll[1:N-1]*DDZt(W, L, M, N-1, hz))*0.5/Re -
                                                                                                                                zt_zColl[1:N-1]*D_Zt(W, L, M, N-1, hz)*W[1:L, 1:M, 1:N-1] -
                      0.25*(U[0:L-1, 1:M, 1:N-1] + U[1:L, 1:M, 1:N-1] + U[1:L, 1:M, 2:N] + U[0:L-1, 1:M, 2:N])*xi_xStag[0:L-1, npax, npax]*D_Xi(W, L, M, N-1, hx) -
                      0.25*(V[1:L, 0:M-1, 1:N-1] + V[1:L, 1:M, 1:N-1] + V[1:L, 1:M, 2:N] + V[1:L, 0:M-1, 2:N])*et_yStag[0:M-1, npax]*D_Et(W, L, M, N-1, hy))

    # Calculating guessed values of U implicitly
    Hx[1:L-1, 1:M, 1:N] = U[1:L-1, 1:M, 1:N] + dt*(Hx[1:L-1, 1:M, 1:N] - xi_xColl[1:L-1, npax, npax]*(P[2:L, 1:M, 1:N] - P[1:L-1, 1:M, 1:N])/hx)
    Up = uJacobi(Hx, hx, hy, hz)

    # Calculating guessed values of V implicitly
    Hy[1:L, 1:M-1, 1:N] = V[1:L, 1:M-1, 1:N] + dt*(Hy[1:L, 1:M-1, 1:N] - et_yColl[1:M-1, npax]*(P[1:L, 2:M, 1:N] - P[1:L, 1:M-1, 1:N])/hy)
    Vp = vJacobi(Hy, hx, hy, hz)

    # Calculating guessed values of W implicitly
    Hz[1:L, 1:M, 1:N-1] = W[1:L, 1:M, 1:N-1] + dt*(Hz[1:L, 1:M, 1:N-1] - zt_zColl[1:N-1]*(P[1:L, 1:M, 2:N] - P[1:L, 1:M, 1:N-1])/hz)
    Wp = wJacobi(Hz, hx, hy, hz)

    # Calculating pressure correction term
    rhs = np.zeros([L+1, M+1, N+1])
    rhs[1:L, 1:M, 1:N] = (xi_xStag[0:L-1, npax, npax]*(Up[1:L, 1:M, 1:N] - Up[0:L-1, 1:M, 1:N])/hx +
                          et_yStag[0:M-1, npax]*(Vp[1:L, 1:M, 1:N] - Vp[1:L, 0:M-1, 1:N])/hy +
                          zt_zStag[0:N-1]*(Wp[1:L, 1:M, 1:N] - Wp[1:L, 1:M, 0:N-1])/hz)/dt
    Pp = multigrid(rhs, hx, hy, hz)

    # Add pressure correction.
    P = P + Pp

    # Update new values for U, V and W
    U[1:L-1, 1:M, 1:N] = Up[1:L-1, 1:M, 1:N] - dt*xi_xColl[1:L-1, npax, npax]*(Pp[2:L, 1:M, 1:N] - Pp[1:L-1, 1:M, 1:N])/hx
    V[1:L, 1:M-1, 1:N] = Vp[1:L, 1:M-1, 1:N] - dt*et_yColl[1:M-1, npax]*(Pp[1:L, 2:M, 1:N] - Pp[1:L, 1:M-1, 1:N])/hy
    W[1:L, 1:M, 1:N-1] = Wp[1:L, 1:M, 1:N-1] - dt*zt_zColl[1:N-1]*(Pp[1:L, 1:M, 2:N] - Pp[1:L, 1:M, 1:N-1])/hz

    # Impose no-slip BC on new values of U, V and W
    U = imposeUBCs(U)
    V = imposeVBCs(V)
    W = imposeWBCs(W)

    return U, V, W, P


def DDXi(inpFld, Nx, Ny, Nz, hx):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[0:Nx-1, 1:Ny, 1:Nz] - 2.0*inpFld[1:Nx, 1:Ny, 1:Nz] + inpFld[2:Nx+1, 1:Ny, 1:Nz])/(hx*hx)

    return outFld[1:Nx, 1:Ny, 1:Nz]


def DDEt(inpFld, Nx, Ny, Nz, hy):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 0:Ny-1, 1:Nz] - 2.0*inpFld[1:Nx, 1:Ny, 1:Nz] + inpFld[1:Nx, 2:Ny+1, 1:Nz])/(hy*hy)

    return outFld[1:Nx, 1:Ny, 1:Nz]


def DDZt(inpFld, Nx, Ny, Nz, hz):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 1:Ny, 0:Nz-1] - 2.0*inpFld[1:Nx, 1:Ny, 1:Nz] + inpFld[1:Nx, 1:Ny, 2:Nz+1])/(hz*hz)

    return outFld[1:Nx, 1:Ny, 1:Nz]


def D_Xi(inpFld, Nx, Ny, Nz, hx):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[2:Nx+1, 1:Ny, 1:Nz] - inpFld[0:Nx-1, 1:Ny, 1:Nz])*0.5/hx

    return outFld[1:Nx, 1:Ny, 1:Nz]


def D_Et(inpFld, Nx, Ny, Nz, hy):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 2:Ny+1, 1:Nz] - inpFld[1:Nx, 0:Ny-1, 1:Nz])*0.5/hy

    return outFld[1:Nx, 1:Ny, 1:Nz]


def D_Zt(inpFld, Nx, Ny, Nz, hz):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 1:Ny, 2:Nz+1] - inpFld[1:Nx, 1:Ny, 0:Nz-1])*0.5/hz

    return outFld[1:Nx, 1:Ny, 1:Nz]


# No-slip and no-penetration BCs
def imposeUBCs(U):
    global xyPeriodic

    # Periodic BCs along X and Y directions
    if xyPeriodic:
        # Left wall
        U[0, :, :] = U[-2, :, :]

        # Right wall
        U[-1, :, :] = U[1, :, :]

        # Front wall
        U[:, 0, :] = U[:, -3, :]

        # Back wall
        U[:, -1, :] = U[:, 2, :]

    # No-slip and no-penetration BCs
    else:
        # Left wall
        U[0, :, :] = -U[1, :, :]

        # Right wall
        U[-1, :, :] = -U[-2, :, :]

        # Front wall
        U[:, 0, :] = 0.0

        # Back wall
        U[:, -1, :] = 0.0

    # Bottom wall
    U[:, :, 0] = 0.0

    # Top wall - Moving lid on top
    U[:, :, -1] = 1.0

    return U


# No-slip and no-penetration BCs
def imposeVBCs(V):
    global xyPeriodic

    # Periodic BCs along X and Y directions
    if xyPeriodic:
        # Left wall
        V[0, :, :] = V[-3, :, :]

        # Right wall
        V[-1, :, :] = V[2, :, :]

        # Front wall
        V[:, 0, :] = V[:, -2, :]

        # Back wall
        V[:, -1, :] = V[:, 1, :]

    # No-slip and no-penetration BCs
    else:
        # Left wall
        V[0, :, :] = 0.0

        # Right wall
        V[-1, :, :] = 0.0

        # Front wall
        V[:, 0, :] = -V[:, 1, :]

        # Back wall
        V[:, -1, :] = -V[:, -2, :]

    # Bottom wall
    V[:, :, 0] = 0.0

    # Top wall
    V[:, :, -1] = 0.0

    return V


# No-slip and no-penetration BCs
def imposeWBCs(W):
    global xyPeriodic

    # Periodic BCs along X and Y directions
    if xyPeriodic:
        # Left wall
        W[0, :, :] = W[-3, :, :]

        # Right wall
        W[-1, :, :] = W[2, :, :]

        # Front wall
        W[:, 0, :] = W[:, -3, :]

        # Back wall
        W[:, -1, :] = W[:, 2, :]

    # No-slip and no-penetration BCs
    else:
        # Left wall
        W[0, :, :] = 0.0

        # Right wall
        W[-1, :, :] = 0.0

        # Front wall
        W[:, 0, :] = 0.0

        # Back wall
        W[:, -1, :] = 0.0

    # Bottom wall
    W[:, :, 0] = -W[:, :, 1]

    # Top wall
    W[:, :, -1] = -W[:, :, -2]

    return W


def imposePBCs(P):
    global xyPeriodic

    # Periodic BCs along X and Y directions
    if xyPeriodic:
        # Left wall
        P[0, :, :] = P[-3, :, :]

        # Right wall
        P[-1, :, :] = P[2, :, :]

        # Front wall
        P[:, 0, :] = P[:, -3, :]

        # Back wall
        P[:, -1, :] = P[:, 2, :]

    # Neumann boundary condition on pressure
    else:
        # Left wall
        P[0, :, :] = P[2, :, :]

        # Right wall
        P[-1, :, :] = P[-3, :, :]

        # Front wall
        P[:, 0, :] = P[:, 2, :]

        # Back wall
        P[:, -1, :] = P[:, -3, :]

    # Bottom wall
    P[:, :, 0] = P[:, :, 2]

    # Top wall
    P[:, :, -1] = P[:, :, -3]

    return P


#Jacobi iterative solver for U
def uJacobi(rho, hx, hy, hz):
    global N, M
    global Re, dt
    global tolerance
    global iCnt, opInt
    global xix2Coll, ety2Stag, ztz2Stag

    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    test_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L-1, 1:M, 1:N] = (((hy*hy)*(hz*hz)*xix2Coll[1:L-1, npax, npax]*(prev_sol[0:L-2, 1:M, 1:N] + prev_sol[  2:L,   1:M, 1:N]) +
                                      (hx*hx)*(hz*hz)*ety2Stag[0:M-1, npax]*(prev_sol[1:L-1, 0:M-1, 1:N] + prev_sol[1:L-1, 2:M+1, 1:N]) +
                                      (hx*hx)*(hy*hy)*ztz2Stag[0:N-1]*(prev_sol[1:L-1, 1:M, 0:N-1] + prev_sol[1:L-1, 1:M, 2:N+1]))*
                                       dt/((hx*hx)*(hy*hy)*(hz*hz)*2.0*Re) + rho[1:L-1, 1:M, 1:N])/ \
                                (1.0 + dt*((hy*hy)*(hz*hz)*xix2Coll[1:L-1, npax, npax] +
                                           (hx*hx)*(hz*hz)*ety2Stag[0:M-1, npax] +
                                           (hx*hx)*(hy*hy)*ztz2Stag[0:N-1])/(Re*(hx*hx)*(hy*hy)*(hz*hz)))

        # IMPOSE BOUNDARY CONDITION AND COPY TO PREVIOUS SOLUTION ARRAY
        next_sol = imposeUBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L-1, 1:M, 1:N] = next_sol[1:L-1,   1:M, 1:N] - (
                                    xix2Coll[1:L-1, npax, npax]*DDXi(next_sol, L-1, M, N, hx) + \
                                    ety2Stag[0:M-1, npax]*DDEt(next_sol, L-1, M, N, hy) + \
                                    ztz2Stag[0:N-1]*DDZt(next_sol, L-1, M, N, hz))*0.5*dt/Re

        error_temp = np.fabs(rho[1:L-1, 1:M, 1:N] - test_sol[1:L-1, 1:M, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            if iCnt % opInt == 0:
                print "Jacobi solver for U converged in ", jCnt, " iterations"
            break

        jCnt += 1
        if jCnt > 10*N*M*L:
            print "ERROR: Jacobi not converging in U. Aborting"
            print "Maximum error: ", maxErr
            quit()

    return prev_sol


#Jacobi iterative solver for V
def vJacobi(rho, hx, hy, hz):
    global N, M
    global Re, dt
    global tolerance
    global iCnt, opInt
    global xix2Stag, ety2Coll, ztz2Stag

    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    test_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L, 1:M-1, 1:N] = (((hy*hy)*(hz*hz)*xix2Stag[0:L-1, npax, npax]*(prev_sol[0:L-1, 1:M-1, 1:N] + prev_sol[2:L+1, 1:M-1, 1:N]) +
                                      (hx*hx)*(hz*hz)*ety2Coll[1:M-1, npax]*(prev_sol[1:L, 0:M-2,   1:N] + prev_sol[    1:L, 2:M, 1:N]) +
                                      (hx*hx)*(hy*hy)*ztz2Stag[0:N-1]*(prev_sol[1:L, 1:M-1, 0:N-1] + prev_sol[1:L, 1:M-1, 2:N+1]))*
                                       dt/((hx*hx)*(hy*hy)*(hz*hz)*2.0*Re) + rho[1:L, 1:M-1, 1:N])/ \
                                (1.0 + dt*((hy*hy)*(hz*hz)*xix2Stag[0:L-1, npax, npax] +
                                           (hx*hx)*(hz*hz)*ety2Coll[1:M-1, npax] +
                                           (hx*hx)*(hy*hy)*ztz2Stag[0:N-1])/(Re*(hx*hx)*(hy*hy)*(hz*hz)))

        # IMPOSE BOUNDARY CONDITION AND COPY TO PREVIOUS SOLUTION ARRAY
        next_sol = imposeVBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L, 1:M-1, 1:N] = next_sol[  1:L, 1:M-1, 1:N] - (
                                    xix2Stag[0:L-1, npax, npax]*DDXi(next_sol, L, M-1, N, hx) + \
                                    ety2Coll[1:M-1, npax]*DDEt(next_sol, L, M-1, N, hy) + \
                                    ztz2Stag[0:N-1]*DDZt(next_sol, L, M-1, N, hz))*0.5*dt/Re

        error_temp = np.fabs(rho[1:L, 1:M-1, 1:N] - test_sol[1:L, 1:M-1, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            if iCnt % opInt == 0:
                print "Jacobi solver for V converged in ", jCnt, " iterations"
            break

        jCnt += 1
        if jCnt > 10*N*M*L:
            print "ERROR: Jacobi not converging in V. Aborting"
            print "Maximum error: ", maxErr
            quit()

    return prev_sol


#Jacobi iterative solver for W
def wJacobi(rho, hx, hy, hz):
    global N, M
    global Re, dt
    global tolerance
    global iCnt, opInt
    global xix2Stag, ety2Stag, ztz2Coll

    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    test_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L, 1:M, 1:N-1] = (((hy*hy)*(hz*hz)*xix2Stag[0:L-1, npax, npax]*(prev_sol[0:L-1, 1:M, 1:N-1] + prev_sol[2:L+1, 1:M, 1:N-1]) +
                                      (hx*hx)*(hz*hz)*ety2Stag[0:M-1, npax]*(prev_sol[1:L, 0:M-1, 1:N-1] + prev_sol[1:L, 2:M+1, 1:N-1]) +
                                      (hx*hx)*(hy*hy)*ztz2Coll[1:N-1]*(prev_sol[1:L, 1:M, 0:N-2] + prev_sol[  1:L,   1:M, 2:N]))*
                                       dt/((hx*hx)*(hy*hy)*(hz*hz)*2.0*Re) + rho[1:L, 1:M, 1:N-1])/ \
                                (1.0 + dt*((hy*hy)*(hz*hz)*xix2Stag[0:L-1, npax, npax] +
                                           (hx*hx)*(hz*hz)*ety2Stag[0:M-1, npax] +
                                           (hx*hx)*(hy*hy)*ztz2Coll[1:N-1])/(Re*(hx*hx)*(hy*hy)*(hz*hz)))

        # IMPOSE BOUNDARY CONDITION AND COPY TO PREVIOUS SOLUTION ARRAY
        next_sol = imposeWBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L, 1:M, 1:N-1] = next_sol[  1:L, 1:M, 1:N-1] - (
                                    xix2Stag[0:L-1, npax, npax]*DDXi(next_sol, L, M, N-1, hx) + \
                                    ety2Stag[0:M-1, npax]*DDEt(next_sol, L, M, N-1, hy) + \
                                    ztz2Coll[1:N-1]*DDZt(next_sol, L, M, N-1, hz))*0.5*dt/Re

        error_temp = np.fabs(rho[1:L, 1:M, 1:N-1] - test_sol[1:L, 1:M, 1:N-1])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            if iCnt % opInt == 0:
                print "Jacobi solver for W converged in ", jCnt, " iterations"
            break

        jCnt += 1
        if jCnt > 10*N*M*L:
            print "ERROR: Jacobi not converging in W. Aborting"
            print "Maximum error: ", maxErr
            quit()

    return prev_sol


#Multigrid solver
def multigrid(H, hx, hy, hz):
    global N, M, L, vcCnt

    P = np.zeros([L+1, M+1, N+1])
    chMat = np.ones([L+1, M+1, N+1])
    for i in range(vcCnt):
        P = v_cycle(P, H, hx, hy, hz)
        chMat = laplace(P, hx, hy, hz)

    print "Error after multigrid is ", np.amax(np.abs(H[1:L, 1:M, 1:N] - chMat[1:L, 1:M, 1:N]))

    return P


#Multigrid solution without the use of recursion
def v_cycle(P, H, hx, hy, hz):
    global sInd, VDepth
    global preSm, pstSm, proSm

    # Pre-smoothing
    P = smooth(P, H, hx, hy, hz, preSm, 0)

    H_rsdl = H - laplace(P, hx, hy, hz)

    # Restriction operations
    for i in range(VDepth):
        sInd -= 1
        H_rsdl = restrict(H_rsdl)

    # Solving the system after restriction
    P_corr = solve(H_rsdl, (2.0**VDepth)*hx, (2.0**VDepth)*hy, (2.0**VDepth)*hz)

    # Prolongation operations
    for i in range(VDepth):
        sInd += 1
        P_corr = prolong(P_corr)
        H_rsdl = prolong(H_rsdl)
        P_corr = smooth(P_corr, H_rsdl, hx, hy, hz, proSm, VDepth-i-1)

    P += P_corr

    # Post-smoothing
    P = smooth(P, H, hx, hy, hz, pstSm, 0)

    return P


#Uses jacobi iteration to smooth the solution passed to it.
def smooth(function, rho, hx, hy, hz, iteration_times, vLevel):
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    smoothed = np.copy(function)

    # 1 subtracted from shape to account for ghost points
    [L, M, N] = np.array(np.shape(function)) - 1

    for i in range(iteration_times):
        toSmooth = imposePBCs(smoothed)

        smoothed[1:L, 1:M, 1:N] = (
                        (hy*hy)*(hz*hz)*xix2Stag[0::2**vLevel, npax, npax]*(toSmooth[2:L+1, 1:M, 1:N] + toSmooth[0:L-1, 1:M, 1:N])*2.0 +
                        (hy*hy)*(hz*hz)*xixxStag[0::2**vLevel, npax, npax]*(toSmooth[2:L+1, 1:M, 1:N] - toSmooth[0:L-1, 1:M, 1:N])*hx +
                        (hx*hx)*(hz*hz)*ety2Stag[0::2**vLevel, npax]*(toSmooth[1:L, 2:M+1, 1:N] + toSmooth[1:L, 0:M-1, 1:N])*2.0 +
                        (hx*hx)*(hz*hz)*etyyStag[0::2**vLevel, npax]*(toSmooth[1:L, 2:M+1, 1:N] - toSmooth[1:L, 0:M-1, 1:N])*hy +
                        (hx*hx)*(hy*hy)*ztz2Stag[0::2**vLevel]*(toSmooth[1:L, 1:M, 2:N+1] + toSmooth[1:L, 1:M, 0:N-1])*2.0 +
                        (hx*hx)*(hy*hy)*ztzzStag[0::2**vLevel]*(toSmooth[1:L, 1:M, 2:N+1] - toSmooth[1:L, 1:M, 0:N-1])*hz -
                    2.0*(hx*hx)*(hy*hy)*(hz*hz)*rho[1:L, 1:M, 1:N])/ \
                  (4.0*((hy*hy)*(hz*hz)*xix2Stag[0::2**vLevel, npax, npax] +
                        (hx*hx)*(hz*hz)*ety2Stag[0::2**vLevel, npax] +
                        (hx*hx)*(hy*hy)*ztz2Stag[0::2**vLevel]))

    return smoothed


#Reduces the size of the array to a lower level, 2^(n-1)+1.
def restrict(function):
    global sInd, sLst

    [rx, ry, rz] = [sLst[sInd[0]], sLst[sInd[1]], sLst[sInd[2]]]
    restricted = np.zeros([rx + 1, ry + 1, rz + 1])

    for i in range(1, rx):
        for j in range(1, ry):
            for k in range(1, rz):
                restricted[i, j, k] = function[2*i - 1, 2*j - 1, 2*k - 1]

    return restricted


#Increases the size of the array to a higher level, 2^(n+1)+1.
def prolong(function):
    global sInd, sLst

    [rx, ry, rz] = [sLst[sInd[0]], sLst[sInd[1]], sLst[sInd[2]]]
    prolonged = np.zeros([rx + 1, ry + 1, rz + 1])

    [lx, ly, lz] = np.shape(function)
    for i in range(1, lx-1):
        for j in range(1, ly-1):
            for k in range(1, lz-1):
                prolonged[i*2 - 1, j*2 - 1, k*2 - 1] = function[i, j, k]
    
    for i in range(1, rx, 2):
        for j in range(1, ry, 2):
            for k in range(2, rz, 2):
                prolonged[i, j, k] = (prolonged[i, j, k-1] + prolonged[i, j, k+1])/2

    for i in range(1, rx, 2):
        for j in range(2, ry, 2):
            for k in range(1, rz):
                prolonged[i, j, k] = (prolonged[i, j-1, k] + prolonged[i, j+1, k])/2

    for i in range(2, rx, 2):
        for j in range(1, ry):
            for k in range(1, rz):
                prolonged[i, j, k] = (prolonged[i-1, j, k] + prolonged[i+1, j, k])/2

    return prolonged


#This function uses the Jacobi iterative solver, using the grid spacing
def solve(rho, hx, hy, hz):
    global tolerance
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    # 1 subtracted from shape to account for ghost points
    [L, M, N] = np.array(np.shape(rho)) - 1
    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L, 1:M, 1:N] = (
            (hy*hy)*(hz*hz)*xix2Stag[0::2**VDepth, npax, npax]*(prev_sol[2:L+1, 1:M, 1:N] + prev_sol[0:L-1, 1:M, 1:N])*2.0 +
            (hy*hy)*(hz*hz)*xixxStag[0::2**VDepth, npax, npax]*(prev_sol[2:L+1, 1:M, 1:N] - prev_sol[0:L-1, 1:M, 1:N])*hx +
            (hx*hx)*(hz*hz)*ety2Stag[0::2**VDepth, npax]*(prev_sol[1:L, 2:M+1, 1:N] + prev_sol[1:L, 0:M-1, 1:N])*2.0 +
            (hx*hx)*(hz*hz)*etyyStag[0::2**VDepth, npax]*(prev_sol[1:L, 2:M+1, 1:N] - prev_sol[1:L, 0:M-1, 1:N])*hy +
            (hx*hx)*(hy*hy)*ztz2Stag[0::2**VDepth]*(prev_sol[1:L, 1:M, 2:N+1] + prev_sol[1:L, 1:M, 0:N-1])*2.0 +
            (hx*hx)*(hy*hy)*ztzzStag[0::2**VDepth]*(prev_sol[1:L, 1:M, 2:N+1] - prev_sol[1:L, 1:M, 0:N-1])*hz -
        2.0*(hx*hx)*(hy*hy)*(hz*hz)*rho[1:L, 1:M, 1:N])/ \
      (4.0*((hy*hy)*(hz*hz)*xix2Stag[0::2**VDepth, npax, npax] +
            (hx*hx)*(hz*hz)*ety2Stag[0::2**VDepth, npax] +
            (hx*hx)*(hy*hy)*ztz2Stag[0::2**VDepth]))

        solLap = np.zeros_like(next_sol)
        solLap[1:L, 1:M, 1:N] = xix2Stag[0::2**VDepth, npax, npax]*DDXi(next_sol, L, M, N, hx) + \
                                xixxStag[0::2**VDepth, npax, npax]*D_Xi(next_sol, L, M, N, hx) + \
                                ety2Stag[0::2**VDepth, npax]*DDEt(next_sol, L, M, N, hy) + \
                                etyyStag[0::2**VDepth, npax]*D_Et(next_sol, L, M, N, hy) + \
                                ztz2Stag[0::2**VDepth]*DDZt(next_sol, L, M, N, hz) + \
                                ztzzStag[0::2**VDepth]*D_Zt(next_sol, L, M, N, hz)

        error_temp = np.abs(rho[1:L, 1:M, 1:N] - solLap[1:L, 1:M, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > 10*N*M*L:
            print "ERROR: Jacobi not converging. Aborting"
            print "Maximum error: ", maxErr
            quit()

        prev_sol = np.copy(next_sol)

    return prev_sol


def laplace(function, hx, hy, hz):
    '''
Function to calculate the Laplacian for a given field of values.
INPUT:  function: 3D matrix of double precision values
        hx, hy, hz: Grid spacing of the uniform grid along x, y and z directions
OUTPUT: gradient: 3D matrix of double precision values with same size as input matrix
    '''

    # 1 subtracted from shape to account for ghost points
    [L, M, N] = np.array(np.shape(function)) - 1
    gradient = np.zeros_like(function)

    gradient[1:L, 1:M, 1:N] = xix2Stag[0:L-1, npax, npax]*DDXi(function, L, M, N, hx) + \
                              xixxStag[0:L-1, npax, npax]*D_Xi(function, L, M, N, hx) + \
                                    ety2Stag[0:M-1, npax]*DDEt(function, L, M, N, hy) + \
                                    etyyStag[0:M-1, npax]*D_Et(function, L, M, N, hy) + \
                                          ztz2Stag[0:N-1]*DDZt(function, L, M, N, hz) + \
                                          ztzzStag[0:N-1]*D_Zt(function, L, M, N, hz)

    return gradient


def getDiv(U, V, W):
    '''
Function to calculate the divergence within the domain (excluding walls)
INPUT:  U, V, W: Velocity values
OUTPUT: The maximum value of divergence in double precision
    '''
    global N, M, L
    global xColl, yColl, zColl

    divMat = np.zeros([L+1, M+1, N+1])
    for i in range(1, L):
        for j in range(1, M):
            for k in range(1, N):
                divMat[i, j, k] = (U[i, j, k] - U[i-1, j, k])/(xColl[i] - xColl[i-1]) + \
                                  (V[i, j, k] - V[i, j-1, k])/(yColl[j] - yColl[j-1]) + \
                                  (W[i, j, k] - W[i, j, k-1])/(zColl[k] - zColl[k-1])

    return np.unravel_index(divMat.argmax(), divMat.shape), np.amax(divMat)


def writeSoln(U, V, W, P, time):
    global fwMode
    global N, M, L
    global xColl, yColl, zColl

    if fwMode == "ASCII":
        fName = "Soln_" + "{0:09.5f}".format(time) + ".dat"
        print "Writing solution file: ", fName

        ofFile = open(fName, 'w')
        ofFile.write("VARIABLES = X, Y, Z, U, V, W, P\n")
        ofFile.write("ZONE T=S\tI={0}\tJ={1}\tK={2}\tF=POINT\tSOLUTIONTIME={3}\n".format(L, M, N, time))
        for i in range(0, N):
            for j in range(0, M):
                for k in range(0, L):
                    ofFile.write("{0:23.16f}\t{1:23.16f}\t{2:23.16f}\t{3:23.16f}\t{4:23.16f}\t{5:23.16f}\t{6:23.16f}\n".format(
                                xColl[k], yColl[j], zColl[i],
                                (U[k, j, i] + U[k, j+1, i] + U[k, j, i+1] + U[k, j+1, i+1])/4.0,
                                (V[k, j, i] + V[k+1, j, i] + V[k, j, i+1] + V[k+1, j, i+1])/4.0,
                                (W[k, j, i] + W[k+1, j, i] + W[k, j+1, i] + W[k+1, j+1, i])/4.0,
                                (P[k,   j, i] + P[k+1,   j, i] + P[k,   j+1, i] + P[k+1,   j+1, i] +
                                P[k, j, i+1] + P[k+1, j, i+1] + P[k, j+1, i+1] + P[k+1, j+1, i+1])/8.0))

        ofFile.close()

    elif fwMode == "HDF5":
        fName = "Soln_" + "{0:09.5f}.h5".format(time)
        print "Writing solution file: ", fName

        f = hp.File(fName, "w")

        dset = f.create_dataset("U",data = U)
        dset = f.create_dataset("V",data = V)
        dset = f.create_dataset("W",data = W)
        dset = f.create_dataset("P",data = P)

        f.close()


def calculateMetrics(hx, hy, hz):
    global beta
    global xColl, yColl, zColl
    global xStag, yStag, zStag
    global xi_xColl, et_yColl, zt_zColl
    global xi_xStag, et_yStag, zt_zStag
    global xixxColl, xix2Coll, etyyColl, ety2Coll, ztzzColl, ztz2Coll
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    xi = np.linspace(0.0, 1.0, L)
    et = np.linspace(0.0, 1.0, M)
    zt = np.linspace(0.0, 1.0, N)

    # Calculate grid and its metrics
    xColl = [(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in xi]
    yColl = [(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in et]
    zColl = [(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in zt]

    xStag = [(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in [(xi[j] + xi[j+1])/2 for j in range(len(xi) - 1)]]
    yStag = [(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in [(et[j] + et[j+1])/2 for j in range(len(et) - 1)]]
    zStag = [(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in [(zt[j] + zt[j+1])/2 for j in range(len(zt) - 1)]]

    # Grid metrics for both staggered and collocated grids
    xi_xColl = np.array([np.tanh(beta)/(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0) for k in xColl])
    xixxColl = np.array([((8.0*k - 4.0)*np.tanh(beta)**3.0)/(beta*(((2.0*k - 1.0)*np.tanh(beta))**2.0 - 1.0)**2.0) for k in xColl])
    xix2Coll = np.array([k*k for k in xi_xColl])

    xi_xStag = np.array([np.tanh(beta)/(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0) for k in xStag])
    xixxStag = np.array([((8.0*k - 4.0)*np.tanh(beta)**3.0)/(beta*(((2.0*k - 1.0)*np.tanh(beta))**2.0 - 1.0)**2.0) for k in xStag])
    xix2Stag = np.array([k*k for k in xi_xStag])

    et_yColl = np.array([np.tanh(beta)/(1.0 - (np.tanh(beta)*(1.0 - 2.0*j))**2.0) for j in yColl])
    etyyColl = np.array([((8.0*j - 4.0)*np.tanh(beta)**3.0)/(beta*(((2.0*j - 1.0)*np.tanh(beta))**2.0 - 1.0)**2.0) for j in yColl])
    ety2Coll = np.array([j*j for j in et_yColl])

    et_yStag = np.array([np.tanh(beta)/(1.0 - (np.tanh(beta)*(1.0 - 2.0*j))**2.0) for j in yStag])
    etyyStag = np.array([((8.0*j - 4.0)*np.tanh(beta)**3.0)/(beta*(((2.0*j - 1.0)*np.tanh(beta))**2.0 - 1.0)**2.0) for j in yStag])
    ety2Stag = np.array([j*j for j in et_yStag])

    zt_zColl = np.array([np.tanh(beta)/(1.0 - (np.tanh(beta)*(1.0 - 2.0*i))**2.0) for i in zColl])
    ztzzColl = np.array([((8.0*i - 4.0)*np.tanh(beta)**3.0)/(beta*(((2.0*i - 1.0)*np.tanh(beta))**2.0 - 1.0)**2.0) for i in zColl])
    ztz2Coll = np.array([i*i for i in zt_zColl])

    zt_zStag = np.array([np.tanh(beta)/(1.0 - (np.tanh(beta)*(1.0 - 2.0*i))**2.0) for i in zStag])
    ztzzStag = np.array([((8.0*i - 4.0)*np.tanh(beta)**3.0)/(beta*(((2.0*i - 1.0)*np.tanh(beta))**2.0 - 1.0)**2.0) for i in zStag])
    ztz2Stag = np.array([i*i for i in zt_zStag])


# Main segment of code.
def main():
    global dt, tMax, fwInt, opInt
    global time, iCnt
    global L, M, N

    # Create and initialize U, V and P arrays
    # The arrays have two extra points
    # These act as ghost points on either sides of the domain
    P = np.ones([L+1, M+1, N+1])

    # U is staggered in Y and Z directions and hence has one extra point along these directions
    U = np.zeros([L, M+1, N+1])

    # V is staggered in X and Z directions and hence has one extra point along these directions
    V = np.zeros([L+1, M, N+1])

    # W is staggered in X and Y directions and hence has one extra point along these directions
    W = np.zeros([L+1, M+1, N])

    # Define grid spacings
    hx = 1.0/(L-1)
    hy = 1.0/(M-1)
    hz = 1.0/(N-1)

    calculateMetrics(hx, hy, hz)

    # BC for moving top lid - U = 1.0 on lid and 2.0 on ghost point
    U[:, :, N] = 2.0

    fwTime = 0.0

    while True:
        if abs(fwTime - time) < 0.5*dt:
            writeSoln(U, V, W, P, time)
            fwTime += fwInt

        U, V, W, P = euler(U, V, W, P, hx, hy, hz)

        maxDiv = getDiv(U, V, W)
        if maxDiv[1] > 10.0:
            print "ERROR: Divergence has exceeded permissible limits. Aborting"
            quit()

        iCnt += 1
        time += dt
        if iCnt % opInt == 0:
            print "Time: {0:9.5f}".format(time)
            print "Maximum divergence: {0:8.5f} at ({1:d}, {2:d}, {3:d})\n".format(maxDiv[1], maxDiv[0][0], maxDiv[0][1], maxDiv[0][2])

        if time > tMax:
            break

    print "Simulation completed"


###########################################################################################################################################

# N should be of the form 2^n+2 so that staggered pressure points will be 2^n + 3 including ghost points
sLst = [2**x + 2 for x in range(12)]

# Limits along each direction
# L - Along X
# M - Along Y
# N - Along Z
# Data stored in arrays accessed by data[1:L, 1:M, 1:N]
# In Python and C, the rightmost index varies fastest
# Therefore indices in Z direction vary fastest, then along Y and finally along X

L = sLst[sInd[0]]
M = sLst[sInd[1]]
N = sLst[sInd[2]]

# Grid metric arrays. Used by the enitre program at various points
xColl = np.zeros(L)
yColl = np.zeros(M)
zColl = np.zeros(N)
xStag = np.zeros(L-1)
yStag = np.zeros(M-1)
zStag = np.zeros(N-1)

xi_xColl = np.zeros(L)      #-- dXi/dX at all x-grid nodes
xixxColl = np.zeros(L)      #-- d2Xi/dX2 at all x-grid nodes
xix2Coll = np.zeros(L)      #-- (dXi/dX)**2 at all x-grid nodes

xi_xStag = np.zeros(L-1)    #-- dXi/dX at all x-grid nodes
xixxStag = np.zeros(L-1)    #-- d2Xi/dX2 at all x-grid nodes
xix2Stag = np.zeros(L-1)    #-- (dXi/dX)**2 at all x-grid nodes

et_yColl = np.zeros(M)      #-- dEt/dY at all y-grid nodes
etyyColl = np.zeros(M)      #-- d2Et/dY2 at all y-grid nodes
ety2Coll = np.zeros(M)      #-- (dEt/dY)**2 at all y-grid nodes

et_yStag = np.zeros(M-1)    #-- dEt/dY at all y-grid nodes
etyyStag = np.zeros(M-1)    #-- d2Et/dY2 at all y-grid nodes
ety2Stag = np.zeros(M-1)    #-- (dEt/dY)**2 at all y-grid nodes

zt_zColl = np.zeros(N)      #-- dZt/dZ at all z-grid nodes
ztzzColl = np.zeros(N)      #-- d2Zt/dZ2 at all z-grid nodes
ztz2Coll = np.zeros(N)      #-- (dZt/dZ)**2 at all z-grid nodes

zt_zStag = np.zeros(N-1)    #-- dZt/dZ at all z-grid nodes
ztzzStag = np.zeros(N-1)    #-- d2Zt/dZ2 at all z-grid nodes
ztz2Stag = np.zeros(N-1)    #-- (dZt/dZ)**2 at all z-grid nodes

time = 0.0
iCnt = 0

main()
