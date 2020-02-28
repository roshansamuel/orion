#!/usr/bin/python

####################################################################################################
# Orion
# 
# Copyright (C) 2020, Roshan J. Samuel
#
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
####################################################################################################

# Import all necessary modules
import boundaryConditions as bc
import multiprocessing as mp
import globalVars as gv
import numpy as np
import h5py as hp

############################################ USER PARAMETERS #######################################

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# [2, 4, 6, 10, 18, 34, 66, 130, 258, 514, 1026, 2050]
#  0  1  2  3   4   5   6    7    8    9    10    11
sInd = np.array([5, 5, 5])

# Time-step
dt = 0.01

# File writing interval
fwInt = 1.0

# Final time
tMax = 0.1

# Reynolds number
Re = 1000

# Tolerance value in Jacobi iterations
tolerance = 0.00001

# Depth of each V-cycle in multigrid
VDepth = 3

# Number of V-cycles to be computed
vcCnt = 10

# Number of iterations during pre-smoothing
preSm = 10

# Number of iterations during post-smoothing
pstSm = 40

# Number of iterations during smoothing in between prolongation operators
proSm = 30

######################################## END OF USER PARAMETERS ####################################

# Redefine frequently used numpy object
npax = np.newaxis

def euler():
    global dt
    global N, M, L
    global rListColl
    global U, V, W, P
    global Hx, Hy, Hz
    global hx, hy, hz
    global xi_xColl, et_yColl, zt_zColl
    global xi_xStag, et_yStag, zt_zStag

    Hx.fill(0.0)
    Hy.fill(0.0)
    Hz.fill(0.0)

    computeNLinDiff_X()
    computeNLinDiff_Y()
    computeNLinDiff_Z()

    ############
    #pool = mp.Pool(processes=gv.nProcs)
    #poolRes = [pool.apply_async(addTurbViscosity, args=(x[0], x[1])) for x in rListColl]
    #cosVals = [x.get() for x in poolRes]
    #print(cosVals[0][:,0,5,5])
    #exit(0)
    # Flatten the lists
    #cos1 = [x for y in cosVals for x in y[0]]
    #cos2 = [x for y in cosVals for x in y[1]]
    #cos3 = [x for y in cosVals for x in y[2]]

    # Add constant pressure gradient forcing for channel flows
    if gv.probType == 1:
        Hx[:, :, :] += 1.0

    # Calculating guessed values of U implicitly
    Hx[1:L-1, 1:M, 1:N] = U[1:L-1, 1:M, 1:N] + dt*(Hx[1:L-1, 1:M, 1:N] - xi_xColl[1:L-1, npax, npax]*(P[2:L, 1:M, 1:N] - P[1:L-1, 1:M, 1:N])/hx)
    Up = uJacobi(Hx)

    # Calculating guessed values of V implicitly
    Hy[1:L, 1:M-1, 1:N] = V[1:L, 1:M-1, 1:N] + dt*(Hy[1:L, 1:M-1, 1:N] - et_yColl[1:M-1, npax]*(P[1:L, 2:M, 1:N] - P[1:L, 1:M-1, 1:N])/hy)
    Vp = vJacobi(Hy)

    # Calculating guessed values of W implicitly
    Hz[1:L, 1:M, 1:N-1] = W[1:L, 1:M, 1:N-1] + dt*(Hz[1:L, 1:M, 1:N-1] - zt_zColl[1:N-1]*(P[1:L, 1:M, 2:N] - P[1:L, 1:M, 1:N-1])/hz)
    Wp = wJacobi(Hz)

    # Calculating pressure correction term
    rhs = np.zeros([L+1, M+1, N+1])
    rhs[1:L, 1:M, 1:N] = (xi_xStag[0:L-1, npax, npax]*(Up[1:L, 1:M, 1:N] - Up[0:L-1, 1:M, 1:N])/hx +
                          et_yStag[0:M-1, npax]*(Vp[1:L, 1:M, 1:N] - Vp[1:L, 0:M-1, 1:N])/hy +
                          zt_zStag[0:N-1]*(Wp[1:L, 1:M, 1:N] - Wp[1:L, 1:M, 0:N-1])/hz)/dt
    Pp = multigrid(rhs)

    # Add pressure correction.
    P = P + Pp

    # Update new values for U, V and W
    U[1:L-1, 1:M, 1:N] = Up[1:L-1, 1:M, 1:N] - dt*xi_xColl[1:L-1, npax, npax]*(Pp[2:L, 1:M, 1:N] - Pp[1:L-1, 1:M, 1:N])/hx
    V[1:L, 1:M-1, 1:N] = Vp[1:L, 1:M-1, 1:N] - dt*et_yColl[1:M-1, npax]*(Pp[1:L, 2:M, 1:N] - Pp[1:L, 1:M-1, 1:N])/hy
    W[1:L, 1:M, 1:N-1] = Wp[1:L, 1:M, 1:N-1] - dt*zt_zColl[1:N-1]*(Pp[1:L, 1:M, 2:N] - Pp[1:L, 1:M, 1:N-1])/hz

    # Impose no-slip BC on new values of U, V and W
    U = bc.imposeUBCs(U)
    V = bc.imposeVBCs(V)
    W = bc.imposeWBCs(W)


def computeNLinDiff_X():
    global Hx, Re
    global N, M, L
    global U, V, W
    global xixxColl, xix2Coll, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    Hx[1:L-1, 1:M, 1:N] = ((xixxColl[1:L-1, npax, npax]*D_Xi(U, L-1, M, N) + etyyStag[0:M-1, npax]*D_Et(U, L-1, M, N) + ztzzStag[0:N-1]*D_Zt(U, L-1, M, N))/Re +
                           (xix2Coll[1:L-1, npax, npax]*DDXi(U, L-1, M, N) + ety2Stag[0:M-1, npax]*DDEt(U, L-1, M, N) + ztz2Stag[0:N-1]*DDZt(U, L-1, M, N))*0.5/Re -
                            xi_xColl[1:L-1, npax, npax]*D_Xi(U, L-1, M, N)*U[1:L-1, 1:M, 1:N] -
                      0.25*(V[1:L-1, 0:M-1, 1:N] + V[1:L-1, 1:M, 1:N] + V[2:L, 1:M, 1:N] + V[2:L, 0:M-1, 1:N])*et_yStag[0:M-1, npax]*D_Et(U, L-1, M, N) - 
                      0.25*(W[1:L-1, 1:M, 0:N-1] + W[1:L-1, 1:M, 1:N] + W[2:L, 1:M, 1:N] + W[2:L, 1:M, 0:N-1])*zt_zStag[0:N-1]*D_Zt(U, L-1, M, N))


def computeNLinDiff_Y():
    global Hy, Re
    global N, M, L
    global U, V, W
    global xixxStag, xix2Stag, etyyColl, ety2Coll, ztzzStag, ztz2Stag

    Hy[1:L, 1:M-1, 1:N] = ((xixxStag[0:L-1, npax, npax]*D_Xi(V, L, M-1, N) + etyyColl[1:M-1, npax]*D_Et(V, L, M-1, N) + ztzzStag[0:N-1]*D_Zt(V, L, M-1, N))/Re +
                           (xix2Stag[0:L-1, npax, npax]*DDXi(V, L, M-1, N) + ety2Coll[1:M-1, npax]*DDEt(V, L, M-1, N) + ztz2Stag[0:N-1]*DDZt(V, L, M-1, N))*0.5/Re -
                                                                             et_yColl[1:M-1, npax]*D_Et(V, L, M-1, N)*V[1:L, 1:M-1, 1:N] -
                      0.25*(U[0:L-1, 1:M-1, 1:N] + U[1:L, 1:M-1, 1:N] + U[1:L, 2:M, 1:N] + U[0:L-1, 2:M, 1:N])*xi_xStag[0:L-1, npax, npax]*D_Xi(V, L, M-1, N) -
                      0.25*(W[1:L, 1:M-1, 0:N-1] + W[1:L, 1:M-1, 1:N] + W[1:L, 2:M, 1:N] + W[1:L, 2:M, 0:N-1])*zt_zStag[0:N-1]*D_Zt(V, L, M-1, N))


def computeNLinDiff_Z():
    global Hz, Re
    global N, M, L
    global U, V, W
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzColl, ztz2Coll

    Hz[1:L, 1:M, 1:N-1] = ((xixxStag[0:L-1, npax, npax]*D_Xi(W, L, M, N-1) + etyyStag[0:M-1, npax]*D_Et(W, L, M, N-1) + ztzzColl[1:N-1]*D_Zt(W, L, M, N-1))/Re +
                           (xix2Stag[0:L-1, npax, npax]*DDXi(W, L, M, N-1) + ety2Stag[0:M-1, npax]*DDEt(W, L, M, N-1) + ztz2Coll[1:N-1]*DDZt(W, L, M, N-1))*0.5/Re -
                                                                                                                        zt_zColl[1:N-1]*D_Zt(W, L, M, N-1)*W[1:L, 1:M, 1:N-1] -
                      0.25*(U[0:L-1, 1:M, 1:N-1] + U[1:L, 1:M, 1:N-1] + U[1:L, 1:M, 2:N] + U[0:L-1, 1:M, 2:N])*xi_xStag[0:L-1, npax, npax]*D_Xi(W, L, M, N-1) -
                      0.25*(V[1:L, 0:M-1, 1:N-1] + V[1:L, 1:M, 1:N-1] + V[1:L, 1:M, 2:N] + V[1:L, 0:M-1, 2:N])*et_yStag[0:M-1, npax]*D_Et(W, L, M, N-1))


def DDXi(inpFld, Nx, Ny, Nz):
    global hx

    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[0:Nx-1, 1:Ny, 1:Nz] - 2.0*inpFld[1:Nx, 1:Ny, 1:Nz] + inpFld[2:Nx+1, 1:Ny, 1:Nz])/(hx*hx)

    return outFld[1:Nx, 1:Ny, 1:Nz]


def DDEt(inpFld, Nx, Ny, Nz):
    global hy

    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 0:Ny-1, 1:Nz] - 2.0*inpFld[1:Nx, 1:Ny, 1:Nz] + inpFld[1:Nx, 2:Ny+1, 1:Nz])/(hy*hy)

    return outFld[1:Nx, 1:Ny, 1:Nz]


def DDZt(inpFld, Nx, Ny, Nz):
    global hz

    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 1:Ny, 0:Nz-1] - 2.0*inpFld[1:Nx, 1:Ny, 1:Nz] + inpFld[1:Nx, 1:Ny, 2:Nz+1])/(hz*hz)

    return outFld[1:Nx, 1:Ny, 1:Nz]


def D_Xi(inpFld, Nx, Ny, Nz):
    global hx

    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[2:Nx+1, 1:Ny, 1:Nz] - inpFld[0:Nx-1, 1:Ny, 1:Nz])*0.5/hx

    return outFld[1:Nx, 1:Ny, 1:Nz]


def D_Et(inpFld, Nx, Ny, Nz):
    global hy

    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 2:Ny+1, 1:Nz] - inpFld[1:Nx, 0:Ny-1, 1:Nz])*0.5/hy

    return outFld[1:Nx, 1:Ny, 1:Nz]


def D_Zt(inpFld, Nx, Ny, Nz):
    global hz

    outFld = np.zeros_like(inpFld)
    outFld[1:Nx, 1:Ny, 1:Nz] = (inpFld[1:Nx, 1:Ny, 2:Nz+1] - inpFld[1:Nx, 1:Ny, 0:Nz-1])*0.5/hz

    return outFld[1:Nx, 1:Ny, 1:Nz]


#Jacobi iterative solver for U
def uJacobi(rho):
    global Re, dt
    global L, N, M
    global maxCount
    global tolerance
    global hx, hy, hz
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
        next_sol = bc.imposeUBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L-1, 1:M, 1:N] = next_sol[1:L-1,   1:M, 1:N] - (
                                    xix2Coll[1:L-1, npax, npax]*DDXi(next_sol, L-1, M, N) + \
                                    ety2Stag[0:M-1, npax]*DDEt(next_sol, L-1, M, N) + \
                                    ztz2Stag[0:N-1]*DDZt(next_sol, L-1, M, N))*0.5*dt/Re

        error_temp = np.fabs(rho[1:L-1, 1:M, 1:N] - test_sol[1:L-1, 1:M, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            if gv.iCnt % gv.opInt == 0:
                print "Jacobi solver for U converged in ", jCnt, " iterations"
            break

        jCnt += 1
        if jCnt > maxCount:
            print "ERROR: Jacobi not converging in U. Aborting"
            print "Maximum error: ", maxErr
            quit()

    return prev_sol


#Jacobi iterative solver for V
def vJacobi(rho):
    global Re, dt
    global L, N, M
    global maxCount
    global tolerance
    global hx, hy, hz
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
        next_sol = bc.imposeVBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L, 1:M-1, 1:N] = next_sol[  1:L, 1:M-1, 1:N] - (
                                    xix2Stag[0:L-1, npax, npax]*DDXi(next_sol, L, M-1, N) + \
                                    ety2Coll[1:M-1, npax]*DDEt(next_sol, L, M-1, N) + \
                                    ztz2Stag[0:N-1]*DDZt(next_sol, L, M-1, N))*0.5*dt/Re

        error_temp = np.fabs(rho[1:L, 1:M-1, 1:N] - test_sol[1:L, 1:M-1, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            if gv.iCnt % gv.opInt == 0:
                print "Jacobi solver for V converged in ", jCnt, " iterations"
            break

        jCnt += 1
        if jCnt > maxCount:
            print "ERROR: Jacobi not converging in V. Aborting"
            print "Maximum error: ", maxErr
            quit()

    return prev_sol


#Jacobi iterative solver for W
def wJacobi(rho):
    global Re, dt
    global L, N, M
    global maxCount
    global tolerance
    global hx, hy, hz
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
        next_sol = bc.imposeWBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L, 1:M, 1:N-1] = next_sol[  1:L, 1:M, 1:N-1] - (
                                    xix2Stag[0:L-1, npax, npax]*DDXi(next_sol, L, M, N-1) + \
                                    ety2Stag[0:M-1, npax]*DDEt(next_sol, L, M, N-1) + \
                                    ztz2Coll[1:N-1]*DDZt(next_sol, L, M, N-1))*0.5*dt/Re

        error_temp = np.fabs(rho[1:L, 1:M, 1:N-1] - test_sol[1:L, 1:M, 1:N-1])
        maxErr = np.amax(error_temp)
        if maxErr < tolerance:
            if gv.iCnt % gv.opInt == 0:
                print "Jacobi solver for W converged in ", jCnt, " iterations"
            break

        jCnt += 1
        if jCnt > maxCount:
            print "ERROR: Jacobi not converging in W. Aborting"
            print "Maximum error: ", maxErr
            quit()

    return prev_sol


######################################## MULTIGRID SOLVER ##########################################


#Multigrid solver
def multigrid(H):
    global N, M, L, vcCnt

    Pp = np.zeros([L+1, M+1, N+1])
    chMat = np.ones([L+1, M+1, N+1])
    for i in range(vcCnt):
        Pp = v_cycle(Pp, H)
        chMat = laplace(Pp)

    print "Error after multigrid is ", np.amax(np.abs(H[1:L, 1:M, 1:N] - chMat[1:L, 1:M, 1:N]))

    return Pp


#Multigrid solution without the use of recursion
def v_cycle(P, H):
    global hx, hy, hz
    global sInd, VDepth
    global preSm, pstSm, proSm

    # Pre-smoothing
    P = smooth(P, H, hx, hy, hz, preSm, 0)

    H_rsdl = H - laplace(P)

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
        toSmooth = bc.imposePBCs(smoothed)

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
        solLap[1:L, 1:M, 1:N] = xix2Stag[0::2**VDepth, npax, npax]*DDXi(next_sol, L, M, N)/((2**VDepth)**2) + \
                                xixxStag[0::2**VDepth, npax, npax]*D_Xi(next_sol, L, M, N)/(2**VDepth) + \
                                ety2Stag[0::2**VDepth, npax]*DDEt(next_sol, L, M, N)/((2**VDepth)**2) + \
                                etyyStag[0::2**VDepth, npax]*D_Et(next_sol, L, M, N)/(2**VDepth) + \
                                ztz2Stag[0::2**VDepth]*DDZt(next_sol, L, M, N)/((2**VDepth)**2) + \
                                ztzzStag[0::2**VDepth]*D_Zt(next_sol, L, M, N)/(2**VDepth)

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


def laplace(function):
    '''
Function to calculate the Laplacian for a given field of values.
INPUT:  function: 3D matrix of double precision values
OUTPUT: gradient: 3D matrix of double precision values with same size as input matrix
    '''
    global hx, hy, hz

    # 1 subtracted from shape to account for ghost points
    [L, M, N] = np.array(np.shape(function)) - 1
    gradient = np.zeros_like(function)

    gradient[1:L, 1:M, 1:N] = xix2Stag[0:L-1, npax, npax]*DDXi(function, L, M, N) + \
                              xixxStag[0:L-1, npax, npax]*D_Xi(function, L, M, N) + \
                                    ety2Stag[0:M-1, npax]*DDEt(function, L, M, N) + \
                                    etyyStag[0:M-1, npax]*D_Et(function, L, M, N) + \
                                          ztz2Stag[0:N-1]*DDZt(function, L, M, N) + \
                                          ztzzStag[0:N-1]*D_Zt(function, L, M, N)

    return gradient


############################################ LES CODE ##############################################


def addTurbViscosity(xStr, xEnd):
    global nu
    global U, V, W
    global L, M, N
    global hx, hy, hz

    xInd = 0
    subNx = xEnd - xStr

    vList = np.zeros((subNx, M-2, N-2))

    for i in range(xStr, xEnd):
        for j in range(1, M-1):
            for k in range(1, N-1):
                # Compute all the necessary velocity gradients in each cell
                dudx = (U[i+1, j, k] - U[i, j, k])/hx
                dudy = (U[i, j+1, k] - U[i, j, k])/hy
                dudz = (U[i, j, k+1] - U[i, j, k])/hz

                dvdx = (V[i+1, j, k] - V[i, j, k])/hx
                dvdy = (V[i, j+1, k] - V[i, j, k])/hy
                dvdz = (V[i, j, k+1] - V[i, j, k])/hz

                dwdx = (W[i+1, j, k] - W[i, j, k])/hx
                dwdy = (W[i, j+1, k] - W[i, j, k])/hy
                dwdz = (W[i, j, k+1] - W[i, j, k])/hz

                # Construct the resolved strain-rate tensor
                S = np.matrix([[dudx, (dudy + dvdx)*0.5, (dudz + dwdx)*0.5],
                               [(dvdx + dudy)*0.5, dvdy, (dvdz + dwdy)*0.5],
                               [(dwdx + dudz)*0.5, (dwdy + dvdz)*0.5, dwdz]])

                # Obtain the eigenvalues and eigenvectors of the strain-rate tensor
                eVals, eVecs = np.linalg.eig(S)

                # Sort the eigenvalues and eigenvectors in decreasing order
                sortedIndices = np.argsort(eVals)[::-1]
                eVecs = eVecs[:, sortedIndices]

                # Get the intermediate eigenvector along which the sub-grid vortex is assumed to be aligned
                ev = eVecs[:,1]

                # Compute the stretching felt along subgrid vortex axis
                a = 0.0
                for i in range(3):
                    for j in range(3):
                        a += S[i,j]*ev[i]*ev[j]

                # Compute the lambda coefficient used to compute the 
                lambda_v = 2.0*nu/(3.0*a)

        xInd += 1

    return vList


def getDiv():
    '''
Function to calculate the divergence within the domain (excluding walls)
INPUT:  U, V, W: Velocity values
OUTPUT: The maximum value of divergence in double precision
    '''
    global N, M, L
    global U, V, W
    global xColl, yColl, zColl

    divMat = np.zeros([L+1, M+1, N+1])
    for i in range(1, L):
        for j in range(1, M):
            for k in range(1, N):
                divMat[i, j, k] = (U[i, j, k] - U[i-1, j, k])/(xColl[i] - xColl[i-1]) + \
                                  (V[i, j, k] - V[i, j-1, k])/(yColl[j] - yColl[j-1]) + \
                                  (W[i, j, k] - W[i, j, k-1])/(zColl[k] - zColl[k-1])

    return np.unravel_index(divMat.argmax(), divMat.shape), np.amax(divMat)


def writeSoln(time):
    global N, M, L
    global U, V, W, P
    global xColl, yColl, zColl

    if gv.fwMode == "ASCII":
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
                                (P[k, j, i] + P[k+1, j, i] + P[k, j+1, i] + P[k+1, j+1, i] +
                                 P[k, j, i+1] + P[k+1, j, i+1] + P[k, j+1, i+1] + P[k+1, j+1, i+1])/8.0))

        ofFile.close()

    elif gv.fwMode == "HDF5":
        fName = "Soln_" + "{0:09.5f}.h5".format(time)
        print "Writing solution file: ", fName

        f = hp.File(fName, "w")

        dset = f.create_dataset("U", data = U)
        dset = f.create_dataset("V", data = V)
        dset = f.create_dataset("W", data = W)
        dset = f.create_dataset("P", data = P)

        f.close()


def calculateMetrics():
    global xColl, yColl, zColl
    global xStag, yStag, zStag
    global xi_xColl, et_yColl, zt_zColl
    global xi_xStag, et_yStag, zt_zStag
    global xixxColl, xix2Coll, etyyColl, ety2Coll, ztzzColl, ztz2Coll
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    xi = np.linspace(0.0, 1.0, L)
    et = np.linspace(0.0, 1.0, M)
    zt = np.linspace(0.0, 1.0, N)

    xLen = gv.dLen[0]
    yLen = gv.dLen[1]
    zLen = gv.dLen[2]

    xBeta = gv.beta[0]
    yBeta = gv.beta[1]
    zBeta = gv.beta[2]

    # Calculate grid and its metrics
    xColl = [xLen*(1.0 - np.tanh(xBeta*(1.0 - 2.0*i))/np.tanh(xBeta))/2.0 for i in xi]
    yColl = [yLen*(1.0 - np.tanh(yBeta*(1.0 - 2.0*i))/np.tanh(yBeta))/2.0 for i in et]
    zColl = [zLen*(1.0 - np.tanh(zBeta*(1.0 - 2.0*i))/np.tanh(zBeta))/2.0 for i in zt]

    xStag = [xLen*(1.0 - np.tanh(xBeta*(1.0 - 2.0*i))/np.tanh(xBeta))/2.0 for i in [(xi[j] + xi[j+1])/2 for j in range(len(xi) - 1)]]
    yStag = [yLen*(1.0 - np.tanh(yBeta*(1.0 - 2.0*i))/np.tanh(yBeta))/2.0 for i in [(et[j] + et[j+1])/2 for j in range(len(et) - 1)]]
    zStag = [zLen*(1.0 - np.tanh(zBeta*(1.0 - 2.0*i))/np.tanh(zBeta))/2.0 for i in [(zt[j] + zt[j+1])/2 for j in range(len(zt) - 1)]]

    # Grid metrics for both staggered and collocated grids
    xi_xColl = np.array([np.tanh(xBeta)/(xBeta*xLen*(1.0 - ((1.0 - 2.0*k/xLen)*np.tanh(xBeta))**2.0)) for k in xColl])
    xixxColl = np.array([-4.0*(np.tanh(xBeta)**3.0)*(1.0 - 2.0*k/xLen)/(xBeta*xLen*xLen*(1.0 - (np.tanh(xBeta)*(1.0 - 2.0*k/xLen)**2.0)**2.0)) for k in xColl])
    xix2Coll = np.array([k*k for k in xi_xColl])

    xi_xStag = np.array([np.tanh(xBeta)/(xBeta*xLen*(1.0 - ((1.0 - 2.0*k/xLen)*np.tanh(xBeta))**2.0)) for k in xStag])
    xixxStag = np.array([-4.0*(np.tanh(xBeta)**3.0)*(1.0 - 2.0*k/xLen)/(xBeta*xLen*xLen*(1.0 - (np.tanh(xBeta)*(1.0 - 2.0*k/xLen)**2.0)**2.0)) for k in xStag])
    xix2Stag = np.array([k*k for k in xi_xStag])

    et_yColl = np.array([np.tanh(yBeta)/(yBeta*yLen*(1.0 - ((1.0 - 2.0*j/yLen)*np.tanh(yBeta))**2.0)) for j in yColl])
    etyyColl = np.array([-4.0*(np.tanh(yBeta)**3.0)*(1.0 - 2.0*j/yLen)/(yBeta*yLen*yLen*(1.0 - (np.tanh(yBeta)*(1.0 - 2.0*j/yLen)**2.0)**2.0)) for j in yColl])
    ety2Coll = np.array([j*j for j in et_yColl])

    et_yStag = np.array([np.tanh(yBeta)/(yBeta*yLen*(1.0 - ((1.0 - 2.0*j/yLen)*np.tanh(yBeta))**2.0)) for j in yStag])
    etyyStag = np.array([-4.0*(np.tanh(yBeta)**3.0)*(1.0 - 2.0*j/yLen)/(yBeta*yLen*yLen*(1.0 - (np.tanh(yBeta)*(1.0 - 2.0*j/yLen)**2.0)**2.0)) for j in yStag])
    ety2Stag = np.array([j*j for j in et_yStag])

    zt_zColl = np.array([np.tanh(zBeta)/(zBeta*zLen*(1.0 - ((1.0 - 2.0*i/zLen)*np.tanh(zBeta))**2.0)) for i in zColl])
    ztzzColl = np.array([-4.0*(np.tanh(zBeta)**3.0)*(1.0 - 2.0*i/zLen)/(zBeta*zLen*zLen*(1.0 - (np.tanh(zBeta)*(1.0 - 2.0*i/zLen)**2.0)**2.0)) for i in zColl])
    ztz2Coll = np.array([i*i for i in zt_zColl])

    zt_zStag = np.array([np.tanh(zBeta)/(zBeta*zLen*(1.0 - ((1.0 - 2.0*i/zLen)*np.tanh(zBeta))**2.0)) for i in zStag])
    ztzzStag = np.array([-4.0*(np.tanh(zBeta)**3.0)*(1.0 - 2.0*i/zLen)/(zBeta*zLen*zLen*(1.0 - (np.tanh(zBeta)*(1.0 - 2.0*i/zLen)**2.0)**2.0)) for i in zStag])
    ztz2Stag = np.array([i*i for i in zt_zStag])


# Main segment of code.
def main():
    global dt, tMax, fwInt, opInt
    global L, M, N
    global time
    global U

    maxProcs = mp.cpu_count()
    if gv.nProcs > maxProcs:
        print("\nERROR: " + str(gv.nProcs) + " exceeds the available number of processors (" + str(maxProcs) + ")\n")
        exit(0)
    else:
        print("\nUsing " + str(gv.nProcs) + " out of " + str(maxProcs) + " processors\n")

    calculateMetrics()

    if gv.probType == 0:
        # BC for moving top lid - U = 1.0 on lid
        U[:, :, N] = 1.0
    elif gv.probType == 1:
        #h = np.linspace(0.0, zLen, N+1)
        #U = 0.1*np.random.rand(L, M+1, N+1)
        U[:, :, :] = 1.0

    fwTime = 0.0

    while True:
        if abs(fwTime - time) < 0.5*dt:
            writeSoln(time)
            fwTime += fwInt

        euler()

        maxDiv = getDiv()
        if maxDiv[1] > 10.0:
            print "ERROR: Divergence has exceeded permissible limits. Aborting"
            quit()

        gv.iCnt += 1
        time += dt
        if gv.iCnt % gv.opInt == 0:
            print "Time: {0:9.5f}".format(time)
            print "Maximum divergence: {0:8.5f} at ({1:d}, {2:d}, {3:d})\n".format(maxDiv[1], maxDiv[0][0], maxDiv[0][1], maxDiv[0][2])

        if time > tMax:
            break

    print "Simulation completed"


####################################################################################################

# N should be of the form 2^n + 2 so that there will be 2^n + 3 staggered pressure points, including ghost points
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

# Define grid spacings
hx = 1.0/(L-1)
hy = 1.0/(M-1)
hz = 1.0/(N-1)

# Maximum number of jacobi iterations
maxCount = 10*N*M*L

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
nu = 1.0/Re

# Create list of ranges (in terms of indices) along X direction, which is the direction of parallelization
rangeDivs = [int(x) for x in np.linspace(1, L-1, gv.nProcs+1)]
rListColl = [(rangeDivs[x], rangeDivs[x+1]) for x in range(gv.nProcs)]
rangeDivs = [int(x) for x in np.linspace(1, L, gv.nProcs+1)]
rListStag = [(rangeDivs[x], rangeDivs[x+1]) for x in range(gv.nProcs)]

# Create and initialize U, V and P arrays
# The arrays have two extra points
# These act as ghost points on either sides of the domain
P = np.ones([L+1, M+1, N+1])

# U and Hx (RHS of X component of NSE) are staggered in Y and Z directions and hence has one extra point along these directions
U = np.zeros([L, M+1, N+1])
Hx = np.zeros([L, M+1, N+1])

# V and Hy (RHS of Y component of NSE) are staggered in X and Z directions and hence has one extra point along these directions
V = np.zeros([L+1, M, N+1])
Hy = np.zeros([L+1, M, N+1])

# W and Hz (RHS of Z component of NSE) are staggered in X and Y directions and hence has one extra point along these directions
W = np.zeros([L+1, M+1, N])
Hz = np.zeros([L+1, M+1, N])

main()
