#!/usr/bin/python3

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
from orion.solvers import poissonSolverNU_d3 as ps
from orion import boundaryConditions as bc
from orion import calculateFD as fd
from orion import meshData as grid
from orion import globalVars as gv
from orion import vortexLES as les
from orion import writeData as dw
import numpy as np
import time

# Redefine frequently used numpy object
npax = np.newaxis

# Get limits from grid object
L, M, N = grid.L, grid.M, grid.N

def initFields():
    global L, M, N
    global U, V, W, P
    global Hx, Hy, Hz, Pp

    # Create and initialize U, V, W and P arrays
    # The arrays have two extra points
    # These act as ghost points on either sides of the domain
    P = np.ones([L + 2, M + 2, N + 2])

    # U is collocated along X direction and hence has one less point along that direction
    U = np.zeros([L + 1, M + 2, N + 2])

    # V is collocated along Y direction and hence has one less point along that direction
    V = np.zeros([L + 2, M + 1, N + 2])

    # W is collocated along Z direction and hence has one less point along that direction
    W = np.zeros([L + 2, M + 2, N + 1])

    # Define arrays for storing RHS of NSE
    Hx = np.zeros_like(U)
    Hy = np.zeros_like(V)
    Hz = np.zeros_like(W)
    Pp = np.zeros_like(P)

    if gv.probType == 0:
        # For moving top lid, U = 1.0 on lid, and second last point lies on the wall
        U[:, :, -2] = 1.0
    elif gv.probType == 1:
        # Initial condition for forced channel flow
        U[:, :, :] = 1.0

    ps.initVariables()


def euler():
    global N, M, L
    global U, V, W, P
    global Hx, Hy, Hz, Pp

    Hx.fill(0.0)
    Hy.fill(0.0)
    Hz.fill(0.0)

    computeNLinDiff_X(U, V, W)
    computeNLinDiff_Y(U, V, W)
    computeNLinDiff_Z(U, V, W)

    # Add constant pressure gradient forcing for channel flows
    if gv.probType == 1:
        Hx[:, :, :] += 1.0

    # Calculating guessed values of U implicitly
    Hx[1:L, 1:M+1, 1:N+1] = U[1:L, 1:M+1, 1:N+1] + gv.dt*(Hx[1:L, 1:M+1, 1:N+1] - grid.xi_xColl[:, npax, npax]*(P[2:L+1, 1:M+1, 1:N+1] - P[1:L, 1:M+1, 1:N+1])/grid.hx)
    uJacobi(Hx)

    # Calculating guessed values of V implicitly
    Hy[1:L+1, 1:M, 1:N+1] = V[1:L+1, 1:M, 1:N+1] + gv.dt*(Hy[1:L+1, 1:M, 1:N+1] - grid.et_yColl[:, npax]*(P[1:L+1, 2:M+1, 1:N+1] - P[1:L+1, 1:M, 1:N+1])/grid.hy)
    vJacobi(Hy)

    # Calculating guessed values of W implicitly
    Hz[1:L+1, 1:M+1, 1:N] = W[1:L+1, 1:M+1, 1:N] + gv.dt*(Hz[1:L+1, 1:M+1, 1:N] - grid.zt_zColl[:]*(P[1:L+1, 1:M+1, 2:N+1] - P[1:L+1, 1:M+1, 1:N])/grid.hz)
    wJacobi(Hz)

    # Calculating pressure correction term
    rhs = np.zeros([L+2, M+2, N+2])
    rhs[1:L+1, 1:M+1, 1:N+1] = ((U[1:L+1, 1:M+1, 1:N+1] - U[0:L, 1:M+1, 1:N+1])*grid.xi_xStag[:, npax, npax]/grid.hx +
                                (V[1:L+1, 1:M+1, 1:N+1] - V[1:L+1, 0:M, 1:N+1])*grid.et_yStag[:, npax]/grid.hy +
                                (W[1:L+1, 1:M+1, 1:N+1] - W[1:L+1, 1:M+1, 0:N])*grid.zt_zStag[:]/grid.hz)/gv.dt

    ps.multigrid(Pp, rhs)

    # Add pressure correction.
    P = P + Pp

    # Update new values for U, V and W
    U[1:L, 1:M+1, 1:N+1] = U[1:L, 1:M+1, 1:N+1] - gv.dt*(Pp[2:L+1, 1:M+1, 1:N+1] - Pp[1:L, 1:M+1, 1:N+1])*grid.xi_xColl[:, npax, npax]/grid.hx
    V[1:L+1, 1:M, 1:N+1] = V[1:L+1, 1:M, 1:N+1] - gv.dt*(Pp[1:L+1, 2:M+1, 1:N+1] - Pp[1:L+1, 1:M, 1:N+1])*grid.et_yColl[:, npax]/grid.hy
    W[1:L+1, 1:M+1, 1:N] = W[1:L+1, 1:M+1, 1:N] - gv.dt*(Pp[1:L+1, 1:M+1, 2:N+1] - Pp[1:L+1, 1:M+1, 1:N])*grid.zt_zColl[:]/grid.hz

    # Impose no-slip BC on new values of U, V and W
    U = bc.imposeUBCs(U)
    V = bc.imposeVBCs(V)
    W = bc.imposeWBCs(W)

    print(U[30, 30, 30], V[30, 30, 30], W[30, 30, 30])


def computeNLinDiff_X(U, V, W):
    global Hx
    global N, M, L

    Hx[1:-1, 1:-1, 1:-1] = ((grid.xixxColl[:, npax, npax]*fd.D_Xi(U) + grid.etyyStag[:, npax]*fd.D_Et(U) + grid.ztzzStag[:]*fd.D_Zt(U))/gv.Re +
                            (grid.xix2Coll[:, npax, npax]*fd.DDXi(U) + grid.ety2Stag[:, npax]*fd.DDEt(U) + grid.ztz2Stag[:]*fd.DDZt(U))*0.5/gv.Re -
                             grid.xi_xColl[:, npax, npax]*fd.D_Xi(U)*U[1:-1, 1:-1, 1:-1] -
                      0.25*(V[1:L, 0:M, 1:N+1] + V[1:L, 1:M+1, 1:N+1] + V[2:L+1, 1:M+1, 1:N+1] + V[2:L+1, 0:M, 1:N+1])*grid.et_yStag[:, npax]*fd.D_Et(U) - 
                      0.25*(W[1:L, 1:M+1, 0:N] + W[1:L, 1:M+1, 1:N+1] + W[2:L+1, 1:M+1, 1:N+1] + W[2:L+1, 1:M+1, 0:N])*grid.zt_zStag[:]*fd.D_Zt(U))


def computeNLinDiff_Y(U, V, W):
    global Hy
    global N, M, L

    Hy[1:-1, 1:-1, 1:-1] = ((grid.xixxStag[:, npax, npax]*fd.D_Xi(V) + grid.etyyColl[:, npax]*fd.D_Et(V) + grid.ztzzStag[:]*fd.D_Zt(V))/gv.Re +
                            (grid.xix2Stag[:, npax, npax]*fd.DDXi(V) + grid.ety2Coll[:, npax]*fd.DDEt(V) + grid.ztz2Stag[:]*fd.DDZt(V))*0.5/gv.Re -
                                                                       grid.et_yColl[:, npax]*fd.D_Et(V)*V[1:-1, 1:-1, 1:-1] -
                      0.25*(U[0:L, 1:M, 1:N+1] + U[1:L+1, 1:M, 1:N+1] + U[1:L+1, 2:M+1, 1:N+1] + U[0:L, 2:M+1, 1:N+1])*grid.xi_xStag[:, npax, npax]*fd.D_Xi(V) -
                      0.25*(W[1:L+1, 1:M, 0:N] + W[1:L+1, 1:M, 1:N+1] + W[1:L+1, 2:M+1, 1:N+1] + W[1:L+1, 2:M+1, 0:N])*grid.zt_zStag[:]*fd.D_Zt(V))


def computeNLinDiff_Z(U, V, W):
    global Hz
    global N, M, L

    Hz[1:-1, 1:-1, 1:-1] = ((grid.xixxStag[:, npax, npax]*fd.D_Xi(W) + grid.etyyStag[:, npax]*fd.D_Et(W) + grid.ztzzColl[:]*fd.D_Zt(W))/gv.Re +
                            (grid.xix2Stag[:, npax, npax]*fd.DDXi(W) + grid.ety2Stag[:, npax]*fd.DDEt(W) + grid.ztz2Coll[:]*fd.DDZt(W))*0.5/gv.Re -
                                                                                                           grid.zt_zColl[:]*fd.D_Zt(W)*W[1:-1, 1:-1, 1:-1] -
                      0.25*(U[0:L, 1:M+1, 1:N] + U[1:L+1, 1:M+1, 1:N] + U[1:L+1, 1:M+1, 2:N+1] + U[0:L, 1:M+1, 2:N+1])*grid.xi_xStag[:, npax, npax]*fd.D_Xi(W) -
                      0.25*(V[1:L+1, 0:M, 1:N] + V[1:L+1, 1:M+1, 1:N] + V[1:L+1, 1:M+1, 2:N+1] + V[1:L+1, 0:M, 2:N+1])*grid.et_yStag[:, npax]*fd.D_Et(W))


#Jacobi iterative solver for U
def uJacobi(rho):
    global U
    global L, M, N

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[1:L, 2:M, 2:N] = ((grid.hy2hz2*(U[0:L-1, 2:M, 2:N] + U[2:L+1, 2:M, 2:N])*grid.xix2Coll[:, npax, npax] +
                                    grid.hz2hx2*(U[1:L, 1:M-1, 2:N] + U[1:L, 3:M+1, 2:N])*grid.ety2Stag[1:-1, npax] +
                                    grid.hx2hy2*(U[1:L, 2:M, 1:N-1] + U[1:L, 2:M, 3:N+1])*grid.ztz2Stag[1:-1])*
                             gv.dt/(grid.hx2hy2hz2*2.0*gv.Re) + rho[1:L, 2:M, 2:N])/ \
                      (1.0 + gv.dt*(grid.hy2hz2*grid.xix2Coll[:, npax, npax] +
                                    grid.hz2hx2*grid.ety2Stag[1:-1, npax] +
                                    grid.hx2hy2*grid.ztz2Stag[1:-1])/(gv.Re*grid.hx2hy2hz2))

        # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
        U, temp_sol = temp_sol, U
        U = bc.imposeUBCs(U)

        maxErr = np.amax(np.fabs(rho[1:L, 2:M, 2:N] - (U[1:L, 2:M, 2:N] - 0.5*gv.dt*(
                        (U[0:L-1, 2:M, 2:N] - 2.0*U[1:L, 2:M, 2:N] + U[2:L+1, 2:M, 2:N])*grid.xix2Coll[:, npax, npax]/grid.hx2 +
                        (U[1:L, 1:M-1, 2:N] - 2.0*U[1:L, 2:M, 2:N] + U[1:L, 3:M+1, 2:N])*grid.ety2Stag[1:-1, npax]/grid.hy2 +
                        (U[1:L, 2:M, 1:N-1] - 2.0*U[1:L, 2:M, 2:N] + U[1:L, 2:M, 3:N+1])*grid.ztz2Stag[1:-1]/grid.hz2)/gv.Re)))

        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in U. Aborting")
            print("Maximum error: ", maxErr)
            quit()


#Jacobi iterative solver for V
def vJacobi(rho):
    global V
    global L, M, N

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[2:L, 1:M, 2:N] = ((grid.hy2hz2*(V[1:L-1, 1:M, 2:N] + V[3:L+1, 1:M, 2:N])*grid.xix2Stag[1:-1, npax, npax] +
                                    grid.hz2hx2*(V[2:L, 0:M-1, 2:N] + V[2:L, 2:M+1, 2:N])*grid.ety2Coll[:, npax] +
                                    grid.hx2hy2*(V[2:L, 1:M, 1:N-1] + V[2:L, 1:M, 3:N+1])*grid.ztz2Stag[1:-1])*
                             gv.dt/(grid.hx2hy2hz2*2.0*gv.Re) + rho[2:L, 1:M, 2:N])/ \
                      (1.0 + gv.dt*(grid.hy2hz2*grid.xix2Stag[1:-1, npax, npax] +
                                    grid.hz2hx2*grid.ety2Coll[:, npax] +
                                    grid.hx2hy2*grid.ztz2Stag[1:-1])/(gv.Re*grid.hx2hy2hz2))

        # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
        V, temp_sol = temp_sol, V
        V = bc.imposeVBCs(V)

        maxErr = np.amax(np.fabs(rho[2:L, 1:M, 2:N] - (V[2:L, 1:M, 2:N] - 0.5*gv.dt*(
                        (V[1:L-1, 1:M, 2:N] - 2.0*V[2:L, 1:M, 2:N] + V[3:L+1, 1:M, 2:N])*grid.xix2Stag[1:-1, npax, npax]/grid.hx2 +
                        (V[2:L, 0:M-1, 2:N] - 2.0*V[2:L, 1:M, 2:N] + V[2:L, 2:M+1, 2:N])*grid.ety2Coll[:, npax]/grid.hy2 +
                        (V[2:L, 1:M, 1:N-1] - 2.0*V[2:L, 1:M, 2:N] + V[2:L, 1:M, 3:N+1])*grid.ztz2Stag[1:-1]/grid.hz2)/gv.Re)))

        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            print("Maximum error: ", maxErr)
            quit()


#Jacobi iterative solver for W
def wJacobi(rho):
    global W
    global L, M, N

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[2:L, 2:M, 1:N] = ((grid.hy2hz2*(W[1:L-1, 2:M, 1:N] + W[3:L+1, 2:M, 1:N])*grid.xix2Stag[1:-1, npax, npax] +
                                    grid.hz2hx2*(W[2:L, 1:M-1, 1:N] + W[2:L, 3:M+1, 1:N])*grid.ety2Stag[1:-1, npax] +
                                    grid.hx2hy2*(W[2:L, 2:M, 0:N-1] + W[2:L, 2:M, 2:N+1])*grid.ztz2Coll[:])*
                             gv.dt/(grid.hx2hy2hz2*2.0*gv.Re) + rho[2:L, 2:M, 1:N])/ \
                      (1.0 + gv.dt*(grid.hy2hz2*grid.xix2Stag[1:-1, npax, npax] +
                                    grid.hz2hx2*grid.ety2Stag[1:-1, npax] +
                                    grid.hx2hy2*grid.ztz2Coll[:])/(gv.Re*grid.hx2hy2hz2))

        # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
        W, temp_sol = temp_sol, W
        W = bc.imposeWBCs(W)

        maxErr = np.amax(np.fabs(rho[2:L, 2:M, 1:N] - (W[2:L, 2:M, 1:N] - 0.5*gv.dt*(
                        (W[1:L-1, 2:M, 1:N] - 2.0*W[2:L, 2:M, 1:N] + W[3:L+1, 2:M, 1:N])*grid.xix2Stag[1:-1, npax, npax]/grid.hx2 +
                        (W[2:L, 1:M-1, 1:N] - 2.0*W[2:L, 2:M, 1:N] + W[2:L, 3:M+1, 1:N])*grid.ety2Stag[1:-1, npax]/grid.hy2 +
                        (W[2:L, 2:M, 0:N-1] - 2.0*W[2:L, 2:M, 1:N] + W[2:L, 2:M, 2:N+1])*grid.ztz2Coll[:]/grid.hz2)/gv.Re)))

        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()


def getDiv():
    global N, M, L
    global U, V, W

    divMat = np.zeros([L, M, N])
    # This excludes the velocity difference across the wall. Hence limits start from 2
    for i in range(2, L):
        for j in range(2, M):
            for k in range(2, N):
                divMat[i, j, k] = (U[i, j, k] - U[i-1, j, k])/(grid.xColl[i-1] - grid.xColl[i-2]) + \
                                  (V[i, j, k] - V[i, j-1, k])/(grid.yColl[j-1] - grid.yColl[j-2]) + \
                                  (W[i, j, k] - W[i, j, k-1])/(grid.zColl[k-1] - grid.zColl[k-2])

    return np.unravel_index(divMat.argmax(), divMat.shape), np.amax(divMat)

def writeSoln(solTime):
    global U, V, W, P

    dw.writeSoln3D(U, V, W, P, solTime)

