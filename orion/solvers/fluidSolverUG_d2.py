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
from orion.solvers import poissonSolverUG_d2 as ps
from orion import boundaryConditions as bc
from orion import calculateFD as fd
from orion import meshData as grid
from orion import globalVars as gv
from orion import vortexLES as les
from orion import writeData as dw
import numpy as np
import time

# Get limits from grid object
L, N = grid.L, grid.N

def initFields():
    global L, N
    global Hx, Hz
    global U, W, P

    # Create and initialize U, W and P arrays
    # The arrays have two extra points
    # These act as ghost points on either sides of the domain
    P = np.ones([L + 2, N + 2])

    # U is collocated along X direction and hence has one less point along that direction
    U = np.zeros([L + 1, N + 2])

    # W is collocated along Z direction and hence has one less point along that direction
    W = np.zeros([L + 2, N + 1])

    # Define arrays for storing RHS of NSE
    Hx = np.zeros_like(U)
    Hz = np.zeros_like(W)

    if gv.probType == 0:
        # For moving top lid, U = 1.0 on lid, and second last point lies on the wall
        U[:, -2] = 1.0
    elif gv.probType == 1:
        # Initial condition for forced channel flow
        U[:, :] = 1.0

    ps.initVariables()

    if gv.testPoisson:
        ps.initDirichlet()


def euler():
    global N, L
    global Hx, Hz
    global U, W, P

    Hx.fill(0.0)
    Hz.fill(0.0)

    computeNLinDiff_X(U, W)
    computeNLinDiff_Z(U, W)

    # Add constant pressure gradient forcing for channel flows
    if gv.probType == 1:
        Hx[:, :] += 1.0

    # Calculating guessed values of U implicitly
    Hx[1:L, 1:N+1] = U[1:L, 1:N+1] + gv.dt*(Hx[1:L, 1:N+1] - (P[2:L+1, 1:N+1] - P[1:L, 1:N+1])/grid.hx)
    uJacobi(Hx)

    # Calculating guessed values of W implicitly
    Hz[1:L+1, 1:N] = W[1:L+1, 1:N] + gv.dt*(Hz[1:L+1, 1:N] - (P[1:L+1, 2:N+1] - P[1:L+1, 1:N])/grid.hz)
    wJacobi(Hz)

    # Calculating pressure correction term
    rhs = np.zeros([L+2, N+2])
    rhs[1:L+1, 1:N+1] = ((U[1:L+1, 1:N+1] - U[0:L, 1:N+1])/grid.hx +
                         (W[1:L+1, 1:N+1] - W[1:L+1, 0:N])/grid.hz)/gv.dt

    Pp = ps.multigrid(rhs)

    # Add pressure correction.
    P = P + Pp

    # Update new values for U and W
    U[1:L, 1:N+1] = U[1:L, 1:N+1] - gv.dt*(Pp[2:L+1, 1:N+1] - Pp[1:L, 1:N+1])/grid.hx
    W[1:L+1, 1:N] = W[1:L+1, 1:N] - gv.dt*(Pp[1:L+1, 2:N+1] - Pp[1:L+1, 1:N])/grid.hz

    # Impose no-slip BC on new values of U and W
    U = bc.imposeUBCs(U)
    W = bc.imposeWBCs(W)

    print(U[30, 30], W[30, 30])


def computeNLinDiff_X(U, W):
    global Hx
    global N, L

    Hx[1:L, 1:N+1] = ((fd.DDXi(U) + fd.DDZt(U))*0.5/gv.Re -
                       fd.D_Xi(U)*U[1:L, 1:N+1] -
                      0.25*(W[1:L, 0:N] + W[1:L, 1:N+1] + W[2:L+1, 1:N+1] + W[2:L+1, 0:N])*fd.D_Zt(U))


def computeNLinDiff_Z(U, W):
    global Hz
    global N, L

    Hz[1:L+1, 1:N] = ((fd.DDXi(W) + fd.DDZt(W))*0.5/gv.Re -
                       fd.D_Zt(W)*W[1:L+1, 1:N] -
                      0.25*(U[0:L, 1:N] + U[1:L+1, 1:N] + U[1:L+1, 2:N+1] + U[0:L, 2:N+1])*fd.D_Xi(W))


#Jacobi iterative solver for U
def uJacobi(rho):
    global U
    global L, N

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[1:L, 2:N] = ((grid.hz2*(U[0:L-1, 2:N] + U[2:L+1, 2:N]) +
                               grid.hx2*(U[1:L, 1:N-1] + U[1:L, 3:N+1]))*
                        gv.dt/(grid.hz2hx2*2.0*gv.Re) + rho[1:L, 2:N]) / \
                 (1.0 + gv.dt*(grid.hz2 + grid.hx2)/(gv.Re*grid.hz2hx2))

        # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
        U, temp_sol = temp_sol, U
        U = bc.imposeUBCs(U)

        maxErr = np.amax(np.fabs(rho[1:L, 2:N] - (U[1:L, 2:N] - 0.5*gv.dt*(
                        (U[0:L-1, 2:N] - 2.0*U[1:L, 2:N] + U[2:L+1, 2:N])/grid.hx2 +
                        (U[1:L, 1:N-1] - 2.0*U[1:L, 2:N] + U[1:L, 3:N+1])/grid.hz2)/gv.Re)))

        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in U. Aborting")
            print("Maximum error: ", maxErr)
            quit()


#Jacobi iterative solver for W
def wJacobi(rho):
    global W
    global L, N

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[2:L, 1:N] = ((grid.hz2*(W[1:L-1, 1:N] + W[3:L+1, 1:N]) +
                               grid.hx2*(W[2:L, 0:N-1] + W[2:L, 2:N+1]))*
                        gv.dt/(grid.hz2hx2*2.0*gv.Re) + rho[2:L, 1:N]) / \
                 (1.0 + gv.dt*(grid.hz2 + grid.hx2)/(gv.Re*grid.hz2hx2))

        # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
        W, temp_sol = temp_sol, W
        W = bc.imposeWBCs(W)

        maxErr = np.amax(np.fabs(rho[2:L, 1:N] - (W[2:L, 1:N] - 0.5*gv.dt*(
                        (W[1:L-1, 1:N] - 2.0*W[2:L, 1:N] + W[3:L+1, 1:N])/grid.hx2 +
                        (W[2:L, 0:N-1] - 2.0*W[2:L, 1:N] + W[2:L, 2:N+1])/grid.hz2)/gv.Re)))

        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()


def getDiv():
    global N, L
    global U, W

    divMat = np.zeros([L, N])
    for i in range(1, L):
        for k in range(1, N):
            divMat[i, k] = (U[i, k] - U[i-1, k])/grid.hx + (W[i, k] - W[i, k-1])/grid.hz

    return np.unravel_index(divMat.argmax(), divMat.shape), np.amax(divMat)

def writeSoln(solTime):
    global U, W, P

    dw.writeSoln2D(U, W, P, solTime)

