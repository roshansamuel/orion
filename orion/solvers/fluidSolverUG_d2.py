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
    global U, W, P
    global Hx, Hz

    # Create and initialize U and P arrays
    # The arrays have two extra points
    # These act as ghost points on either sides of the domain
    P = np.ones([grid.L + 1, grid.N + 1])

    # U is staggered in Y and Z directions and hence has one extra point along these directions
    U = np.zeros([grid.L, grid.N + 1])

    # W is staggered in X and Y directions and hence has one extra point along these directions
    W = np.zeros([grid.L + 1, grid.N])

    # Define arrays for storing RHS of NSE
    Hx = np.zeros_like(U)
    Hz = np.zeros_like(W)

    if gv.probType == 0:
        # BC for moving top lid - U = 1.0 on lid
        U[:, grid.N] = 1.0
    elif gv.probType == 1:
        # Initial condition for channel flow
        U[:, :] = 1.0


def euler():
    global N, L
    global U, W, P
    global Hx, Hz

    Hx.fill(0.0)
    Hz.fill(0.0)

    computeNLinDiff_X(U, W)
    computeNLinDiff_Z(U, W)

    # Add constant pressure gradient forcing for channel flows
    if gv.probType == 1:
        Hx[:, :] += 1.0

    # Calculating guessed values of U implicitly
    Hx[1:L-1, 1:N] = U[1:L-1, 1:N] + gv.dt*(Hx[1:L-1, 1:N] - (P[2:L, 1:N] - P[1:L-1, 1:N])/grid.hx)
    Up = uJacobi(Hx)

    # Calculating guessed values of W implicitly
    Hz[1:L, 1:N-1] = W[1:L, 1:N-1] + gv.dt*(Hz[1:L, 1:N-1] - (P[1:L, 2:N] - P[1:L, 1:N-1])/grid.hz)
    Wp = wJacobi(Hz)

    # Calculating pressure correction term
    rhs = np.zeros([L+1, N+1])
    rhs[1:L, 1:N] = ((Up[1:L, 1:N] - Up[0:L-1, 1:N])/grid.hx + (Wp[1:L, 1:N] - Wp[1:L, 0:N-1])/grid.hz)/gv.dt
    Pp = ps.multigrid(rhs)

    # Add pressure correction.
    P = P + Pp

    # Update new values for U and W
    U[1:L-1, 1:N] = Up[1:L-1, 1:N] - gv.dt*(Pp[2:L, 1:N] - Pp[1:L-1, 1:N])/grid.hx
    W[1:L, 1:N-1] = Wp[1:L, 1:N-1] - gv.dt*(Pp[1:L, 2:N] - Pp[1:L, 1:N-1])/grid.hz

    # Impose no-slip BC on new values of U and W
    U = bc.imposeUBCs(U)
    W = bc.imposeWBCs(W)


def computeNLinDiff_X(U, W):
    global Hx
    global N, L

    Hx[1:L-1, 1:N] = ((fd.DDXi(U, L-1, N) + fd.DDZt(U, L-1, N))*0.5/gv.Re -
                       fd.D_Xi(U, L-1, N)*U[1:L-1, 1:N] -
                      0.25*(W[1:L-1, 0:N-1] + W[1:L-1, 1:N] + W[2:L, 1:N] + W[2:L, 0:N-1])*fd.D_Zt(U, L-1, N))


def computeNLinDiff_Z(U, W):
    global Hz
    global N, L

    Hz[1:L, 1:N-1] = ((fd.DDXi(W, L, N-1) + fd.DDZt(W, L, N-1))*0.5/gv.Re -
                       fd.D_Zt(W, L, N-1)*W[1:L, 1:N-1] -
                      0.25*(U[0:L-1, 1:N-1] + U[1:L, 1:N-1] + U[1:L, 2:N] + U[0:L-1, 2:N])*fd.D_Xi(W, L, N-1))


#Jacobi iterative solver for U
def uJacobi(rho):
    global L, N

    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    test_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L-1, 1:N] = ((grid.hz2*(prev_sol[0:L-2,   1:N] + prev_sol[  2:L,   1:N]) +
                                 grid.hx2*(prev_sol[1:L-1, 0:N-1] + prev_sol[1:L-1, 2:N+1]))*
                                       gv.dt/(grid.hz2hx2*2.0*gv.Re) + rho[1:L-1, 1:N])/ \
                                (1.0 + gv.dt*(grid.hz2 + grid.hx2)/(gv.Re*grid.hz2hx2))

        # IMPOSE BOUNDARY CONDITION AND COPY TO PREVIOUS SOLUTION ARRAY
        next_sol = bc.imposeUBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L-1, 1:N] = next_sol[1:L-1, 1:N] - (fd.DDXi(next_sol, L-1, N) + fd.DDZt(next_sol, L-1, N))*0.5*gv.dt/gv.Re

        error_temp = np.fabs(rho[1:L-1, 1:N] - test_sol[1:L-1, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < gv.tolerance:
            if gv.iCnt % gv.opInt == 0:
                print("Jacobi solver for U converged in ", jCnt, " iterations")
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in U. Aborting")
            print("Maximum error: ", maxErr)
            quit()

    return prev_sol


#Jacobi iterative solver for W
def wJacobi(rho):
    global L, N

    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    test_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L, 1:N-1] = ((grid.hz2*(prev_sol[0:L-1, 1:N-1] + prev_sol[2:L+1, 1:N-1]) +
                                 grid.hx2*(prev_sol[1:L,   0:N-2] + prev_sol[1:L,     2:N]))*
                                       gv.dt/(grid.hz2hx2*2.0*gv.Re) + rho[1:L, 1:N-1])/ \
                                (1.0 + gv.dt*(grid.hz2 + grid.hx2)/(gv.Re*grid.hz2hx2))

        # IMPOSE BOUNDARY CONDITION AND COPY TO PREVIOUS SOLUTION ARRAY
        next_sol = bc.imposeWBCs(next_sol)
        prev_sol = np.copy(next_sol)

        test_sol[1:L, 1:N-1] = next_sol[1:L, 1:N-1] - (fd.DDXi(next_sol, L, N-1) + fd.DDZt(next_sol, L, N-1))*0.5*gv.dt/gv.Re

        error_temp = np.fabs(rho[1:L, 1:N-1] - test_sol[1:L, 1:N-1])
        maxErr = np.amax(error_temp)
        if maxErr < gv.tolerance:
            if gv.iCnt % gv.opInt == 0:
                print("Jacobi solver for W converged in ", jCnt, " iterations")
            break

        jCnt += 1
        if jCnt > grid.maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()

    return prev_sol


def getDiv():
    '''
Function to calculate the divergence within the domain (excluding walls)
INPUT:  U, W: Velocity values
OUTPUT: The maximum value of divergence in double precision
    '''
    global N, L
    global U, W

    divMat = np.zeros([L+1, N+1])
    for i in range(1, L):
        for k in range(1, N):
            divMat[i, k] = (U[i, j, k] - U[i-1, j, k])/(grid.xColl[i] - grid.xColl[i-1]) + \
                           (W[i, j, k] - W[i, j, k-1])/(grid.zColl[k] - grid.zColl[k-1])

    return np.unravel_index(divMat.argmax(), divMat.shape), np.amax(divMat)

def writeSoln(solTime):
    global U, W, P

    dw.writeSoln(U, W, P, solTime)

