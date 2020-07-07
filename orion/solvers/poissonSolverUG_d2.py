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
from orion import boundaryConditions as bc
from orion import calculateFD as fd
from orion import meshData as grid
from orion import globalVars as gv
import numpy as np

# Get limits from grid object
L, N = grid.L, grid.N

def multigrid(H):
    global N, L

    Pp = np.zeros([L+1, N+1])
    chMat = np.ones([L+1, N+1])
    for i in range(gv.vcCnt):
        Pp = v_cycle(Pp, H)

        chMat = laplace(Pp)
        print("Residual after V-Cycle ", i, " is ", np.amax(np.abs(H[1:L, 1:N] - chMat[1:L, 1:N])))

    return Pp


#Multigrid solution without the use of recursion
def v_cycle(P, H):
    # Pre-smoothing
    P = smooth(P, H, grid.hx, grid.hz, gv.preSm, 0)

    H_rsdl = H - laplace(P)

    # Restriction operations
    for i in range(gv.VDepth):
        gv.sInd -= 1
        H_rsdl = restrict(H_rsdl)

    # Solving the system after restriction
    P_corr = solve(H_rsdl, (2.0**gv.VDepth)*grid.hx, (2.0**gv.VDepth)*grid.hz)

    # Prolongation operations
    for i in range(gv.VDepth):
        gv.sInd += 1
        P_corr = prolong(P_corr)
        H_rsdl = prolong(H_rsdl)
        P_corr = smooth(P_corr, H_rsdl, grid.hx, grid.hz, gv.proSm, gv.VDepth-i-1)

    P += P_corr

    # Post-smoothing
    P = smooth(P, H, grid.hx, grid.hz, gv.pstSm, 0)

    return P


#Uses jacobi iteration to smooth the solution passed to it.
def smooth(function, rho, hx, hz, iteration_times, vLevel):
    smoothed = np.copy(function)

    # 1 subtracted from shape to account for ghost points
    [L, N] = np.array(np.shape(function)) - 1

    for i in range(iteration_times):
        toSmooth = bc.imposePBCs(smoothed)

        smoothed[1:L, 1:N] = ((hz*hz)*(toSmooth[2:L+1, 1:N] + toSmooth[0:L-1, 1:N]) +
                              (hx*hx)*(toSmooth[1:L, 2:N+1] + toSmooth[1:L, 0:N-1]) -
                              (hx*hx)*(hz*hz)*rho[1:L, 1:N]) / (2.0*((hz*hz) + (hx*hx)))

    return smoothed


#Reduces the size of the array to a lower level, 2^(n-1)+1.
def restrict(function):
    [rx, rz] = [grid.sLst[gv.sInd[0]], grid.sLst[gv.sInd[2]]]
    restricted = np.zeros([rx + 1, rz + 1])

    for i in range(1, rx):
        i2 = i*2
        for k in range(1, rz):
            k2 = k*2
            restricted[i, k] = 0.25*(function[i2 - 1, k2 - 1]) + \
                              0.125*(function[i2 + 1, k2] + function[i2 - 1, k2] + function[i2, k2 + 1] + function[i2, k2 - 1]) + \
                             0.0625*(function[i2 + 1, k2 + 1] + function[i2 + 1, k2 - 1] + function[i2 - 1, k2 + 1] + function[i2 - 1, k2 - 1])

    return restricted


# Increases the size of the array to a higher level, 2^(n + 1) + 1
def prolong(function):
    [rx, rz] = [grid.sLst[gv.sInd[0]], grid.sLst[gv.sInd[2]]]
    prolonged = np.zeros([rx + 1, rz + 1])

    [lx, lz] = np.shape(function)
    for i in range(1, rx):
        i2 = i/2;
        if isOdd(i):
            for k in range(1, rz):
                k2 = k/2;
                if isOdd(k):
                    prolonged[i, k] = (function[i2, k2] + function[i2, k2 + 1] + function[i2 + 1, k2] + function[i2 + 1, k2 + 1])/4.0;
                else:
                    prolonged[i, k] = (function[i2, k2] + function[i2 + 1, k2])/2.0;
        else:
            for k in range(1, rz):
                k2 = k/2;
                if isOdd(k):
                    prolonged[i, k] = (function[i2, k2] + function[i2, k2 + 1])/2.0;
                else:
                    prolonged[i, k] = function[i2, k2];

    return prolonged


#This function uses the Jacobi iterative solver, using the grid spacing
def solve(rho, hx, hz):
    # 1 subtracted from shape to account for ghost points
    [L, N] = np.array(np.shape(rho)) - 1
    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L, 1:N] = ((hz*hz)*(prev_sol[2:L+1, 1:N] + prev_sol[0:L-1, 1:N]) +
                              (hx*hx)*(prev_sol[1:L, 2:N+1] + prev_sol[1:L, 0:N-1]) -
                              (hx*hx)*(hz*hz)*rho[1:L, 1:N]) / (2.0*((hz*hz) + (hx*hx)))

        solLap = np.zeros_like(next_sol)
        solLap[1:L, 1:N] = (fd.DDXi(next_sol, L, N) + fd.DDZt(next_sol, L, N))/((2**gv.VDepth)**2)

        error_temp = np.abs(rho[1:L, 1:N] - solLap[1:L, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > 10*N*L:
            print("ERROR: Jacobi not converging. Aborting")
            print("Maximum error: ", maxErr)
            quit()

        prev_sol = np.copy(next_sol)

    return prev_sol


def laplace(function):
    '''
Function to calculate the Laplacian for a given field of values.
INPUT:  function: 3D matrix of double precision values
OUTPUT: gradient: 3D matrix of double precision values with same size as input matrix
    '''

    # 1 subtracted from shape to account for ghost points
    [L, N] = np.array(np.shape(function)) - 1
    gradient = np.zeros_like(function)

    gradient[1:L, 1:N] = fd.DDXi(function, L, N) + fd.DDZt(function, L, N)

    return gradient

