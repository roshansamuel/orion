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

# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(grid.sLst[x[0]] - 1, grid.sLst[x[2]] - 1) for x in [gv.sInd - y for y in range(gv.VDepth + 1)]]

# Define array of grid spacings along X
hx = [1.0/(x[0]-1) for x in N]

# Define array of grid spacings along X
hz = [1.0/(x[1]-1) for x in N]

# Square of hx, used in finite difference formulae
hx2 = [x*x for x in hx]

# Square of hz, used in finite difference formulae
hz2 = [x*x for x in hz]

# Cross product of hx and hz, used in finite difference formulae
hzhx = [hx[i]*hz[i] for i in range(gv.VDepth + 1)]

# Maximum number of iterations while solving at coarsest level
maxCount = 10*N[-1][0]*N[-1][1]

# Integer specifying the level of V-cycle at any point while solving
vLev = 0

# Flag to determine if non-zero homogenous BC has to be applied or not
zeroBC = False


def multigrid(H):
    global N
    global pData, rData

    n = N[0]
    rData[0] = H[1:-1, 1:-1]
    chMat = np.zeros(n)

    for i in range(gv.vcCnt):
        v_cycle()

        chMat = laplace(pData[0])
        resVal = np.amax(np.abs(H[1:-1, 1:-1] - chMat))

        print("Residual after V-Cycle {0:2d} is {1:.4e}".format(i+1, resVal))

    return pData[0]


# Initialize the arrays used in MG algorithm
def initVariables():
    global N
    global pData, rData, sData, iTemp

    nList = np.array(N)

    rData = [np.zeros(tuple(x)) for x in nList]
    pData = [np.zeros(tuple(x)) for x in nList + 2]

    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in pData]


#Multigrid solution without the use of recursion
def v_cycle():
    global vLev, zeroBC

    vLev = 0
    zeroBC = False

    # Pre-smoothing
    smooth(gv.preSm)

    zeroBC = True
    for i in range(gv.VDepth):
        # Compute residual
        calcResidual()

        # Copy smoothed pressure for later use
        sData[vLev] = np.copy(pData[vLev])


    print(pData[0].shape)
    print(rData[0].shape)
    exit()

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
def smooth(sCount):
    global N
    global hx2
    global vLev
    global rData, pData

    n = N[vLev]
    for iCnt in range(sCount):
        bc.imposePBCs(pData[vLev])

        # Gauss-Seidel smoothing
        for i in range(1, n[0]+1):
            for j in range(1, n[1]+1):
                pData[vLev][i, j] = (hz2[vLev]*(pData[vLev][i+1, j] + pData[vLev][i-1, j]) +
                                     hx2[vLev]*(pData[vLev][i, j+1] + pData[vLev][i, j-1]) -
                                     hzhx[vLev]*rData[vLev][i-1, j-1]) / (2.0*(hx2[vLev] + hz2[vLev]))


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev][1:-1, 1:-1] = rData[vLev] - laplace(pData[vLev])


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
                    prolonged[i, k] = (function[i2, k2] + function[i2, k2 + 1] + function[i2 + 1, k2] + function[i2 + 1, k2 + 1])/4.0
                else:
                    prolonged[i, k] = (function[i2, k2] + function[i2 + 1, k2])/2.0
        else:
            for k in range(1, rz):
                k2 = k/2;
                if isOdd(k):
                    prolonged[i, k] = (function[i2, k2] + function[i2, k2 + 1])/2.0
                else:
                    prolonged[i, k] = function[i2, k2]

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
    global vLev
    global N, hx2

    n = N[vLev]

    gradient = np.zeros(n)
    gradient = ((function[:n[0], 1:-1] - 2.0*function[1:n[0]+1, 1:-1] + function[2:, 1:-1])/hx2[vLev] + 
                (function[1:-1, :n[1]] - 2.0*function[1:-1, 1:n[1]+1] + function[1:-1, 2:])/hz2[vLev])

    return gradient

