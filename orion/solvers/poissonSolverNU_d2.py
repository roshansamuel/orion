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
from orion import meshData as grid
from orion import globalVars as gv
import numpy as np

############################### GLOBAL VARIABLES ################################

# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(grid.sLst[x[0]], grid.sLst[x[2]]) for x in [gv.sInd - y for y in range(gv.VDepth + 1)]]

# Define array of grid spacings along X
hx = [1.0/(x[0]-1) for x in N]

# Define array of grid spacings along Z
hz = [1.0/(x[1]-1) for x in N]

# Square of hx, used in finite difference formulae
hx2 = [x*x for x in hx]

# Square of hz, used in finite difference formulae
hz2 = [x*x for x in hz]

# Cross product of hx and hz, used in finite difference formulae
hzhx = [hx2[i]*hz2[i] for i in range(gv.VDepth + 1)]

# Maximum number of iterations while solving at coarsest level
maxCount = 10*N[-1][0]*N[-1][1]

# Integer specifying the level of V-cycle at any point while solving
vLev = 0

# Flag to determine if non-zero homogenous BC has to be applied or not
zeroBC = False

############################## MULTI-GRID SOLVER ###############################

def multigrid(P, H):
    global N
    global pData, rData

    chMat = np.zeros(N[0])
    for i in range(gv.VDepth):
        pData[i].fill(0.0)
        rData[i].fill(0.0)
        sData[i].fill(0.0)

    pData[0][1:-1, 1:-1] = P[1:-1, 1:-1]
    rData[0] = H[1:-1, 1:-1]

    for i in range(gv.vcCnt):
        v_cycle()

        if gv.testPoisson:
            chMat = laplace(pData[0])

            resVal = np.amax(np.abs(H[1:-1, 1:-1] - chMat))
            print("Residual after V-Cycle {0:2d} is {1:.4e}".format(i+1, resVal))

            errVal = np.amax(np.abs(pAnlt[1:-1, 1:-1] - pData[0][1:-1, 1:-1]))
            print("Error after V-Cycle {0:2d} is {1:.4e}\n".format(i+1, errVal))

    P[1:-1, 1:-1] = pData[0][1:-1, 1:-1]


# Multigrid V-cycle without the use of recursion
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

        # Restrict to coarser level
        restrict()

        # Reinitialize pressure at coarser level to 0 - this is critical!
        pData[vLev].fill(0.0)

        # If the coarsest level is reached, solve. Otherwise, keep smoothing!
        if vLev == gv.VDepth:
            if gv.solveSol:
                solve()
            else:
                smooth(gv.preSm + gv.pstSm)
        else:
            smooth(gv.preSm)

    # Prolongation operations
    for i in range(gv.VDepth):
        # Prolong pressure to next finer level
        prolong()

        # Add previously stored smoothed data
        pData[vLev] += sData[vLev]

        # Apply homogenous BC so long as we are not at finest mesh (at which vLev = 0)
        if vLev:
            zeroBC = True
        else:
            zeroBC = False

        # Post-smoothing
        smooth(gv.pstSm)


# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(sCount):
    global N
    global vLev
    global rData, pData
    global hx2, hz2, hzhx
    global xixx, xix2, ztzz, ztz2

    n = N[vLev]
    for iCnt in range(sCount):
        imposeBC(pData[vLev])

        # Gauss-Seidel smoothing
        for i in range(1, n[0]+1):
            for j in range(1, n[1]+1):
                pData[vLev][i, j] = (
                    hz2[vLev]*xix2[vLev][i-1]*(pData[vLev][i+1, j] + pData[vLev][i-1, j])*2.0 +
                    hz2[vLev]*xixx[vLev][i-1]*(pData[vLev][i+1, j] - pData[vLev][i-1, j])*hx[vLev] +
                    hx2[vLev]*ztz2[vLev][j-1]*(pData[vLev][i, j+1] + pData[vLev][i, j-1])*2.0 +
                    hx2[vLev]*ztzz[vLev][j-1]*(pData[vLev][i, j+1] - pData[vLev][i, j-1])*hz[vLev] -
                2.0*hzhx[vLev]*rData[vLev][i-1, j-1]) / \
            (4.0*(hz2[vLev]*xix2[vLev][i-1] + hx2[vLev]*ztz2[vLev][j-1]))

    imposeBC(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev][1:-1, 1:-1] = rData[vLev] - laplace(pData[vLev])


# Reduces the size of the array to a lower level, 2^(n - 1) + 1
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    n = N[vLev]
    for i in range(1, n[0] + 1):
        i2 = i*2
        for k in range(1, n[1] + 1):
            k2 = k*2
            rData[vLev][i-1, k-1] = 0.25*(iTemp[pLev][i2 - 1, k2 - 1]) + \
                                   0.125*(iTemp[pLev][i2 - 2, k2 - 1] + iTemp[pLev][i2, k2 - 1] + iTemp[pLev][i2 - 1, k2 - 2] + iTemp[pLev][i2 - 1, k2]) + \
                                  0.0625*(iTemp[pLev][i2 - 2, k2 - 2] + iTemp[pLev][i2, k2 - 2] + iTemp[pLev][i2 - 2, k2] + iTemp[pLev][i2, k2])


# Solves at coarsest level using an iterative solver
def solve():
    global N, vLev
    global maxCount
    global pData, rData
    global hx2, hz2, hzhx
    global xixx, xix2, ztzz, ztz2

    n = N[vLev]
    solLap = np.zeros(n)

    jCnt = 0
    while True:
        imposeBC(pData[vLev])

        # Gauss-Seidel iterative solver
        for i in range(1, n[0]+1):
            for j in range(1, n[1]+1):
                pData[vLev][i, j] = (
                    hz2[vLev]*xix2[vLev][i-1]*(pData[vLev][i+1, j] + pData[vLev][i-1, j])*2.0 +
                    hz2[vLev]*xixx[vLev][i-1]*(pData[vLev][i+1, j] - pData[vLev][i-1, j])*hx[vLev] +
                    hx2[vLev]*ztz2[vLev][j-1]*(pData[vLev][i, j+1] + pData[vLev][i, j-1])*2.0 +
                    hx2[vLev]*ztzz[vLev][j-1]*(pData[vLev][i, j+1] - pData[vLev][i, j-1])*hz[vLev] -
                2.0*hxhyhz[vLev]*rData[vLev][i-1, j-1]) / \
            (4.0*(hz2[vLev]*xix2[vLev][i-1] + hx2[vLev]*ztz2[vLev][j-1]))

        maxErr = np.amax(np.abs(rData[vLev] - laplace(pData[vLev])))
        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging. Aborting")
            quit()

    imposeBC(pData[vLev])


# Increases the size of the array to a higher level, 2^(n + 1) + 1
def prolong():
    global N
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    pData[vLev].fill(0.0)

    n = N[vLev]
    for i in range(1, n[0] + 1):
        i2 = int(i/2) + 1
        if i % 2:
            for k in range(1, n[1] + 1):
                k2 = int(k/2) + 1
                if k % 2:
                    pData[vLev][i, k] = pData[pLev][i2, k2]
                else:
                    pData[vLev][i, k] = (pData[pLev][i2, k2] + pData[pLev][i2, k2 - 1])*0.5
        else:
            for k in range(1, n[1] + 1):
                k2 = int(k/2) + 1
                if k % 2:
                    pData[vLev][i, k] = (pData[pLev][i2, k2] + pData[pLev][i2 - 1, k2])*0.5
                else:
                    pData[vLev][i, k] = (pData[pLev][i2, k2] + pData[pLev][i2, k2 - 1] + pData[pLev][i2 - 1, k2] + pData[pLev][i2 - 1, k2 - 1])*0.25


# Computes the 2D laplacian of function
def laplace(function):
    global N, vLev
    global hx2, hz2

    n = N[vLev]

    laplacian = xix2[vLev]*(function[2:, 1:-1] - 2.0*function[1:n[0]+1, 1:-1] + function[:n[0], 1:-1]) / hx2[vLev] + \
                xixx[vLev]*(function[2:, 1:-1] - function[:n[0], 1:-1]) / (2.0*hx[vLev]) + \
                ztz2[vLev]*(function[1:-1, 2:] - 2.0*function[1:-1, 1:n[1]+1] + function[1:-1, :n[1]]) / hz2[vLev] + \
                ztzz[vLev]*(function[1:-1, 2:] - function[1:-1, :n[1]]) / (2.0*hz[vLev])

    return laplacian


# Initialize the arrays used in MG algorithm
def initVariables():
    global N
    global pData, rData, sData, iTemp

    nList = np.array(N)

    rData = [np.zeros(tuple(x)) for x in nList]
    pData = [np.zeros(tuple(x)) for x in nList + 2]

    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in pData]

    initGrid()


# Initialize the grid metric terms at each V-level
def initGrid():
    global N
    global xixx, xix2
    global ztzz, ztz2

    # Uniform grid default values
    xPts = [np.linspace(0.0, 1.0, n[0]) for n in N]
    zPts = [np.linspace(0.0, 1.0, n[1]) for n in N]

    xix2 = [np.ones_like(i) for i in xPts]
    xixx = [np.zeros_like(i) for i in xPts]

    ztz2 = [np.ones_like(i) for i in zPts]
    ztzz = [np.zeros_like(i) for i in zPts]

    # Copy the values for finest grid from meshData
    xPts[0] = np.copy(grid.xStag)
    xixx[0] = np.copy(grid.xixxStag)
    xix2[0] = np.copy(grid.xix2Stag)

    zPts[0] = np.copy(grid.zStag)
    ztzz[0] = np.copy(grid.ztzzStag)
    ztz2[0] = np.copy(grid.ztz2Stag)

    # For coarser grids, simply use the values at every even index of the finer grid array.
    for i in range(1, gv.VDepth+1):
        xPts[i] = xPts[i-1][::2]
        xixx[i] = xixx[i-1][::2]
        xix2[i] = xix2[i-1][::2]

        zPts[i] = zPts[i-1][::2]
        ztzz[i] = ztzz[i-1][::2]
        ztz2[i] = ztz2[i-1][::2]

    # Reshape arrays to make it easier to multiply with 3D arrays
    xixx = [x[:, np.newaxis] for x in xixx]
    xix2 = [x[:, np.newaxis] for x in xix2]


############################## BOUNDARY CONDITION ###############################


# The name of this function is self-explanatory. It imposes BC on P
def imposeBC(P):
    global zeroBC
    global pWallX, pWallZ

    if gv.testPoisson:
        # Dirichlet BC
        if zeroBC:
            # Homogenous BC
            # Left Wall
            P[0, :] = -P[2, :]

            # Right Wall
            P[-1, :] = -P[-3, :]

            # Bottom wall
            P[:, 0] = -P[:, 2]

            # Top wall
            P[:, -1] = -P[:, -3]

        else:
            # Non-homogenous BC
            # Left Wall
            P[0, :] = 2.0*pWallX - P[2, :]

            # Right Wall
            P[-1, :] = 2.0*pWallX - P[-3, :]

            # Bottom wall
            P[:, 0] = 2.0*pWallZ - P[:, 2]

            # Top wall
            P[:, -1] = 2.0*pWallZ - P[:, -3]

    else:
        # Periodic BCs along X and Y directions
        if gv.xyPeriodic:
            # Left wall
            P[0, :] = P[-3, :]

            # Right wall
            P[-1, :] = P[2, :]

        # Neumann boundary condition on pressure
        else:
            # Left wall
            P[0, :] = P[2, :]

            # Right wall
            P[-1, :] = P[-3, :]

        # Bottom wall
        P[:, 0] = P[:, 2]

        # Top wall
        P[:, -1] = P[:, -3]

