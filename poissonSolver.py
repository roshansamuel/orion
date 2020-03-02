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
import boundaryConditions as bc
import calculateFD as fd
import meshData as grid
import globalVars as gv
import numpy as np

# Redefine frequently used numpy object
npax = np.newaxis

def multigrid(H):
    global N, M, L

    Pp = np.zeros([L+1, M+1, N+1])
    chMat = np.ones([L+1, M+1, N+1])
    for i in range(gv.vcCnt):
        Pp = v_cycle(Pp, H)
        chMat = laplace(Pp)

    print("Error after multigrid is ", np.amax(np.abs(H[1:L, 1:M, 1:N] - chMat[1:L, 1:M, 1:N])))

    return Pp


#Multigrid solution without the use of recursion
def v_cycle(P, H):
    # Pre-smoothing
    P = smooth(P, H, grid.hx, grid.hy, grid.hz, gv.preSm, 0)

    H_rsdl = H - laplace(P)

    # Restriction operations
    for i in range(gv.VDepth):
        gv.sInd -= 1
        H_rsdl = restrict(H_rsdl)

    # Solving the system after restriction
    P_corr = solve(H_rsdl, (2.0**gv.VDepth)*grid.hx, (2.0**gv.VDepth)*grid.hy, (2.0**gv.VDepth)*grid.hz)

    # Prolongation operations
    for i in range(gv.VDepth):
        gv.sInd += 1
        P_corr = prolong(P_corr)
        H_rsdl = prolong(H_rsdl)
        P_corr = smooth(P_corr, H_rsdl, grid.hx, grid.hy, grid.hz, gv.proSm, gv.VDepth-i-1)

    P += P_corr

    # Post-smoothing
    P = smooth(P, H, grid.hx, grid.hy, grid.hz, gv.pstSm, 0)

    return P


#Uses jacobi iteration to smooth the solution passed to it.
def smooth(function, rho, hx, hy, hz, iteration_times, vLevel):
    smoothed = np.copy(function)

    # 1 subtracted from shape to account for ghost points
    [L, M, N] = np.array(np.shape(function)) - 1

    for i in range(iteration_times):
        toSmooth = bc.imposePBCs(smoothed)

        smoothed[1:L, 1:M, 1:N] = (
                        (hy*hy)*(hz*hz)*grid.xix2Stag[0::2**vLevel, npax, npax]*(toSmooth[2:L+1, 1:M, 1:N] + toSmooth[0:L-1, 1:M, 1:N])*2.0 +
                        (hy*hy)*(hz*hz)*grid.xixxStag[0::2**vLevel, npax, npax]*(toSmooth[2:L+1, 1:M, 1:N] - toSmooth[0:L-1, 1:M, 1:N])*hx +
                        (hx*hx)*(hz*hz)*grid.ety2Stag[0::2**vLevel, npax]*(toSmooth[1:L, 2:M+1, 1:N] + toSmooth[1:L, 0:M-1, 1:N])*2.0 +
                        (hx*hx)*(hz*hz)*grid.etyyStag[0::2**vLevel, npax]*(toSmooth[1:L, 2:M+1, 1:N] - toSmooth[1:L, 0:M-1, 1:N])*hy +
                        (hx*hx)*(hy*hy)*grid.ztz2Stag[0::2**vLevel]*(toSmooth[1:L, 1:M, 2:N+1] + toSmooth[1:L, 1:M, 0:N-1])*2.0 +
                        (hx*hx)*(hy*hy)*grid.ztzzStag[0::2**vLevel]*(toSmooth[1:L, 1:M, 2:N+1] - toSmooth[1:L, 1:M, 0:N-1])*hz -
                    2.0*(hx*hx)*(hy*hy)*(hz*hz)*rho[1:L, 1:M, 1:N])/ \
                  (4.0*((hy*hy)*(hz*hz)*grid.xix2Stag[0::2**vLevel, npax, npax] +
                        (hx*hx)*(hz*hz)*grid.ety2Stag[0::2**vLevel, npax] +
                        (hx*hx)*(hy*hy)*grid.ztz2Stag[0::2**vLevel]))

    return smoothed


#Reduces the size of the array to a lower level, 2^(n-1)+1.
def restrict(function):
    [rx, ry, rz] = [grid.sLst[gv.sInd[0]], grid.sLst[gv.sInd[1]], grid.sLst[gv.sInd[2]]]
    restricted = np.zeros([rx + 1, ry + 1, rz + 1])

    for i in range(1, rx):
        for j in range(1, ry):
            for k in range(1, rz):
                restricted[i, j, k] = function[2*i - 1, 2*j - 1, 2*k - 1]

    return restricted


#Increases the size of the array to a higher level, 2^(n+1)+1.
def prolong(function):
    [rx, ry, rz] = [grid.sLst[gv.sInd[0]], grid.sLst[gv.sInd[1]], grid.sLst[gv.sInd[2]]]
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
    # 1 subtracted from shape to account for ghost points
    [L, M, N] = np.array(np.shape(rho)) - 1
    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:L, 1:M, 1:N] = (
            (hy*hy)*(hz*hz)*grid.xix2Stag[0::2**gv.VDepth, npax, npax]*(prev_sol[2:L+1, 1:M, 1:N] + prev_sol[0:L-1, 1:M, 1:N])*2.0 +
            (hy*hy)*(hz*hz)*grid.xixxStag[0::2**gv.VDepth, npax, npax]*(prev_sol[2:L+1, 1:M, 1:N] - prev_sol[0:L-1, 1:M, 1:N])*hx +
            (hx*hx)*(hz*hz)*grid.ety2Stag[0::2**gv.VDepth, npax]*(prev_sol[1:L, 2:M+1, 1:N] + prev_sol[1:L, 0:M-1, 1:N])*2.0 +
            (hx*hx)*(hz*hz)*grid.etyyStag[0::2**gv.VDepth, npax]*(prev_sol[1:L, 2:M+1, 1:N] - prev_sol[1:L, 0:M-1, 1:N])*hy +
            (hx*hx)*(hy*hy)*grid.ztz2Stag[0::2**gv.VDepth]*(prev_sol[1:L, 1:M, 2:N+1] + prev_sol[1:L, 1:M, 0:N-1])*2.0 +
            (hx*hx)*(hy*hy)*grid.ztzzStag[0::2**gv.VDepth]*(prev_sol[1:L, 1:M, 2:N+1] - prev_sol[1:L, 1:M, 0:N-1])*hz -
        2.0*(hx*hx)*(hy*hy)*(hz*hz)*rho[1:L, 1:M, 1:N])/ \
      (4.0*((hy*hy)*(hz*hz)*grid.xix2Stag[0::2**gv.VDepth, npax, npax] +
            (hx*hx)*(hz*hz)*grid.ety2Stag[0::2**gv.VDepth, npax] +
            (hx*hx)*(hy*hy)*grid.ztz2Stag[0::2**gv.VDepth]))

        solLap = np.zeros_like(next_sol)
        solLap[1:L, 1:M, 1:N] = grid.xix2Stag[0::2**gv.VDepth, npax, npax]*fd.DDXi(next_sol, L, M, N)/((2**gv.VDepth)**2) + \
                                grid.xixxStag[0::2**gv.VDepth, npax, npax]*fd.D_Xi(next_sol, L, M, N)/(2**gv.VDepth) + \
                                grid.ety2Stag[0::2**gv.VDepth, npax]*fd.DDEt(next_sol, L, M, N)/((2**gv.VDepth)**2) + \
                                grid.etyyStag[0::2**gv.VDepth, npax]*fd.D_Et(next_sol, L, M, N)/(2**gv.VDepth) + \
                                grid.ztz2Stag[0::2**gv.VDepth]*fd.DDZt(next_sol, L, M, N)/((2**gv.VDepth)**2) + \
                                grid.ztzzStag[0::2**gv.VDepth]*fd.D_Zt(next_sol, L, M, N)/(2**gv.VDepth)

        error_temp = np.abs(rho[1:L, 1:M, 1:N] - solLap[1:L, 1:M, 1:N])
        maxErr = np.amax(error_temp)
        if maxErr < gv.tolerance:
            break

        jCnt += 1
        if jCnt > 10*N*M*L:
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
    [L, M, N] = np.array(np.shape(function)) - 1
    gradient = np.zeros_like(function)

    gradient[1:L, 1:M, 1:N] = grid.xix2Stag[0:L-1, npax, npax]*fd.DDXi(function, L, M, N) + \
                              grid.xixxStag[0:L-1, npax, npax]*fd.D_Xi(function, L, M, N) + \
                                    grid.ety2Stag[0:M-1, npax]*fd.DDEt(function, L, M, N) + \
                                    grid.etyyStag[0:M-1, npax]*fd.D_Et(function, L, M, N) + \
                                          grid.ztz2Stag[0:N-1]*fd.DDZt(function, L, M, N) + \
                                          grid.ztzzStag[0:N-1]*fd.D_Zt(function, L, M, N)

    return gradient


####################################################################################################

# Limits along each direction
# L - Along X
# M - Along Y
# N - Along Z
# Data stored in arrays accessed by data[1:L, 1:M, 1:N]
# In Python and C, the rightmost index varies fastest
# Therefore indices in Z direction vary fastest, then along Y and finally along X

L = grid.sLst[gv.sInd[0]]
M = grid.sLst[gv.sInd[1]]
N = grid.sLst[gv.sInd[2]]
