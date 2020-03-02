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
import multiprocessing as mp
import meshData as grid
import globalVars as gv
import numpy as np

# Redefine frequently used numpy object
npax = np.newaxis

def addTurbViscosity(xStr, xEnd):
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
                dudx = (U[i+1, j, k] - U[i, j, k])/grid.hx
                dudy = (U[i, j+1, k] - U[i, j, k])/grid.hy
                dudz = (U[i, j, k+1] - U[i, j, k])/grid.hz

                dvdx = (V[i+1, j, k] - V[i, j, k])/grid.hx
                dvdy = (V[i, j+1, k] - V[i, j, k])/grid.hy
                dvdz = (V[i, j, k+1] - V[i, j, k])/grid.hz

                dwdx = (W[i+1, j, k] - W[i, j, k])/grid.hx
                dwdy = (W[i, j+1, k] - W[i, j, k])/grid.hy
                dwdz = (W[i, j, k+1] - W[i, j, k])/grid.hz

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
                lambda_v = 2.0*gv.nu/(3.0*a)

        xInd += 1

    return vList
