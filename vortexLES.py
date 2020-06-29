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
import meshData as grid
import globalVars as gv
import numpy as np

# Function to compute and add turbulent viscosity from LES model
def addTurbViscosity(xStr, xEnd):
    global U, V, W
    global L, M, N

    xInd = 0
    subNx = xEnd - xStr

    vList = np.zeros((subNx, M-2, N-2))

    for i in range(xStr, xEnd):
        for j in range(1, M-1):
            for k in range(1, N-1):
                hx = grid.xColl[i+1] - grid.xColl[i]
                hy = grid.yColl[j+1] - grid.yColl[j]
                hz = grid.zColl[k+1] - grid.zColl[k]

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

                # Compute the lambda coefficient used to compute the subgrid energy
                # Warning: Use a default value here if a = 0
                lambda_v = 2.0*gv.nu/(3.0*a)

                # Warning: Test with non-uniform grid spacing
                delta_c = np.cbrt(hx*hy*hz)
                k_c = np.pi/delta_c
                kappa_c = k_c*lambda_v

        xInd += 1

    return vList
