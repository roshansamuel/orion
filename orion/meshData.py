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

from orion.globalVars import sInd, dLen, beta
import numpy as np

# Redefine frequently used numpy object
npax = np.newaxis

# N should be of the form 2^n + 1 so that there will be 2^n + 3 staggered points, including two ghost points
sLst = [2**x + 1 for x in range(12)]

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

# Domain lengths along each direction
xLen = dLen[0]
yLen = dLen[1]
zLen = dLen[2]

# Stretching parameters along each direction
xBeta = beta[0]
yBeta = beta[1]
zBeta = beta[2]

# Define grid spacings
hx = 1.0/(L-1)
hy = 1.0/(M-1)
hz = 1.0/(N-1)

# Other frequently used grid constants
hx2 = hx*hx
hy2 = hy*hy
hz2 = hz*hz

hx2hy2 = hx2*hy2
hy2hz2 = hy2*hz2
hz2hx2 = hz2*hx2

hx2hy2hz2 = hx2*hy2*hz2

# Maximum number of jacobi iterations
maxCount = 10*N*M*L

xStag = np.zeros(L)
yStag = np.zeros(M)
zStag = np.zeros(N)
xColl = np.zeros(L-1)
yColl = np.zeros(M-1)
zColl = np.zeros(N-1)

# Grid metric arrays. Used by the non uniform solvers
# The default values of arrays initialized below corresponds to uniform grid values
xi_xColl = np.ones_like(xColl)       #-- dXi/dX at all face-centered points along X direction
xixxColl = np.zeros_like(xColl)      #-- d2Xi/dX2 at all face-centered points along X direction
xix2Coll = np.ones_like(xColl)       #-- (dXi/dX)**2 at all face-centered points along X direction

xi_xStag = np.ones_like(xStag)       #-- dXi/dX at all cell-centered points along X direction
xixxStag = np.zeros_like(xStag)      #-- d2Xi/dX2 at all cell-centered points along X direction
xix2Stag = np.ones_like(xStag)       #-- (dXi/dX)**2 at all cell-centered points along X direction

et_yColl = np.ones_like(yColl)       #-- dEt/dY at all face-centered points along Y direction
etyyColl = np.zeros_like(yColl)      #-- d2Et/dY2 at all face-centered points along Y direction
ety2Coll = np.ones_like(yColl)       #-- (dEt/dY)**2 at all face-centered points along Y direction

et_yStag = np.ones_like(yStag)       #-- dEt/dY at all cell-centered points along Y direction
etyyStag = np.zeros_like(yStag)      #-- d2Et/dY2 at all cell-centered points along Y direction
ety2Stag = np.ones_like(yStag)       #-- (dEt/dY)**2 at all cell-centered points along Y direction

zt_zColl = np.ones_like(zColl)       #-- dZt/dZ at all face-centered points along Z direction
ztzzColl = np.zeros_like(zColl)      #-- d2Zt/dZ2 at all face-centered points along Z direction
ztz2Coll = np.ones_like(zColl)       #-- (dZt/dZ)**2 at all face-centered points along Z direction

zt_zStag = np.ones_like(zStag)       #-- dZt/dZ at all cell-centered points along Z direction
ztzzStag = np.zeros_like(zStag)      #-- d2Zt/dZ2 at all cell-centered points along Z direction
ztz2Stag = np.ones_like(zStag)       #-- (dZt/dZ)**2 at all cell-centered points along Z direction


def initializeGrid():
    global L, M, N
    global xLen, yLen, zLen
    global xBeta, yBeta, zBeta
    global xColl, yColl, zColl
    global xStag, yStag, zStag

    xi = np.linspace(0.0, 1.0, L)
    et = np.linspace(0.0, 1.0, M)
    zt = np.linspace(0.0, 1.0, N)

    # Initialize face-centered and cell-centered grids
    xStag = [xLen*(1.0 - np.tanh(xBeta*(1.0 - 2.0*i))/np.tanh(xBeta))/2.0 for i in xi]
    yStag = [yLen*(1.0 - np.tanh(yBeta*(1.0 - 2.0*i))/np.tanh(yBeta))/2.0 for i in et]
    zStag = [zLen*(1.0 - np.tanh(zBeta*(1.0 - 2.0*i))/np.tanh(zBeta))/2.0 for i in zt]

    xColl = [xLen*(1.0 - np.tanh(xBeta*(1.0 - 2.0*i))/np.tanh(xBeta))/2.0 for i in [(xi[j] + xi[j+1])/2 for j in range(len(xi) - 1)]]
    yColl = [yLen*(1.0 - np.tanh(yBeta*(1.0 - 2.0*i))/np.tanh(yBeta))/2.0 for i in [(et[j] + et[j+1])/2 for j in range(len(et) - 1)]]
    zColl = [zLen*(1.0 - np.tanh(zBeta*(1.0 - 2.0*i))/np.tanh(zBeta))/2.0 for i in [(zt[j] + zt[j+1])/2 for j in range(len(zt) - 1)]]


def calculateMetrics():
    global xLen, yLen, zLen
    global xBeta, yBeta, zBeta
    global xColl, yColl, zColl
    global xStag, yStag, zStag
    global xi_xColl, et_yColl, zt_zColl
    global xi_xStag, et_yStag, zt_zStag
    global xixxColl, xix2Coll, etyyColl, ety2Coll, ztzzColl, ztz2Coll
    global xixxStag, xix2Stag, etyyStag, ety2Stag, ztzzStag, ztz2Stag

    # Grid metrics for both staggered and collocated grids
    xi_xColl = np.array([np.tanh(xBeta)/(xBeta*xLen*(1.0 - ((1.0 - 2.0*k/xLen)*np.tanh(xBeta))**2.0)) for k in xColl])
    xixxColl = np.array([-4.0*(np.tanh(xBeta)**3.0)*(1.0 - 2.0*k/xLen)/(xBeta*xLen*xLen*(1.0 - (np.tanh(xBeta)*(1.0 - 2.0*k/xLen))**2.0)**2.0) for k in xColl])
    xix2Coll = np.array([k*k for k in xi_xColl])

    xi_xStag = np.array([np.tanh(xBeta)/(xBeta*xLen*(1.0 - ((1.0 - 2.0*k/xLen)*np.tanh(xBeta))**2.0)) for k in xStag])
    xixxStag = np.array([-4.0*(np.tanh(xBeta)**3.0)*(1.0 - 2.0*k/xLen)/(xBeta*xLen*xLen*(1.0 - (np.tanh(xBeta)*(1.0 - 2.0*k/xLen))**2.0)**2.0) for k in xStag])
    xix2Stag = np.array([k*k for k in xi_xStag])

    et_yColl = np.array([np.tanh(yBeta)/(yBeta*yLen*(1.0 - ((1.0 - 2.0*j/yLen)*np.tanh(yBeta))**2.0)) for j in yColl])
    etyyColl = np.array([-4.0*(np.tanh(yBeta)**3.0)*(1.0 - 2.0*j/yLen)/(yBeta*yLen*yLen*(1.0 - (np.tanh(yBeta)*(1.0 - 2.0*j/yLen))**2.0)**2.0) for j in yColl])
    ety2Coll = np.array([j*j for j in et_yColl])

    et_yStag = np.array([np.tanh(yBeta)/(yBeta*yLen*(1.0 - ((1.0 - 2.0*j/yLen)*np.tanh(yBeta))**2.0)) for j in yStag])
    etyyStag = np.array([-4.0*(np.tanh(yBeta)**3.0)*(1.0 - 2.0*j/yLen)/(yBeta*yLen*yLen*(1.0 - (np.tanh(yBeta)*(1.0 - 2.0*j/yLen))**2.0)**2.0) for j in yStag])
    ety2Stag = np.array([j*j for j in et_yStag])

    zt_zColl = np.array([np.tanh(zBeta)/(zBeta*zLen*(1.0 - ((1.0 - 2.0*i/zLen)*np.tanh(zBeta))**2.0)) for i in zColl])
    ztzzColl = np.array([-4.0*(np.tanh(zBeta)**3.0)*(1.0 - 2.0*i/zLen)/(zBeta*zLen*zLen*(1.0 - (np.tanh(zBeta)*(1.0 - 2.0*i/zLen))**2.0)**2.0) for i in zColl])
    ztz2Coll = np.array([i*i for i in zt_zColl])

    zt_zStag = np.array([np.tanh(zBeta)/(zBeta*zLen*(1.0 - ((1.0 - 2.0*i/zLen)*np.tanh(zBeta))**2.0)) for i in zStag])
    ztzzStag = np.array([-4.0*(np.tanh(zBeta)**3.0)*(1.0 - 2.0*i/zLen)/(zBeta*zLen*zLen*(1.0 - (np.tanh(zBeta)*(1.0 - 2.0*i/zLen))**2.0)**2.0) for i in zStag])
    ztz2Stag = np.array([i*i for i in zt_zStag])

