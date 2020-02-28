#!/usr/bin/python

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

from globalVars import sInd
import numpy as np

# Redefine frequently used numpy object
npax = np.newaxis

# N should be of the form 2^n + 2 so that there will be 2^n + 3 staggered pressure points, including ghost points
sLst = [2**x + 2 for x in range(12)]

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

# Define grid spacings
hx = 1.0/(L-1)
hy = 1.0/(M-1)
hz = 1.0/(N-1)

# Maximum number of jacobi iterations
maxCount = 10*N*M*L

# Grid metric arrays. Used by the enitre program at various points
xColl = np.zeros(L)
yColl = np.zeros(M)
zColl = np.zeros(N)
xStag = np.zeros(L-1)
yStag = np.zeros(M-1)
zStag = np.zeros(N-1)

xi_xColl = np.zeros(L)      #-- dXi/dX at all x-grid nodes
xixxColl = np.zeros(L)      #-- d2Xi/dX2 at all x-grid nodes
xix2Coll = np.zeros(L)      #-- (dXi/dX)**2 at all x-grid nodes

xi_xStag = np.zeros(L-1)    #-- dXi/dX at all x-grid nodes
xixxStag = np.zeros(L-1)    #-- d2Xi/dX2 at all x-grid nodes
xix2Stag = np.zeros(L-1)    #-- (dXi/dX)**2 at all x-grid nodes

et_yColl = np.zeros(M)      #-- dEt/dY at all y-grid nodes
etyyColl = np.zeros(M)      #-- d2Et/dY2 at all y-grid nodes
ety2Coll = np.zeros(M)      #-- (dEt/dY)**2 at all y-grid nodes

et_yStag = np.zeros(M-1)    #-- dEt/dY at all y-grid nodes
etyyStag = np.zeros(M-1)    #-- d2Et/dY2 at all y-grid nodes
ety2Stag = np.zeros(M-1)    #-- (dEt/dY)**2 at all y-grid nodes

zt_zColl = np.zeros(N)      #-- dZt/dZ at all z-grid nodes
ztzzColl = np.zeros(N)      #-- d2Zt/dZ2 at all z-grid nodes
ztz2Coll = np.zeros(N)      #-- (dZt/dZ)**2 at all z-grid nodes

zt_zStag = np.zeros(N-1)    #-- dZt/dZ at all z-grid nodes
ztzzStag = np.zeros(N-1)    #-- d2Zt/dZ2 at all z-grid nodes
ztz2Stag = np.zeros(N-1)    #-- (dZt/dZ)**2 at all z-grid nodes
