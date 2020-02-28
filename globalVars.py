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

# Import all necessary modules
import multiprocessing as mp
import numpy as np
import h5py as hp

############################################ USER PARAMETERS #######################################

# Set the number of processors for parallel computing with multiprocessing module
nProcs = 8

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# [2, 4, 6, 10, 18, 34, 66, 130, 258, 514, 1026, 2050]
#  0  1  2  3   4   5   6    7    8    9    10    11
sInd = np.array([5, 5, 5])

# Domain lengths - along X, Y and Z directions respectively
dLen = [1.0, 1.0, 1.0]

# Tangent-hyperbolic grid stretching factor along X, Y and Z directions respectively
beta = [1.0, 1.0, 1.0]

# The hydrodynamic problem to be solved can be chosen from below
# 0 - Lid-driven cavity
# 1 - Forced channel flow
probType = 0

# Flag for setting periodicity along X and Y directions of the domain
xyPeriodic = False

# Toggle fwMode between ASCII and HDF5  to write output data in corresponding format
fwMode = "HDF5"

# Time-step
dt = 0.01

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 1.0

# Final time
tMax = 0.1

# Reynolds number
Re = 1000

# Tolerance value in Jacobi iterations
tolerance = 0.00001

# Depth of each V-cycle in multigrid
VDepth = 3

# Number of V-cycles to be computed
vcCnt = 10

# Number of iterations during pre-smoothing
preSm = 10

# Number of iterations during post-smoothing
pstSm = 40

# Number of iterations during smoothing in between prolongation operators
proSm = 30

######################################## END OF USER PARAMETERS ####################################

######################################## OTHER GLOBAL PARAMETERS ###################################

time = 0.0
iCnt = 0
nu = 1.0/Re

