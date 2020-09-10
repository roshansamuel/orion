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
import numpy as np

############################################ USER PARAMETERS #######################################

# Set whether the simulation is going to be 2D or 3D
# If 2D, set the below flag to True
planar = False

# Set below flag to True if the Poisson solver is being tested
testPoisson = True

# Set the number of processors for parallel computing (under development) with multiprocessing module
nProcs = 8

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# [2, 4, 6, 10, 18, 34, 66, 130, 258, 514, 1026, 2050]
#  0  1  2  3   4   5   6    7    8    9    10    11
sInd = np.array([5, 5, 5])

# Domain lengths - along X, Y and Z directions respectively
dLen = [1.0, 1.0, 1.0]

# Turn below flag on when using uniform grid
uniformGrid = True

# If above flag is True, set tangent-hyperbolic stretching factors along X, Y and Z directions respectively
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

# Final time
tMax = 0.5

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 1.0

# Reynolds number
Re = 1000

# Flag to check if Poisson solver should solve or smooth at coarsest level
solveSol = False

# Tolerance value in Jacobi iterations
tolerance = 1.0e-6

# Depth of each V-cycle in multigrid
VDepth = 4

# Number of V-cycles to be computed
vcCnt = 5

# Number of iterations during pre-smoothing
preSm = 3

# Number of iterations during post-smoothing
pstSm = 3

######################################## END OF USER PARAMETERS ####################################

######################################## OTHER GLOBAL PARAMETERS ###################################

iCnt = 0

nu = 1.0/Re

########################################### PRINT PARAMETERS #######################################

def printParams():
    if testPoisson:
        print("Testing Poisson solver\n")
    else:
        if probType == 0:
            print("Solving for lid-driven cavity\n")
        elif probType == 1:
            print("Solving for forced channel flow\n")

        print("Format used to write output data is {}\n".format(fwMode))
        print("Time-step is {}\n".format(dt))
        print("Number of iterations after which output must be printed to standard I/O is {}\n".format(opInt))
        print("File writing interval is {}\n".format(fwInt))
        print("Final time is {}\n".format(tMax))
        print("Reynolds number is {}\n".format(Re))

    nList = [1, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049]
    gList = [nList[x] for x in sInd]
    if planar:
        print("Running solver with 2D grid of size {} x {} over a domain of size {} x {}\n".format(gList[0], gList[2], dLen[0], dLen[1]))
    else:
        print("Running solver with 3D grid of size {} x {} x {} over a domain of size {} x {} x {}\n".format(*(tuple(gList + dLen))))

    if uniformGrid:
        print("Using uniform mesh in domain\n")
    else:
        print("Using non-uniform mesh with tangent-hyperbolic stretching factors {}, {} and {} along X, Y and Z respectively\n".format(*beta))

    print("Tolerance value in Jacobi iterations is {}\n".format(tolerance))
    print("Depth of each V-cycle in multigrid is {}\n".format(VDepth))
    print("Number of V-cycles to be computed is {}\n".format(vcCnt))
    print("Number of iterations during pre-smoothing is {}\n".format(preSm))
    print("Number of iterations during post-smoothing is {}\n".format(pstSm))

########################################### CHECK PARAMETERS #######################################

def checkParams():
    # AN EMPTY LINE FOR AESTHETICS
    print("")

    if VDepth >= min(sInd):
        print("ERROR: V-Cycle depth exceeds the allowable depth for given grid size")
        exit()

    if testPoisson and not uniformGrid:
        print("ERROR: Poisson testing subroutines are presently available only for uniform grid solvers")
        exit()
