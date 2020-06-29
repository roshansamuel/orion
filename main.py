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
import fluidSolver as fs
import globalVars as gv
import numpy as np
import time

# Main segment of code.
def main():
    global U, V, W, P

    maxProcs = mp.cpu_count()
    if gv.nProcs > maxProcs:
        print("\nERROR: " + str(gv.nProcs) + " exceeds the available number of processors (" + str(maxProcs) + ")\n")
        exit(0)
    else:
        print("\nUsing " + str(gv.nProcs) + " out of " + str(maxProcs) + " processors\n")

    fs.grid.calculateMetrics()

    fs.initFields()

    ndTime = 0.0
    fwTime = 0.0

    tStart = time.process_time()

    while True:
        if abs(fwTime - ndTime) < 0.5*gv.dt:
            fs.writeSoln(ndTime)
            fwTime += gv.fwInt

        fs.euler()

        maxDiv = fs.getDiv()
        if maxDiv[1] > 10.0:
            print("ERROR: Divergence has exceeded permissible limits. Aborting")
            quit()

        gv.iCnt += 1
        ndTime += gv.dt
        if gv.iCnt % gv.opInt == 0:
            print("Time: {0:9.5f}".format(ndTime))
            print("Maximum divergence: {0:8.5f} at ({1:d}, {2:d}, {3:d})\n".format(maxDiv[1], maxDiv[0][0], maxDiv[0][1], maxDiv[0][2]))

        if ndTime > gv.tMax:
            break

    tEnd = time.process_time()
    tElap = tEnd - tStart

    print("Time elapsed = ", tElap)
    print("Simulation completed")


####################################################################################################


# Create list of ranges (in terms of indices) along X direction, which is the direction of parallelization
rangeDivs = [int(x) for x in np.linspace(1, fs.grid.L-1, gv.nProcs+1)]
rListColl = [(rangeDivs[x], rangeDivs[x+1]) for x in range(gv.nProcs)]
rangeDivs = [int(x) for x in np.linspace(1, fs.grid.L, gv.nProcs+1)]
rListStag = [(rangeDivs[x], rangeDivs[x+1]) for x in range(gv.nProcs)]

if __name__ == "__main__":
    main()

