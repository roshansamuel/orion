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
from meshData import xColl, yColl, zColl
from globalVars import fwMode
import h5py as hp

def writeSoln(U, V, W, P, time):
    L, M ,N = tuple(map(lambda i: i - 1, P.shape))

    if fwMode == "ASCII":
        fName = "Soln_" + "{0:09.5f}".format(time) + ".dat"
        print("Writing solution file: ", fName)

        ofFile = open(fName, 'w')
        ofFile.write("VARIABLES = X, Y, Z, U, V, W, P\n")
        ofFile.write("ZONE T=S\tI={0}\tJ={1}\tK={2}\tF=POINT\tSOLUTIONTIME={3}\n".format(L, M, N, time))
        for i in range(0, N):
            for j in range(0, M):
                for k in range(0, L):
                    ofFile.write("{0:23.16f}\t{1:23.16f}\t{2:23.16f}\t{3:23.16f}\t{4:23.16f}\t{5:23.16f}\t{6:23.16f}\n".format(
                                xColl[k], yColl[j], zColl[i],
                                (U[k, j, i] + U[k, j+1, i] + U[k, j, i+1] + U[k, j+1, i+1])/4.0,
                                (V[k, j, i] + V[k+1, j, i] + V[k, j, i+1] + V[k+1, j, i+1])/4.0,
                                (W[k, j, i] + W[k+1, j, i] + W[k, j+1, i] + W[k+1, j+1, i])/4.0,
                                (P[k, j, i] + P[k+1, j, i] + P[k, j+1, i] + P[k+1, j+1, i] +
                                 P[k, j, i+1] + P[k+1, j, i+1] + P[k, j+1, i+1] + P[k+1, j+1, i+1])/8.0))

        ofFile.close()

    elif fwMode == "HDF5":
        fName = "Soln_" + "{0:09.5f}.h5".format(time)
        print("Writing solution file: ", fName)

        f = hp.File(fName, "w")

        dset = f.create_dataset("U", data = U)
        dset = f.create_dataset("V", data = V)
        dset = f.create_dataset("W", data = W)
        dset = f.create_dataset("P", data = P)

        f.close()

