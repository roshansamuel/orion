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
from orion.meshData import hx, hy, hz, hx2, hy2, hz2
from orion.globalVars import planar
import numpy as np


if planar:
############################# 2D VERSIONS #############################
    def DDXi(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1] = (inpFld[0:-2, 1:-1] - 2.0*inpFld[1:-1, 1:-1] + inpFld[2:, 1:-1])/hx2

        return outFld[1:-1, 1:-1]


    def DDZt(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1] = (inpFld[1:-1, 0:-2] - 2.0*inpFld[1:-1, 1:-1] + inpFld[1:-1, 2:])/hz2

        return outFld[1:-1, 1:-1]


    def D_Xi(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1] = (inpFld[2:, 1:-1] - inpFld[0:-2, 1:-1])*0.5/hx

        return outFld[1:-1, 1:-1]


    def D_Zt(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1] = (inpFld[1:-1, 2:] - inpFld[1:-1, 0:-2])*0.5/hz

        return outFld[1:-1, 1:-1]

else:
############################# 3D VERSIONS #############################
    def DDXi(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1, 1:-1] = (inpFld[0:-2, 1:-1, 1:-1] - 2.0*inpFld[1:-1, 1:-1, 1:-1] + inpFld[2:, 1:-1, 1:-1])/hx2

        return outFld[1:-1, 1:-1, 1:-1]


    def DDEt(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1, 1:-1] = (inpFld[1:-1, 0:-2, 1:-1] - 2.0*inpFld[1:-1, 1:-1, 1:-1] + inpFld[1:-1, 2:, 1:-1])/hy2

        return outFld[1:-1, 1:-1, 1:-1]


    def DDZt(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1, 1:-1] = (inpFld[1:-1, 1:-1, 0:-2] - 2.0*inpFld[1:-1, 1:-1, 1:-1] + inpFld[1:-1, 1:-1, 2:])/hz2

        return outFld[1:-1, 1:-1, 1:-1]


    def D_Xi(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1, 1:-1] = (inpFld[2:, 1:-1, 1:-1] - inpFld[0:-2, 1:-1, 1:-1])*0.5/hx

        return outFld[1:-1, 1:-1, 1:-1]


    def D_Et(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1, 1:-1] = (inpFld[1:-1, 2:, 1:-1] - inpFld[1:-1, 0:-2, 1:-1])*0.5/hy

        return outFld[1:-1, 1:-1, 1:-1]


    def D_Zt(inpFld):
        outFld = np.zeros_like(inpFld)
        outFld[1:-1, 1:-1, 1:-1] = (inpFld[1:-1, 1:-1, 2:] - inpFld[1:-1, 1:-1, 0:-2])*0.5/hz

        return outFld[1:-1, 1:-1, 1:-1]

