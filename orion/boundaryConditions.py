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

from orion.globalVars import xyPeriodic, probType, planar

if planar:
    # No-slip and no-penetration BCs
    def imposeUBCs(U):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            U[0, :] = U[-2, :]

            # Right wall
            U[-1, :] = U[1, :]

        # No-slip and no-penetration BCs
        else:
            # Left wall
            U[0, :] = -U[1, :]

            # Right wall
            U[-1, :] = -U[-2, :]

        # Bottom wall
        U[:, 0] = 0.0

        # Top wall
        if probType == 0:
            # Moving lid on top for LDC
            U[:, -1] = 1.0
        elif probType == 1:
            U[:, -1] = 0.0

        return U


    # No-slip and no-penetration BCs
    def imposeWBCs(W):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            W[0, :] = W[-3, :]

            # Right wall
            W[-1, :] = W[2, :]

        # No-slip and no-penetration BCs
        else:
            # Left wall
            W[0, :] = 0.0

            # Right wall
            W[-1, :] = 0.0

        # Bottom wall
        W[:, 0] = -W[:, 1]

        # Top wall
        W[:, -1] = -W[:, -2]

        return W


    def imposePBCs(P):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            P[0, :] = P[-3, :]

            # Right wall
            P[-1, :] = P[2, :]

        # Neumann boundary condition on pressure
        else:
            # Left wall
            P[0, :] = P[2, :]

            # Right wall
            P[-1, :] = P[-3, :]

        # Bottom wall
        P[:, 0] = P[:, 2]

        # Top wall
        P[:, -1] = P[:, -3]

        return P
else:

    # No-slip and no-penetration BCs
    def imposeUBCs(U):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            U[0, :, :] = U[-2, :, :]

            # Right wall
            U[-1, :, :] = U[1, :, :]

            # Front wall
            U[:, 0, :] = U[:, -3, :]

            # Back wall
            U[:, -1, :] = U[:, 2, :]

        # No-slip and no-penetration BCs
        else:
            # Left wall
            U[0, :, :] = -U[1, :, :]

            # Right wall
            U[-1, :, :] = -U[-2, :, :]

            # Front wall
            U[:, 1, :] = 0.0

            # Back wall
            U[:, -2, :] = 0.0

        # Bottom wall
        U[:, :, 1] = 0.0

        # Top wall
        if probType == 0:
            # Moving lid on top for LDC
            U[:, :, -2] = 1.0
        elif probType == 1:
            U[:, :, -2] = 0.0

        return U


    # No-slip and no-penetration BCs
    def imposeVBCs(V):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            V[0, :, :] = V[-3, :, :]

            # Right wall
            V[-1, :, :] = V[2, :, :]

            # Front wall
            V[:, 0, :] = V[:, -2, :]

            # Back wall
            V[:, -1, :] = V[:, 1, :]

        # No-slip and no-penetration BCs
        else:
            # Left wall
            V[1, :, :] = 0.0

            # Right wall
            V[-2, :, :] = 0.0

            # Front wall
            V[:, 0, :] = -V[:, 1, :]

            # Back wall
            V[:, -1, :] = -V[:, -2, :]

        # Bottom wall
        V[:, :, 1] = 0.0

        # Top wall
        V[:, :, -2] = 0.0

        return V


    # No-slip and no-penetration BCs
    def imposeWBCs(W):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            W[0, :, :] = W[-3, :, :]

            # Right wall
            W[-1, :, :] = W[2, :, :]

            # Front wall
            W[:, 0, :] = W[:, -3, :]

            # Back wall
            W[:, -1, :] = W[:, 2, :]

        # No-slip and no-penetration BCs
        else:
            # Left wall
            W[1, :, :] = 0.0

            # Right wall
            W[-2, :, :] = 0.0

            # Front wall
            W[:, 1, :] = 0.0

            # Back wall
            W[:, -2, :] = 0.0

        # Bottom wall
        W[:, :, 0] = -W[:, :, 1]

        # Top wall
        W[:, :, -1] = -W[:, :, -2]

        return W


    def imposePBCs(P):
        # Periodic BCs along X and Y directions
        if xyPeriodic:
            # Left wall
            P[0, :, :] = P[-3, :, :]

            # Right wall
            P[-1, :, :] = P[2, :, :]

            # Front wall
            P[:, 0, :] = P[:, -3, :]

            # Back wall
            P[:, -1, :] = P[:, 2, :]

        # Neumann boundary condition on pressure
        else:
            # Left wall
            P[0, :, :] = P[2, :, :]

            # Right wall
            P[-1, :, :] = P[-3, :, :]

            # Front wall
            P[:, 0, :] = P[:, 2, :]

            # Back wall
            P[:, -1, :] = P[:, -3, :]

        # Bottom wall
        P[:, :, 0] = P[:, :, 2]

        # Top wall
        P[:, :, -1] = P[:, :, -3]

        return P
