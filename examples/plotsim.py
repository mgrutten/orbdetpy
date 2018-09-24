# plotsim.py - Plot simulated data.
# Copyright (C) 2018 Shiva Iyer <shiva.iyer AT utexas DOT edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import json
import numpy
from numpy.linalg import norm
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbdetpy import init
init()
from orbdetpy.orekit import *
from orbdetpy.utils import *

if (len(sys.argv) < 3):
    print("Usage: python %s config_file simulated_data_file"
          % sys.argv[0])
    exit()

with open(sys.argv[1], "r") as f:
    cfg = json.load(f)
with open(sys.argv[2], "r") as f:
    out = json.load(f)

mu = 398600.4418
mass = cfg["SpaceObject"]["Mass"]
tstamp, hvec, hmag, ener, alt, ecc, inc, accel = [], [], [], [], [], [], [], []
ecefList, eciList, velList = [], [], []
sdim = len(out[0]["State"])
frame = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

for o in out:
    tim = datetime.strptime(o["Time"], "%Y-%m-%dT%H:%M:%S.%fZ")
    javatim = strtodate(o["Time"])
    tstamp.append(tim)

    state = o["State"]
    
    javaState = SpacecraftState(CartesianOrbit(
        PVCoordinates(
            Vector3D(state[0], state[1], state[2]), 
            Vector3D(state[3], state[4], state[5])),
        frame, javatim, Constants.EGM96_EARTH_MU), mass)
    eci = javaState.getPVCoordinates(frame).getPosition().toArray()
    ecef = javaState.getPVCoordinates(itrf).getPosition().toArray()
    vel = javaState.getPVCoordinates(itrf).getVelocity().toArray()

    eciList.append(eci)
    ecefList.append(ecef)
    velList.append(vel)

hvec = numpy.array(hvec)
accel = numpy.array(accel)
eciList = numpy.array(eciList)/1000.0
ecefList = numpy.array(ecefList)/1000.0
velList = numpy.array(velList)
tim = [(t - tstamp[0]).total_seconds()/3600 for t in tstamp]

fig = plt.figure(0)
plt.subplot(311)
plt.plot(tim, eciList[:,0], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("x [km]")
plt.subplot(312)
plt.plot(tim, eciList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("y [km]")
plt.subplot(313)
plt.plot(tim, eciList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("z [km]")
plt.suptitle("ECI Position")
plt.savefig('img/eci-position.png')

fig = plt.figure(1)
plt.subplot(311)
plt.plot(tim, ecefList[:,0], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("x [km]")
plt.subplot(312)
plt.plot(tim, ecefList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("y [km]")
plt.subplot(313)
plt.plot(tim, ecefList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("z [km]")
plt.suptitle("ECEF Position")
plt.savefig('img/ecef-position.png')

fig = plt.figure(2)
plt.subplot(311)
plt.plot(tim, velList[:,0], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("v_x [m/s]")
plt.subplot(312)
plt.plot(tim, velList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("v_y [m/s]")
plt.subplot(313)
plt.plot(tim, velList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("v_z [m/s]")
plt.suptitle("Velocity in ECEF")
plt.savefig('img/velocity.png')
