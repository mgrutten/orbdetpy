# plotodet.py - Module to plot OD output.
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

if (len(sys.argv) < 4):
    print("Usage: python %s config_file measurement_file output_file"
          % sys.argv[0])
    exit()

with open(sys.argv[1], "r") as f:
    cfg = json.load(f)
with open(sys.argv[2], "r") as f:
    inp = json.load(f)
with open(sys.argv[3], "r") as f:
    out = json.load(f)["Estimation"]

mu = 398600.4418
mass = cfg["SpaceObject"]["Mass"]
tstamp, prefit, posfit, inocov, params = [], [], [], [], []
hvec, hmag, ener, alt, ecc, inc, accel = [], [], [], [], [], [], []
trueECIList, estECIList, trueECEFList, estECEFList, trueVelList, estVelList = [], [], [], [], [], []
trueAccList, estAccList = [], []
trueRICList, estRICList, trueRICvelList, trueRICaccList, estRICvelList, estRICaccList = [], [], [], [], [], []
sdim = len(out[0]["EstimatedState"])
frame = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

for i, o in zip(inp, out):
    tim = datetime.strptime(i["Time"], "%Y-%m-%dT%H:%M:%S.%fZ")
    javatim = strtodate(i["Time"])
    tstamp.append(tim)

    p = []
    for k, v in o["PreFit"].items():
        p.append(i[k] - v)
    prefit.append(p)

    p = []
    for k, v in o["PostFit"].items():
        p.append(i[k] - v)
    posfit.append(p)

    if (len(o["EstimatedState"]) > 6):
        params.append(o["EstimatedState"][6:])

    if ("InnovationCovariance" in o):
        p = []
        for j in range(len(o["InnovationCovariance"])):
            p.append(3.0*numpy.sqrt(o["InnovationCovariance"][j][j]))
        inocov.append(p)

    trueState = i["State"]
    estimatedState = o["EstimatedState"]

    trueJavaState = SpacecraftState(CartesianOrbit(
        PVCoordinates(
            Vector3D(trueState[0], trueState[1], trueState[2]), 
            Vector3D(trueState[3], trueState[4], trueState[5])),
        frame, javatim, Constants.EGM96_EARTH_MU), mass)
    trueECI = trueJavaState.getPVCoordinates(frame).getPosition().toArray()
    trueVel = trueJavaState.getPVCoordinates(frame).getVelocity().toArray()
    trueECEF = trueJavaState.getPVCoordinates(itrf).getPosition().toArray()
    trueAcc = trueJavaState.getPVCoordinates(frame).getAcceleration().toArray()

    trueHvec = numpy.cross(trueECI, trueVel)
    radial = trueECI/norm(trueECI)
    crosstrack = trueHvec/norm(trueHvec)
    intrack = numpy.cross(crosstrack, radial)
    RIC = numpy.array([radial, intrack, crosstrack])

    mu = trueJavaState.getMu()
    trueAcc = numpy.array(trueAcc) + mu/norm(trueECI)**3 * numpy.array(trueECI)

    estimatedJavaState = SpacecraftState(CartesianOrbit(
        PVCoordinates(
            Vector3D(estimatedState[0], estimatedState[1], estimatedState[2]), 
            Vector3D(estimatedState[3], estimatedState[4], estimatedState[5])),
        frame, javatim, Constants.EGM96_EARTH_MU), mass)

    estECI = estimatedJavaState.getPVCoordinates(frame).getPosition().toArray()
    estECEF = estimatedJavaState.getPVCoordinates(itrf).getPosition().toArray()
    estVel = trueJavaState.getPVCoordinates(frame).getVelocity().toArray()

    if sdim == 9 :
        estAcc = numpy.array(estimatedState[6:])
        estAccList.append(estAcc)
        trueRICacc = RIC.dot(trueAcc)
        estRICacc = RIC.dot(estAcc)
        trueRICaccList.append(trueRICacc)
        estRICaccList.append(estRICacc)

    estRIC = RIC.dot(estECI)
    trueRICvel = RIC.dot(trueVel)
    estRICvel = RIC.dot(estVel)

    trueECIList.append(trueECI)
    estECIList.append(estECI)
    trueRICList.append(norm(trueECI))
    estRICList.append(estRIC)
    trueECEFList.append(trueECEF)
    estECEFList.append(estECEF)
    trueVelList.append(trueVel)
    estVelList.append(estVel)
    trueAccList.append(trueAcc)

    trueRICvelList.append(trueRICvel)
    estRICvelList.append(estRICvel)

pre = numpy.array(prefit)
pos = numpy.array(posfit)
cov = numpy.array(inocov)
par = numpy.array(params)
hvec = numpy.array(hvec)

trueECIList = numpy.array(trueECIList)/1000.0
estECIList = numpy.array(estECIList)/1000.0
trueECEFList = numpy.array(trueECEFList)/1000.0
estECEFList = numpy.array(estECEFList)/1000.0
trueVelList = numpy.array(trueVelList)/1000.0
estVelList = numpy.array(estVelList)/1000.0
trueAccList = numpy.array(trueAccList)
estAccList = numpy.array(estAccList)
key = list(cfg["Measurements"].keys())
tim = [(t - tstamp[0]).total_seconds()/3600 for t in tstamp]

trueRICList = numpy.array(trueRICList)/1000.0
estRICList = numpy.array(estRICList)/1000.0
trueRICvelList = numpy.array(trueRICvelList)/1000.0
estRICvelList = numpy.array(estRICvelList)/1000.0
trueRICaccList = numpy.array(trueRICaccList)
estRICaccList = numpy.array(estRICaccList)

fig = plt.figure(0)
plt.subplot(211)
plt.plot(tim, pre[:,0], "ob")
plt.xlabel("Time [hr]")
plt.ylabel("%s residual" % key[0])
plt.subplot(212)
plt.plot(tim, pre[:,1], "ob")
plt.xlabel("Time [hr]")
plt.ylabel("%s residual" % key[1])
plt.suptitle("Pre-fit residuals")
plt.savefig('prefit-residuals')

fig = plt.figure(1)
plt.subplot(211)
plt.plot(tim, pos[:,0], "ob", label="Residual")
if (len(cov) > 0):
    plt.plot(tim, -cov[:,0], "-r", label="Covariance")
    plt.plot(tim,  cov[:,0], "-r")
plt.xlabel("Time [hr]")
plt.ylabel("%s residual" % key[0])
plt.legend(bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure)
plt.subplot(212)
plt.plot(tim, pos[:,1], "ob")
if (len(cov) > 0):
    plt.plot(tim, -cov[:,1], "-r")
    plt.plot(tim,  cov[:,1], "-r")
plt.xlabel("Time [hr]")
plt.ylabel("%s residual" % key[1])
plt.suptitle("Post-fit residuals")

#for i in range(par.shape[-1]):
#    if (i == 0):
#        fig = plt.figure(2)
#    plt.subplot(par.shape[1], 1, i + 1)
#    plt.plot(tim, par[:,i], "ob")
#    plt.xlabel("Time [hr]")
#    plt.ylabel("Parameter %d" % (i + 1))

plt.savefig('residuals')

fig = plt.figure(3)
plt.subplot(311)
plt.plot(tim, trueECEFList[:,0], "-r", label="Truth")
plt.plot(tim, estECEFList[:,0], "-b", label="Estimate")
plt.xlabel("Time [hr]")
plt.ylabel("x [km]")
plt.legend(bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure)
plt.subplot(312)
plt.plot(tim, trueECEFList[:,1], "-r")
plt.plot(tim, estECEFList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("y [km]")
plt.subplot(313)
plt.plot(tim, trueECEFList[:,2], "-r")
plt.plot(tim, estECEFList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("z [km]")
plt.suptitle("ECEF Position")
plt.savefig('ecef')

fig = plt.figure(4)
plt.subplot(311)
plt.plot(tim, trueECIList[:,0], "-r", label="Truth")
plt.plot(tim, estECIList[:,0], "-b", label="Estimate")
plt.xlabel("Time [hr]")
plt.ylabel("x [km]")
plt.legend(bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure)
plt.subplot(312)
plt.plot(tim, trueECIList[:,1], "-r")
plt.plot(tim, estECIList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("y [km]")
plt.subplot(313)
plt.plot(tim, trueECIList[:,2], "-r")
plt.plot(tim, estECIList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("z [km]")
plt.suptitle("ECI Position")
plt.savefig('eci')

fig = plt.figure(5)
plt.subplot(311)
plt.plot(tim, trueVelList[:,0], "-r", label="Truth")
plt.plot(tim, estVelList[:,0], "-b", label="Estimate")
plt.xlabel("Time [hr]")
plt.ylabel("$v_x$ [m/s]")
plt.legend(bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure)
plt.subplot(312)
plt.plot(tim, trueVelList[:,1], "-r")
plt.plot(tim, estVelList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("$v_y$ [m/s]")
plt.subplot(313)
plt.plot(tim, trueVelList[:,2], "-r")
plt.plot(tim, estVelList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("$v_z$ [m/s]")
plt.suptitle("Velocity")
plt.savefig('vel')

fig = plt.figure(7)
plt.subplot(311)
plt.plot(tim, trueRICvelList[:,0], "-r", label="Truth")
plt.plot(tim, estRICvelList[:,0], "-b", label="Estimate")
plt.xlabel("Time [hr]")
plt.ylabel("$v_R$ [m/s]")
plt.legend(bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure)
plt.subplot(312)
plt.plot(tim, trueRICvelList[:,1], "-r")
plt.plot(tim, estRICvelList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("$v_I$ [m/s]")
plt.subplot(313)
plt.plot(tim, trueRICvelList[:,2], "-r")
plt.plot(tim, estRICvelList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("$v_C$ [m/s]")
plt.suptitle("Velocity in RIC")
plt.savefig('ric-vel')

fig = plt.figure(8)
plt.subplot(311)
plt.plot(tim, trueRICList, "-r", label="Truth")
plt.plot(tim, estRICList[:,0], "-b", label="Estimate")
plt.xlabel("Time [hr]")
plt.ylabel("Radial [km]")
plt.legend(bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure)
plt.subplot(312)
plt.plot(tim, estRICList[:,1], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("Intrack [km]")
plt.subplot(313)
plt.plot(tim, estRICList[:,2], "-b")
plt.xlabel("Time [hr]")
plt.ylabel("Crosstrack [km]")
plt.suptitle("RIC Position")
plt.savefig('ric')

if sdim == 9 :
    fig = plt.figure(9)
    plt.subplot(311)
    plt.plot(tim, trueAccList[:,0], "-r", label="Truth")
    plt.plot(tim, estAccList[:,0], "-b", label="Estimate")
    plt.xlabel("Time [hr]")
    plt.ylabel("$w_x$")
    plt.legend(bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure)
    plt.subplot(312)
    plt.plot(tim, trueAccList[:,0], "-r")
    plt.plot(tim, estAccList[:,1], "-b")
    plt.xlabel("Time [hr]")
    plt.ylabel("$w_y$")
    plt.subplot(313)
    plt.plot(tim, trueAccList[:,0], "-r")
    plt.plot(tim, estAccList[:,2], "-b")
    plt.xlabel("Time [hr]")
    plt.ylabel("$w_z$")
    plt.suptitle("Components of Acceleration")
    plt.savefig('acc')

    fig = plt.figure(10)
    plt.subplot(311)
    plt.plot(tim, trueRICaccList[:,0], "-r", label="Truth")
    plt.plot(tim, estRICaccList[:,0], "-b", label="Estimate")
    plt.xlabel("Time [hr]")
    plt.ylabel("$w_R$")
    plt.legend(bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure)
    plt.subplot(312)
    plt.plot(tim, trueRICaccList[:,0], "-r")
    plt.plot(tim, estRICaccList[:,1], "-b")
    plt.xlabel("Time [hr]")
    plt.ylabel("$w_I$")
    plt.subplot(313)
    plt.plot(tim, trueRICaccList[:,0], "-r")
    plt.plot(tim, estRICaccList[:,2], "-b")
    plt.xlabel("Time [hr]")
    plt.ylabel("$w_C$")
    plt.suptitle("Acceleration in RIC")
    plt.savefig('ric-acc')
