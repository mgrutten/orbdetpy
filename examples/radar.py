# radar.py - Example to demonstrate OD using radar measurements.
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
import time
import json

if (len(sys.argv) < 4):
    print("Usage: python %s config_file measurement_file output_file [EKF|UKF|DMC]"
          % sys.argv[0])
    exit()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbdetpy import init

with open(sys.argv[1], "r") as f:
    init(json.load(f))

if len(sys.argv) > 4:
    filt = sys.argv[4].upper()
    if filt == "EKF":
        from orbdetpy.ekf import estimate
    elif filt == "DMC":
        from orbdetpy.dmc import estimate
    else:
        filt = "UKF"
        from orbdetpy.ukf import estimate
else:
    filt = "UKF"
    from orbdetpy.ukf import estimate

print("%s start : %s" % (filt, time.strftime("%Y-%m-%d %H:%M:%S")))
with open(sys.argv[2], "r") as fin:
    res = estimate(json.load(fin))
    with open(sys.argv[3], "w") as fout:
        json.dump(res, fout, indent = 1)
print("%s end   : %s" % (filt, time.strftime("%Y-%m-%d %H:%M:%S")))
