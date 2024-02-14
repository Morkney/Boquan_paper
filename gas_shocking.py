import numpy as np
import sys

import h5py
from auriga_config import *
import auriga_functions as auriga
import snapHDF5 as snap
import readhaloHDF5
import readsubfHDF5
from get_halo import get_data
import misc_funcs as misc

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
plt.ion()

from scipy.interpolate import make_interp_spline

if sys.version[:5] != '3.7.3':
  print('Change python dist.')
  sys.exit()
from sphviewer.tools import QuickView

# Select the simulation:
#--------------------------------------------------------------------
snapshot = 73
subhalo = 0
halo = int(sys.argv[1])
z_cut = 5
r_cut = (0.3, 30)
#--------------------------------------------------------------------

# Time information:
#--------------------------------------------------------------------
tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % 127
f = h5py.File(tree_path, "r")['Tree0']
ID = np.where((f['SnapNum'][:]==snapshot) & (f['SubhaloNumber'][:]==subhalo))
redshifts = np.array(h5py.File(tree_path, 'r')['Header']['Redshifts'][:])
a = 1/(1+redshifts)
time = np.array([auriga.age_time(i) for i in a])
lookback_time = auriga.age_time(1) - time
#--------------------------------------------------------------------

# Load stellar data to find disc orientation:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, stellar_data = get_data(halo, snapshot, SubhaloNumber=f['SubhaloNumber'][ID], \
                                         data_fields=data_fields, cat_fields=cat_fields, align_on_disc=True)
#--------------------------------------------------------------------

# Load gas data:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'EDIS', 'MACH', 'U   ', 'NE  ', 'RHO ']
header, cat, lf, data = get_data(halo, snapshot, SubhaloNumber=f['SubhaloNumber'][ID], \
                                 data_fields=data_fields, cat_fields=cat_fields, align_on_disc=False, single_sub=False, PtType=0)
data['POS '] = np.dot(stellar_data['DROT'], [*data['POS '].T]).T
data['VEL '] = np.dot(stellar_data['DROT'], [*data['VEL '].T]).T

# Calculate temperature in Kelvin:
data['TEMP'] = auriga.temperature(data['U   '], data['NE  '])
#--------------------------------------------------------------------

# Look at what the various properties look like:
#--------------------------------------------------------------------
# Temperature looks sort of ok. Ranges from <1e4 to >1e7. No signs of filaments, though.
# Mach number could be OK. It is difficult to tell.

slice = np.abs(data['POS '][:,2]) < 20
lim = 100
bins = [np.linspace(-lim, lim, 100), np.linspace(-lim, lim, 100)]
extent = np.array([-lim, lim, -lim, lim])

rho_img = QuickView(data['POS '][slice], mass=data['RHO '][slice], \
                    r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
density_img = QuickView(data['POS '][slice], mass=data['MASS'][slice], \
                        r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
temp_img = QuickView(data['POS '][slice], mass=data['RHO '][slice]*data['TEMP'][slice], \
                     r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / rho_img
edis_img = QuickView(data['POS '][slice], mass=data['EDIS'][slice], \
                     r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
mach_img = QuickView(data['POS '][slice], mass=data['EDIS'][slice]*data['TEMP'][slice], \
                     r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / edis_img

f, ax = plt.subplots(figsize=(8,4), ncols=2)

img1 = ax[0].imshow(np.log10(temp_img).T, origin='lower')
img2 = ax[1].imshow(np.log10(mach_img).T, origin='lower')
#--------------------------------------------------------------------
