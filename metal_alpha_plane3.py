import numpy as np
import sys

import h5py
from auriga_config import *
import auriga_functions as auriga
import snapHDF5 as snap
import readhaloHDF5
import readsubfHDF5
from get_halo import get_data

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
#mpl.use('Agg')
#plt.ioff()
plt.ion()

from scipy.ndimage import gaussian_filter

# Select the simulation:
#--------------------------------------------------------------------
snapshot = 127
halo = int(sys.argv[1])
#--------------------------------------------------------------------

# Load the snapshot at z=0 and get the metal ratios:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE', 'POT ', 'GMET', 'GSPH']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, data_fields=data_fields, cat_fields=cat_fields, align_on_disc=True)
#--------------------------------------------------------------------

# Identify disc stars:
#--------------------------------------------------------------------
data['DISC'] = auriga.identify_disc(data['POS '], data['VEL '], data['POT '])
#--------------------------------------------------------------------

# Make a histogram of the chemical plane:
#--------------------------------------------------------------------
fs = 12
fig, ax = plt.subplots(figsize=(6,6))

N_bins = 250
binrange = [np.nanpercentile(data['FeH '], [1, 99.9]), \
            np.nanpercentile(data['MgFe'], [1, 99])]
binrange[0] += np.array([-0.1, 0.1])*np.diff(binrange[0])
binrange[1] += np.array([-0.1, 0.1])*np.diff(binrange[1])
FeH_bins = np.linspace(*binrange[0], N_bins)
MgFe_bins = np.linspace(*binrange[1], N_bins)
extent = np.ravel([FeH_bins[[0,-1]], MgFe_bins[[0,-1]]])
z_cut = 2
r_cut = (0, 30)
cut = data['DISC'] * (np.abs(data['POS '][:,2]) <= z_cut) * (data['RG  '] >= r_cut[0]) * (data['RG  '] <= r_cut[1]) * data['IN  ']
hist = np.histogram2d(data['FeH '][cut], data['MgFe'][cut], weights=data['MASS'][cut], bins=[FeH_bins, MgFe_bins])[0]
hist[hist==0] = np.nan
#img = ax.imshow(hist.T, origin='lower', extent=extent, norm=LogNorm(), aspect=np.diff(FeH_bins[[0,-1]])/np.diff(MgFe_bins[[0,-1]]), cmap=cm.viridis, rasterized=True)
#img = ax.imshow(hist.T, origin='lower', extent=extent, aspect=np.diff(FeH_bins[[0,-1]])/np.diff(MgFe_bins[[0,-1]]), cmap=cm.viridis, rasterized=True)
img = ax.imshow(hist.T, origin='lower', extent=extent, aspect=np.diff(FeH_bins[[0,-1]])/np.diff(MgFe_bins[[0,-1]]), cmap=cm.YlGnBu, rasterized=True, norm=LogNorm())

ax.set_xlabel('[Fe/H]', fontsize=fs)
ax.set_ylabel('[Mg/Fe]', fontsize=fs)
ax.tick_params(axis='both', labelsize=fs-2)
#--------------------------------------------------------------------

# Add time contours:
#--------------------------------------------------------------------
N_bins = 50
FeH_bins = np.linspace(*binrange[0], N_bins)
MgFe_bins = np.linspace(*binrange[1], N_bins)
hist = np.histogram2d(data['FeH '][cut], data['MgFe'][cut], weights=data['RG  '][cut], bins=[FeH_bins, MgFe_bins])[0]
hist_norm = np.histogram2d(data['FeH '][cut], data['MgFe'][cut], bins=[FeH_bins, MgFe_bins])[0]
hist /= hist_norm

# Smooth histogram:
def nan_smooth(U, sigma=1):
  V = U.copy()
  V[np.isnan(U)] = 0
  VV = gaussian_filter(V, sigma=1)
  W = 0*U.copy()+1
  W[np.isnan(U)] = 0
  WW = gaussian_filter(W, sigma=1)
  Z = VV/WW
  Z[np.isnan(U)] = np.nan
  return Z
hist = nan_smooth(hist, sigma=2)

# Plot contours:
contours = [1, 3, 5, 7, 11, 15]
CS = ax.contour(hist.T, extent=extent, levels=contours, colors=['k'], zorder=99, linewidths=[1])

# Remove disjointed contour segments:
for rung in ax.collections:
  segments = rung.get_paths()
  if len(segments):
    main_segment = np.argmax([len(segment) for segment in segments])
    rung._paths = [segments[main_segment]]

# Define locations for clabels as the midpoints of the contours:
manual_locations = []
for contour in CS.collections:
  if len(contour.get_paths()):
    path = contour.get_paths()[0]._vertices
    manual_locations.append(tuple(path[int(len(path)/2)]))

ax.clabel(CS, inline=True, fmt='%i', fontsize=fs-3, manual=manual_locations)
ax.set_facecolor('lightgrey')

string = r'$%i<R/\rm{kpc}<%i$' % (r_cut[0], r_cut[1]) + '\n' r'$\Delta Z = %i\,$kpc' % (z_cut * 2)
ax.text(0.95, 0.95, string, ha='right', va='top', transform=ax.transAxes, fontsize=fs)
fig.suptitle(r'Au-%i' % halo, fontsize=fs, y=0.925)
#--------------------------------------------------------------------

plt.savefig('./images/metal_plane_Au-%i_r%i-%i_Rg.pdf' % (halo, r_cut[0], r_cut[1]), bbox_inches='tight')
