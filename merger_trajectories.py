import numpy as np
from scipy.interpolate import make_interp_spline
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
plt.ioff()
#plt.ion()

# Select the simulation:
#--------------------------------------------------------------------
snapshot = 127
halo = int(sys.argv[1])
#--------------------------------------------------------------------

# Step 1; load the snapshot at z=0 and get the metal ratios:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE', 'GMET']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, data_fields=data_fields, cat_fields=cat_fields)
#--------------------------------------------------------------------

# Step 2; find the IDs of all mergers that contribute to the galaxy:
#--------------------------------------------------------------------
tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % snapshot
f = h5py.File(tree_path, "r")['Tree0']
redshifts = np.array(h5py.File(tree_path, 'r')['Header']['Redshifts'][:])

pkmassids = np.unique(lf.mdata['Exsitu']['PeakMassIndex'])
pkmassids = pkmassids[pkmassids != -1]
pkmasses = f['SubhaloMassType'][:][pkmassids].sum(axis=1) * 1e10/h
ID = 0
IDs = {}
while ID != -1:
  IDs[f['SnapNum'][ID]] = ID
  ID = f['FirstProgenitor'][ID]
if halo==9:
  IDs[51] = IDs[50]
main_IDs = [IDs[i] for i in f['SnapNum'][:][pkmassids]]
mass_ratios = f['SubhaloMassType'][:][pkmassids].sum(axis=1) / \
              f['SubhaloMassType'][:][main_IDs].sum(axis=1)

mass_range = ((mass_ratios > 1/20.) | (pkmasses > 1e10)) & (pkmasses > 1e9)
pkmassids = pkmassids[mass_range]
pkmasses = pkmasses[mass_range]
#--------------------------------------------------------------------

# Step 3; find the progenitor histories of all mergers:
#--------------------------------------------------------------------
# Find the index of the main halo progenitors:
ID = np.where((f['SnapNum'][:]==snapshot) & (f['SubhaloNumber'][:]==0))[0][0]

# Find the tree indices for all the progenitors of this halo object:
print('>    Host')
IDs = []
while ID != -1:
  IDs.append(ID)
  ID = f['FirstProgenitor'][ID]
IDs = IDs[::-1]

# Find the indices of the merger haloes:
merger_IDs = {}
for pkmassid in pkmassids:
  print('>    Merger %i' % pkmassid)
  merger_IDs[pkmassid] = []
  ID = pkmassid

  # Wind the object forwards until merger time:
  while f['FirstProgenitor'][f['Descendant'][ID]] == ID:
    ID = f['Descendant'][ID]

  # Wind the object backwards and append the IDs:
  while ID != -1:
    merger_IDs[pkmassid].append(ID)
    ID = f['FirstProgenitor'][ID]
  merger_IDs[pkmassid] = merger_IDs[pkmassid][::-1]
#--------------------------------------------------------------------

# Step 3; plot everything together:
#--------------------------------------------------------------------
fs = 10
fig, ax = plt.subplots(figsize=(10, 5), nrows=3, ncols=1, gridspec_kw={'hspace':0.0, 'wspace':0.0, 'height_ratios':[0.25, 0.25, 0.50]})

# Time binning:
time_bins = np.linspace(0.3, 13.8, 500)
a_bins = auriga.time_age(time_bins)
end_time = auriga.age_time(1)
time_bins = end_time - time_bins

# Metal abundance binning:
metal_range = np.nanpercentile(data['MgFe'][data['IN  '] & \
                              (data['GAGE'] < a_bins[-1]) & \
                              (data['GAGE'] > a_bins[0])], [0.1,99.95])
metal_range += [-0.025, 0.025]
metal_bins = np.linspace(*metal_range, 125)

data['IN  '] *= data['RG  '] <= 15

hist = np.histogram2d(data['GAGE'][data['IN  ']], data['MgFe'][data['IN  ']], bins=[a_bins, metal_bins], weights=data['MASS'][data['IN  ']])
aspect = np.abs(np.diff([*time_bins[[-1,0]]])) / np.abs(np.diff([*metal_range])) / 4.
ax[2].imshow(hist[0].T, extent=[time_bins[0],time_bins[-1], *metal_range], norm=LogNorm(), aspect=aspect*0.99, origin='lower')
ax[2].set_ylabel('[Mg/Fe]', fontsize=fs)

# Star formation history:
bin_size = time_bins[0] - time_bins[1]
cuts = [data['IN  '], (data['RG  '] <= 5) * data['IN  ']]
hyphen = u"\u2010"
labels = [r'$In%ssitu$' % hyphen, r'$In%ssitu$, $R_{\rm G}<5\,$kpc' % hyphen]
colours = ['mediumblue', 'cornflowerblue']
for cut, label, colour in zip(cuts, labels, colours):
  SFR_hist = np.histogram(data['GAGE'][cut], bins=a_bins, weights=data['MASS'][cut])[0] / (bin_size*1e9)
  ax[1].fill_between(time_bins[1:], SFR_hist, label=label, color=colour, step='post')
ax[1].set_ylabel(r'SFR [M$_{\odot}\,$yr$^{-1}$]', fontsize=fs)
ax[1].set_yscale('log')
ax[1].set_ylim(ymin=1e-1)

ax[1].legend(fontsize=fs-4, loc='lower right')

# Trajectories:
cmap = cm.gnuplot
factor = 5
a = 1 / (1+redshifts)
times = np.array([auriga.age_time(i) for i in a])
r200s = f['Group_R_Crit200'][:][f['FirstHaloInFOFGroup'][:][IDs]] * 1e3/h
for pkmassid, pkmass in zip(pkmassids, pkmasses):
  if len(merger_IDs[pkmassid]) < 5:
    continue

  overlap1 = np.in1d(f['SnapNum'][:][IDs], f['SnapNum'][:][merger_IDs[pkmassid]])
  overlap2 = np.in1d(f['SnapNum'][:][merger_IDs[pkmassid]], f['SnapNum'][:][IDs])
  orig_res = f['SnapNum'][:][merger_IDs[pkmassid]][overlap2]
  spline_res = np.linspace(*orig_res[[0,-1]], len(orig_res)*factor)

  # Orbital radius:
  pos = (f['SubhaloPos'][:][merger_IDs[pkmassid]][overlap2] - f['SubhaloPos'][:][IDs][overlap1])*1e3/h
  vel = (f['SubhaloVel'][:][merger_IDs[pkmassid]][overlap2] - f['SubhaloVel'][:][IDs][overlap1])#*np.vstack(np.sqrt(1/a[orig_res]))
  pos = make_interp_spline(orig_res, pos, k=3)(spline_res)
  RG = np.linalg.norm(pos, axis=1)

  # Time:
  time = end_time - np.interp(spline_res, orig_res, times[orig_res])

  # Subhalo star formation rate:
  SFR_min = -4
  SFR_max = 1
  SFR = f['SubhaloSFR'][:][merger_IDs[pkmassid]][overlap2]
  SFR = np.log10(np.interp(spline_res, orig_res, SFR))
  SFR[np.isinf(SFR) | np.isnan(SFR)] = SFR_min
  line_colour = cmap(auriga.normalise(SFR, 0,1, SFR_min,SFR_max))

  # Subhalo total mass:
  sub_mass = f['SubhaloMassType'][:][merger_IDs[pkmassid]][overlap2].sum(axis=1) * 1e10/h
  host_mass = f['SubhaloMassType'][:][IDs][overlap1].sum(axis=1) * 1e10/h
  mass_ratio = make_interp_spline(orig_res, sub_mass/host_mass, k=3)(spline_res)
  linewidths = auriga.normalise(mass_ratio, 1,6, (1/20.),(1/4.))

  line_segments = LineCollection([np.column_stack([[time[i], time[i+1]], [RG[i], RG[i+1]]]) for i in range(len(time)-1)], \
                                 linewidths=linewidths, capstyle='round', color=line_colour, rasterized=False, joinstyle='round')
  ax[0].add_collection(line_segments)
  #ax[0].plot(time, RG)
  #ax[0].plot(time[-1], RG[-1], 'kx', markersize=4)

  # Find intercept at top of plot:
  intercept = abs(RG - r200s.max()*1.1).argmin()
  if any(i > 0 for i in RG - r200s.max()*1.1) & any(i < 0 for i in RG - r200s.max()*1.1):
    intercept = np.where((np.sign((RG - r200s.max()*1.1)[:-1])==1) & \
                         (np.sign((RG - r200s.max()*1.1)[1:])==-1))[0][0] + 1
  elif any(i < 0 for i in RG - r200s.max()*1.1):
    intercept = 0
  else:
    continue
  string = r'%s' % pkmassid
  ax[0].text(time[intercept], 1.03, string, va='bottom', ha='center', \
             transform=ax[0].get_xaxis_transform(), fontsize=fs-8, rotation=90)

# Add a colourbar:
norm = mpl.colors.Normalize(vmin=SFR_min, vmax=SFR_max)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
l, b, w, h = ax[0].get_position().bounds
cax = fig.add_axes([l + w*1.01, b, w*0.01, h])
cbar = plt.colorbar(sm, cax=cax)
string = r'$\log_{10}$ SFR' + '\n' + r'[M$_{\odot}\,\rm{yr}^{-1}$]'
cbar.set_label(string, fontsize=fs)
cbar.ax.tick_params(labelsize=fs-2)

ax[0].set_ylabel(r'$R_{\rm G}$ [ckpc]', fontsize=fs)

ax[0].fill_between(end_time - times[f['SnapNum'][:][IDs]], r200s, np.ones_like(r200s)*r200s.max()*1.1, color='lightgrey', zorder=0)
ax[0].set_ylim(0, r200s.max()*1.1)
#--------------------------------------------------------------------

for i in range(3):
  ax[i].tick_params(axis='both', labelsize=fs-2)
  ax[i].label_outer()
  ax[i].set_xlim([13.5, 0])
ax[2].set_xlabel('Lookback time [Gyr]', fontsize=fs)

fig.suptitle(r'Au-%i' % halo, fontsize=fs, y=0.95)

plt.savefig('./images/infalls/merger_infalls_Au-%i.pdf' % halo, bbox_inches='tight')
