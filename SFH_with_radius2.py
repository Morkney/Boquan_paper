import numpy as np
import sys

import h5py
from auriga_config import *
import auriga_functions as auriga
import snapHDF5 as snap
import readhaloHDF5
import readsubfHDF5
import misc_funcs as misc
from get_halo import get_data

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
mpl.use('Agg')
#plt.ion()

from scipy.interpolate import make_interp_spline

# Select the simulation:
#--------------------------------------------------------------------
snapshot = 251
subhalo = 0
halo = int(sys.argv[1])
z_cut = 5
r_cut = (0.3, 30)
basepath = lvl4rerun_basepath
#--------------------------------------------------------------------

# Time information:
#--------------------------------------------------------------------
tree_path = basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % snapshot
f = h5py.File(tree_path, "r")['Tree0']
ID = np.where((f['SnapNum'][:]==snapshot) & (f['SubhaloNumber'][:]==subhalo))
redshifts = np.array(h5py.File(tree_path, 'r')['Header']['Redshifts'][:])
a = 1/(1+redshifts)
time = np.array([auriga.age_time(i) for i in a])
lookback_time = auriga.age_time(1) - time
#--------------------------------------------------------------------

# Load halo data:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE', 'POT ', 'GMET', 'GSPH']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, SubhaloNumber=f['SubhaloNumber'][ID], \
                                 data_fields=data_fields, cat_fields=cat_fields, align_on_disc=True, basepath=basepath)

# Identify disc stars:
data = auriga.identify_disc(data)
data['DISC'] &= data['IN  ']

cut = data['DISC'] * (np.abs(data['POS '][:,2]) < z_cut)
#--------------------------------------------------------------------

# Property bins:
#--------------------------------------------------------------------
time_bin_width = 0.05 # [Gyrs]
a_start = 0.1
a_end = 1.0
time_bins = np.arange(auriga.age_time(a_start), auriga.age_time(a_end), time_bin_width)
a_bins = auriga.time_age(time_bins)
time_bins = auriga.age_time(1) - time_bins

N_bins = int((len(a_bins)/3.) * (6/10.))
r_bins = np.linspace(0, 30, N_bins)

binrange = [np.nanpercentile(data['FeH '][cut], [1, 99.9]), \
            np.nanpercentile(data['MgFe'][cut], [0.5, 99])]
binrange[0] += np.array([-0.1, 0.1])*np.diff(binrange[0])
binrange[1] += np.array([-0.1, 0.1])*np.diff(binrange[1])
FeH_bins = np.linspace(*binrange[0], N_bins)
MgFe_bins = np.linspace(*binrange[1], N_bins)
#--------------------------------------------------------------------

# Find all notable mergers:
#--------------------------------------------------------------------
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

# Find the progenitor line of each merger:
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

# Make a plot:
#--------------------------------------------------------------------
fs = 12
fig, ax = plt.subplots(figsize=(10,6), nrows=3, ncols=1, gridspec_kw={'hspace':0.0, 'wspace':0.0})

# Add r200 limit:
r200s = f['Group_R_Crit200'][:][IDs] * 1e3/h
times = auriga.age_time(1) - time[f['SnapNum'][:][IDs]]
times = times[r200s>0]
r200s = r200s[r200s>0]
ax[0].plot(times, r200s, color='silver', ls='--', lw=1, zorder=100)
ax[0].set_ylim(0, r200s.max()*1.1)
ax[0].set_aspect('auto')

# Merger paths:
factor = 10
cmap = cm.gnuplot
cmap.set_bad('grey') # Needs matplotlib version >3.2
exsitu_alphas = np.nanpercentile(data['MgFe'][~cut], [0.5, 99])
for pkmassid, pkmass in zip(pkmassids, pkmasses):
  if len(merger_IDs[pkmassid]) < 5:
    continue
  print('>    Plotting merger %i' % pkmassid)

  overlap1 = np.in1d(f['SnapNum'][:][IDs], f['SnapNum'][:][merger_IDs[pkmassid]])
  overlap2 = np.in1d(f['SnapNum'][:][merger_IDs[pkmassid]], f['SnapNum'][:][IDs])
  orig_res = f['SnapNum'][:][merger_IDs[pkmassid]][overlap2]
  spline_res = np.linspace(*orig_res[[0,-1]], len(orig_res)*factor)

  # Orbital radius:
  pos = (f['SubhaloPos'][:][merger_IDs[pkmassid]][overlap2] - f['SubhaloPos'][:][IDs][overlap1])*1e3/h
  #pos *= np.vstack(a[f['SnapNum'][:][IDs][overlap1]])
  pos = make_interp_spline(orig_res, pos, k=3)(spline_res)
  RG = np.linalg.norm(pos, axis=1)

  # Time:
  times = auriga.age_time(1) - np.interp(spline_res, orig_res, time[orig_res])

  # Subhalo total mass:
  sub_mass = f['SubhaloMassType'][:][merger_IDs[pkmassid]][overlap2].sum(axis=1) * 1e10/h
  host_mass = f['SubhaloMassType'][:][IDs][overlap1].sum(axis=1) * 1e10/h
  mass_ratio = make_interp_spline(orig_res, sub_mass/host_mass, k=3)(spline_res)
  linewidths = auriga.normalise(mass_ratio, 1,6, (1/20.),(1/4.))

  # Subhalo gas metals through time:
  cat_fields = ['SubhaloGasMetalFractions']
  MgFe = np.empty(np.sum(overlap2))
  for i, (snapshot, id) in enumerate(zip(f['SnapNum'][:][merger_IDs[pkmassid]][overlap2], np.array(merger_IDs[pkmassid])[overlap2])):
    cat = readsubfHDF5.subfind_catalog(basepath+'/halo_%i/output/' % halo, snapshot, long_ids=True, keysel=cat_fields)
    MgFe[i] = auriga.metal_ratio(np.array([cat.SubhaloGasMetalFractions[f['SubhaloNumber'][id]]]), 'Mg', 'Fe') + 0.4
  MgFe = np.interp(spline_res, orig_res, MgFe)
  line_colour = cmap(auriga.normalise(MgFe, 0,1, *exsitu_alphas))

  line_segments = LineCollection([np.column_stack([[times[i], times[i+1]], [RG[i], RG[i+1]]]) for i in range(len(times)-1)], \
                                 linewidths=linewidths, capstyle='round', color=line_colour, rasterized=False, joinstyle='round')
  ax[0].add_collection(line_segments)

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
  ax[0].text(times[intercept], 1.03, string, va='bottom', ha='center', \
             transform=ax[0].get_xaxis_transform(), fontsize=fs-8, rotation=90)

ax[0].set_xlim(*time_bins[[0,-1]])

# FeH histogram:
data['RG  '][data['RG  ']==0] = 1e-1
hist, proprange, densrange, stamp = misc.cmap_2D_hist(prop=np.log10(data['RG  '][cut]), x=data['GAGE'][cut], y=data['FeH '][cut], z=data['MASS'][cut], \
                                                      bins=[a_bins, FeH_bins], cmap='rainbow', proprange=[*np.log10(r_cut)], densrange=[10,99.9])
extent = [*time_bins[[0,-1]], *FeH_bins[[0,-1]]]
ax[1].imshow(hist, origin='lower', norm=LogNorm(), aspect='auto', extent=extent)

# MgFe histogram:
hist, proprange, densrange, stamp = misc.cmap_2D_hist(prop=np.log10(data['RG  '][cut]), x=data['GAGE'][cut], y=data['MgFe'][cut], z=data['MASS'][cut], \
                                                      bins=[a_bins,MgFe_bins], cmap='rainbow', proprange=[*np.log10(r_cut)], densrange=[10,99.9])
extent = [*time_bins[[0,-1]], *MgFe_bins[[0,-1]]]
ax[2].imshow(hist, origin='lower', norm=LogNorm(), aspect='auto', extent=extent)

ax[0].set_ylabel(r'Radius [ckpc]', fontsize=fs)
ax[1].set_ylabel(r'[Fe/H]', fontsize=fs)
ax[2].set_ylabel(r'[Mg/Fe]', fontsize=fs)
ax[2].set_xlabel(r'Lookback time [Gyr]', fontsize=fs)

for i in range(3):
  ax[i].label_outer()
  ax[i].tick_params(axis='both', labelsize=fs-2)
#--------------------------------------------------------------------

# Add a merger ratio legend:
#--------------------------------------------------------------------
string = r'$M(\rm{Merger})$:$M(\rm{Host})'
ax[0].plot([],[], c='k', lw=6, label=string + r' = 1:4$', solid_capstyle='butt')
ax[0].plot([],[], c='k', lw=2.25, label=string + r' = 1:10$', solid_capstyle='butt')
ax[0].plot([],[], c='k', lw=1, label=string + r' = 1:20$', solid_capstyle='butt')
ax[0].legend(fontsize=fs-6.5, loc='upper right', facecolor='w', framealpha=0.5, labelspacing=0.2)
#--------------------------------------------------------------------

# Add colourbars:
#--------------------------------------------------------------------
padding = 0.05 * (6/10.)
size = 0.05 * (6/10.)

norm = mpl.colors.Normalize(vmin=exsitu_alphas[0], vmax=exsitu_alphas[1])
sm = cm.ScalarMappable(norm=norm, cmap=cm.gnuplot)
sm.set_array([])
l, b, w, h = ax[0].get_position().bounds
cax = fig.add_axes([l+w+padding*w, b+padding/2., w*size, h-padding])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label(r'Gas [Mg/Fe]', fontsize=fs)
cbar.ax.tick_params(labelsize=fs-2)

l1, b1, w1, h1 = ax[1].get_position().bounds
l2, b2, w2, h2 = ax[2].get_position().bounds
cax = fig.add_axes([l1+w1+padding*w1, b2+padding/2., w1*size*3, (b1+h1) - b2 - padding])
cax.imshow(np.transpose(stamp.cmap, axes=[1,0,2]), origin='lower', aspect='auto', extent=[*densrange, *proprange])
cax.tick_params(axis='both', labelsize=fs-2)
cax.set_ylabel(r'$\log_{10}(R_{\rm G})$ [kpc]', fontsize=fs)
cax.set_xlabel(r'$\log_{10}(\rho)$', fontsize=fs)
cax.yaxis.set_label_position("right")
cax.yaxis.tick_right()
cax.xaxis.tick_bottom()
#--------------------------------------------------------------------

ax[2].text(0.05/3., 0.05, 'Au-%i' % halo, fontsize=fs-2, ha='left', va='bottom', transform=ax[2].transAxes)

plt.savefig('./images/SFH/SFH_histograms_Au-%i_rerun.pdf' % halo, bbox_inches='tight')
