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
mpl.use('Agg')
#plt.ion()

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

# Identify different groups:
#--------------------------------------------------------------------
# data['IN  ']
# data['EX  ']
data['DISC'] = auriga.identify_disc2(data['POS '], data['VEL '], data['POT '], data['FeH '])

num_sats = 4
tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % snapshot
f = h5py.File(tree_path, 'r')['Tree0']
accreted_subhaloes = (lf.mdata['Exsitu']['PeakMassIndex'] > 0) & (lf.mdata['Exsitu']['AccretedFlag'] == 0)
sat_pkmassids, sat_Nstars = np.unique(lf.mdata['Exsitu']['PeakMassIndex'][accreted_subhaloes], return_counts=True)
contribution_order = np.argsort(sat_Nstars)[::-1][:num_sats]
sat_pkmassids = sat_pkmassids[contribution_order]
sat_masses = f['SubhaloMassType'][:][sat_pkmassids,4] * 1e10/h
for i, sat_pkmassid in enumerate(sat_pkmassids, 1):
  print('>    %i' % i)
  sat_IDs = lf.mdata['Exsitu']['ParticleIDs'][lf.mdata['Exsitu']['PeakMassIndex'] == sat_pkmassid]
  data['M%i  ' % i] = np.in1d(data['ID  '], sat_IDs)
#--------------------------------------------------------------------

# Make a histogram of the chemical plane:
#--------------------------------------------------------------------
fs = 12
fig, ax = plt.subplots(figsize=(6,6))

N_bins = 250
binrange = [np.nanpercentile(data['FeH '], [0.1, 99.9]), \
            np.nanpercentile(data['MgFe'], [0.1, 99.9])]
binrange[0] += np.array([-0.1, 0.1])*np.diff(binrange[0])
binrange[1] += np.array([-0.1, 0.1])*np.diff(binrange[1])
FeH_bins = np.linspace(*binrange[0], N_bins)
MgFe_bins = np.linspace(*binrange[1], N_bins)
extent = np.ravel([FeH_bins[[0,-1]], MgFe_bins[[0,-1]]])
#cut = (data['RG  '] < 30) * data['DISC']
cut = np.ones_like(data['ID  ']).astype('bool')
hist = np.histogram2d(data['FeH '][cut], data['MgFe'][cut], weights=data['MASS'][cut], bins=[FeH_bins, MgFe_bins])[0]
img = ax.imshow(hist.T, origin='lower', extent=extent, norm=LogNorm(), aspect=np.diff(FeH_bins[[0,-1]])/np.diff(MgFe_bins[[0,-1]]), cmap=cm.Greys, rasterized=True)
img.set_clim(vmax=img.get_clim()[1]*3)

ax.set_xlabel('[Fe/H]', fontsize=fs)
ax.set_ylabel('[Mg/Fe]', fontsize=fs)
ax.tick_params(axis='both', labelsize=fs-2)
#--------------------------------------------------------------------

# Add contours to show distinct populations:
#--------------------------------------------------------------------
N_bins = 100
FeH_bins = np.linspace(*binrange[0], N_bins)
MgFe_bins = np.linspace(*binrange[1], N_bins)

fraction = 0.9
lw = 1.5
cuts = ['IN  ', 'DISC'] + ['M%i  ' % i for i in range(1,num_sats+1)]
colours = ['k', 'r', 'royalblue', 'forestgreen', 'darkorange', 'blueviolet']
hyphen = u"\u2010"
labels = ['$In%ssitu$' % hyphen, 'Disc'] + \
         [r'M%i: %i, $%s\,$M$_{\odot}$' % (i, j, auriga.latex_float(k)) for i,(j,k) in enumerate(zip(sat_pkmassids, sat_masses), 1)]
for cut, colour, label in zip(cuts, colours, labels):

  print('>    %s' % cut)

  hist,xbins,ybins = np.histogram2d(data['FeH '][data[cut]], data['MgFe'][data[cut]], \
                                    weights=data['MASS'][data[cut]], bins=[FeH_bins, MgFe_bins], density=True)
  hist = gaussian_filter(hist**(1/10), sigma=1.5)**(10) # Smoothing
  hist_sort = np.argsort(np.ravel(hist))

  contour = np.ravel(hist)[hist_sort][np.argmin(np.abs(np.cumsum(np.ravel(hist)[hist_sort]) - np.sum(hist)*(1-fraction)))]
  CS = ax.contour(hist.T, extent=np.ravel(binrange), levels=[contour], colors=['w'], zorder=50, linewidths=[lw+1])
  CS = ax.contour(hist.T, extent=np.ravel(binrange), levels=[contour], colors=[colour], zorder=100, linewidths=[lw])
  ax.plot([],[], color=colour, lw=lw, label=label)

  ax.scatter(np.nanmedian(data['FeH '][data[cut]]), np.nanmedian(data['MgFe'][data[cut]]), s=100, \
             marker='P', color=colour, edgecolors='w', linewidths=1, zorder=101)

ax.legend(frameon=True, fontsize=fs-4, loc='upper right')
#--------------------------------------------------------------------

fig.suptitle(r'Au-%i' % halo, fontsize=fs, y=0.925)
plt.savefig('./images/metal_planes/metal_plane_Au-%i.pdf' % halo, bbox_inches='tight')
