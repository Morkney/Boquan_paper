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
plt.ion()

from scipy.ndimage import gaussian_filter

# (Use Au-24 or Au-18 for Boquan!)
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
data = auriga.identify_disc(data)
#--------------------------------------------------------------------

# Make a temporal cut:
#--------------------------------------------------------------------
t_extension = 1 # [Gyr]
#peri_t_lookbacks = {5:6.95, 9:10.1, 10:6.95, 10:6.2, 15:5.4, \
#                    17:10.9, 18:8.9, 22:10.7, 24:8.7, 27:9.4}
#burst_t_lookback = {1:9.8, 9:10.25, 12:8, 18:9.25, 23:11, 24:8.75, 26:9, 27:9.25, 29:11.25}
burst_t_lookback = {9:10.25, 12:8.1, 18:9.15, 23:10.25, 24:8.75, 26:9.25, 27:9.4, 29:11.75}
time_before = data['GAGE'] <= auriga.time_age([auriga.age_time(1) - burst_t_lookback[halo]])
#time_after = (data['GAGE'] > auriga.time_age([auriga.age_time(1) - burst_t_lookback[halo]]))
time_after = (data['GAGE'] > auriga.time_age([auriga.age_time(1) - burst_t_lookback[halo]])) & \
             (data['GAGE'] < auriga.time_age([auriga.age_time(1) - burst_t_lookback[halo]]) + 1)
#--------------------------------------------------------------------

fs = 10
fig, ax = plt.subplots(figsize=(6,6), nrows=2, ncols=2, gridspec_kw={'hspace':0.0, 'wspace':0.0})
N_bins = 100
FeH_range = (-2, 0)
FeH_bins = np.linspace(*FeH_range, N_bins)

####
# In order to find the merger-induced peak, I must search for the star formation in the 'red tail'.
# I.e., stars formed from the merger as it fell in.
# Are these truly in-situ, anyway?
# This can be revealed in even more detail by then also removing the disc...
# The disc overlaps in R and in [Fe/H] at its early low-metallicity end.
####

hyphen = u"\u2010"
r_ranges = [(0,1), (1,6), (6,10), (10,30)]
colours = ['k', 'mediumblue', 'royalblue', 'cornflowerblue']

for i in range(2):
  for j in range(2):
    print(i, j)

    if j==0:
      cut = data['IN  '].copy()
      x_label = '$In%ssitu$' % hyphen
    elif j==1:
      cut = data['EX  '].copy()
      x_label = '$Ex%ssitu$' % hyphen
    if i==0:
      cut &= time_before
      y_label = 'Before merger'
    elif i==1:
      cut &= time_after
      y_label = r'After merger + $1\,{\rm Gyr}$'
      #y_label = r'After merger'
    #cut &= ~data['DISC']


    hist = np.histogram(data['FeH '][cut], bins=FeH_bins, weights=data['MASS'][cut], density=True)[0]
    norm = hist/np.histogram(data['FeH '][cut], bins=FeH_bins, weights=data['MASS'][cut], density=False)[0]
    hist = np.append(hist, hist[-1])
    ax[i,j].fill_between(FeH_bins, hist, color='lightsteelblue', step='post', lw=0)

    for r_range, colour in zip(r_ranges, colours):
      r_cut = cut & (data['RG  '] >= r_range[0]) & (data['RG  '] < r_range[1])
      hist = np.histogram(data['FeH '][r_cut], bins=FeH_bins, weights=data['MASS'][r_cut], density=False)[0] * norm
      hist = np.append(hist, hist[-1])
      ax[i,j].step(FeH_bins, hist, where='post', color=colour, lw=1)

    ax[i,j].set_xlim(*FeH_range)
    ax[i,j].set_ylim(ymin=0)

    ax[i,j].tick_params(axis='both', labelsize=fs-2)
    ax[i,j].set_ylabel(r'$df/d{\rm [Fe/H]}$ [dex$^{-1}$]', fontsize=fs)
    ax[i,j].set_xlabel(r'[Fe/H]', fontsize=fs)
    ax[i,j].label_outer()

    if i==0:
      ax[i,j].text(0.5, 1.025, x_label, va='bottom', ha='center', transform=ax[i,j].transAxes, fontsize=fs)
    if j==1:
      ax[i,j].text(1.025, 0.5, y_label, va='center', ha='left', rotation=90, transform=ax[i,j].transAxes, fontsize=fs)

for r_range, colour in zip(r_ranges, colours):
  ax[0,0].plot([], color=colour, lw=1, label=r'$%i<R_{\rmG}/{\rm kpc}<%i$' % (r_range[0], r_range[1]))
ax[0,0].legend(loc='upper left', fontsize=fs-2, labelspacing=0.1, handlelength=1.5, frameon=False)

# Reset the axis limits:
max_y = min(2, np.max([i.get_ylim()[1] for i in np.ravel(fig.get_axes())]))
for axes in np.ravel(fig.get_axes()):
  axes.set_ylim(0, max_y)

ax[0,0].text(0.95, 0.95, r'Au-%i' % halo, va='top', ha='right', transform=ax[0,0].transAxes, fontsize=fs)

ax[1,0].set_yticks(ax[1,0].get_yticks()[:-1])
ax[1,0].set_xticks(ax[1,0].get_xticks()[:-1])

plt.savefig('./images/MDFsquareplot_Au-%i_nodisccut.pdf' % halo, bbox_inches='tight')  
#--------------------------------------------------------------------
