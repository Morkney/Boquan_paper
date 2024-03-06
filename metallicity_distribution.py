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
peri_t_lookbacks = {5:6.95, 9:10.1, 10:6.95, 10:6.2, 15:5.4, \
                    17:10.9, 18:8.9, 22:10.7, 24:8.7, 27:9.4}
peri_GAGE = auriga.time_age([auriga.age_time(1) - peri_t_lookbacks[halo] + t_extension])
time_cut = data['GAGE'] <= peri_GAGE
#--------------------------------------------------------------------

# Find solar region cuts:
#--------------------------------------------------------------------
solar_r = 8 # [kpc]
solar_r_cut1 = (data['RG  '] <= (solar_r+1)) & (data['RG  '] >= (solar_r-1))
solar_r_cut2 = (data['RG  '] <= (solar_r+2)) & (data['RG  '] >= (solar_r-2))
solar_torus_cut1 = auriga.solar_torus(*data['POS '].T, a=1)
solar_torus_cut2 = auriga.solar_torus(*data['POS '].T, a=2)
inner_r_cut = data['RG  '] <= 6
#--------------------------------------------------------------------

# Make a 1D histogram of iron metals:
#--------------------------------------------------------------------
cuts = [time_cut, time_cut & solar_r_cut2, time_cut & solar_r_cut2 & data['IN  '], \
        time_cut & ~data['DISC'], time_cut & data['IN  '] & ~data['DISC']]
hyphen = u"\u2010"
clbs = [r'$t < t_{\rm GSE}+1\,$Gyr', \
        r'$%i<R_{\rm G}/{\rm kpc}<%i$' % (8-2, 8+2), \
         '$In%ssitu$' % hyphen, r'Stellar halo', \
        r'$R_{\rm G} < 6\,$kpc']
labels = [clbs[0], '\n'.join([clbs[0], clbs[1]]), '\n'.join([clbs[0], clbs[1], clbs[2]]), \
          '\n'.join([clbs[0], clbs[3]]), '\n'.join([clbs[0], clbs[3], clbs[2]])]

cuts = [time_cut & inner_r_cut, inner_r_cut, time_cut & inner_r_cut & data['IN  '], \
        time_cut & inner_r_cut & data['IN  '] & ~data['DISC']]
labels = ['\n'.join([clbs[0], clbs[4]]), clbs[4], '\n'.join([clbs[0], clbs[1], clbs[3]]), \
          '\n'.join([clbs[0], clbs[4], clbs[2], clbs[3]])]

for i, (cut, label) in enumerate(zip(cuts, labels)):

  #cut = data['IN  '] & solar_r_cut2 & time_cut

  fs = 12
  fig, ax = plt.subplots(figsize=(6,4))

  N_bins = 100
  FeH_range = (-2, 0)
  FeH_bins = np.linspace(*FeH_range, N_bins)
  hist = np.histogram(data['FeH '][cut], bins=FeH_bins, weights=data['MASS'][cut], density=True)[0]
  hist = np.append(hist, hist[-1])
  ax.fill_between(FeH_bins, hist, color='cornflowerblue', step='post')

  ax.set_xlim(*FeH_range)
  ax.set_ylim(ymin=0)

  ax.tick_params(axis='both', labelsize=fs-2)
  ax.set_ylabel(r'$df/d{\rm [Fe/H]}$ [dex$^{-1}$]', fontsize=fs)
  ax.set_xlabel(r'[Fe/H]', fontsize=fs)

  ax.text(1-0.1/3., 0.95, 'Au-%i' % halo, va='top', ha='right', transform=ax.transAxes, fontsize=fs-2)
  ax.text(0.1/3., 0.95, label, va='top', ha='left', transform=ax.transAxes, fontsize=fs-2)

  plt.savefig('./images/MDF_Au-%i_%i.pdf' % (halo, i), bbox_inches='tight')  
#--------------------------------------------------------------------
