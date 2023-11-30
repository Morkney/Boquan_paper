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
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
plt.ion()

paths = [path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]

# Select the simulation:
#--------------------------------------------------------------------
snapshot = 127
halo = int(sys.argv[1])
#--------------------------------------------------------------------
density_cmap = cm.cubehelix
metal_cmap = cm.viridis
alpha_cmap = cm.plasma
v_cmap = cm.gnuplot
clabels = [r'$\log_{10}$ Density [M$_{\odot}\,$kpc$^{-2}$]', \
           r'$\log_{10}$ SFR [M$_{\odot}\,$yr$^{-1}\,$kpc$^{-2}$]', \
           r'[Fe/H]', \
           r'[Mg/Fe]']
# Turn SFR into SFE?

# Step 1; load the snapshot at z=0 and get the metal ratios:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, data_fields=data_fields, cat_fields=cat_fields)

# Find disc alignment:
r200 = cat.Group_R_Crit200[0] * 1e3/h
disc_r = 0.1 * r200 # [kpc]
data['POS '], data['VEL '], disc_R = auriga.orient_on_disc(data['POS '], data['VEL '], data['MASS'], disc_r, data['IN  '])

del data
print('Found alignment of stars...')

# Load gas:
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GMET', 'SFR ', 'RHO ']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, data_fields=data_fields, cat_fields=cat_fields, PtType=0)

# Align on stellar disc:
data['POS '] = np.dot(disc_R, [*data['POS '].T]).T
data['VEL '] = np.dot(disc_R, [*data['VEL '].T]).T
print('Loaded gas...')
#--------------------------------------------------------------------

# Create an attractive plot of the gas metallicity:
#--------------------------------------------------------------------
# SWITCH TO PYNBODY ONLY?
from sphviewer.tools import QuickView

slice = np.abs(data['POS '][:,2]) <= 0.5 # [kpc]

width = 100
extent = np.array([-1,1,-1,1]) * width/2.

elements = ['Fe', 'H', 'Mg']
metal_img = {}
for i in elements:
  abundance = data['GMET'][:,metals[i]['ID']]
  metal_img[i] = QuickView(data['POS '][slice], mass=(data['MASS']*abundance)[slice], \
                 r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T

FeH_img = np.log10((metal_img['Fe']/metals['Fe']['mass']) / (metal_img['H']/metals['H']['mass'])) \
          - (metals['Fe']['solar'] - metals['H']['solar']) - 0.4
MgFe_img = np.log10((metal_img['Mg']/metals['Mg']['mass']) / (metal_img['Fe']/metals['Fe']['mass'])) \
           - (metals['Mg']['solar'] - metals['Fe']['solar']) + 0.4
FeH_img[np.isinf(FeH_img)] = np.nanmin(FeH_img[FeH_img != -np.inf])
MgFe_img[np.isinf(MgFe_img)] = np.nanmin(MgFe_img[MgFe_img != -np.inf])

density_img = QuickView(data['POS '][slice], mass=data['MASS'][slice], \
                        r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T

#velocity_img = QuickView(data['POS '][slice], mass=np.linalg.norm(data['VEL '][slice], axis=1) * data['MASS'][slice], \
#                         r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
velocity_img = QuickView(data['POS '][slice], mass=data['SFR '][slice] * data['MASS'][slice], \
                         r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
#--------------------------------------------------------------------

# Plot the result:
#--------------------------------------------------------------------
fs = 10
fig, ax = plt.subplots(figsize=(6,6), nrows=2, ncols=2, gridspec_kw={'hspace':0.0, 'wspace':0.0})

# Add a map for SFR!

img1 = ax[0,0].imshow(np.log10(density_img).T, origin='lower', extent=extent, cmap=density_cmap)
img2 = ax[0,1].imshow(np.log10(velocity_img/density_img).T, origin='lower', extent=extent, cmap=v_cmap)
img3 = ax[1,0].imshow(FeH_img.T, origin='lower', extent=extent, cmap=metal_cmap)
img4 = ax[1,1].imshow(MgFe_img.T, origin='lower', extent=extent, cmap=alpha_cmap)

# Still need: colourbars, distance bars, etc
# Remove margins and ticks:
for i, axes in enumerate(np.ravel(fig.get_axes())):
  axes.set_facecolor('black')
  axes.set_xticks([])
  axes.set_yticks([])
  axes.set_xlim(extent[[0,1]])
  axes.set_ylim(extent[[2,3]])

ruler = int('%.0f' % float('%.1g' % (width/5)))
corner1 = width/2. - (0.1*width/2.) - ruler
corner2 = 0.9*width/2.
corner1 = 0.1*width/2. - width/2.
cap = 0.025 * width/2.

for lw, color, order, capstyle in zip([3,1], ['k', 'w'], [100, 101], ['projecting', 'butt']):
  _, _, caps = ax[0,0].errorbar([corner1, corner1+ruler], np.ones(2)*corner2, yerr=np.ones(2)*cap, \
                                color=color, linewidth=lw, ecolor=color, elinewidth=lw, zorder=order)
  caps[0].set_capstyle(capstyle)

# Distance bar labels:
ax[0,0].text(corner1 + ruler/2., corner2 - 0.025*width/2.,  r'$%.0f\,$kpc' % ruler, \
             va='top', ha='center', color='w', fontsize=fs-2, path_effects=paths)
#--------------------------------------------------------------------

# Colorbars:
#--------------------------------------------------------------------
for i, (axes, img, clabel) in enumerate(zip(np.ravel(fig.get_axes()), [img1, img2, img3, img4], clabels)):
  pad = 0.025
  l, b, w, h = axes.get_position().bounds
  if i > 1:
    position = 'bottom'
    cax = fig.add_axes([l+pad, b - h*0.05, w-2*pad, h*0.05])
  else:
    position = 'top'
    cax = fig.add_axes([l+pad, b+h, w-2*pad, h*0.05])
  cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
  cbar.set_label(clabel, fontsize=fs)
  cbar.ax.tick_params(labelsize=fs-2)
  cax.xaxis.set_label_position(position)
  cax.xaxis.set_ticks_position(position)
#--------------------------------------------------------------------

# Find a 3D power spectrum equivalent:
#--------------------------------------------------------------------
import pynbody

# Make a pynbody version of my data:
s = pynbody.new(gas=len(data['ID  ']))
s.g['pos'] = data['POS ']
s.g['FeH'] = data['FeH ']
s.g['MgFe'] = data['MgFe']
s.g['mass'] = data['MASS']
s.g['rho'] = data['RHO ']
s.g['Fe'] = s.g['mass'] * data['GMET'][:,metals['Fe']['ID']]
s.g['Mg'] = s.g['mass'] * data['GMET'][:,metals['Mg']['ID']]
s.g['H'] = s.g['mass'] * data['GMET'][:,metals['H']['ID']]

# Make a 3D datacube for the metals:
fs = 12
fig, ax = plt.subplots(figsize=(6,6))

width = 100
N_bins = 500

for qty, label in zip(['rho', 'FeH', 'MgFe'], ['Density', '[Fe/H]', '[Mg/Fe]']):
  print('>    %s' % qty)

  if qty is 'rho':
    data_cube = pynbody.sph.to_3d_grid(s, qty=qty, nx=N_bins, x2=width/2.)
    data_cube = np.log10(data_cube)
  else:
    data_cube1 = pynbody.sph.to_3d_grid(s, qty=qty[:2], nx=N_bins, x2=width/2.)
    data_cube2 = pynbody.sph.to_3d_grid(s, qty=qty[2:], nx=N_bins, x2=width/2.)
    data_cube = np.log10((data_cube1/metals[qty[:2]]['mass']) / (data_cube2/metals[qty[2:]]['mass'])) \
                - (metals[qty[:2]]['solar'] - metals[qty[2:]]['solar'])
    data_cube -= 0.4 if qty == 'FeH' else -0.4
  data_cube = np.nan_to_num(data_cube)
  data_cube = auriga.NormaliseData(data_cube)

  # Perform fourier analys1is as in https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/:
  fourier_img = np.fft.fftn(data_cube)
  fourier_amps = np.abs(fourier_img)**2

  # Get wavenumber:
  k_freq = np.fft.fftfreq(len(data_cube)) * len(data_cube)
  k_freq_3D = np.meshgrid(k_freq, k_freq, k_freq)

  k_norm = np.sqrt(k_freq_3D[0]**2 + k_freq_3D[1]**2 + k_freq_3D[2]**2)
  k_norm = k_norm.flatten()
  fourier_amps = fourier_amps.flatten()

  # Bin amplitudes in k-space:
  k_bins = np.arange(0.5, len(data_cube)//2+1, 1.)

  k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])

  import scipy.stats as stats
  A_bins, _, _ = stats.binned_statistic(k_norm, fourier_amps, statistic='mean', bins=k_bins)
  A_bins *= 4./3. * np.pi * (k_bins[1:]**3 - k_bins[:-1]**3)

  ax.loglog(width/k_vals, A_bins, label=label)

ax.set_xlabel(r'Wavelength [kpc]', fontsize=fs)
ax.set_ylabel(r'$P(k)$', fontsize=fs)
ax.tick_params(axis='both', labelsize=fs-2)
ax.legend(fontsize=fs-2)
#--------------------------------------------------------------------

