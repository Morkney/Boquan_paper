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
#mpl.use('Agg')
plt.ion()

from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

def remove_nan(U, sigma=1):
  V = U.copy()
  V[np.isnan(U)] = 0
  VV = gaussian_filter(V, sigma=sigma)
  W = 0*U.copy()+1
  W[np.isnan(U)] = 0
  WW = gaussian_filter(W, sigma=sigma)
  V[np.isnan(U)] = (VV/WW)[np.isnan(U)]
  return V

paths = [path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]

# Select the simulation:
#--------------------------------------------------------------------
halo = int(sys.argv[1])
snapshot = int(sys.argv[2])

# Find the halo information:
tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % 127
f = h5py.File(tree_path, "r")['Tree0']
ID = 0
while f['SnapNum'][ID] != snapshot:
  ID = f['FirstProgenitor'][ID]
  if ID == -1:
    sys.exit()
GrNr = f['SubhaloGrNr'][ID]
SubhaloNumber = f['SubhaloNumber'][ID]
#--------------------------------------------------------------------
density_cmap = cm.cubehelix
metal_cmap = cm.viridis
alpha_cmap = cm.plasma
temp_cmap = cm.gist_heat
clabels = [r'$\log_{10}$ Density [M$_{\odot}\,$kpc$^{-2}$]', \
           r'Temperature [K]', \
           r'[Fe/H]', \
           r'[Mg/Fe]']

# Step 1; load the snapshot at z=0 and get the metal ratios:
#--------------------------------------------------------------------
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, GrNr=GrNr, SubhaloNumber=SubhaloNumber, data_fields=data_fields, cat_fields=cat_fields)

# Find disc alignment:
r200 = cat.Group_R_Crit200[0] * header.time * 1e3/h
disc_r = 0.1 * r200 # [kpc]
data['POS '], data['VEL '], disc_R = auriga.orient_on_disc(data['POS '], data['VEL '], data['MASS'], disc_r, data['IN  '])

del data
print('Found alignment of stars...')

# Load gas:
data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GMET', 'SFR ', 'RHO ', 'NE  ', 'U   ']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, GrNr=GrNr, SubhaloNumber=SubhaloNumber, data_fields=data_fields, cat_fields=cat_fields, PtType=0, single_sub=False)

# Align on stellar disc:
data['POS '] = np.dot(disc_R, [*data['POS '].T]).T
data['VEL '] = np.dot(disc_R, [*data['VEL '].T]).T
print('Loaded gas...')

# Calculate temperature in Kelvin:
data['TEMP'] = auriga.temperature(data['U   '], data['NE  '])
#--------------------------------------------------------------------

# Create an attractive plot of the gas metallicity:
#--------------------------------------------------------------------
from sphviewer.tools import QuickView

delta = 100 # [kpc]
slice = np.abs(data['POS '][:,2]) <= delta/2. # [kpc]

width = 100
extent = np.array([-1,1,-1,1]) * width/2.
rho_img = QuickView(data['POS '][slice], mass=data['RHO '][slice], \
                    r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T

elements = ['Fe', 'H', 'Mg']
metal_img = {}
for i in elements:
  abundance = data['GMET'][:,metals[i]['ID']]
  metal_img[i] = QuickView(data['POS '][slice], mass=(data['MASS']*abundance*data['RHO '])[slice], \
                 r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / rho_img

FeH_img = np.log10((metal_img['Fe']/metals['Fe']['mass']) / (metal_img['H']/metals['H']['mass'])) \
          - (metals['Fe']['solar'] - metals['H']['solar']) - 0.4
MgFe_img = np.log10((metal_img['Mg']/metals['Mg']['mass']) / (metal_img['Fe']/metals['Fe']['mass'])) \
           - (metals['Mg']['solar'] - metals['Fe']['solar']) + 0.4
FeH_img[np.isinf(FeH_img)] = np.nanmin(FeH_img[FeH_img != -np.inf])
MgFe_img[np.isinf(MgFe_img)] = np.nanmin(MgFe_img[MgFe_img != -np.inf])

density_img = QuickView(data['POS '][slice], mass=data['MASS'][slice], \
                        r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T

temp_img = QuickView(data['POS '][slice], mass=data['RHO '][slice] * data['TEMP'][slice], \
           r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / rho_img

# Plug gaps in metal arrays:
FeH_img = remove_nan(FeH_img, sigma=1)
MgFe_img = remove_nan(MgFe_img, sigma=1)
temp_img = remove_nan(temp_img, sigma=1)
#--------------------------------------------------------------------

# Plot the result:
#--------------------------------------------------------------------
fs = 10
fig, ax = plt.subplots(figsize=(10,5), nrows=2, ncols=4, gridspec_kw={'hspace':0.0, 'wspace':0.0})
gs = ax[1,3].get_gridspec()
# Remove right-hand axes:
for axes in np.ravel(ax[:,2:]):
  axes.remove()
ax2 = fig.add_subplot(gs[:,2:])
# Shift to allow space for a y-axis title:
box = ax2.get_position()
box.x0 += 0.075
box.x1 += 0.075
ax2.set_position(box)

# Add a map for SFR!
counts = np.histogram2d(data['POS '][:,0], data['POS '][:,1], range=[[*extent[[0,1]]], [*extent[[2,3]]]], bins=500)[0]
filled = counts > 0

density_img = np.log10(density_img)
img1 = ax[0,0].imshow(density_img.T, origin='lower', extent=extent, cmap=density_cmap, vmin=2.5)
img1.set_clim(*np.nanpercentile(density_img[filled], [0.1,99.9]))

img2 = ax[0,1].imshow(np.log10(temp_img).T, origin='lower', extent=extent, cmap=temp_cmap)
img2.set_clim(*np.nanpercentile(np.log10(temp_img[filled]), [0.5,99.5]))

img3 = ax[1,0].imshow(FeH_img.T, origin='lower', extent=extent, cmap=metal_cmap)
img3.set_clim(vmin=max(-3, np.nanpercentile(FeH_img[filled], 1)), vmax=np.nanmax(FeH_img[filled]))

img4 = ax[1,1].imshow(MgFe_img.T, origin='lower', extent=extent, cmap=alpha_cmap)
img4.set_clim(*np.nanpercentile(MgFe_img[filled], [0.1,99.9]))

# Still need: colourbars, distance bars, etc
# Remove margins and ticks:
for i, axes in enumerate(np.ravel(ax[:,:2])):
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
             va='top', ha='center', color='w', fontsize=fs-4, path_effects=paths)

# Add delta:
ax[0,0].text(0.95, 0.95, r'$\Delta Z=%s\,$kpc' % delta, va='top', ha='right', color='w', \
             fontsize=fs-4, path_effects=paths, transform=ax[0,0].transAxes)

# Add time:
string = r'$z=%.2f$' % header.redshift + '\n' +  r'$\tau=%.2f\,$Gyr' % (auriga.age_time(1)-auriga.age_time(1/(1+header.redshift)))
ax[0,0].text(0.05, 0.05, string, va='bottom', ha='left', color='w', \
             fontsize=fs-4, path_effects=paths, transform=ax[0,0].transAxes)

# Add halo identifier:
ax[0,0].text(0.95, 0.05, r'Au-%s' % halo, va='bottom', ha='right', color='w', \
             fontsize=fs-4, path_effects=paths, transform=ax[0,0].transAxes)
#--------------------------------------------------------------------

# Colorbars:
#--------------------------------------------------------------------
for i, (axes, img, clabel) in enumerate(zip(np.ravel(ax[:,:2]), [img1, img2, img3, img4], clabels)):
  pad = 0.025/2.
  l, b, w, h = axes.get_position().bounds
  if i > 1.5:
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
import scipy.stats as stats

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
s.g['temp'] = data['TEMP']

# Find the maximum width:
min_box_width = np.abs(np.percentile(data['POS '], [1,99], axis=0)).min()
width = min(min_box_width, 30)
N_bins = 250
power = {}

labels = ['Density', '[Fe/H]', '[Mg/Fe]', 'Temperature']
colours = ['k', '#0aa185', '#e320d6', '#c71e00']
for qty, colour, label in zip(['rho', 'FeH', 'MgFe', 'temp'], colours, labels):
  print('>    %s' % qty)

  # Is it possible to weight these by density? Try it!
  # Possible improvements to representing the power spectra, comparing between different power spectra?

  if (qty is 'rho') or (qty is 'temp'):
    data_cube = pynbody.sph.to_3d_grid(s, qty=qty, nx=N_bins, x2=width/2.)
  elif (qty is 'FeH') or (qty is 'MgFe'):
    data_cube1 = pynbody.sph.to_3d_grid(s, qty=qty[:2], nx=N_bins, x2=width/2.)
    data_cube2 = pynbody.sph.to_3d_grid(s, qty=qty[2:], nx=N_bins, x2=width/2.)
    data_cube = np.log10((data_cube1/metals[qty[:2]]['mass']) / (data_cube2/metals[qty[2:]]['mass'])) \
                - (metals[qty[:2]]['solar'] - metals[qty[2:]]['solar'])
    data_cube -= 0.4 if qty == 'FeH' else -0.4
  data_cube = np.nan_to_num(data_cube)

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

  A_bins, _, _ = stats.binned_statistic(k_norm, fourier_amps, statistic='mean', bins=k_bins)
  A_bins *= 4./3. * np.pi * (k_bins[1:]**3 - k_bins[:-1]**3)
  power[qty] = A_bins

  normed_power = power[qty] / np.trapz(power[qty], dx=0.1)
  ax2.loglog(width/k_vals, normed_power, label=label, color=colour)

ax2.set_xlabel(r'Wavelength [kpc]', fontsize=fs)
ax2.set_ylabel(r'Normalised $P(k)$', fontsize=fs)
ax2.tick_params(axis='both', labelsize=fs-2)
ax2.legend(fontsize=fs)
#--------------------------------------------------------------------

plt.savefig('./images/gas_images/frame_%i.png' % snapshot, bbox_inches='tight', dpi=300)

# Make an animation:
