import numpy as np
import itertools
import h5py
import sys
import gc
import os

from auriga_config import *
import auriga_functions as auriga
import snapHDF5 as snap
import readhaloHDF5
import readsubfHDF5
from get_halo import get_data
import misc_funcs as misc

import default_setup
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
#plt.ion()

from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.interpolate import make_interp_spline
# Ensure correct python version first:
if sys.version[:5] != '3.7.3':
  print('Change python dist.')
  sys.exit()
from sphviewer.tools import QuickView

import pickle

# Select halo:
#--------------------------------------------------------------------
halo = int(sys.argv[1])
snapshot = 127

with open('./files/merger_dict.pk1', 'rb') as file:
  props = pickle.load(file)
#--------------------------------------------------------------------

# Plotting preferences:
#--------------------------------------------------------------------
cwidth = 150 # [ckpc]
z_cut = 10

density_cmap = cm.cubehelix
metal_cmap = cm.viridis
alpha_cmap = cm.inferno
v_cmap = cm.magma
temp_cmap = cm.gist_heat
mach_cmap = cm.gist_stern
vr_cmap = cm.PuOr_r
sfr_cmap = cm.winter

clabels = [r'$\log_{10}$ Density [M$_{\odot}\,$kpc$^{-2}$]', \
           r'$\log_{10}$ SFR [M$_{\odot}\,$yr$^{-1}$]', \
           r'$\log_{10}$ Temperature [k]', \
           r'[Fe/H]', \
           r'[Mg/Fe]', \
           r'Mach number $\mathcal{M}$']

clabels = [r'$\log_{10}$ Density [M$_{\odot}\,$kpc$^{-2}$]', \
           r'[Mg/Fe]', \
           r'Velocity [$\rm{km}\,\rm{s}^{-1}$]', \
           r'$\log_{10}$ Temperature [k]', \
           r'$\log_{10}$ SFE [M$_{\odot}\,$yr$^{-1}\,$kpc$^{-2}$]', \
           r'[Fe/H]', \
           r'$v_{r}$ [$\rm{km}\,\rm{s}^{-1}$]', \
           r'Mach Number $M$']
paths = [path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]
#--------------------------------------------------------------------

tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % snapshot
f = h5py.File(tree_path, 'r')['Tree0']
redshifts = np.array(h5py.File(tree_path, 'r')['Header']['Redshifts'][:])
a = 1/(1+redshifts)

def remove_nan(U, sigma=1):
  V = U.copy()
  V[np.isnan(U)] = 0
  VV = gaussian_filter(V, sigma=sigma)
  W = 0*U.copy()+1
  W[np.isnan(U)] = 0
  WW = gaussian_filter(W, sigma=sigma)
  V[np.isnan(U)] = (VV/WW)[np.isnan(U)]
  return V

if not os.path.isdir('./images/gas_plumes/Au-%i_gas_plume' % halo):
  os.mkdir('./images/gas_plumes/Au-%i_gas_plume' % halo)

# Find all of the mergers that passed near the centre of the host:
#--------------------------------------------------------------------
merger_ids = np.array(list(props['Au-%i' % halo].keys()))
peris = np.array([props['Au-%i' % halo][merger_ID]['peri'] for merger_ID in merger_ids])
peri_ts = np.array([props['Au-%i' % halo][merger_ID]['t_peri'] for merger_ID in merger_ids])
v_ratios = np.array([props['Au-%i' % halo][merger_ID]['v_ratio'] for merger_ID in merger_ids])
merger_masses = f['SubhaloMassType'][:][merger_ids].sum(axis=1)
host_masses = f['SubhaloMassType'][:][f['FirstHaloInFOFGroup'][:][f['Descendant'][:][merger_ids]]].sum(axis=1)
mass_ratios = merger_masses / host_masses

# Mergers must have small pericentres, not too early, high-ratio, and radial:
merger_ids = merger_ids[(peris <= (50*auriga.time_age(peri_ts))) & \
                        ((auriga.age_time(1)-peri_ts) < 12.) & \
                        (mass_ratios > 1/40.) & \
                        (v_ratios > 0.)]
# Get host IDs:
host_ID = 0
host_IDs = []
while host_ID != -1:
  host_IDs.append(host_ID)
  host_ID = f['FirstProgenitor'][host_ID]
host_IDs = np.array(host_IDs)
host_map = {}
for host_ID in host_IDs:
  host_map[f['SnapNum'][host_ID]] = host_ID
#--------------------------------------------------------------------

snapshot_pad = 3
for merger_pkmassID in merger_ids:

  # Wind the merger back to entry:
  merger_ID = merger_pkmassID
  while f['FirstHaloInFOFGroup'][merger_ID] != merger_ID:
    merger_ID = f['FirstProgenitor'][merger_ID]
    if merger_ID==-1:
      break
  if merger_ID==-1:
      break
  host_ID = f['FirstProgenitor'][f['FirstHaloInFOFGroup'][f['Descendant'][merger_ID]]]

  # Find the position and velocity of the merger up until dissolution:
  #--------------------------------------------------------------------
  # Find merger IDs:
  merger_IDs = []
  while True:
    if f['SnapNum'][merger_ID] in list(host_map.keys()):
      merger_IDs.append(merger_ID)
      merger_ID = f['Descendant'][merger_ID]
    else:
      merger_ID = f['Descendant'][merger_ID]
      continue
    if f['FirstProgenitor'][f['Descendant'][merger_ID]] != merger_ID:
      break
  snapshots = f['SnapNum'][:][merger_IDs]
  host_IDs = [host_map[i] for i in f['SnapNum'][:][merger_IDs]]
  pos = (f['SubhaloPos'][:][merger_IDs] - f['SubhaloPos'][:][host_IDs]) * np.vstack(a[snapshots]) * 1e3/h
  vel = (f['SubhaloVel'][:][merger_IDs] - f['SubhaloVel'][:][host_IDs]) * np.vstack(np.sqrt(a[snapshots]))
  #--------------------------------------------------------------------

  # Make a spline of the merger trajectory:
  #--------------------------------------------------------------------
  factor = 20
  k = min(3, len(pos)-1)
  orig_res = f['SnapNum'][:][merger_IDs]
  spline_res = np.linspace(*orig_res[[0,-1]], len(orig_res)*factor)
  pos = make_interp_spline(orig_res, pos, k=k)(spline_res)
  if props['Au-%i' % halo][merger_pkmassID]['z_peri'] < redshifts[orig_res][-1]:
    step = np.diff([pos[-2], pos[-1]], axis=0)[0]
    extrapolate = np.linspace(step, step*200, 200)
    pos = np.append(pos, pos[-1]+extrapolate, axis=0)

  # Find the pericentre:
  diff = np.diff(np.linalg.norm(pos, axis=1))
  peri = np.where((diff[1:] > 0) & (diff[:-1] < 0))[0] + 1
  if not len(peri):
    peri = -1
  else:
    peri = peri[0]

  # Movement vector:
  vec = pos[peri] - pos[peri-1]

  # Alignment:
  merger_R = np.dot(misc.R_z(-np.pi/2.), misc.align_orbit(vec, vec))
  pos = np.dot(merger_R, [*pos.T]).T
  theta = misc.angle(pos[peri][[1,2]], [1., 0.])
  rotate_x = misc.R_x(-np.sign(pos[peri][2]) * theta)
  pos = np.dot(rotate_x, [*pos.T]).T
  #--------------------------------------------------------------------

  # Calculate the rotation matrix to align with the merger:
  #--------------------------------------------------------------------
  # Find snapshot closest to pericentre:
  peri_snap = np.abs(redshifts - props['Au-%i' % halo][merger_pkmassID]['z_peri']).argmin()

  # Find host ID N snapshots before pericentre:
  host_ID = host_map[peri_snap]
  for i in range(snapshot_pad):
    host_ID = f['FirstProgenitor'][host_ID]
  #--------------------------------------------------------------------

  # Create list of snapshots, where middle snapshot is first in the list:
  #--------------------------------------------------------------------
  host_IDs = []
  for i in range(1+snapshot_pad*2):
    host_IDs.append(host_ID)
    host_ID = f['Descendant'][host_ID]
  host_IDs = np.array(host_IDs)
  host_IDs[[0,snapshot_pad]] = host_IDs[[snapshot_pad,0]]
  #--------------------------------------------------------------------

  # Loop over each snapshot, starting from before pericentre passage and ending several snapshots afterwards:
  for i, host_ID in enumerate(host_IDs):
  #--------------------------------------------------------------------
    print('>    %i' % i)

    # Retrieve halo and stellar data:
    #------------------------------------------------------------------
    data_fields = ['POS ', 'VEL ', 'MASS', 'GMET', 'ID  ', 'RHO ', 'SFR ', 'EDIS', 'MACH', 'NE  ', 'U   ']
    cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
    header, cat, lf, data = get_data(halo, f['SnapNum'][host_ID], GrNr=f['SubhaloGrNr'][host_ID], SubhaloNumber=f['SubhaloNumber'][host_ID], \
                                     data_fields=data_fields, cat_fields=cat_fields, align_on_disc=False, single_sub=False, PtType=0)
    # Orient on incoming merger:
    data['POS '] = np.dot(merger_R, [*data['POS '].T]).T
    data['VEL '] = np.dot(merger_R, [*data['VEL '].T]).T

    # Calculate temperature in Kelvin:
    data['TEMP'] = auriga.temperature(data['U   '], data['NE  '])

    # Calculate vr:
    r, theta, phi, data['VR  '], v_th, v_p = auriga.cartesian_to_spherical(*data['POS '].T, *data['VEL '].T)
    #------------------------------------------------------------------

    # Gas density, gas velocity, [Fe/H], [Mg/Fe]:
    #------------------------------------------------------------------
    # Prepare a thin slice in z:
    width = cwidth * header.time
    extent = np.array([-width, width, -width, width])
    slice = (data['POS '][:,0] < width*1.1) & (data['POS '][:,1] < width*1.1) & (np.abs(data['POS '][:,2]) < z_cut)

    rho_img = QuickView(data['POS '][slice], mass=data['RHO '][slice], \
                        r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
    density_img = QuickView(data['POS '][slice], mass=data['MASS'][slice], \
                  r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T

    elements = ['Fe', 'H', 'Mg']
    metal_img = {}
    for j in elements:
      abundance = data['GMET'][:,metals[j]['ID']]
      metal_img[j] = QuickView(data['POS '][slice], mass=(abundance*data['MASS']*data['RHO '])[slice], \
                     r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / rho_img

    FeH_img = np.log10((metal_img['Fe']/metals['Fe']['mass']) / (metal_img['H']/metals['H']['mass'])) \
              - (metals['Fe']['solar'] - metals['H']['solar']) - 0.4
    MgFe_img = np.log10((metal_img['Mg']/metals['Mg']['mass']) / (metal_img['Fe']/metals['Fe']['mass'])) \
               - (metals['Mg']['solar'] - metals['Fe']['solar']) + 0.4
    FeH_img[np.isinf(FeH_img)] = np.nanmin(FeH_img[FeH_img != -np.inf])
    MgFe_img[np.isinf(MgFe_img)] = np.nanmin(MgFe_img[MgFe_img != -np.inf])

    velocity_img = QuickView(data['POS '][slice], mass=np.linalg.norm(data['VEL '][slice], axis=1) * data['MASS'][slice], \
                   r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / density_img

    vr_img = QuickView(data['POS '][slice], mass=data['VR  '][slice] * data['MASS'][slice], \
             r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / density_img

    SFR_img = QuickView(data['POS '][slice], mass=data['SFR '][slice] * data['RHO '][slice], \
                        r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False, nb=64).get_image().T / rho_img

    temp_img = QuickView(data['POS '][slice], mass=data['RHO '][slice] * data['TEMP'][slice], \
               r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / rho_img

    edis_img = QuickView(data['POS '][slice], mass=data['EDIS'][slice], \
               r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T
    mach_img = QuickView(data['POS '][slice], mass=data['EDIS'][slice] * data['TEMP'][slice], \
               r='infinity', x=0, y=0, z=0, extent=list(extent), plot=False, logscale=False).get_image().T / edis_img

    # Plug gaps in img arrays:
    FeH_img = remove_nan(FeH_img, sigma=1)
    MgFe_img = remove_nan(MgFe_img, sigma=1)
    temp_img = remove_nan(temp_img, sigma=1)
    velocity_img = remove_nan(velocity_img, sigma=1)
    vr_img = remove_nan(vr_img, sigma=1)
    #------------------------------------------------------------------

    # Prepare a plot with 4 panels:
    #------------------------------------------------------------------
    fs = 12
    fig, ax = plt.subplots(figsize=(12,6), nrows=2, ncols=4, gridspec_kw={'hspace':0.0, 'wspace':0.0})

    counts = np.histogram2d(data['POS '][slice,0], data['POS '][slice,1], range=[[*extent[[0,1]]], [*extent[[2,3]]]], bins=500)[0]
    filled = counts > 1

    img1 = ax[0,0].imshow(np.log10(density_img).T, origin='lower', extent=extent, cmap=density_cmap)
    img2 = ax[0,1].imshow(MgFe_img.T, origin='lower', extent=extent, cmap=alpha_cmap)
    img3 = ax[0,2].imshow(velocity_img.T, origin='lower', extent=extent, cmap=v_cmap)
    img4 = ax[0,3].imshow(np.log10(temp_img).T, origin='lower', extent=extent, cmap=temp_cmap)
    img5 = ax[1,0].imshow(np.log10(SFR_img).T, origin='lower', extent=extent, cmap=sfr_cmap)
    img6 = ax[1,1].imshow(FeH_img.T, origin='lower', extent=extent, cmap=metal_cmap)
    img7 = ax[1,2].imshow(vr_img.T, origin='lower', extent=extent, cmap=vr_cmap)
    img8 = ax[1,3].imshow(np.log10(mach_img).T, origin='lower', extent=extent, cmap=mach_cmap)

    # Set colourbar limits:
    if i==0:
      dens_clim = np.nanpercentile(np.log10(density_img)[filled][np.abs(np.log10(density_img)[filled])!=np.inf], [0.1,99.9])
      MgFe_clim = np.nanpercentile(MgFe_img[filled], [0.25,99.9])
      temp_clim = np.nanpercentile(np.log10(temp_img[filled]), [0.5,99.5])
      SFR_clim = [-6, np.nanmax(np.log10(SFR_img))-0.2]
      FeH_clim = [max(-3, np.nanpercentile(FeH_img[filled], 1)), np.nanpercentile(FeH_img[filled], 99.9)]
      vr_clim = np.array([-1,1]) * 400 # np.nanpercentile(vr_img[filled], 99)
      v_clim = np.nanpercentile(velocity_img[filled], [0.1,99])
      mach_clim = np.nanpercentile(np.log10(mach_img[filled]), [1,99])
    for img, clim in zip([img1,img2,img3,img4,img5,img6,img7,img8], \
                         [dens_clim,MgFe_clim,v_clim,temp_clim,SFR_clim,FeH_clim,vr_clim,mach_clim]):
      img.set_clim(*clim)

    # Remove margins and ticks:
    for j, axes in enumerate(np.ravel(fig.get_axes())):
      axes.set_facecolor('black')
      axes.set_xticks([])
      axes.set_yticks([])
      axes.set_xlim(extent[[0,1]])
      axes.set_ylim(extent[[2,3]])

    # Draw the spline on top:
    ax[0,0].plot(pos[:,0], pos[:,1], 'w-', lw=0.5)

    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #------------------------------------------------------------------

    # Only add the following to the first subplot:
    #------------------------------------------------------------------
    # Add a distance bar:
    corner1 = 0.1 * width - width
    corner2 = 0.9 * width
    ruler = int('%.0f' % float('%.1g' % (width/2.5)))

    for lw, color, order, capstyle in zip([3,1], ['k', 'w'], [100, 101], ['projecting', 'butt']):
      _, _, cap = ax[0,0].errorbar([corner1, corner1+ruler], np.ones(2)*corner2, yerr=np.ones(2)*0.025*width, \
                                    color=color, linewidth=lw, ecolor=color, elinewidth=lw, zorder=order)
      cap[0].set_capstyle(capstyle)

    # Distance bar labels:
    ax[0,0].text(corner1 + ruler/2., corner2 - 0.025*width, \
                 r'$%.0f\,$kpc' % ruler, va='top', ha='center', color='w', fontsize=fs-2, path_effects=paths)
    # Add delta:
    ax[0,0].text(0.95, 0.95, r'$\Delta Z=%s\,$kpc' % (z_cut*2), va='top', ha='right', color='w', \
                 fontsize=fs-2, path_effects=paths, transform=ax[0,0].transAxes)
    # Add time:
    string = r'$z=%.2f$' % header.redshift + '\n' +  r'$\tau=%.2f\,$Gyr' % (auriga.age_time(1)-auriga.age_time(1/(1+header.redshift)))
    ax[0,0].text(0.05, 0.05, string, va='bottom', ha='left', color='w', \
                 fontsize=fs-2, path_effects=paths, transform=ax[0,0].transAxes)
    # Add halo identifier:
    string = r'Au-%s' % halo + '\n' + r'%i' % merger_pkmassID
    ax[0,0].text(0.95, 0.05, string, va='bottom', ha='right', color='w', \
                 fontsize=fs-2, path_effects=paths, transform=ax[0,0].transAxes)
    #------------------------------------------------------------------

    # Colorbars:
    #------------------------------------------------------------------
    for j, (axes, img, clabel) in enumerate(zip(np.ravel(fig.get_axes()), [img1,img2,img3,img4,img5,img6,img7,img8], clabels)):
      pad = 0.025
      l, b, w, h = axes.get_position().bounds
      if j > 3.5:
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
    #------------------------------------------------------------------
    plt.savefig('./images/gas_plumes/Au-%i_gas_plume/frame_halo%i_snapshot%i.png' % (halo, merger_pkmassID, f['SnapNum'][host_ID]))
