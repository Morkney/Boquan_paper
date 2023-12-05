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
plt.ion()

# Select the simulation:
#--------------------------------------------------------------------
snapshot = 127
halo = int(sys.argv[1])
#--------------------------------------------------------------------

# Look at the accretions:
#--------------------------------------------------------------------
lf = auriga.listfile(lvl4_basepath, halo)

# Collect all mergers that do not have significant bound remnants:
merger_pkmassids = []
sat_pkmassids = np.unique(lf.mdata['Exsitu']['PeakMassIndex'][lf.mdata['Exsitu']['PeakMassIndex'] >= 0])
for sat_pkmassid in sat_pkmassids:
  sat_cut = lf.mdata['Exsitu']['PeakMassIndex'] == sat_pkmassid
  bound_remnant = lf.mdata['Exsitu']['AccretedFlag'][sat_cut]
  if (bound_remnant.sum()/len(bound_remnant) < 0.05) & (sat_cut.sum() > 50):
    merger_pkmassids.append(sat_pkmassid)

# Trace back each merger to its pre-infall ID:
tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % snapshot
f = h5py.File(tree_path, 'r')['Tree0']
redshifts = np.array(h5py.File(tree_path, 'r')['Header']['Redshifts'][:])
merger_ids = []
for merger_pkmassid in merger_pkmassids:
  if f['FirstProgenitor'][f['Descendant'][merger_pkmassid]] != merger_pkmassid:
    merger_pkmassid = f['FirstProgenitor'][merger_pkmassid]
  # Ignore IDs that have no history:
  if merger_pkmassid != -1:
    merger_ids.append(merger_pkmassid)

# Trace the merger to its penultimate ID:
merger_ids = []
for merger_pkmassid in merger_pkmassids:
  while f['FirstProgenitor'][f['Descendant'][merger_pkmassid]] == merger_pkmassid:
    merger_pkmassid = f['Descendant'][merger_pkmassid]
  merger_ids.append(merger_pkmassid)

# Now reverse and append the entire ID evolution:
IDs = {}
for merger_id, merger_pkmassid in zip(merger_ids, merger_pkmassids):
  IDs[merger_pkmassid] = []
  while merger_id != -1:
    IDs[merger_pkmassid].append(merger_id)
    merger_id = f['FirstProgenitor'][merger_id]
  IDs[merger_pkmassid] = np.array(IDs[merger_pkmassid][::-1])

# Also find the IDs for the main progenitor:
main_IDs = []
main_ID = 0
while main_ID != -1:
  main_IDs.append(main_ID)
  main_ID = f['FirstProgenitor'][main_ID]
main_IDs = np.array(main_IDs[::-1])

# Create a dictionary that maps IDs to snapshots:
main_map = {}
for snap, ID in zip(f['SnapNum'][:][main_IDs], main_IDs):
  main_map[snap] = ID
#--------------------------------------------------------------------

# Find associated merger properties:
#--------------------------------------------------------------------
string = 'ID\tz_inf\tMstar\t\tMgas\t\tfgas\tPeri\tv_ratio\t[Fe/H]\tD[Fe/H]\t[Mg/Fe]\tD[Mg/Fe]\n'
with open('./files/merger_properties_Au-%i.txt' % halo, 'w') as file:
  file.write(string)

from scipy.interpolate import BSpline, make_interp_spline
k = 3
factor = 20

for merger_pkmassid in merger_pkmassids:
  print('>    %i' % merger_pkmassid)

  if len(IDs[merger_pkmassid]) < 10:
    print('Broken history.')
    continue

  if not np.sum(f['SubhaloLenType'][:,0][IDs[merger_pkmassid]]):
    print('No gas.')
    continue

  # Get the host halo IDs at the same snapshots as the merger IDs:
  main_IDs_cropped = []
  for i, ID in enumerate(IDs[merger_pkmassid]):
    if f['SnapNum'][ID] in list(main_map.keys()):
      main_IDs_cropped.append(main_map[f['SnapNum'][ID]])
    else:
      IDs[merger_pkmassid] = IDs[merger_pkmassid][IDs[merger_pkmassid] != ID]
  k = min(3, len(main_IDs_cropped))

  orig_res = np.linspace(*f['SnapNum'][:][IDs[merger_pkmassid][[0,-1]]], len(IDs[merger_pkmassid]))
  spline_res = np.linspace(orig_res[0], orig_res[-1], (len(IDs[merger_pkmassid])) * factor)

  merger_pos = f['SubhaloPos'][:][IDs[merger_pkmassid]]
  main_pos = f['SubhaloPos'][:][main_IDs_cropped]
  pos = make_interp_spline(orig_res, (merger_pos-main_pos)*1e3/h, k=k)(spline_res)
  r = np.linalg.norm(pos, axis=1)

  main_r200 = f['Group_R_Crit200'][:][[main_map[i] for i in f['SnapNum'][:][IDs[merger_pkmassid]]]]
  r200 = make_interp_spline(orig_res, main_r200*1e3/h, k=k)(spline_res)

  merger_vel = f['SubhaloVel'][:][IDs[merger_pkmassid]]
  main_vel = f['SubhaloVel'][:][main_IDs_cropped]
  vel = make_interp_spline(orig_res, (merger_vel-main_vel)*1e3/h, k=k)(spline_res)

  r, theta, phi, v_r, v_th, v_p = auriga.cartesian_to_spherical(*pos.T, *vel.T)
  v2 = (vel**2).sum(axis=1)
  v_tan = np.sqrt(v2 - v_r**2)
  v_ratio = (np.abs(v_r)-np.abs(v_tan)) / np.linalg.norm(vel, axis=1)

  merger_smass = f['SubhaloMassType'][:,4][IDs[merger_pkmassid]]
  smass = np.interp(spline_res, orig_res, merger_smass*1e10/h)

  merger_gmass = f['SubhaloMassType'][:,0][IDs[merger_pkmassid]]
  gmass = np.interp(spline_res, orig_res, merger_gmass*1e10/h)

  gas_fraction = gmass / (gmass + smass)

  z = make_interp_spline(orig_res, redshifts[f['SnapNum'][:][IDs[merger_pkmassid]]], k=k)(spline_res)
  try:
    z_interp = np.where(((r-r200)[1:]<0) & ((r-r200)[:-1]>0))[0][0]
  except:
    z_interp = -1
  nearest_snap = int(np.floor(spline_res[z_interp]))
  nearest_ID = IDs[merger_pkmassid][f['SnapNum'][:][IDs[merger_pkmassid]]==nearest_snap][0]
  nearest_snum = f['SubhaloNumber'][nearest_ID]
  nearest_GrNr = f['SubhaloGrNr'][nearest_ID]

  if not f['SubhaloMassType'][nearest_ID,0]:
    print('No gas.')
    continue

  # Calculate pericentre passages:
  peri_index = np.where((np.diff(r[1:])>0) & (np.diff(r[:-1])<0) & (r<r200)[1:-1])[0] + 1
  if len(peri_index):
    first_peri = r[peri_index[0]]
  else:
    first_peri = np.nan

  # Load the halo at pre-infall and retrieve metallicity data:
  header, cat, lf, data = get_data(halo, snapshot=nearest_snap, GrNr=nearest_GrNr, SubhaloNumber=nearest_snum, \
                                   data_fields=['ID  ', 'GMET'], PtType=0)
  FeH = np.nanmedian(data['FeH '])
  FeH_std = np.nanstd(data['FeH '])
  MgFe = np.nanmedian(data['MgFe'])
  MgFe_std = np.nanstd(data['MgFe'])

  separator = '\t'
  string = separator.join(['%s' % merger_pkmassid, '%.2f' % z[z_interp], '%.2e' % smass[z_interp], '%.2e' % gmass[z_interp], '%.2f' % gas_fraction[z_interp], '%.2f' % first_peri, '%.2f' % v_ratio[z_interp], '%.3f' % FeH, '%.3f' % FeH_std, '%.3f' % MgFe, '%.3f' % MgFe_std]) + '\n'
  with open('./files/merger_properties_Au-%i.txt' % halo, 'a') as file:
    file.write(string)
#--------------------------------------------------------------------
