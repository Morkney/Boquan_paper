import numpy as np
from auriga_config import *
import auriga_functions as auriga
import snapHDF5 as snap
import readhaloHDF5
import readsubfHDF5

# Retrieve halo and stellar data:
def get_data(halo, snapshot=127, GrNr=0, SubhaloNumber=0, PtType=4, data_fields=[], cat_fields=[], align_on_disc=False, single_sub=True):
  #--------------------------------------------------------------------
  header = snap.snapshot_header(lvl4_basepath + '/halo_%i/output/snapdir_%03i/snapshot_%03i.0.hdf5' % (halo, snapshot, snapshot))
  h, a = header.hubble, header.time
  if 'GroupFirstSub' not in cat_fields:
    cat_fields += ['GroupFirstSub']
  cat = readsubfHDF5.subfind_catalog(lvl4_basepath+'/halo_%i/output/' % halo, snapshot, long_ids=True, keysel=cat_fields)

  if single_sub:
    snum = SubhaloNumber-cat.GroupFirstSub[GrNr]
  else:
    snum = -1

  data = {}
  if 'ID  ' not in data_fields:
    data_fields += ['ID  ']
  readhaloHDF5.reset()
  for field in data_fields:
    data[field] = readhaloHDF5.readhalo(lvl4_basepath+'/halo_%i/output/' % halo, snapbase, snapshot, field, PtType, GrNr, snum, long_ids=True)

  # Correct units:
  if 'SubhaloPos' in cat_fields:
    data['POS '] -= cat.SubhaloPos[SubhaloNumber]
  if 'SubhaloVel' in cat_fields:
    data['VEL '] -= cat.SubhaloVel[SubhaloNumber] * np.sqrt(1/a)
  auriga.physical_units(data, a, h)

  # Remove wind particles:
  if 'GAGE' in data_fields:
    wind_particles = data['GAGE'] < 0.
    for field in data_fields:
      data[field] = data[field][(~wind_particles)]

  if 'POS ' in data_fields:
    data['RG  '] = np.linalg.norm(data['POS '], axis=1)

  # Calculate default metal abundance ratios:
  if 'GMET' in data_fields:
    data['FeH '] = auriga.metal_ratio(data['GMET'], 'Fe', 'H') - 0.4
    data['MgFe'] = auriga.metal_ratio(data['GMET'], 'Mg', 'Fe') + 0.4

  # Find in/ex-situ:
  lf = auriga.listfile(lvl4_basepath, halo)
  if PtType==4:
    data['IN  '] = np.in1d(data['ID  '], lf.mdata['Insitu']['ParticleIDs'])
    data['EX  '] = np.in1d(data['ID  '], lf.mdata['Exsitu']['ParticleIDs'])

  # Align on the disc:
  if align_on_disc & ('POS ' in data_fields) & ('VEL ' in data_fields) & ('MASS' in data_fields):
    r200 = cat.Group_R_Crit200[GrNr] * header.time * 1e3/h
    disc_r = 0.1 * r200 # [kpc]
    data['POS '], data['VEL '], data['DROT'] = auriga.orient_on_disc(data['POS '], data['VEL '], data['MASS'], disc_r, data['IN  '])

  return header, cat, lf, data
