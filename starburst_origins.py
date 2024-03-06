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
burst_epoch = (8.25, 8.75) # lookback time in Gyrs
merger = 68307
z_cut = 5 # [kpc]

halo = int(sys.argv[1])
ID = 0
snapshot = 251
basepath = lvl4rerun_basepath
#--------------------------------------------------------------------

# Load the stars and tracers at z=0:
#--------------------------------------------------------------------
tree_path = basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % snapshot
f = h5py.File(tree_path, "r")['Tree0']
redshifts = np.array(h5py.File(tree_path, 'r')['Header']['Redshifts'][:])

cat_fields = ['GroupFirstSub']
data_fields = ['TRFQ', 'TRID', 'TRPR']
header, cat, lf, tdata = get_data(halo, snapshot, SubhaloNumber=f['SubhaloNumber'][ID], single_sub=False, \
                                  data_fields=data_fields, cat_fields=cat_fields, PtType=6, basepath=basepath)
# Need to edit tracers in Auriga_python/subfind/readsubfHDF5.py 

data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE', 'POT ', 'GMET', 'GSPH']
cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
header, cat, lf, data = get_data(halo, snapshot, SubhaloNumber=f['SubhaloNumber'][ID], \
                                 data_fields=data_fields, cat_fields=cat_fields, align_on_disc=True, basepath=basepath)
#--------------------------------------------------------------------

# Find stars/tracers associated with a particular starburst:
#--------------------------------------------------------------------
data = auriga.identify_disc(data)
data['DISC'] &= data['IN  ']
cut = data['DISC'] * (np.abs(data['POS '][:,2]) < z_cut)

# Convert to scalefactor:
burst_epoch = auriga.time_age(auriga.age_time(1.) - np.array(burst_epoch))
burst_cut = (data['GAGE'] > burst_epoch.min()) & (data['GAGE'] <= burst_epoch.max())

#overlap = np.in1d(tdata['TRPR'], data['ID  '][cut & burst_cut])
overlap = np.in1d(tdata['TRPR'], data['ID  '][burst_cut])
starburst_tracers = tdata['TRID'][overlap]
#--------------------------------------------------------------------

# Find where the gas originated at a time prior to merger infall:
#--------------------------------------------------------------------
# Wind merger ID back to a time before infall:
ID = merger
while f['FirstHaloInFOFGroup'][ID] != ID:
  ID = f['FirstProgenitor'][ID]

# Wind the host halo ID back to this same snapshot:
host_ID = 0
while f['SnapNum'][host_ID] > f['SnapNum'][ID]:
  host_ID = f['FirstProgenitor'][host_ID]

merger_tracers = np.zeros_like(starburst_tracers).astype('bool')

# Load the gas tracers for both objects:
cat_fields = ['GroupFirstSub']
data_fields = ['ID  ']
IDs = []

# Loop over a few snapshots just to pick up stray gas tracers that were stripped early:
for i in range(3):

  print('>    Snapshot %i, Shared Tracers=%i' % (f['SnapNum'][ID], np.sum(merger_tracers)))

  # Apparently, this tracer ID loader is completely untrustworthy! Don't even use it...
  # Instead, load the halo gas and all the tracers, and then do a npin1d with the parent IDs to find the tracers.

  # Load gas for this halo:
  header, cat, lf, data = get_data(halo, f['SnapNum'][ID], GrNr=f['SubhaloGrNr'][ID], SubhaloNumber=f['SubhaloNumber'][ID], single_sub=False, \
                                   data_fields=data_fields, cat_fields=cat_fields, PtType=0, basepath=basepath)

  # Load all tracer parent IDs:
  path = basepath + 'halo_%i/output/snapdir_%03d/snapshot_%03d' % (halo, f['SnapNum'][ID], f['SnapNum'][ID])
  tracer_parents = snap.read_block(path, 'TRPR', parttype=6)
  tracer_ID = snap.read_block(path, 'TRID', parttype=6)

  # Some gas IDs have multiple tracers in them, need to be careful to do the in1d this way round:
  tracers_in_halo = np.in1d(tracer_parents, data['ID  '])
  tracer_parents_in_halo = tracer_parents[tracers_in_halo]
  tracer_IDs_in_halo = tracer_ID[tracers_in_halo]

  merger_tracers = np.logical_or(merger_tracers, np.in1d(starburst_tracers, tracer_IDs_in_halo))

  IDs.append(ID)
  ID = f['FirstProgenitor'][ID]

IDs = np.array(IDs)
redshift = redshifts[f['SnapNum'][:][IDs]]

# Specifically check the metallicity of the gas:
cat_fields = ['GroupFirstSub', 'SubhaloPos']
data_fields = ['ID  ', 'POS ', 'MASS', 'GMET']
header, cat, lf, data = get_data(halo, f['SnapNum'][ID], GrNr=f['SubhaloGrNr'][ID], SubhaloNumber=f['SubhaloNumber'][ID], single_sub=False, \
                                 data_fields=data_fields, cat_fields=cat_fields, PtType=0, basepath=basepath)
donated_gas = np.in1d(data['ID  '], tracer_parents_in_halo[np.in1d(tracer_IDs_in_halo, starburst_tracers)])


# Result: The gas seems to be thoroughly mixed by the time it forms stars in the host galaxy.
