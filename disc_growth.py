import numpy as np
import sys

import h5py
from auriga_config import *
import auriga_functions as auriga
import snapHDF5 as snap
import readhaloHDF5
import readsubfHDF5
from get_halo import get_data

# Select the simulation:
#--------------------------------------------------------------------
halo = int(sys.argv[1])
ID = 0

tree_path = lvl4_basepath + 'mergertrees/halo_%i/' % halo + 'trees_sf1_%i.0.hdf5' % 127
f = h5py.File(tree_path, "r")['Tree0']
#--------------------------------------------------------------------

# Loop over each snapshot and find the disc fraction:
#--------------------------------------------------------------------
while ID != -1:

  # Load the stellar data:
  GrNr = f['SubhaloGrNr'][ID]
  SubhaloNumber = f['SubhaloNumber'][ID]
  data_fields = ['POS ', 'VEL ', 'MASS', 'ID  ', 'GAGE']
  cat_fields = ['GroupFirstSub', 'Group_R_Crit200', 'SubhaloPos', 'SubhaloVel']
  header, cat, lf, data = get_data(halo, snapshot, GrNr=GrNr, SubhaloNumber=SubhaloNumber, data_fields=data_fields, cat_fields=cat_fields, align_on_disc=True)

  # Select disc stars:
  data['DISC'] = auriga.identify_disc(data['POS '], data['VEL '], data['POT ']) & \
                 (data['POS '][:,2] <= 10) & (data['RG  '] <= 30)

  # Calculate the disc fraction:
  disc_mass = data['MASS'][data['DISC']].sum()
  total_mass = data['MASS'].sum()
  disc_frac = disc_mass / total_mass

  # Find the times:
  z = 1/header.time - 1
  t = auriga.age_time(header.time)

  # Append to file:
  with open('./files/disc_growth_Au-%i.txt' % halo, 'a') as f:
    f.write('%.2f\t%.2f\t%.6f\t%.6f\t%.2f' % (t, z, disc_mass, total_mass, disc_frac))

  # Next snapshot:
  ID = f['FirstProgenitor'][ID]
#--------------------------------------------------------------------
