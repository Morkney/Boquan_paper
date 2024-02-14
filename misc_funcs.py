import numpy as np
import auriga_functions as auriga
from scipy.stats import binned_statistic_2d

# Function to create a variable-alpha cmap:
#--------------------------------------------------------------------
def density_map(prop, x, y, z, bins, cmap, proprange=[None, None], stat='mean'):

  no_infs = ~(np.isinf(prop) | np.isnan(prop))
  if stat=='mean':
    prop = np.histogram2d(x[no_infs], y[no_infs], bins=bins, weights=prop[no_infs])[0]
    prop_N = np.histogram2d(x[no_infs], y[no_infs], bins=bins)[0]
    prop /= prop_N
  else:
    prop = binned_statistic_2d(x[no_infs], y[no_infs], prop[no_infs], bins=bins, statistic=stat)[0]

  if None in proprange:
    proprange = np.nanpercentile(prop, [5, 99])

  density = np.log10(np.histogram2d(x, y, bins=bins, weights=z)[0])
  if np.isinf(density.max()): density[0,0] = 1e-10
  density_min = np.min(density[density>0])
  density_max = np.nanpercentile(density[~np.isinf(density)], 99)
  density[density==-np.inf] = density_min

  # Create a 2D cmap image:
  prop_to_cmap = np.nan_to_num(auriga.normalise(prop, 0, 255, *proprange)).astype('int')
  prop_img = auriga.NormaliseData(cmap(prop_to_cmap)) * 254.99
  prop_img[:,:,3] = auriga.normalise(density, 0, 254.99, density_min, density_max)
  prop_correct = 1 - (auriga.normalise(density, 0, 0.75, density_min, density_max))**2
  for i in range(3):
    prop_img[:,:,i] *= prop_correct
  combined_img = prop_img.astype(np.uint8)
  combined_img = np.dstack([combined_img[:,:,i].T for i in range(4)])
  return combined_img, proprange
#--------------------------------------------------------------------

# Function for 2D colourmap plots:
#--------------------------------------------------------------------
def cmap_2D_hist(prop, x, y, z, bins, cmap, proprange=[None, None], densrange=[None, None], stat='mean', divisor=1.):
  import colorstamps
  import PIL

  no_infs = ~(np.isinf(prop) | np.isnan(prop))
  if stat=='mean':
    prop = np.histogram2d(x[no_infs], y[no_infs], bins=bins, weights=prop[no_infs])[0]
    prop_N = np.histogram2d(x[no_infs], y[no_infs], bins=bins)[0]
    prop /= prop_N
  else:
    prop = binned_statistic_2d(x[no_infs], y[no_infs], prop[no_infs], bins=bins, statistic=stat)[0]
  if None in proprange:
    proprange = np.nanpercentile(prop, [5, 99])

  density = np.histogram2d(x, y, bins=bins, weights=z)[0]
  density = np.log10(density/divisor)
  if None in densrange:
    densrange = np.nanpercentile(density[~np.isinf(density)], [5, 99.9])
  else:
    densrange = np.nanpercentile(density[~np.isinf(density)], densrange)
  density[density==-np.inf] = densrange[0]

  # https://colorstamps.readthedocs.io/en/latest/index.html
  if cmap=='rainbow':
    cmap = np.asarray(PIL.Image.open('./files/SFH_cmap_256.png'))
  rgb, stamp = colorstamps.apply_stamp(density.T, prop.T, cmap, \
                                       vmin_0=densrange[0], vmax_0=densrange[1], \
                                       vmin_1=proprange[0], vmax_1=proprange[1])

  return rgb, proprange, densrange, stamp
#--------------------------------------------------------------------
