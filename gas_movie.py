# Make an animation:
#--------------------------------------------------------------------
import multiprocessing
import imageio
from PIL import Image
import numpy as np
import os, sys

halo = int(sys.argv[1])

frames = np.array([imageio.imread('./images/gas_images/frame_%i.png' % i) for i in range(40, 127+1, 1)])
writer = imageio.get_writer('./images/gas_movie_Au-%i.mp4' % halo, fps=4)
for frame in frames:
  writer.append_data(frame)
writer.close()

# Remove temporary frame storage:
#os.system('rm -rf ./pngs')
#--------------------------------------------------------------------
