import numpy as np
import h5py
import sys

from auriga_config import *
import auriga_functions as auriga

import default_setup
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
plt.ion()

# Ensure correct python version first:
if sys.version[:5] != '3.7.3':
  print('Wrong python version!')
  sys.exit()
import pydot

# The graphviz module must be loaded for ytree to produce a plot.

halo = 5
MaxSnapNum = 127

# Set plot parameters:
#--------------------------------------------------------------------
path = lvl4_basepath + 'mergertrees/halo_%i/' % halo
TreePath = path + 'trees_sf1_%03d.0.hdf5' % MaxSnapNum

# Find the main halo:
MainHalo = 0

MinMassRatio = 1/20. # Minimum merger ratio
MinMass = 1e9
cmap = cm.Spectral_r

dpi = 5
fs = 10
aspect = None
TimeAxis = True
LabelHalos = True
LabelRatios = True
LineColour = 'lightgrey'
MainProgLineColour = 'lightcoral'
direction = 'left to right' # "left to right" or "right to left"

# Pydot keyword arguments:
dot_kwargs = {'rankdir': 'LR', 'nodesep': 1, 'ranksep': '1', 'splines': 'spline', 'outputorder': 'edgesfirst', 'dpi': dpi, 'pad': 1}
node_kwargs = {"label":'', "shape": "square", "width": 0.01, "height": 0.01}
edge_kwargs = {"penwidth": 0}
#--------------------------------------------------------------------

# Load tree:
#--------------------------------------------------------------------
f = h5py.File(TreePath, 'r')['Tree0']
TreeRedshifts = np.array(h5py.File(TreePath, 'r')['Header']['Redshifts'][:])
max_ID = h5py.File(TreePath, 'r')['Header']['TreeNHalos'][0]

# Define the subsampling of the outputs:
SubRedshifts = np.append(TreeRedshifts[:-56], TreeRedshifts[-55::2])
SubRedshifts = TreeRedshifts
SubSnaps = np.array([np.argmin(np.abs(i - TreeRedshifts)) for i in SubRedshifts])
#--------------------------------------------------------------------

# Functions:
#--------------------------------------------------------------------
# Sigmoid function for smooth pydot edges:
def sigmoid(x, beta=3.):
  return 1 / (1 + (x / (1-x))**(-beta))
x = sigmoid(np.linspace(0, 1, 20))

# Convenient normalisation function:
def normalise(a, normmin, normmax, vmin, vmax):
  normed = (a-vmin) / (vmax-vmin) * (normmax-normmin) + normmin
  normed[normed > normmax] = normmax
  normed[normed < normmin] = normmin
  return normed

# Translate numbers into hex colours:
def get_colour(cprops, cpropMin, cpropMax):
  cprop_normed = np.nan_to_num(normalise(cprops, 0, 99, cpropMin, cpropMax))
  color = colours[np.round(cprop_normed).astype('int')]
  color[np.isnan(cprops)] = np.array([1,1,1,1])
  return np.array(list(map(matplotlib.colors.rgb2hex, color)))

# Reformat scientific notation:
def latex_float(f):
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

# Get the tree indexes for the main progenitor line:
def ProgLine(halo=0):
  progs = []
  while halo != -1:
    progs.append(halo)
    halo = f['FirstProgenitor'][halo]
  return progs

# Add pydot node:
def plot_node(halo, graph):
  nodeName = f"{halo}"
  node = graph.get_node(nodeName)
  if len(node) == 0:
    node = pydot.Node(nodeName, **node_kwargs)
    graph.add_node(node)
  else:
    node = node[0]
  return node

# Walk through tree and find the necessary properties:
def get_ancestors(halo, i, graph, node, DescID):

  if TreeRedshifts[f['SnapNum'][halo]] > 10:
    return

  if f['SnapNum'][halo] in SubSnaps:
    node = plot_node(halo, graph)
    DescID = halo
    i += 1

  # Find all of the ancestors to the current halo:
  ancestors = np.where(((f['SnapNum'][:] == f['SnapNum'][halo] - 1) | \
                        (f['SnapNum'][:] == f['SnapNum'][halo] - 2)) & \
                        (f['Descendant'][:] == halo))[0]

  if not len(ancestors):
    return

  for ancestor in ancestors:
    if f['FirstProgenitor'][f['Descendant'][ancestor]] != ancestor:
      if MergerRatios[ancestor] < MinMassRatio:
        continue
    if PeakProgMass[ancestor] < MinMass:
      continue
    if TreeRedshifts[f['SnapNum'][ancestor]] > 10:
      return

    # Add an intermediary node if there is a skipped snapshot:
    SnapJump = f['SnapNum'][f['Descendant'][ancestor]] - f['SnapNum'][ancestor]
    if (SnapJump > 1) & (f['SnapNum'][ancestor] in SubSnaps):
      ancestorNode = plot_node(ancestor + max_ID, graph)
      graph.add_edge(pydot.Edge(node, ancestorNode, **edge_kwargs))
      node = ancestorNode

    # Add node and connecting pydot edges:
    ancestorNode = plot_node(ancestor, graph)
    graph.add_edge(pydot.Edge(node, ancestorNode, **edge_kwargs))
    if f['SnapNum'][ancestor] in SubSnaps:
      ID.append(ancestor)
      Desc.append(DescID)
    else:
      node = ancestorNode

    get_ancestors(ancestor, i-1, graph, node, DescID)
#--------------------------------------------------------------------

# Generate total mass, peak mass, and peak merger mass ratios:
#--------------------------------------------------------------------
# Find the total halo+galaxy mass:
TreeMass = f['SubhaloMassType'][:].sum(axis=1) * 1e10/0.6777
TreeMass[-1] = 0.

# Find the most massive progenitor and merger ratio:
Step = np.arange(max_ID)
HostStep = f['FirstHaloInFOFGroup'][:][f['FirstProgenitor'][:][f['Descendant'][Step]]]
PeakProgMass = np.zeros(max_ID)
PeakHostMass = np.ones(max_ID) * np.inf

for i in range(MaxSnapNum):
  MoreMass = TreeMass[Step] > PeakProgMass
  PeakProgMass[MoreMass] = TreeMass[Step][MoreMass]

  InHostFoF = f['FirstHaloInFOFGroup'][:][Step] == HostStep
  PeakHostMass[MoreMass & InHostFoF] = TreeMass[HostStep][MoreMass & InHostFoF]

  HostStep = f['FirstProgenitor'][:][HostStep]
  Step = f['FirstProgenitor'][:][Step]

  # Catch any misaligned host/merger snapshots:
  for j in range(2):
    OffSnap = f['SnapNum'][:][Step] < f['SnapNum'][:][HostStep]
    HostStep[OffSnap] = f['FirstProgenitor'][:][HostStep][OffSnap]
    OffSnap = f['SnapNum'][:][Step] > f['SnapNum'][:][HostStep]
    Step[OffSnap] = f['FirstProgenitor'][:][Step][OffSnap]
MergerRatios = PeakProgMass / PeakHostMass
#--------------------------------------------------------------------

# Find tree properties:
#--------------------------------------------------------------------
print('>    Finding tree properties.')

# Initialise data storage lists:
ID = [MainHalo]
Desc = [f['Descendant'][MainHalo]]

# Set up a pydot graph object:
graph = pydot.Dot(graph_type='graph', **dot_kwargs)

# Walk through the tree:
MainProgs = ProgLine(MainHalo)
get_ancestors(MainHalo, f['SnapNum'][MainHalo], graph, None, None)

# Get property arrays:
ID = np.array(ID)
Desc = np.array(Desc)
Mass = TreeMass[ID]
SnapNum = f['SnapNum'][:][ID]
HaloNum = f['SubhaloNumber'][:][ID]
PosG = f['SubhaloPos'][:][ID] - f['SubhaloPos'][:][f['FirstHaloInFOFGroup'][:][ID]]
Rg = np.linalg.norm(PosG, axis=1)
satellite = (Rg <= f['Group_R_Crit200'][:][f['FirstHaloInFOFGroup'][:][ID]]) & (f['FirstHaloInFOFGroup'][:][ID] != ID)
MergerRatios = MergerRatios[ID]

# Find the main progenitor ID at every snapshot:
HostMap = {}
for prog in MainProgs:
  HostMap[f['SnapNum'][prog]] = prog
for snap in np.setdiff1d(SnapNum, list(HostMap.keys())):
  HostMap[snap] = MainProgs[np.abs(np.array(list(HostMap.keys())) - snap).argmin()]
ID_host = np.array([HostMap[snap] for snap in SnapNum])

# Calculate a velocity parameter where -1 is tangential and +1 is radial:
Pos = f['SubhaloPos'][:][ID] - f['SubhaloPos'][:][ID_host]
Vel = f['SubhaloVel'][:][ID] - f['SubhaloVel'][:][ID_host]
r, theta, phi, v_r, v_th, v_p = auriga.cartesian_to_spherical(*Pos.T, *Vel.T)
v2 = (Vel**2).sum(axis=1)
v_tan = np.sqrt(v2 - v_r**2)
cprops = (np.abs(v_r)-np.abs(v_tan)) / np.linalg.norm(Vel, axis=1)

# Convert the snapshot array to the subsampled snapshot format:
SnapDict = {}
for i, j in enumerate(SubSnaps):
  SnapDict[j] = i
SubSnapNum = np.array([SnapDict[i] for i in SnapNum])

# Render:
filename = 'tree.txt'
func = getattr(graph, f"write_plain", None)
func(filename)

# Interpret the node positions:
with open('tree.txt', 'r') as file:
  lines = file.readlines()
lines = np.array(lines)[['node' in i for i in lines]]
nodeID = [int(line.split(' ')[1]) for line in lines]
Order = [float(line.split(' ')[3]) for line in lines]
Order = np.array(Order)[[np.where(nodeID == i)[0][0] for i in ID]]

# Fix the resolution to be on halves:
Order = 0.75 * np.round(Order*2)/2.
#--------------------------------------------------------------------

# Make the plot:
#--------------------------------------------------------------------
print('>    Building plot.')

fig_size = (len(np.unique(SubSnapNum))-1) / 5.
if aspect is not None:
  AxisRatio = aspect
else:
  AxisRatio = (SubSnapNum.max() - SubSnapNum.min()) / (Order.max() - Order.min())
  AxisRatio = min(AxisRatio, 6)
#fig, ax = plt.subplots(figsize=(fig_size, 2*fig_size/AxisRatio), nrows=2, ncols=1, gridspec_kw={'hspace':0.0, 'wspace':0.0, 'height_ratios':[0.2, 0.6]})
fig, ax = plt.subplots(figsize=(fig_size, fig_size/AxisRatio))

# Define the size of the nodes:
MinDotSize = 1
MaxDotSize = 9
MaxMass = np.log10(Mass.max())
MinMass = np.log10(MinMass)
DotSize = normalise(np.log10(Mass), MinDotSize, MaxDotSize, MinMass, MaxMass)

# Set up the colourbar properties:
colours = cmap(np.linspace(0, 1, 100))
cbarLabel = r'$\frac{\left | V_{\rm rad} \right | - \left | V_{\rm tan} \right |}{\bar{V}}$'
cpropMax = 1
cpropMin = -1
c = get_colour(cprops, cpropMin, cpropMax)

scatter_kwargs = {'clip_on':False, 'zorder':3, 'edgecolors':'black'}
joint_kwargs = {'capstyle':'round', 'rasterized':False, 'joinstyle':'round'}

prog_counter = 0
Rank = SubSnapNum == SubSnapNum.max()
ax.scatter(SubSnapNum[Rank], Order[Rank], s=DotSize[Rank]**2, c=c[Rank], linewidths=0.5, **scatter_kwargs)
for SubSnap in range(SubSnapNum.max()-1, SubSnapNum.min()-1, -1):

  # Find the current timestep and the descendants:
  Rank = SubSnapNum == SubSnap
  NextRank = [np.where(ID == j)[0][0] for j in Desc[Rank]]

  # Draw the nodes:
  ax.scatter(SubSnapNum[Rank&~satellite], Order[Rank&~satellite], s=DotSize[Rank&~satellite]**2, c=c[Rank&~satellite], linestyle='-', linewidths=0.5, **scatter_kwargs)
  ax.scatter(SubSnapNum[Rank&satellite], Order[Rank&satellite], s=DotSize[Rank&satellite]**2, c=c[Rank&satellite], linewidths=0, **scatter_kwargs)

  # Add the connecting lines:
  for j in range(np.sum(Rank)):
    if LabelHalos:
      string = '%i' % HaloNum[Rank][j] + '\n' + '%i' % SnapNum[Rank][j]
      ax.text(SubSnapNum[Rank][j], Order[Rank][j], string, fontsize=1, ha='center', va='center')
    edge_c = LineColour

    # Determine the membership of each connecting line:
    if ID[Rank][j] == MainProgs[prog_counter+1]:
      edge_c = MainProgLineColour ; prog_counter += 1

    # Draw the lines between each node:
    y_arr = np.linspace(SubSnapNum[Rank][j], SubSnapNum[NextRank][j], 20)
    x_Rank, x_NextRank = Order[Rank][j], Order[NextRank][j]
    linewidths = 0.7 * np.linspace(DotSize[Rank][j], DotSize[NextRank][j], len(y_arr)) + 0.5
    x_arr = x_Rank + (x * (x_NextRank - x_Rank))
    line_segments = LineCollection([np.column_stack([[y_arr[m], y_arr[m+1]], [x_arr[m], x_arr[m+1]]]) for m in range(len(y_arr)-1)], linewidths=linewidths, color=edge_c, **joint_kwargs)
    ax.add_collection(line_segments)
#--------------------------------------------------------------------

# Time axis:
#--------------------------------------------------------------------
redshift_ticks = [10, 8, 6, 5, 4, 3, 2, 1.5, 1.2, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
n_ticks = 10 # Number of ticks
tick_precision = 2 # Number of significant figures for the ticks
timebar_padding = 0. # Padding between merger tree and axis, vertical fraction

snapshots = np.unique(SubSnapNum)
redshifts = np.abs([SubRedshifts[i] for i in snapshots])

if redshift_ticks is not None:
  new_tick_locations = np.interp(redshift_ticks, redshifts[::-1], snapshots[::-1])
else:
  new_tick_locations = np.linspace(*snapshots[[0,-1]], n_ticks)
new_tick_labels = np.interp(new_tick_locations, snapshots, redshifts)

ax.set_xticks(new_tick_locations)
ax.set_xticklabels(['%.2g' % round(i, tick_precision) for i in new_tick_labels])
ax.set_xlabel(r'Redshift', fontsize=fs)
ax.tick_params(axis='x', labelsize=fs-2)
ax.minorticks_off()

ax.set_yticks([])
ax.tick_params(axis='x', which='both', direction='out')
ax.tick_params(top=False, which='both')
for spine in ['left', 'right', 'top']:
  ax.spines[spine].set_visible(False)

if direction == 'left to right':
  ax.set_xlim(new_tick_locations[[-1,0]])
elif direction == 'right to left':
  ax.set_xlim(new_tick_locations[[0,-1]])

# Twinned axis for linear time:
if TimeAxis:
  times = auriga.age_time(1) - np.array([auriga.age_time(1/(1+i)) for i in new_tick_labels])
  time_ax = ax.twiny()
  time_ax.set_position(ax.get_position())
  time_ax.invert_xaxis()
  time_ax.set_xticks(new_tick_locations)
  time_ax.set_xlim(ax.get_xlim())
  plt.sca(ax)
  time_ax.set_xticklabels([])
  time_ticks = ['%.1f' % x for x in times]
  time_ticks[-1] = '0'
  time_ax.set_xticklabels(time_ticks)
  time_ax.set_xlabel(r'Time [Gyr]', fontsize=fs)
  time_ax.tick_params(axis='x', labelsize=fs-2)
  time_ax.tick_params(axis='x', direction='out')
  time_ax.tick_params(top=False, which='minor')
  time_ax.minorticks_off()
  for spine in ['left', 'right', 'bottom']:
    time_ax.spines[spine].set_visible(False)
#--------------------------------------------------------------------

# Legend:
#--------------------------------------------------------------------
dots = 5 # Number of dots

DotSizes = np.linspace(MinDotSize, MaxDotSize, dots)
DotMasses = np.logspace(np.log10(MinMass), np.log10(MaxMass), dots)

nodeColour = cmap(64)
ax.scatter([],[], s=DotSizes[-1]**2, c=nodeColour, edgecolors='k', linewidths=0.5, label=r'Central')
ax.scatter([],[], s=DotSizes[-1]**2, c=nodeColour, edgecolors='k', linewidths=0., label=r'Satellite $(R_{\rm G}<R_{200})$')
ax.scatter([],[], c='None', label=' ')
ax.scatter([],[], c='None', label=' ')
for DotSize, DotMass in zip(DotSizes, DotMasses):
  ax.scatter([],[], s=DotSize**2, c='black', lw=0, label=r'$%s\,$M$_{\odot}$' % latex_float(10**DotMass))

# Add labels for line types:
ax.plot([],[], color=MainProgLineColour, lw=4, label='Main progenitor line')

if direction == 'left to right':
  location = 'upper left'
elif direction == 'right to left':
  location = 'upper right'
handles, labels = ax.get_legend_handles_labels()
reorder = [5,6,7,8,9,0,1,2,3,4]
ax.legend([handles[idx] for idx in reorder],[labels[idx] for idx in reorder], frameon=False, loc=location, fontsize=fs-1, ncol=2)
#--------------------------------------------------------------------

# Colorbar:
#--------------------------------------------------------------------
norm = mpl.colors.Normalize(vmin=cpropMin, vmax=cpropMax)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
l, b, w, h = ax.get_position().bounds

if direction == 'left to right':
  cax = fig.add_axes([l + w*1.01, b, w*0.01, h])
  cbar = plt.colorbar(sm, cax=cax)
  cax.yaxis.set_label_position('right')
  cax.yaxis.set_ticks_position('right')
elif direction == 'right to left':
  cax = fig.add_axes([l - w*2*1.01, b, w*0.01, h])
  cbar = plt.colorbar(sm, cax=cax)
  cax.yaxis.set_label_position('left')
  cax.yaxis.set_ticks_position('left')

cbar.set_label(cbarLabel, fontsize=fs+2)
cbar.ax.tick_params(labelsize=fs-2)
cbar.set_ticks([-1,-0.5,0,0.5,1])
#--------------------------------------------------------------------

# Add markers for the merger mass ratio:
#--------------------------------------------------------------------
if LabelRatios:

  # Lines that connect to the merger node:
  xpad = 0.75
  ypad = xpad

  # IDs of the merger nodes:
  MergeID = np.array([(ID[i] not in MainProgs) & \
                      (Desc[i] in MainProgs) & \
                      (MergerRatios[i] != np.inf) for i in range(len(ID))])

  for MergeSnap in np.unique(f['SnapNum'][:][Desc[MergeID]]):

    preMergeID = np.where(f['SnapNum'][:][Desc[MergeID]] == MergeSnap)[0]
    if len(preMergeID) > 1:
      merge_string = '\n'.join(['1:%i' % np.round(1/MergeRatio) for MergeRatio in MergerRatios[MergeID][preMergeID]])
    else:
      merge_string = '1:%i' % np.round(1/MergerRatios[MergeID][preMergeID])

    MergeOrder = Order[np.where(SubSnapNum == MergeSnap)[0][0]]
    ax.plot([MergeSnap, MergeSnap+xpad], [MergeOrder, MergeOrder+ypad], 'k-', lw=0.5)
    ax.text(MergeSnap+xpad, MergeOrder+ypad, merge_string, fontsize=fs-4, ha='center', va='bottom')  
#--------------------------------------------------------------------

# Optional: Add title:
#--------------------------------------------------------------------
bbox = ax.get_window_extent()
height = (bbox.height - bbox.width/50.) / bbox.height
ax.text(0.5, height, 'Halo %i merger tree' % halo, fontsize=fs, ha='center', va='top', transform=ax.transAxes)
#--------------------------------------------------------------------

plt.savefig('../images/trees/merger_tree_%s.pdf' % halo, bbox_inches='tight')
print('Done.')

