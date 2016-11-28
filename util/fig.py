import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------

def parselog(fname, xi=0, yi=2):
  x, y = [], []
  with open(fname) as f:
    for line in f:
      fields = line.strip().split()
      x.append(float(fields[xi]))
      y.append(float(fields[yi]))
  return np.array(x), np.array(y)

def plot_many(fname, curves, names=None, double=[]):
  plt.figure(figsize=(12,8))
  curves = _flip_and_stretch(curves, double)

  plt_list = [itm for lst in curves for itm in lst]
  plt.plot(*plt_list)

  plt.ylim((0, 0.05))

  if names: plt.legend(names, prop={'size':8})

  plt.savefig(fname)
  plt.close()

def plot_one_vs_many(fname, main_curve, curves, double_main=False):
  plt.figure(figsize=(12,8))
  
  idx = [0] if double_main else []
  main_curve = _flip_and_stretch([main_curve], idx)[0]
  curves = _flip_and_stretch(curves)

  plt.plot(main_curve[0], main_curve[1])

  plt_list = [itm for lst in curves for itm in lst]
  plt.plot(*plt_list, alpha=0.1)

  plt.ylim((0, 0.05))

  plt.savefig(fname)
  plt.close()

def plot_many_vs_many(fname, curves1, curves2, double1=False, double2=False):
  plt.figure(figsize=(12,8))
  
  idx = range(len(curves1)) if double1 else []
  curves1 = _flip_and_stretch(curves1, idx)
  idx = range(len(curves2)) if double2 else []
  curves2 = _flip_and_stretch(curves2, idx)

  plt_list = [itm for lst in curves1 for itm in lst]
  plt.plot(*plt_list, alpha=0.25, color='blue')

  plt_list = [itm for lst in curves2 for itm in lst]
  plt.plot(*plt_list, alpha=0.25, color='green')

  # plt.ylim((0, 0.15))
  plt.ylim((0, 0.05))

  plt.savefig(fname)
  plt.close()

def _flip_and_stretch(curves, idx=[]):
  curves2 = list()
  for i, (x,y) in enumerate(curves):
    if i in idx:
      curves2.append( (2*x, 1-y) )
    else:
      curves2.append( (x,1-y) )
  return curves2