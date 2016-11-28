import numpy as np

# ----------------------------------------------------------------------------
# iteration

def iterate_minibatch_idx(n_inputs, batchsize,):
  for start_idx in range(0, n_inputs - batchsize + 1, batchsize):
    yield start_idx, min(start_idx + batchsize, n_inputs)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
    else:
        excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

def random_subbatch(inputs, targets, batchsize):
  assert len(inputs) == len(targets)
  indices = np.arange(len(inputs))
  np.random.shuffle(indices)
  excerpt = indices[:batchsize]
  return inputs[excerpt], targets[excerpt]

# ----------------------------------------------------------------------------
# eval

def evaluate(eval_f, X, Y, batchsize=1000):
  tot_err, tot_acc, batches = 0, 0, 0
  for inputs, targets in iterate_minibatches(X, Y, batchsize, shuffle=False):
    err, acc = eval_f(inputs, targets)
    tot_err += err
    tot_acc += acc
    batches += 1
  return tot_err / batches, tot_acc / batches

def log_metrics(logname, metrics):
  logfile = '%s.log' % logname
  with open(logfile, 'a') as f:
    f.write('\t'.join([str(m) for m in metrics]) + '\n')