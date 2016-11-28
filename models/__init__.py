from softmax import Softmax
from mlp import MLP
from cnn import CNN
from vae import VAE
from sbn import SBN
from adgm import ADGM

try:
  from resnet import Resnet
except:
  print 'WARNING: Could not import Resnet; you might need to upgrade Lasagne.'