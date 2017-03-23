from softmax import Softmax
from mlp import MLP
from cnn import CNN
from vae import VAE
from sbn import SBN
from adgm import ADGM
from hdgm import HDGM
from dadgm import DADGM

# load some models that require the latest version of Lasagne
try:
  from resnet import Resnet
except:
  print 'WARNING: Could not import the Resnet model'

try:
  from dcgan import DCGAN  
except:
  print 'WARNING: Could not import the DCGAN model'

try:
  from ssadgm import SSADGM
except:
  print 'WARNING: Could not import the SSADGM model'  