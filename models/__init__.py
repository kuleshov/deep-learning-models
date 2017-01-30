from softmax import Softmax
from mlp import MLP
from cnn import CNN
from kcnn import KCNN
from vae import VAE
from sbn import SBN
from adgm import ADGM
from dadgm import DADGM

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