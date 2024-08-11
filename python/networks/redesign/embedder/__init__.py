from .mlp import MLPEmbedder
from .mlp_softmax import MLPSoftmaxEmbedder

model_dict = {'mlp': MLPEmbedder, 'mlp_softmax': MLPSoftmaxEmbedder}