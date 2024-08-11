from .mlp import MLPEmbedder

class MLPSoftmaxEmbedder(MLPEmbedder):
    def __init__(self, dimension=20, *args, **kwargs):
        super().__init__(dimension=dimension, *args, **kwargs)
        self.name = self.name = 'mlp_softmax_embedder_dim{}'.format(dimension)

    def forward(self, x):
        result = super().forward(x)
        return result.softmax(dim=1)