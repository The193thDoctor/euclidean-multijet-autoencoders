from .mlp import MLPEmbedder

class MLPSoftmaxEmbedder(MLPEmbedder):
    def __init__(self, dim=20, *args, **kwargs):
        super().__init__(dim=dim, *args, **kwargs)
        self.name = self.name = 'mlp_softmax_embedder_dim{}'.format(dim)

    def forward(self, x):
        result = super().forward(x)
        return result.softmax(dim=1)