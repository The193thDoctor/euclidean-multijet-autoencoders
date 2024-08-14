import torch
from .base import BaseAutoencoder
from ..embedder.mlp import MLPEmbedder
from ..preprocessor.original import OriginalPreprocessor
from ..postprocessor.original import OriginalPostprocessor

class EmbedderAE(BaseAutoencoder):
    def __init__(self, hidden_dim):
        encoder = MLPEmbedder(4, hidden_dim)
        decoder = MLPEmbedder(hidden_dim, 4)
        preprocessor = OriginalPreprocessor()
        postprocessor = OriginalPostprocessor()
        super().__init__(encoder, decoder, [hidden_dim, 4], preprocessor, postprocessor)