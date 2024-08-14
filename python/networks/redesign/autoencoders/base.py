import torch
from ...utils import FreezableModule
from ..preprocessor.null import NullPreprocessor
from ..postprocessor.null import NullPostprocessor

class BaseAutoencoder(FreezableModule):
    def __init__(self, encoder, decoder, latent_dims,
                 preprocessor=NullPreprocessor(), postprocessor=NullPostprocessor()):
        super(BaseAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.name = 'base_autoencoder'

    def forward(self, j, return_latent=False):
        latent_rep = self.encoder(self.preprocessor(j))
        if __debug__:
            assert(latent_rep.shape[1:] == self.latent_dims)
        result = self.postprocessor(self.decoder(latent_rep))
        if return_latent:
            return result, latent_rep
        else:
            return result