import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import plots
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

torch.manual_seed(0)#make training results repeatable


#
# embed inputs in feature space
#
class Input_Embed(nn.Module):
    def __init__(self, dimension, device='cpu', symmetrize=True, return_masses = False):
        super(Input_Embed, self).__init__()
        self.d = dimension
        self.device = device
        self.symmetrize = symmetrize
        self.return_masses = return_masses

        # embed inputs to dijetResNetBlock in target feature space
        self.jet_embed     = Ghost_Batch_Norm(3, features_out=self.d, conv=True, name='jet embedder', device=self.device) # phi is relative to dijet, mass is zero in toy data. # 3 features -> 8 features
        self.jet_conv      = Ghost_Batch_Norm(self.d, conv=True, name='jet convolution', device = self.device)

        self.dijet_embed   = Ghost_Batch_Norm(4, features_out=self.d, conv=True, name='dijet embedder', device = self.device) # phi is relative to quadjet, # 4 features -> 8 features
        self.dijet_conv    = Ghost_Batch_Norm(self.d, conv=True, name='dijet convolution', device = self.device) 

        self.quadjet_embed = Ghost_Batch_Norm(3 if self.symmetrize else 4, features_out=self.d, conv=True, name='quadjet embedder', device = self.device) # phi is removed. # 3 features -> 8 features
        self.quadjet_conv  = Ghost_Batch_Norm(self.d, conv=True, name='quadjet convolution', device = self.device)

        #self.register_buffer('tau', torch.tensor(math.tau, dtype=torch.float))

    
        
    def data_prep(self, j):
        j = j.clone()# prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        j = j.view(-1,4,4)

        d, dPxPyPzE = addFourVectors(j[:,:,(0,2,0,1,0,1)], # 6 pixels
                                     j[:,:,(1,3,2,3,3,2)])

        q, qPxPyPzE = addFourVectors(d[:,:,(0,2,4)],
                                     d[:,:,(1,3,5)], 
                                     v1PxPyPzE = dPxPyPzE[:,:,(0,2,4)],
                                     v2PxPyPzE = dPxPyPzE[:,:,(1,3,5)])
        
        if self.return_masses:
            m2j = d[:, 3:4, :].clone()
            m4j = q[:, 3:4, :].clone()

        # take log of pt, mass variables which have long tails
        # j = PxPyPzE(j)
        # j = torch.log(1+j.abs())*j.sign()
        j[:,(0,3),:] = torch.log(1+j[:,(0,3),:])
        d[:,(0,3),:] = torch.log(1+d[:,(0,3),:])
        q[:,(0,3),:] = torch.log(1+q[:,(0,3),:])

        # set up all possible jet pairings
        j = torch.cat([j, j[:,:,(0,2,1,3)], j[:,:,(0,3,1,2)]],2)

        if self.symmetrize:
            # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
            j[:,2:3,(0,2,4,6,8,10)] = calcDeltaPhi(d, j[:,:,(0,2,4,6,8,10)]) # replace jet phi with deltaPhi between dijet and jet
            j[:,2:3,(1,3,5,7,9,11)] = calcDeltaPhi(d, j[:,:,(1,3,5,7,9,11)])
        
            d[:,2:3,(0,2,4)] = calcDeltaPhi(q, d[:,:,(0,2,4)])
            d[:,2:3,(1,3,4)] = calcDeltaPhi(q, d[:,:,(1,3,5)])

            q = torch.cat( (q[:,:2,:],q[:,3:,:]) , 1 ) # remove phi from quadjet features

        if self.return_masses:
            return j, d, q, m2j, m4j
        else:
            return j, d, q

    def set_mean_std(self, j):
        if self.return_masses:
            j, d, q, _, _ = self.data_prep(j)
        else:
            j, d, q = self.data_prep(j)

        self    .jet_embed.set_mean_std(j[:,0:3])#mass is always zero in toy data
        self  .dijet_embed.set_mean_std(d)
        self.quadjet_embed.set_mean_std(q)

    def set_ghost_batches(self, n_ghost_batches):
        self.    jet_embed.set_ghost_batches(n_ghost_batches)
        self.  dijet_embed.set_ghost_batches(n_ghost_batches)
        self.quadjet_embed.set_ghost_batches(n_ghost_batches)

        self.    jet_conv.set_ghost_batches(n_ghost_batches)
        self.  dijet_conv.set_ghost_batches(n_ghost_batches)
        self.quadjet_conv.set_ghost_batches(n_ghost_batches)

    def forward(self, j):
        if self.return_masses:
            j, d, q, m2j, m4j = self.data_prep(j)
        else:
            j, d, q = self.data_prep(j)

        j = self    .jet_embed(j[:,0:3])#mass is always zero in toy data
        d = self  .dijet_embed(d)
        q = self.quadjet_embed(q)

        j = self    .jet_conv(NonLU(j))
        d = self  .dijet_conv(NonLU(d))
        q = self.quadjet_conv(NonLU(q))
        
        if self.return_masses:
            return j, d, q, m2j, m4j
        else:
            return j, d, q



class Basic_CNN(nn.Module):
    def __init__(self, dimension, n_classes=2, device='cpu'):
        super(Basic_CNN, self).__init__()
        self.device = device
        self.d = dimension
        self.n_classes = n_classes
        self.n_ghost_batches = 64

        self.name = f'Basic_CNN_{self.d}'

        self.input_embed = Input_Embed(self.d)

        self.jets_to_dijets     = Ghost_Batch_Norm(self.d, stride=2, conv=True, device = self.device)
        self.dijets_to_quadjets = Ghost_Batch_Norm(self.d, stride=2, conv=True, device = self.device)

        self.select_q = Ghost_Batch_Norm(self.d, features_out=1, conv=True, bias=False, device = self.device)
        self.out      = Ghost_Batch_Norm(self.d, features_out=self.n_classes, conv=True, device = self.device)

    def set_mean_std(self, j):
        self.input_embed.set_mean_std(j)

    def set_ghost_batches(self, n_ghost_batches):
        self.input_embed.set_ghost_batches(n_ghost_batches)
        self.jets_to_dijets.set_ghost_batches(n_ghost_batches)
        self.dijets_to_quadjets.set_ghost_batches(n_ghost_batches)
        self.select_q.set_ghost_batches(n_ghost_batches)
        self.out.set_ghost_batches(n_ghost_batches)
        self.n_ghost_batches = n_ghost_batches

    def forward(self, j):
        j, d, q = self.input_embed(j)

        d = d + NonLU(self.jets_to_dijets(j))
        q = q + NonLU(self.dijets_to_quadjets(d))

        #compute a score for each event quadjet
        q_logits = self.select_q(q)

        #convert the score to a 'probability' with softmax. This way the classifier is learning which pairing is most relevant to the classification task at hand.
        q_score = F.softmax(q_logits, dim=-1)
        q_logits = q_logits.view(-1, 3)

        #add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1,2))

        #project the final event-level pixel into the class score space
        c_logits = self.out(e)
        c_logits = c_logits.view(-1, self.n_classes)

        return c_logits, q_logits

class Basic_encoder(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, permute_input_jet = False, phi_rotations = False, return_masses = False, n_ghost_batches = -1, device = 'cpu'):
        super(Basic_encoder, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.permute_input_jet = permute_input_jet
        self.phi_rotations = phi_rotations
        self.return_masses = return_masses
        self.n_ghost_batches = n_ghost_batches
        

        self.name = f'Basic_encoder_{self.d_bottleneck}'

        self.input_embed            = Input_Embed(self.d, symmetrize=self.phi_rotations, return_masses=self.return_masses)
        self.jets_to_dijets         = Ghost_Batch_Norm(self.d, stride=2, conv=True, device=self.device)
        self.dijets_to_quadjets     = Ghost_Batch_Norm(self.d, stride=2, conv=True, device=self.device)
        self.select_q               = Ghost_Batch_Norm(self.d, features_out=1, conv=True, bias=False, device=self.device)

        self.bottleneck_in          = Ghost_Batch_Norm(self.d, features_out=self.d_bottleneck, conv=True, device=self.device)
        
    def set_mean_std(self, j):
        self.input_embed.set_mean_std(j)
  
    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = n_ghost_batches
        # encoder
        self.input_embed.set_ghost_batches(n_ghost_batches)
        self.jets_to_dijets.set_ghost_batches(n_ghost_batches)
        self.dijets_to_quadjets.set_ghost_batches(n_ghost_batches)
        self.select_q.set_ghost_batches(n_ghost_batches)
        # bottleneck_in
        self.bottleneck_in.set_ghost_batches(n_ghost_batches)

    def forward(self, j):
        #
        # Preparation block
        #
        j_rot = j.clone() # j.shape = [batch_size, 4, 4]
        
        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane
        j_rot = setSubleadingPhiPositive(setLeadingPhiTo0(setLeadingEtaPositive(j_rot))) if self.phi_rotations else j_rot
        
        if self.permute_input_jet: 
            for i in range(j.shape[0]): # randomly permute the input jets positions# randomly permute the input jets positions
                j_rot[i] = j[i, :, torch.randperm(4)]

        # convert to PxPyPzE and compute means and variances
        jPxPyPzE = PxPyPzE(j_rot) # j_rot.shape = [batch_size, 4, 4]

        #
        # Encode Block
        #
        if self.return_masses:
            j, d, q, m2j, m4j = self.input_embed(j_rot)                                         # j.shape = [batch_size, self.d, 12] -> 12 = 0 1 2 3 0 2 1 3 0 3 1 2       
        else:
            j, d, q = self.input_embed(j_rot)                                                   # j.shape = [batch_size, self.d, 12] -> 12 = 0 1 2 3 0 2 1 3 0 3 1 2
                                                                                                # d.shape = [batch_size, self.d, 6]  -> 6 = 01 23 02 13 03 12
                                                                                                # q.shape = [batch_size, self.d, 3]  -> 3 = 0123 0213 0312; 3 pixels each with 8 features
        d = d + NonLU(self.jets_to_dijets(j))                                                   # d.shape = [batch_size, self.d, 6]
        q = q + NonLU(self.dijets_to_quadjets(d))                                               # q.shape = [batch_size, self.d, 3]
        # compute a score for each event quadjet
        q_logits = self.select_q(q)                                                             # q_logits.shape = [batch_size, 1, 3] -> 3 = 0123 0213 0312
        # convert the score to a 'probability' with softmax. This way the classifier is learning which pairing is most relevant to the classification task at hand.
        q_score = F.softmax(q_logits, dim=-1)                                                   # q_score.shape = [batch_size, 1, 3]
        q_logits = q_logits.view(-1, 3)                                                         # q_logits.shape = [batch_size, 3, 1]
        # add together the quadjets with their corresponding probability weight
        e_in = torch.matmul(q, q_score.transpose(1,2))                                          # e.shape = [batch_size, self.d, 1] (8x3 · 3x1 = 8x1)



        #
        # Bottleneck
        #
        z = NonLU(self.bottleneck_in(e_in))

        if self.return_masses:
            return jPxPyPzE, j_rot, z, m2j, m4j
        else:
            return jPxPyPzE, j_rot, z

class Basic_decoder(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, correct_DeltaR = False, return_masses = False, n_ghost_batches = -1, device = 'cpu'):
        super(Basic_decoder, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.correct_DeltaR = correct_DeltaR
        self.return_masses = return_masses
        self.n_ghost_batches = n_ghost_batches
        

        self.name = f'Basic_decoder_{self.d_bottleneck}'

        self.bottleneck_out         = Ghost_Batch_Norm(self.d_bottleneck, features_out=self.d, conv=True, device=self.device)

        self.extract_q              = Ghost_Batch_Norm(self.d, features_out=self.d, stride=3, conv_transpose=True, device=self.device)
        self.dijets_from_quadjets   = Ghost_Batch_Norm(self.d, features_out=self.d, stride=2, conv_transpose=True, device=self.device)
        self.jets_from_dijets       = Ghost_Batch_Norm(self.d, features_out=self.d, stride=2, conv_transpose=True, device=self.device)
        self.select_j               = Ghost_Batch_Norm(self.d*4, features_out=1, conv=True, bias=False, device=self.device)

        self.decode_j1 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.jets_res_2 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.jets_res_3 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.decode_j = Ghost_Batch_Norm(self.d, features_out=3, conv=True, device=self.device)
        
        self.decode_j2 = Ghost_Batch_Norm(self.d, features_out=4, conv=True, device=self.device)
        # self.expand_j = Ghost_Batch_Norm(self.d, features_out=128, conv=True, device=self.device)
        # self.decode_j = Ghost_Batch_Norm(128, features_out=3, conv=True, device=self.device)# jet mass is always zero, let's take advantage of this!

        # self.decode_1 = Ghost_Batch_Norm(  self.d, features_out=2*self.d, stride=2, conv_transpose=True, device=self.device)
        # self.decode_2 = Ghost_Batch_Norm(2*self.d, features_out=4*self.d, stride=2, conv_transpose=True, device=self.device)
        # self.decode_3 = Ghost_Batch_Norm(4*self.d, features_out=3,                  conv=True,           device=self.device)

    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = n_ghost_batches

        # bottleneck_out
        self.bottleneck_out.set_ghost_batches(n_ghost_batches)
        # decoder
        self.extract_q.set_ghost_batches(n_ghost_batches)
        self.dijets_from_quadjets.set_ghost_batches(n_ghost_batches)
        self.jets_from_dijets.set_ghost_batches(n_ghost_batches)
        self.select_j.set_ghost_batches(n_ghost_batches)
        self.decode_j1.set_ghost_batches(n_ghost_batches)
        # self.jets_res_2.set_ghost_batches(n_ghost_batches)
        # self.jets_res_3.set_ghost_batches(n_ghost_batches)
        # self.expand_j.set_ghost_batches(n_ghost_batches)
        self.decode_j2.set_ghost_batches(n_ghost_batches)
  
    def forward(self, z):
        #
        # Bottleneck
        #
        e_out = NonLU(self.bottleneck_out(z))



        #
        # Decode Block
        #
        '''
        dec_d = NonLU(self.decode_1(e))     # 1 pixel to 2
        dec_j = NonLU(self.decode_2(dec_d)) # 2 pixel to 4
        dec_j =       self.decode_3(dec_j)  # down to four features per jet. Nonlinearity is sinh, cosh activations below
        '''
        dec_q = NonLU(self.extract_q(e_out))                                                     # dec_q.shape = [batch_size, self.d, 3] 0123 0213 0312
        dec_d = NonLU(self.dijets_from_quadjets(dec_q))                                         # dec_d.shape = [batch_size, self.d, 6] 01 23 02 13 03 12
        dec_j = NonLU(self.jets_from_dijets(dec_d))                                             # dec_j.shape = [batch_size, self.d, 12]; dec_j is interpreted as jets 0 1 2 3 0 2 1 3 0 3 1 2
        
        dec_j = dec_j.view(-1, self.d, 3, 4)                                                    # last index is jet
        dec_j = dec_j.transpose(-1, -2)                                                         # last index is pairing history now 
        dec_j = dec_j.contiguous().view(-1, self.d * 4, 3)                                      # 32 numbers corresponding to each pairing: which means that you have 8 numbers corresponding to each jet in each pairing concatenated along the same dimension
                                                                                                # although this is not exact because now the information between jets is mixed, but thats the idea
        dec_j_logits = self.select_j(dec_j)
        dec_j_score = F.softmax(dec_j_logits, dim = -1)                                         # 1x3

        dec_j = torch.matmul(dec_j, dec_j_score.transpose(1, 2))                                # (32x3 · 3x1 = 32x1)
        dec_j = dec_j.view(-1, self.d, 4)                                                       # 8x4

        # conv kernel 1
        j_res = dec_j.clone()
        dec_j = NonLU(self.decode_j1(dec_j)) + j_res
        # j_res = dec_j.clone()
        # dec_j = NonLU(self.jets_res_2(dec_j))+j_res
        # j_res = dec_j.clone()
        # dec_j = NonLU(self.jets_res_3(dec_j))+j_res        
        # dec_j = self.expand_j(dec_j)
        # dec_j = NonLU(dec_j)
        dec_j = self.decode_j2(dec_j)                                                            # 4x4
        

        # apply the DeltaR correction (in inference) so that jets are separated at least deltaR = 0.4
        dec_j = deltaR_correction(dec_j) if self.correct_DeltaR and not self.training else dec_j

        
        Pt = dec_j[:,0:1].cosh()+39 # ensures pt is >=40 GeV
        Eta = dec_j[:,1:2]
        Phi = dec_j[:,2:3]
        # M  = dec_j[:,3:4].cosh()-1 # >=0, in our case it is always zero for the toy data. we could relax this for real data
        M = dec_j[:,3:4].cosh()-1

        rec_j = torch.cat((Pt, Eta, Phi, M), 1)
        if self.return_masses:
            rec_d, rec_dPxPyPzE = addFourVectors(   rec_j[:,:,(0,2,0,1,0,1)], 
                                                    rec_j[:,:,(1,3,2,3,3,2)])
            rec_q, rec_qPxPyPzE = addFourVectors(   rec_d[:,:,(0,2,4)],
                                                    rec_d[:,:,(1,3,5)])
            rec_m2j = rec_d[:, 3:4, :].clone()
            rec_m4j = rec_q[:, 3:4, :].clone()

        Px = Pt*Phi.cos()
        Py = Pt*Phi.sin()
        Pz = Pt*Eta.sinh()

        
        # E  = (Pt**2+Pz**2+M**2).sqrt()   # ensures E^2>=M^2
        E  = (Pt**2+Pz**2).sqrt() # ensures E^2>=0. In our case M is zero so let's not include it
        
        rec_jPxPyPzE = torch.cat((Px, Py, Pz, E), 1)

        # # Nonlinearity for final output four-vector components
        # rec_jPxPyPzE = torch.cat((dec_j[:,0:3,:].sinh(), dec_j[:,3:4,:].cosh()), dim=1)
        if self.return_masses:
            return rec_jPxPyPzE, rec_j, z, rec_m2j, rec_m4j
        else:
            return rec_jPxPyPzE, rec_j, z

class Basic_CNN_AE(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, permute_input_jet = False, phi_rotations = False, correct_DeltaR = False, return_masses = False, device = 'cpu'):
        super(Basic_CNN_AE, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.permute_input_jet = permute_input_jet
        self.phi_rotations = phi_rotations
        self.correct_DeltaR = correct_DeltaR
        self.return_masses = return_masses
        self.n_ghost_batches = 64
        

        self.name = f'Basic_CNN_AE_{self.d_bottleneck}'

        self.encoder = Basic_encoder(dimension = self.d, bottleneck_dim = self.d_bottleneck, permute_input_jet = self.permute_input_jet, phi_rotations = self.phi_rotations, return_masses = self.return_masses, n_ghost_batches = self.n_ghost_batches, device = self.device)
        self.decoder = Basic_decoder(dimension = self.d, bottleneck_dim = self.d_bottleneck, correct_DeltaR = self.correct_DeltaR, return_masses = self.return_masses, n_ghost_batches = self.n_ghost_batches, device = self.device)

    
    def set_mean_std(self, j):
        self.encoder.set_mean_std(j)
    
    def set_ghost_batches(self, n_ghost_batches):
        self.encoder.set_ghost_batches(n_ghost_batches)
        self.decoder.set_ghost_batches(n_ghost_batches)

    
    def forward(self, j):
        #
        # Encode
        #
        if self.return_masses:
            jPxPyPzE, j_rot, z, m2j, m4j = self.encoder(j)      
        else:
            jPxPyPzE, j_rot, z = self.encoder(j)   
        
        #
        # Decode
        #
        if self.return_masses:
            rec_jPxPyPzE, rec_j, z, rec_m2j, rec_m4j = self.decoder(z)      
        else:
            rec_jPxPyPzE, rec_j, z = self.decoder(z)   



        if self.return_masses:
            return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j, z, m2j, m4j, rec_m2j, rec_m4j
        else:
            return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j, z



class K_Fold(nn.Module):
    def __init__(self, networks, task = 'FvT'):
        super(K_Fold, self).__init__()
        self.networks = networks
        for network in self.networks:
            network.eval()
        self.task = task

    @torch.no_grad()
    def forward(self, j, e):

        if self.task == 'SvB' or self.task == 'FvT': # i.e. if task is classification
            c_logits = torch.zeros(j.shape[0], self.networks[0].n_classes)
            q_logits = torch.zeros(j.shape[0], 3)

            for offset, network in enumerate(self.networks):
                mask = (e==offset)
                c_logits[mask], q_logits[mask] = network(j[mask])

            # shift logits to have mean zero over quadjets/classes. Has no impact on output of softmax, just makes logits easier to interpret
            c_logits = c_logits - c_logits.mean(dim=-1, keepdim=True)
            q_logits = q_logits - q_logits.mean(dim=-1, keepdim=True)

            return c_logits, q_logits  
        elif self.task == 'dec':
            rec_j = torch.zeros(j.shape[0], 4, 4)
            z = torch.zeros(j.shape[0], self.networks[0].d_bottleneck, 1)
            for offset, network in enumerate(self.networks):
                mask = (e==offset)
                if network.return_masses:
                    _, _, _, rec_j[mask], z[mask], _, _, _, _ = network(j[mask])
                else:
                    _, _, _, rec_j[mask], z[mask] = network(j[mask])
            return rec_j, z # save only j in PtEtaPhiM representation
        elif self.task == 'gen':
            rec_j = torch.zeros(j.shape[0], 4, 4)
            for offset, network in enumerate(self.networks):
                mask = (e==offset)
                z_offset = j[mask]
                #z_sampled = GMM_sample(z_offset, max_nb_gaussians = 5, debug = True,  sample = 'fourTag_10x', density = True, offset = offset)
                z_sampled = KDE_sample(z_offset, debug = True, sample = 'fourTag_10x', density = True, offset = offset)
                #import sys; sys.exit()
                if network.return_masses:
                    _, rec_j[mask], _, _, _ = network(z_sampled)
                else:
                    _, rec_j[mask], _ = network(z_sampled)
            return rec_j # save only j in PtEtaPhiM representation
        
        else:
            pass

def KDE_sample(z, debug, **kwargs):
    dimension = z.shape[1]
    z = z.numpy()
    # Flatten the data to perform KDE
    flattened_z = z.reshape(-1, dimension)
    if debug:
        sample = kwargs.get('sample', 'fourTag_10x')
        offset = kwargs.get('offset', None)
        density = kwargs.get("density", True) # default True
        import matplotlib.pyplot as plt
        # create necessary things to plot
        # Determine the grid layout based on the number of features
        if dimension <= 4:
            num_rows = 2
            num_cols = 2
        elif dimension <= 6:
            num_rows = 2
            num_cols = 3
        elif dimension <= 9:
            num_rows = 3
            num_cols = 3 
        elif dimension <= 12:
            num_rows = 3
            num_cols = 4
        elif dimension <= 16:
            num_rows = 4
            num_cols = 4
        else:
            raise ValueError("The number of features is too high to display in a reasonable way.")
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20,8))
        axs = axs.flatten()
        if (dimension < num_rows * num_cols):
            for j in range(1, num_rows*num_cols - dimension + 1):
                axs[-j].axis('off')  # Hide any empty subplots
        h, bins = np.zeros_like(axs), np.zeros_like(axs)

    # Create the Kernel Density Estimation model with the Gaussian kernel
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde.fit(flattened_z)

    # Generate a new sample from the density-weighted distribution
    num_samples = z.shape[0]  # Number of samples to generate
    z_sampled = kde.sample(num_samples).astype(z.dtype)

    # Reshape the sampled data back to the original shape
    z_sampled = z_sampled.reshape(num_samples, dimension, 1)

    if debug:
        for d in range(dimension):
            h[d], bins[d], _ = axs[d].hist(z[:,d], density=density, color='black', bins=32, histtype = 'step', lw = 3, ls = 'solid', label = 'True $z$')
            axs[d].hist(z_sampled[:,d], bins = bins[d], color = "red", density = density, histtype = 'step', lw = 3, ls = 'dashed', label = "Sampled $z$")
            axs[d].set_title(f'Feature {d+1}', fontsize = 20)
            axs[d].set_xlabel("Value", fontsize = 20)
            axs[d].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 20)
            axs[d].minorticks_on()
            axs[d].tick_params(which = 'minor', axis = 'both', direction='in', length = 3)
            if not density:
                axs[d].set_ylabel(f"Counts / {(bins[d][1]-bins[d][0]):.1f}", fontsize = 20)
            else:
                axs[d].set_ylabel(f"Counts / {(bins[d][1]-bins[d][0]):.1f} (norm)", fontsize = 20)
            if d == 0:
                fig.legend(loc = 'center', bbox_to_anchor=(0.25, 0.5), ncol=3, fontsize = 20)
        fig.tight_layout()
        path = f"plots/redec/{sample}/"
        plots.mkpath(path)
        path_fig = f'{path}{sample}_zsampled_{dimension}_offset_{offset}.pdf' if offset is not None else f'{path}{sample}_zsampled_{dimension}.pdf'
        fig.savefig(path_fig)
        print(f'KDE plot saved to {path_fig}')
    return torch.tensor(z_sampled)

def GMM_sample(z, max_nb_gaussians = 2, debug = False, **kwargs):
    dimension = z.shape[1]
    z = z.numpy()
    z_sampled = np.zeros_like(z) # create the final sampled activations
    if debug:
        sample = kwargs.get('sample', 'fourTag_10x')
        offset = kwargs.get('offset', None)
        density = kwargs.get("density", True) # default True
        import matplotlib.pyplot as plt
        # create necessary things to plot
        # Determine the grid layout based on the number of features
        if dimension <= 4:
            num_rows = 2
            num_cols = 2
        elif dimension <= 6:
            num_rows = 2
            num_cols = 3
        elif dimension <= 9:
            num_rows = 3
            num_cols = 3 
        elif dimension <= 12:
            num_rows = 3
            num_cols = 4
        elif dimension <= 16:
            num_rows = 4
            num_cols = 4
        else:
            raise ValueError("The number of features is too high to display in a reasonable way.")
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20,8))
        axs = axs.flatten()
        if (dimension < num_rows * num_cols):
            for j in range(1, num_rows*num_cols - dimension + 1):
                axs[-j].axis('off')  # Hide any empty subplots
        h, bins = np.zeros_like(axs), np.zeros_like(axs)

    
    for d in range(dimension):
        min_bic = 0
        counter = 1
        gmms, fits, bics = [], [], []
        if offset is not None: print(f'\nOffset {offset}:')
        print(f'Running GMM with max. {max_nb_gaussians} gaussians for {d}-th feature')
        for i in range (max_nb_gaussians): # test the AIC/BIC metric between 1 and max_nb_gaussians components
            gmm = GMM(n_components = counter, random_state=10, covariance_type = 'full')
            gmms.append(gmm)
            fits.append(gmm.fit(z[:,d]))
            #labels = fit.predict(z[:,0])
            bic = gmm.bic(z[:,d])
            bics.append(bic)
            if bic < min_bic or min_bic == 0:
                min_bic = bic
                n_opt = counter
            counter = counter + 1
        # get optimal GMM model
        gmm_opt = gmms[n_opt - 1]
        # get optimal parameters
        means_opt = fits[n_opt - 1].means_
        covs_opt  = fits[n_opt - 1].covariances_
        weights_opt = fits[n_opt - 1].weights_

        if debug:
            h[d], bins[d], _ = axs[d].hist(z[:,d], density=density, color='black', bins=32, histtype = 'step', lw = 3, label = 'True $z$')
            density_factor = np.sum(h[d]*(bins[d][1:]-bins[d][:-1])) if not density else 1.
            x_ax = np.linspace(bins[d][0], bins[d][-1], 1000)
            y_axs = []
            for i in range(n_opt):
                y_axs.append(density_factor*norm.pdf(x_ax, float(means_opt[i][0]), np.sqrt(float(covs_opt[i][0][0])))*weights_opt[i]) # ith gaussian
                axs[d].plot(x_ax, y_axs[i], lw = 1.5)
                axs[d].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 20)
                axs[d].minorticks_on()
                axs[d].tick_params(which = 'minor', axis = 'both', direction='in', length = 3)
            axs[d].plot(x_ax, np.sum(y_axs, axis = 0), lw = 1.5, ls='dashed', label = "GMM estimated PDF")
            axs[d].set_title(f'Feature {d+1}; opt. comp. = {n_opt}', fontsize = 20)
            axs[d].set_xlabel("Value", fontsize = 20)
            if not density:
                axs[d].set_ylabel(f"Counts / {(bins[d][1]-bins[d][0]):.1f}", fontsize = 20)
            else:
                axs[d].set_ylabel(f"Counts / {(bins[d][1]-bins[d][0]):.1f} (norm)", fontsize = 20)


        # Sampling
        r_values = np.random.uniform(0, 1, len(z[:,d]))
        # Cumulative sum of weights to sample the identity of the gaussian
        weights_cum = np.cumsum(weights_opt)
        # Find the indices of the values in weights_cumulative that are immediately higher than 'r'
        gaussian_indices = np.searchsorted(weights_cum, r_values[:, np.newaxis], side='right')[:,0]
        # Use list comprehension to get the parameters for the corresponding Gaussian distributions
        mu = [float(means_opt[i][0]) for i in gaussian_indices]
        sigma = [np.sqrt(float(covs_opt[i][0][0])) for i in gaussian_indices]

        # Sample from the corresponding Gaussian distributions
        z_sampled[:,d,0] = np.random.normal(mu, sigma)
        if debug:
            axs[d].hist(z_sampled[:,d], bins = bins[d], color = "red", density = density, histtype = 'step', lw = 3, ls = 'solid', label = "Sampled $z$")
            if d == 0:
                fig.legend(loc='center', bbox_to_anchor=(0.25, 0.5), ncol=3, fontsize = 20)

    # layout
    fig.tight_layout()
    path = f"plots/redec/{sample}/"
    plots.mkpath(path)
    path_fig = f'{path}{sample}_zsampled_{dimension}_offset_{offset}.pdf' if offset is not None else f'{path}{sample}_zsampled_{dimension}.pdf'
    fig.savefig(path_fig)
    print(f'GMM plot saved to {path_fig}')
    
    return torch.tensor(z_sampled)

