from glob import glob
from coffea import util
import awkward as ak
import torch
import numpy as np
import uproot

sample = 'fourTag'
custom_selection = 'event.preselection'  # region on which you want to train

train_valid_modulus = 3
def coffea_to_tensor(event, device = 'cpu', decode = False, kfold=False):
    j = torch.FloatTensor( event['Jet',('pt','eta','phi','mass')].to_numpy().view(np.float32).reshape(-1, 4, 4) ) # [event,jet,feature]
    j = j.transpose(1,2).contiguous() # [event,feature,jet]
    e = torch.LongTensor( np.asarray(event['event'], dtype=np.uint8) )%train_valid_modulus
    if kfold:
        return j, e
    w = torch.FloatTensor( event['weight'].to_numpy().view(np.float32) )
    R  = 1*torch.LongTensor( event['SB'].to_numpy().view(np.uint8) )
    R += 2*torch.LongTensor( event['SR'].to_numpy().view(np.uint8) )
    if device != 'cpu':
        j, w, R, e = j.to(device), w.to(device), R.to(device), e.to(device)
    return j, w, R, e

def load(cfiles, selection=''):
    event_list = []
    for cfile in cfiles:
        print(cfile, selection)
        event = util.load(cfile)
        if selection:
            event = event[eval(selection)]
        event_list.append(event)
    return ak.concatenate(event_list)

def torch2ak(x):
    return ak.Array(ak.Array(x.cpu().detach().numpy()))

def jet2dict(jet):
    return {'pt': torch2ak(jet[:,:,0]), 'eta': torch2ak(jet[:,:,1]), 'phi': torch2ak(jet[:,:,2]), 'mass': torch2ak(jet[:,:,3])}

coffea_file = sorted(glob(f'data/{sample}_picoAOD*.coffea')) # file used for autoencoding

# Load data
event = load(coffea_file, selection=custom_selection)
j, w, R, e = coffea_to_tensor(event, device='cpu')

# store data
file = uproot.recreate(f'data/toy_data.root')
file['jet'] = jet2dict(j)

# symmetrize the data



