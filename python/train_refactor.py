'''
Imports and config
'''
from glob import glob
import sys, os, argparse, re
from copy import copy, deepcopy
from coffea import util
import awkward as ak
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import plots
import json
import itertools
import networks

device_def = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = networks.original.Basic_CNN_AE

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #this doesn't work, need to run `conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1` and then reactivate the conda environment


np.random.seed(0)
torch.manual_seed(0)#make training results repeatable 
plots.update_rcparams()
SCREEN_WIDTH = 100

'''
Labels for classifier
'''
class ClassInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = ClassInfo(abbreviation='d4', name= 'FourTag Data', color='red')
d3 = ClassInfo(abbreviation='d3', name='ThreeTag Data', color='orange')


S = ClassInfo(abbreviation='S', name='Signal Data', color='red')
BG = ClassInfo(abbreviation='BG', name='Background Data', color='orange')

FvT_classes = [d4, d3]
SvB_classes = [BG, S]
for i, c in enumerate(FvT_classes): c.index = i
for i, c in enumerate(SvB_classes): c.index = i


'''
Load coffea files and convert to tensor
'''
def load(cfiles, selection=''):
    event_list = []
    for cfile in cfiles:
        print(cfile, selection)
        event = util.load(cfile)
        if selection:
            event = event[eval(selection)]
        event_list.append(event)
    return ak.concatenate(event_list)

# convert coffea objects in to pytorch tensors
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

    if decode == False:
        y = torch.LongTensor( np.asarray(event['class'], dtype=np.uint8) )
        y = y.to(device)
        dataset = TensorDataset(j, y, w, R, e)
    else:
        y = None
        dataset = TensorDataset(j, w, R, e)
    return dataset


'''
Architecture hyperparameters
'''
bottleneck_dim = 6
permutations = list(itertools.permutations([0,1,2,3]))
loss_pt = False                 # whether to add pt to the loss of PxPyPzE
permute_input_jet = False       # whether to randomly permute the positions of input jets
rotate_phi = False              # whether to remove eta-phi invariances in the encoding
correct_DeltaR = False          # whether to correct DeltaR (in inference)

sample = 'fourTag'

testing = False
plot_training_progress = True  # plot training progress
if testing:
    num_epochs = 1
    plot_every = 1
else:
    num_epochs = 25
    plot_every = 5

lr_init  = 0.01
lr_scale = 0.25
bs_scale = 2
bs_milestones =     [1, 3, 6, 10]
lr_milestones =     [10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 100, 200, 250, 300, 400, 450]
#gb_milestones =     [5, 7, 10, 15, 20, 30, 40, 50, 100, 150, 200]

train_batch_size = 2**10
infer_batch_size = 2**14
max_train_batch_size = train_batch_size*64
num_workers=8


############ default hyperparameters for FvT ############
# num_epochs = 20                                       #
# lr_init  = 0.01                                       #
# lr_scale = 0.25                                       #
# bs_scale = 2                                          #
#                                                       #
# bs_milestones = [1,3,6,10]                            #
# lr_milestones = [15,16,17,18,19,20,21,22,23,24]       #
#                                                       #
# train_batch_size = 2**10                              #
# infer_batch_size = 2**14                              #
# max_train_batch_size = train_batch_size*64            #
#                                                       #
# num_workers=8                                         #
#                                                       #
#########################################################

train_loss_tosave = [] # vectors to store loss during training for plotting afterwards (to be implemented)
val_loss_tosave = []



'''
Batch loaders class for inference and training
'''
class Loader_Result:
    def __init__(self, model, dataset, n_classes=2, train=False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.dataset = dataset
        self.infer_loader = DataLoader(dataset=dataset, batch_size=infer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        self.train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True) if train else None
        self.device = device
        self.n = len(dataset)
        self.w = dataset.tensors[1] if model.task == 'dec' else dataset.tensors[2]
        self.w_sum = self.w.sum()
        self.cross_entropy = torch.zeros(self.n)
        self.decoding_loss = torch.zeros(self.n)
        self.j_ = torch.zeros(self.n, 4, 4) # store vectors for plotting at the end of epoch
        self.rec_j_ = torch.zeros(self.n, 4, 4)
        self.z_ = torch.zeros(self.n, model.network.d_bottleneck, 1) # activations in the embedded space
        self.m2j_ = torch.zeros(self.n, 1, 6)
        self.m4j_ = torch.zeros(self.n, 1, 3)
        self.rec_m2j_ = torch.zeros(self.n, 1, 6)
        self.rec_m4j_ = torch.zeros(self.n, 1, 3)
        self.component_weights = torch.tensor([1,1,0.3,0.3]).to(device).view(1,4,1) # adapt magnitude of PxPy versus PzE
        self.n_done = 0
        self.loaded_die_loss = model.loaded_die_loss if hasattr(model, 'loaded_die_loss') else None
        self.loss_estimate = 1.0
        self.history = {'loss': []}
        self.task = model.task
        self.train = train

    def eval(self):
        self.n_done = 0

    def loss_fn(self, jPxPyPzE, rec_jPxPyPzE, j, rec_j, phi_rotations, reduction = 'mean', ignore_perm = True):
        
        if ignore_perm: # do not search for the minimum loss of jets combination
            mse_loss_batch = F.mse_loss(jPxPyPzE*self.component_weights, rec_jPxPyPzE*self.component_weights, reduction = 'none').sum(dim = (1,2)) # sum along jets and features errors
        
        else:
            # compute all the possible 24 reconstruction losses between 0123 in output with 0123 in input
            jPxPyPzE = jPxPyPzE.unsqueeze(3).repeat(1, 1, 1, 24)                    # repeat jPxPyPzE (copy) along the 24-sized permutations dimension
            j = j.unsqueeze(3).repeat(1, 1, 1, 24)                                  # repeat j (copy) along the 24-sized permutations dimension
            
            
            rec_j = torch.swapaxes(rec_j[:, :, permutations], 2, 3)
            rec_jPxPyPzE = torch.swapaxes(rec_jPxPyPzE[:, :, permutations], 2, 3)
        
            self.component_weights = self.component_weights.unsqueeze(2)                                                                                    # add a component for the permutations dimension in the case we are not ignoring perms
            mse_loss_batch_perms = F.mse_loss(jPxPyPzE*self.component_weights, rec_jPxPyPzE*self.component_weights, reduction = 'none').sum(dim = (1,2))    # sum along jets and features errors
            mse_loss_batch, perm_index = mse_loss_batch_perms.min(dim = 1)                                                                                  # dimension 0 is batch and dimension 1 is permutation
            rec_jPxPyPzE = rec_jPxPyPzE[torch.arange(rec_jPxPyPzE.shape[0]), :, :, perm_index]                                                              # re-obtain the [batch_number, 4, 4] tensor with the jet with minimum loss

            # loss on m2j (not employed)
            '''
            d, dPxPyPzE = networks.addFourVectors(0, 0, jPxPyPzE[:,:,(0,2,0,1,0,1),0], jPxPyPzE[:,:,(1,3,2,3,3,2),0])
            rec_d, rec_dPxPyPzE = networks.addFourVectors(0, 0, rec_jPxPyPzE[:,:,(0,2,0,1,0,1)], rec_jPxPyPzE[:,:,(1,3,2,3,3,2)])
            mass_loss = F.mse_loss(d[:, 3:4,:], rec_d[:, 3:4,:], reduction = 'none').sum(dim=(1,2))
            '''             
            # perm_index is a number 0-23 indicating the best permutation: 0123, 0132, 0213, ..., 3210
            



        # taking the sqrt so that [loss] = GeV
        loss_batch = (mse_loss_batch).sqrt() # loss_batch.shape = [batch_size]

        return loss_batch, rec_jPxPyPzE

    def infer_batch_AE(self, jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j, phi_rotations, epoch, plot_every): # expecting same sized j and rec_j
        n_batch = rec_jPxPyPzE.shape[0]
        loss_batch, rec_jPxPyPzE = self.loss_fn(jPxPyPzE, rec_jPxPyPzE, j, rec_j, phi_rotations = phi_rotations)

        if epoch % plot_every == 0 and self.train: # way of ensure we are saving jets and z from training dataset
            self.j_[self.n_done : self.n_done + n_batch] = jPxPyPzE
            self.rec_j_[self.n_done : self.n_done + n_batch] = rec_jPxPyPzE
            self.z_[self.n_done : self.n_done + n_batch] = z
            self.m2j_[self.n_done : self.n_done + n_batch] = m2j
            self.m4j_[self.n_done : self.n_done + n_batch] = m4j
            self.rec_m2j_[self.n_done : self.n_done + n_batch] = rec_m2j
            self.rec_m4j_[self.n_done : self.n_done + n_batch] = rec_m4j

        self.decoding_loss[self.n_done : self.n_done + n_batch] = loss_batch
        self.n_done += n_batch
    
    def infer_done_AE(self):
        self.loss = (self.w * self.decoding_loss).sum() / self.w_sum # 
        self.history['loss'].append(copy(self.loss))
        train_loss_tosave.append(self.loss.item()) if self.train else val_loss_tosave.append(self.loss.item()) # save loss to plot later
        self.n_done = 0

    def train_batch_AE(self, jPxPyPzE, rec_jPxPyPzE, j, rec_j, w, phi_rotations): # expecting same sized j and rec_j
        loss_batch, _ = self.loss_fn(jPxPyPzE, rec_jPxPyPzE, j, rec_j, phi_rotations=phi_rotations) # here rec_jPxPyPzE is not used; we only plot during inference of train dataset
        loss = (w * loss_batch).sum() / w.sum() # multiply weight for all the jet features and recover the original shape of the features 
        loss.backward()
        self.loss_estimate = self.loss_estimate * .02 + 0.98*loss.item()


'''
Model used for autoencoding
'''
class Train_AE:
    def __init__(self, train_valid_offset=0, task='dec', model_file='', sample='', generate_synthetic_dataset=False,
                 network=networks.benchmark_models.original.Basic_CNN_AE, decoder=networks.benchmark_models.original.Basic_decoder, device = device_def):
        self.train_valid_offset = train_valid_offset
        self.task = task
        self.sample = sample
        self.generate_synthetic_dataset = generate_synthetic_dataset
        self.device = device
        self.return_masses = True # whether to return masses from the Input_Embed; this is used by the class member function K_fold
        self.network = network(dimension = 16, bottleneck_dim = bottleneck_dim, permute_input_jet = permute_input_jet, phi_rotations = rotate_phi, correct_DeltaR = correct_DeltaR, return_masses = self.return_masses) if not self.generate_synthetic_dataset else decoder(dimension = 16, bottleneck_dim = bottleneck_dim, correct_DeltaR = correct_DeltaR, return_masses = self.return_masses, n_ghost_batches = 64)
        self.network.to(self.device)
        n_trainable_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f'Network has {n_trainable_parameters} trainable parameters')
        self.epoch = 0
        self.lr_current = lr_init
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr_init, amsgrad=False)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma=lr_scale, last_epoch=-1)
        self.lr_change = []
        self.bs_change = []
        self.n_done = 0

        if model_file:
            print(f'Load {model_file}')
            model_name_parts = re.split(r'[/_]', model_file)
            self.task = model_name_parts[1] if not self.generate_synthetic_dataset else "gen"
            print(f'The task of the model is {self.task}')
            self.train_valid_offset = model_name_parts[model_name_parts.index('offset')+1]
            self.model_pkl = model_file
            self.model_dict = torch.load(self.model_pkl)
            if self.generate_synthetic_dataset: # separate this as we do not need the optimizer
                self.network.load_state_dict(self.model_dict['decoder'])
                self.network.eval()
            else:
                self.network.load_state_dict(self.model_dict['model'])
                self.optimizer.load_state_dict(self.model_dict['optimizer'])
            self.epoch = self.model_dict['epoch']
        
    def make_loaders(self, event):
        '''
        Split into training and validation, define datasets and format into pytorch tensors, declare train_result and valid_result
        '''
        # Split into training and validation sets and format into pytorch tensors
        valid = event.event%train_valid_modulus == self.train_valid_offset
        train = ~valid

        dataset_train = coffea_to_tensor(event[train], device='cpu', decode=True)
        dataset_valid = coffea_to_tensor(event[valid], device='cpu', decode=True)

        self.train_result = Loader_Result(self, dataset_train, device=self.device, train=True)
        self.valid_result = Loader_Result(self, dataset_valid, device=self.device)

        print(f'{self.train_result.n:,} training samples split into {len(self.train_result.train_loader):,} batches of {train_batch_size:,}')
        print(f'{self.valid_result.n:,} validation samples split into {len(self.valid_result.infer_loader):,} batches of {infer_batch_size:,}')
    
    @torch.no_grad()
    def inference(self, result):
        '''
        Reconstruct the jets in inference mode and compute the loss
        '''
        self.network.eval()

        # nb, event jets vector, weight, region, event number
        for batch_number, (j, w, R, e) in enumerate(result.infer_loader):
            j, w, R, e = j.to(self.device), w.to(self.device), R.to(self.device), e.to(self.device)
            jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j = self.network(j)

            
            result.infer_batch_AE(jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j, self.network.phi_rotations, epoch = self.epoch, plot_every = plot_every)
            
            percent = float(batch_number+1)*100/len(result.infer_loader)
            sys.stdout.write(f'\rEvaluating {percent:3.0f}%')
            sys.stdout.flush()

        result.infer_done_AE()

    def train_inference(self):
        self.inference(self.train_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'\n\nEpoch {self.epoch:>2} | Training   | Loss {self.train_result.loss:1.5} GeV')

    def valid_inference(self):
        self.inference(self.valid_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'         | Validation | Loss {self.valid_result.loss:1.5} GeV')

    def train(self, result=None):
        if result is None: result = self.train_result
        self.network.train() # inherited from nn.Module()

        for batch_number, (j, w, R, e) in enumerate(result.train_loader):
            self.optimizer.zero_grad()
            j, w, R, e = j.to(self.device), w.to(self.device), R.to(self.device), e.to(self.device)
            jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j = self.network(j)

            result.train_batch_AE(jPxPyPzE, rec_jPxPyPzE, j, rec_j, w, self.network.phi_rotations)

            percent = float(batch_number+1)*100/len(result.train_loader)
            sys.stdout.write(f'\rTraining {percent:3.0f}% >>> Loss Estimate {result.loss_estimate:1.5f} GeV')
            sys.stdout.flush()
            self.optimizer.step()
        result.loss_estimate = 0

    def increment_train_loader(self, new_batch_size = None):
        current_batch_size = self.train_result.train_loader.batch_size
        if new_batch_size is None: new_batch_size = current_batch_size * bs_scale
        if new_batch_size == current_batch_size: return
        print(f'Change training batch size: {current_batch_size} -> {new_batch_size} ({self.train_result.n//new_batch_size} batches)')
        del self.train_result.train_loader
        self.train_result.train_loader = DataLoader(dataset=self.train_result.dataset, batch_size=new_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        self.bs_change.append(self.epoch)

    def run_epoch(self, plot_training_progress):
        
        self.epoch += 1
        self.train()
        self.train_inference()        
        self.valid_inference()
        self.scheduler.step()
        if plot_training_progress and self.epoch % plot_every == 0:
            plots.plot_training_residuals_PxPyPzEm2jm4jPtm2jvsm4j(jPxPyPzE=self.train_result.j_, rec_jPxPyPzE=self.train_result.rec_j_, phi_rot=self.network.phi_rotations, m2j=self.train_result.m2j_, m4j=self.train_result.m4j_, rec_m2j=self.train_result.rec_m2j_, rec_m4j=self.train_result.rec_m4j_, offset=self.train_valid_offset, epoch=self.epoch, sample=self.sample, network_name=self.network.name) # plot training residuals for pt, eta, phi
            plots.plot_PxPyPzEPtm2jm4j(jPxPyPzE=self.train_result.j_, rec_jPxPyPzE=self.train_result.rec_j_, phi_rot=self.network.phi_rotations, m2j=self.train_result.m2j_, m4j=self.train_result.m4j_, rec_m2j=self.train_result.rec_m2j_, rec_m4j=self.train_result.rec_m4j_, offset=self.train_valid_offset, epoch=self.epoch, sample=self.sample, network_name=self.network.name)
            randomly_plotted_event_number = plots.plot_etaPhi_plane(jPxPyPzE = self.train_result.j_, rec_jPxPyPzE = self.train_result.rec_j_, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)
            plots.plot_PxPy_plane(jPxPyPzE = self.train_result.j_, rec_jPxPyPzE = self.train_result.rec_j_, event_number = randomly_plotted_event_number, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)
            plots.plot_activations_embedded_space(z=self.train_result.z_, offset=self.train_valid_offset, epoch=self.epoch, sample=self.sample, network_name=self.network.name)


        if (self.epoch in bs_milestones or self.epoch in lr_milestones) and self.network.n_ghost_batches:
            gb_decay = 4 #2 if self.epoch in bs_mile
            print(f'set_ghost_batches({self.network.n_ghost_batches//gb_decay})')
            self.network.set_ghost_batches(self.network.n_ghost_batches//gb_decay)
        if self.epoch in bs_milestones:
            self.increment_train_loader()
        if self.epoch in lr_milestones:
            print(f'Decay learning rate: {self.lr_current} -> {self.lr_current*lr_scale}')
            self.lr_current *= lr_scale
            self.lr_change.append(self.epoch)

    def run_training(self, plot_training_progress = False):
        min_val_loss = 1e20
        val_loss_increase_counter = 0
        self.network.set_mean_std(self.train_result.dataset.tensors[0].to(self.device))
        # self.train_inference()
        # self.valid_inference()


        
        for _ in range(num_epochs):
            self.run_epoch(plot_training_progress = plot_training_progress)
            
            if val_loss_tosave[-1] < min_val_loss and _ > 0:
                self.del_prev_model()
                self.save_model()
                min_val_loss = val_loss_tosave[-1]
                val_loss_increase_counter = 0
            else:
                val_loss_increase_counter += 1
            
            if val_loss_increase_counter == 20:
                val_loss_increase_counter = 0
                print(f'Val loss has not decreased in 20 epoch. Decay learning rate: {self.lr_current} -> {self.lr_current*lr_scale}')
                self.lr_current *= lr_scale
                self.lr_change.append(self.epoch)
            #if val_loss_increase_counter == 100: #or min_val_loss < 1.:
                #break
            if self.epoch == num_epochs:
                self.save_model()

        loss_tosave = {"train" : train_loss_tosave, "val" : val_loss_tosave}
        with open("loss.txt", 'w') as file:
            file.write(json.dumps(loss_tosave))
        plots.plot_loss(loss_tosave, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)        
    
    def save_model(self):
        self.history = {'train': self.train_result.history,
                        'valid': self.valid_result.history}
        self.model_dict = {'model': deepcopy(self.network.state_dict()),
                           #'decoder': deepcopy(self.network.decoder.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict()),
                           'epoch': self.epoch,
                           'history': copy(self.history)}
        self.model_pkl = f'models/{self.task}_{self.sample}_{self.network.name}_offset_{self.train_valid_offset}_epoch_{self.epoch:03d}.pkl'
        print(f'Saved model as: {self.model_pkl} with a validation loss {val_loss_tosave[-1]:.2e}')
        torch.save(self.model_dict, self.model_pkl)
    
    def del_prev_model(self):
        self.prev_models = glob(f'models/{self.task}_{self.sample}_{self.network.name}_offset_{self.train_valid_offset}_epoch_*.pkl')
        for model in self.prev_models:
            os.remove(model)




'''
Arguments for execution
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default=False, action='store_true', help='Run model training')
    parser.add_argument('-tk', '--task', default='FvT', type = str, help='Type of classifier (FvT or SvB) to run')
    parser.add_argument('-o', '--offset', default=0, type=int, help='k-folding offset for split between training/validation sets')
    parser.add_argument('-m', '--model', default='', type=str, help='Load these models (* wildcard for offsets)')
    parser.add_argument('-g', '--generate', default=False, action='store_true', help='To be passed with --model and the specific models to generate data')
    args = parser.parse_args()

    
    custom_selection = 'event.preselection' # region on which you want to train


    '''
    Training
    '''
    if args.train:
        if args.task:
            task = args.task
        else:
            # demand the specification of --task if --train to avoid conflicts
            sys.exit("Task not specified. Use FvT, SvB or dec. Exiting...")

        classes = FvT_classes if task == 'FvT' else SvB_classes if task == 'SvB' else None

        # task is autoencoding
        if task == 'dec':
            coffea_file = sorted(glob(f'data/{sample}_picoAOD*.coffea')) # file used for autoencoding
            
            # Load data
            event = load(coffea_file, selection = custom_selection)
            print(event)

            # Load model and run training
            model_args = {  'task': task,
                            'train_valid_offset': args.offset}
            t= Train_AE(sample=sample, network=network, **model_args)
            t.make_loaders(event)
            t.run_training(plot_training_progress = plot_training_progress)


    
    if not args.train and not args.model and not args.generate:
        sys.exit("No --train nor --model specified. Script is not training nor precomputing friend TTrees. Exiting...")
