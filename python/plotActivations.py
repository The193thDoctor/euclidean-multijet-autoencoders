import torch
import os, argparse
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
coralgreen = ["#117a65", "#138D75"]
reddish = ["#c82929", "#e42f2f"]
orangish = ["#d78939", "#e48a2f"]

import matplotlib.cm as cm
import numpy as np
import pickle 

# style
def update_rcparams():
    plt.rcParams.update({
        "figure.figsize": [40, 20],
        "font.weight": "bold",
        'figure.titlesize': 50,
        'axes.titlesize': 50, # changes the axes titles (figure title when you have only one)
        'figure.titleweight': 'bold',
        "text.usetex": True,
        "font.family": "serif",
        
        'legend.fontsize': 30,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 8,
        'xtick.major.width': 1.,
        "ytick.major.size": 8,
        'ytick.major.width': 1.,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 4,
        'xtick.minor.width': 0.8,
        "ytick.minor.size": 4,
        'ytick.minor.width': 0.8,
        'axes.labelpad': 5.0,
        #"xtick.major.pad": 7,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "font.size": 40, # affects axes title size
        "grid.color": "0.5",
        #"grid.linestyle": "-",
        #"grid.linewidth": 5,
        "lines.linewidth": 5,
        #"lines.color": "g",
        })


def mkpath(path, debug=False):
    if os.path.exists(path) and debug:
        print("#",path,"already exists")
        return
        
    thisDir = ''
    for d in path.split('/'):
        thisDir = thisDir+d+"/"
        try:
            os.mkdir(thisDir)
        except FileExistsError:
            if debug: print(f'# {thisDir} already exists')


def plotProjections(thisActivations, thisOffset, suffix = "", pca=None):

        num_rows = 2
        num_cols = 3

        fig, axs = plt.subplots(num_rows, num_cols)
        #fig.suptitle(f"Histograms of {dimension} Different Features", fontsize=16)
        axs = axs.flatten()
    

        # Plot histograms for each feature
        for i in range(dimension):
            ax = axs[i]
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if not pca is None:
                h, bins, _ = ax.hist(thisActivations.dot(pca[i].reshape(-1,1)), bins = 32, alpha=1, color='blue', density = True, histtype='stepfilled', linewidth = plt.rcParams["lines.linewidth"], edgecolor = coralgreen[0], facecolor = coralgreen[1])
            else:
                h, bins, _ = ax.hist(thisActivations[:, i], bins = 32, alpha=1, color='blue', density = True, histtype='stepfilled', linewidth = plt.rcParams["lines.linewidth"], edgecolor = coralgreen[0], facecolor = coralgreen[1])
            ax.set_title(f"Feature {i+1}")
            ax.set_xlabel("Value")
            ax.set_ylabel(f"Counts / {(bins[1]-bins[0]):.1f}")
    

    
        # layout and save 
        #fig.suptitle(f'Epoch {epoch}')
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace = 1.)
        sample = "fourTag"
        network_name = "Basic_CNN_AE_6"
        path = f"plots/redec/{sample}/"
        mkpath(path)
        fig.savefig(f'{path}{sample}_activations_{network_name}_offset_{offset}{suffix}.pdf')
        print(f'Activations saved to {path}')
        plt.close()


def plotAllProjections(allActivations, suffix = "", pcas=None):

        num_rows = 2
        num_cols = 3

        fig, axs = plt.subplots(num_rows, num_cols)
        #fig.suptitle(f"Histograms of {dimension} Different Features", fontsize=16)
        axs = axs.flatten()
    

        # Plot histograms for each feature
        for i in range(dimension):
            ax = axs[i]
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if pcas:
                #print(pcas[0])
                h, bins, _ = ax.hist(allActivations[0].dot(pcas[0][i].reshape(-1,1)), bins = 32, alpha=1, color='blue', density = True, histtype='step', linewidth = plt.rcParams["lines.linewidth"], edgecolor = coralgreen[0], facecolor = coralgreen[1])
                h, bins, _ = ax.hist(allActivations[1].dot(pcas[1][i].reshape(-1,1)), bins = 32, alpha=1, color='blue', density = True, histtype='step', linewidth = plt.rcParams["lines.linewidth"], edgecolor = reddish[0], facecolor = reddish[1])
                h, bins, _ = ax.hist(allActivations[2].dot(pcas[2][i].reshape(-1,1)), bins = 32, alpha=1, color='blue', density = True, histtype='step', linewidth = plt.rcParams["lines.linewidth"], edgecolor = orangish[0], facecolor = orangish[1])
            else:
                h, bins, _ = ax.hist(allActivations[0][:, i], bins = 32, alpha=1, color='blue', density = True, histtype='step', linewidth = plt.rcParams["lines.linewidth"], edgecolor = coralgreen[0], facecolor = coralgreen[1])
                h, bins, _ = ax.hist(allActivations[1][:, i], bins = 32, alpha=1, color='blue', density = True, histtype='step', linewidth = plt.rcParams["lines.linewidth"], edgecolor = reddish[0], facecolor = reddish[1])
                h, bins, _ = ax.hist(allActivations[2][:, i], bins = 32, alpha=1, color='blue', density = True, histtype='step', linewidth = plt.rcParams["lines.linewidth"], edgecolor = orangish[0], facecolor = orangish[1])
            ax.set_title(f"Feature {i+1}")
            ax.set_xlabel("Value")
            ax.set_ylabel(f"Counts / {(bins[1]-bins[0]):.1f}")
    

    
        # layout and save 
        #fig.suptitle(f'Epoch {epoch}')
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace = 1.)
        sample = "fourTag"
        network_name = "Basic_CNN_AE_6"
        path = f"plots/redec/{sample}/"
        mkpath(path)
        fig.savefig(f'{path}{sample}_activations_{network_name}{suffix}.pdf')
        print(f'Activations saved to {path}')
        plt.close()


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPkl', default='activations/fourTag_z_6_epoch_025.pkl', help='File containing hists to be plotted')
    parser.add_argument('--output', default='activationPlots', help='base directory to save plots')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    update_rcparams()
    

    
    mkpath(args.output)
    print("Loading",args.inputPkl)
    with open(args.inputPkl, 'rb') as hfile:
        hdict = torch.load(hfile)
        activations = hdict["activations"].numpy()
        print("initial activations",activations.shape)
        offsets = hdict["offsets"].numpy()

        print(f"activations{activations}")
        activationsTransFormed = activations.reshape(activations.shape[0],activations.shape[1])
        print(f"activationsTransFormed{activationsTransFormed}")

        if args.debug:

            print(activations, type(activations))
            print(activations.shape)
            #print(activations.numpy())
            #activations = activations.numpy()

            print(offsets, type(offsets))

        activationsByOffSet = []
        activationsByOffSet_centered = []
        pcas = []
        for offset in [0,1,2]:

            if args.debug: print(offsets == offset)
            thisActivations = activations[offsets == offset]
            shape = thisActivations.shape
            thisActivations = thisActivations.reshape(shape[0],shape[1])
            activationsByOffSet.append(thisActivations)
            if args.debug: print(thisActivations.shape)
            dimension = thisActivations.shape[1]
            if args.debug:
                print(dimension)
                print(thisActivations[0])

            # 
            # plots
            #
            plotProjections(thisActivations, offset)

            #
            # Center Data
            #
            if args.debug: print(thisActivations.mean(axis=0))
            thisActivations_centered = thisActivations - thisActivations.mean(axis=0)
            activationsByOffSet_centered.append(thisActivations_centered)
            plotProjections(thisActivations_centered, offset,"_centered")
    

            #
            #  Do the SVD
            #
            print("acitivations shape",thisActivations_centered.shape)
            #print(thisActivations_centered)
            #print(thisActivations_centered[0])

            nForSVD = 10000
            #nForSVD = 6
            thisActivations_forSVD = thisActivations_centered[0:nForSVD]
            print("acitivations for SvD shape",thisActivations_forSVD.shape)            
            U, s, Vt = np.linalg.svd(thisActivations_forSVD)
            
            print(s)
            print(Vt.shape)
            c0 = Vt.T[:, 0]
            c1 = Vt.T[:, 1]
            c2 = Vt.T[:, 2]
            c3 = Vt.T[:, 3]
            c4 = Vt.T[:, 4]
            c5 = Vt.T[:, 5]
            if args.debug:
                print(c0)
                print(c0.shape)
                print(c0.reshape(-1,1))

            thisPCA = [c0,c1,c2,c3,c4,c5]
            pcas.append(thisPCA)
            thisPCA = np.array(thisPCA)
            if args.debug: 
                print(activationsTransFormed.shape)
                print (thisPCA.shape)
            tmp = np.matmul(thisActivations_centered,np.transpose(thisPCA))
            if args.debug: print (f"tmp.shape{tmp.shape}")
            activationsTransFormed[offsets == offset] = tmp
            plotProjections(thisActivations_centered, offset,"_pca", pca=thisPCA)

            if args.debug: 
                print("thisActivations_centered",thisActivations_centered)
                print("PCA0",thisActivations_centered.dot(thisPCA[0].reshape(-1,1)))
                print(f"tmp{tmp}")
                
                print("PCA Axis",thisPCA[0].reshape(-1,1))
                print("thisPCA",thisPCA)
            

        #print(f"activationsByOffSet_centered{activationsByOffSet_centered}")
        #print(f"activationsTransFormed{activationsTransFormed}")
        plotAllProjections(activationsByOffSet)
        plotAllProjections(activationsByOffSet_centered, suffix="_centered")
        plotAllProjections(activationsByOffSet_centered, pcas=pcas, suffix="_pca")

        with open('testFile.pkl', 'wb') as f:  # open a text file
            pickle.dump({"activationsTransformed" : activationsTransFormed}, f) 

#data_centered.dot(c1.reshape(-1,1))
