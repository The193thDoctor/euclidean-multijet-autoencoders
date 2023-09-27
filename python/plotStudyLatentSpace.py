import pickle, os, argparse
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networks
import matplotlib.cm as cm



from plots import sample2D, mkpath, update_rcparams




def sample1D(hdict, sample, var, cut='preselection', region='SB', name='', xlim=[], plotsdir='plots'):
    fig = plt.figure(figsize=(12, 12))

    try:
        h = hdict['hists'][var][f'data/{sample}_picoAOD.root', cut, region, ...]
    except KeyError:
        print(f'Could not find hist hdict["hists"][{var}]["data/{sample}_picoAOD.root", {cut}, {region}, ...]')
        return
    
    h.plot(histtype="fill",color="yellow")
    h.plot(histtype="step",color="k")
    
    axes = fig.get_axes()
    if xlim:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)

    outdir = f'{plotsdir}/{cut}/{region}'
    mkpath(outdir)
    if name:
        name = f'{outdir}/{name}.pdf'
    else:
        name = f'{outdir}/{sample}_{var}.pdf'
    print(name)
    fig.savefig(name)
    plt.close()
    dr






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hists', default='data/hists_normalized.pkl', help='File containing hists to be plotted')
    parser.add_argument('--plots', default='plots', help='base directory to save plots')
    args = parser.parse_args()

    with open(args.hists, 'rb') as hfile:
        hdict = pickle.load(hfile)


    update_rcparams()

    for cut in ['preselection']:
        for region in ['inclusive', 'diJetMass', 'SB', 'SR',"posZ0","negZ0"]:
            sample1D(hdict, "fourTag", "m4j",     cut=cut, region=region, plotsdir=args.plots)
            sample1D(hdict, "fourTag", "FvT_rw",     cut=cut, region=region, plotsdir=args.plots)

            for iZ in range(6):
                iStrZ = str(iZ)
                sample1D(hdict, "fourTag", 'z'+iStrZ,     cut=cut, region=region, plotsdir=args.plots)

            for iJ in range(4):
                iStrJ = str(iJ)
                jetVars = ["pt","eta","phi"]
                for v in jetVars:
                    sample1D(hdict, "fourTag", v+'_jet'+iStrJ,     cut=cut, region=region, plotsdir=args.plots)

            for dJ in ["lead","subl"]:
                for v in ["pt","eta","phi","mass","dr"]:
                    sample1D(hdict, "fourTag", v+'_dijet'+dJ,     cut=cut, region=region, plotsdir=args.plots)

            for v in ["pt","eta","phi","mass","dr"]:
                sample1D(hdict, "fourTag", v+'_quadjet',     cut=cut, region=region, plotsdir=args.plots)

            sample1D(hdict, "fourTag", "dPhi_jet0_jet1",     cut=cut, region=region, plotsdir=args.plots)



            xlim = [40,200] if region!='inclusive' else []
            ylim = [40,200] if region!='inclusive' else []
            
            for sample in ['fourTag']:
                sample2D(hdict, sample, 'lead_st_m2j_subl_st_m2j', cut=cut, region=region, xlim=xlim, ylim=ylim, plotsdir=args.plots, figsize=(20,20))

                sample2D(hdict, sample, 'lead_st_dr_subl_st_dr', cut=cut, region=region, plotsdir=args.plots, figsize=(20,20))
                sample2D(hdict, sample, 'lead_jetPhi_subl_jetPhi', cut=cut, region=region, plotsdir=args.plots, figsize=(20,20))


