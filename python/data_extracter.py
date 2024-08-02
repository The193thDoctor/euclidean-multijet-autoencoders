from glob import glob
from coffea import util
import awkward as ak

sample = 'fourTag'
custom_selection = 'event.preselection'  # region on which you want to train

def load(cfiles, selection=''):
    event_list = []
    for cfile in cfiles:
        print(cfile, selection)
        event = util.load(cfile)
        if selection:
            event = event[eval(selection)]
        event_list.append(event)
    return ak.concatenate(event_list)

coffea_file = sorted(glob(f'data/{sample}_picoAOD*.coffea')) # file used for autoencoding

# Load data
event = load(coffea_file, selection=custom_selection)

print(type(event))
print(event.fields)
print(event.layout)
print(event)