train:
	python python/train.py --train --task dec --offset 0
	python python/train.py --train --task dec --offset 1
	python python/train.py --train --task dec --offset 2

writeActivations:
	 python python/train.py  --task dec --model "models/dec_fourTag_Basic_CNN_AE_6_offset_?_epoch_025.pkl"

svd:
	 python python/plotActivations.py --inputPkl activations/fourTag_z_6_epoch_025.pkl 

makeHists:
	 python python/analysis.py  -a

drawHists:
	python python/plotStudyLatentSpace.py --hists data/hists_activationStudy.pkl --plots plots/activationStudy
	tar -czf activationStudy.tar.gz plots/activationStudy
