#!/bin/bash
#SBATCH -p TWIG
#SBATCH -N 1
#SBATCH --gpus=a100-40

#conda init bash
__conda_setup="$('/hildafs/projects/phy210037p/ltang2/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/hildafs/projects/phy210037p/ltang2/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/hildafs/projects/phy210037p/ltang2/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/hildafs/projects/phy210037p/ltang2/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate coffea_torch

echo "conda init success"

#make plots
python python/analysis.py
python python/normalize.py # compute norm for threeTag to fourTag in SB
python python/analysis.py --normalize --save # run again, this time normalizing threeTag to fourTag. Save coffea files with derived quantities for use in classifier training
python python/plots.py
