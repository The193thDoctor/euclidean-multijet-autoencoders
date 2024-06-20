#!/bin/bash
#SBATCH -p TWIG
#SBATCH -N 1
#SBATCH --gpus=a100-40

#argument parser
while getopts ":e:" opt; do
  case "$opt" in
    e) env="$OPTARG";;
    *) echo "Invalid option -$OPTARG" >&2
       exit 1;;
  esac
done

# conda init
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

# main
conda activate "${env}"
conda uninstall -y pytorch
conda install -y pytorch