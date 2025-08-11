#!/bin/sh

helpstring="Usage:
submit.sh [mename] [datasetname] [min_run] [max_run] [miniconda_path] [conda] [metype] [workspace] [outputdir]"
mename=$1
datasetname=$2
min_run=$3
max_run=$4
miniconda_path=$5
conda=$6
metype=$7
workspace=$8
outputdir=$9

# Check inputs
if [ -z ${9+x} ]; then
    echo -e ${helpstring}
    exit
fi

pwd
current_directory=$(pwd)
cd ${miniconda_path}
source etc/profile.d/conda.sh
cd ${current_directory}
conda activate ${conda}

python3 get_MEs.py -w ${workspace} -m ${mename} -d ${datasetname} -t ${metype} --min_run ${min_run} --max_run ${max_run} --outputdir ${outputdir}


