import pandas as pd
import numpy as np
from tqdm import tqdm
import os, subprocess, json
import argparse
import Utilities.database as db

parser = argparse.ArgumentParser()
#parser.add_argument("--out_label", type=str, default="test", help="output label")
parser.add_argument('-m', '--mename', type=str, default="hRHGlobalm2", help="ME name")
parser.add_argument('-p',"--path", type=str, default="../ML4DQM/MEs/", help="parquet file location")
parser.add_argument("--conda_env", type=str, default="/lustrehome/mbuonsante/miniconda3/envs/oms", help="OMS/RR conda env")

args = parser.parse_args()

conda_env = args.conda_env
me = args.mename
path = args.path
job_id, job_path = db.create_new_job_folder(me)

dirs = os.listdir(path)
me_dirs = [path+i for i in dirs if me in i and os.path.isdir(path+i)]
print("The following MEs were found:")
print(me_dirs)

index = 0
for dir in tqdm(me_dirs, desc="Processing MEs"):
    files = os.listdir(dir)
    me_files = [dir+"/"+i for i in files if me in i]
    filtered_files = [file for file in me_files if os.path.exists(file) and os.path.getsize(file) >= 601]
    
    monitoring_elements = pd.read_parquet(filtered_files)
    monitoring_elements = monitoring_elements[monitoring_elements['dataset'].str.contains("StreamExpress")]
    
    run_list = np.sort(np.unique(monitoring_elements["run_number"].unique()))
    command = f"conda run --prefix {conda_env} python3 Utilities/getInfo.py '{json.dumps(run_list.tolist())}' perLSinfo.parquet"
    subprocess.run(command, text=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
    
    lumi_info = pd.read_parquet("perLSinfo.parquet")
    
    monitoring_elements = pd.merge(monitoring_elements, lumi_info, on=['run_number', 'ls_number'], how='left') 
    
    monitoring_elements = monitoring_elements[
        (monitoring_elements["beams_stable"] == True) &
        (monitoring_elements["cms_active"] == True) &
        (monitoring_elements["beam_present"] == True) &
        (monitoring_elements["physics_flag"] == True) &
        (monitoring_elements["cscSTANDBY"] == 0) &
        (monitoring_elements["cscBAD"] == 0) &
        (monitoring_elements["cscGOOD"] != 0) &
        (monitoring_elements["class"].str.contains("Collisions", na=False))
    ]
    
    if "hRHGlobalm" in me:
        monitoring_elements = monitoring_elements[monitoring_elements["cscm_ready"]]
    elif "hRHGlobalp" in me:
        monitoring_elements = monitoring_elements[monitoring_elements["cscp_ready"]]
    
    
    monitoring_elements = monitoring_elements.sort_values(by=['run_number', 'ls_number']).reset_index()
    monitoring_elements = monitoring_elements.drop(columns=["index"])
    
    # ==== Salva ====
    monitoring_elements.to_parquet(job_path+"/"+me+f'_step0_part{index}.parquet', index=False)
    index = index +1
    del monitoring_elements
    
db.update_step(job_id, "S0", job_path+"/"+me+f'_step0_part*.parquet')
print("Your job id is: ", job_id)

