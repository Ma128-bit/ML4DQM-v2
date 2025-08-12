import pandas as pd
import numpy as np
from tqdm import tqdm
import os, subprocess, json, glob
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

max_group_size = 500 * (1024**2)  # maximum total group size in MB
min_file_size = 601     # minimum file size in bytes

# Find all subdirectories
dirs = [os.path.join(path, d) for d in os.listdir(path) if me in d and os.path.isdir(os.path.join(path, d))]
only_names = [os.path.basename(d) for d in dirs]
print("The following MEs were found:")
print(only_names)

# Collect all valid Parquet files using glob
all_files = []
for dir in dirs:
    pattern = os.path.join(dir, '*.parquet')
    for fpath in glob.glob(pattern):
        if os.path.getsize(fpath) >= min_file_size:
            all_files.append(fpath)

print(f"Found {len(all_files)} valid files.")

# Group files by total size not exceeding max_group_size
batches = []
current_batch = []
current_size = 0

for f in all_files:
    fsize = os.path.getsize(f)
    if current_size + fsize > max_group_size and current_batch:
        batches.append(current_batch)
        current_batch = []
        current_size = 0
    current_batch.append(f)
    current_size += fsize

# Add the last batch if it's not empty
if current_batch:
    batches.append(current_batch)

index=0
# Process each batch
for batch_idx, file_group in enumerate(tqdm(batches, desc="Processing batches", unit="batch"), start=1):
    batch_size_mb = sum(os.path.getsize(f) for f in file_group) / (1024 * 1024)
    print(f"\nProcessing batch {batch_idx} ({len(file_group)} files, {batch_size_mb:.2f} MB)")
    monitoring_elements = pd.read_parquet(file_group)
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

