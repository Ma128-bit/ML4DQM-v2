import pandas as pd
import numpy as np
from tqdm import tqdm
import os, subprocess, json, yaml
import argparse, glob
import Utilities.database as db
import Utilities.CSCRebinning as csc

def sum_images(group):
    return pd.Series({
        "recorded_lumi": group["recorded_lumi_per_lumisection"].sum(),
        "scaled_lumi": group["scaled_lumi"].sum(),
        "mean_lumi": group["mean_lumi"].sum(),
        "entries": group["entries"].sum(),
        "run_min": group["run_number"].iloc[0],
        "run_max": group["run_number"].iloc[-1],
        "ls_min": group["ls_number"].iloc[0],
        "ls_max": group["ls_number"].iloc[-1],
        "prescale": [f'{count}x"{val}"' for val, count in zip(*np.unique(group["prescale_name"], return_counts=True))],
        "summed_ls": len(group["entries"]),
        "img": np.sum(group["data"], axis=0).astype(np.float32),
    })

def sum_row_images(row, columns):
    return np.sum([row[col] for col in columns], axis=0)

class MEPreprocessor:
    def __init__(self, job_id):
        self.inpath = glob.glob(db.get_job_info(job_id)["S0"])
        self.me_name = db.get_job_info(job_id)["MEname"]
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id) + os.sep + self.me_name+'step1.parquet'
        self.monitoring_elements = None
        with open("chamber_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.lumimin = config["preprocessing"]["lumimin"]
        self.chamber_params = config["MEs"][self.me_name]

    def update_jobid(self, job_id):
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id) + os.sep + self.me_name+'step1.parquet'

    def load_data(self):
        monitoring_elements = pd.DataFrame()
        for chunk in self.inpath:
            df = pd.read_parquet(chunk)
            df['data'] = df['data'].apply(lambda histo: np.vstack(histo).astype(np.uint32))
            df = df[df['recorded_lumi_per_lumisection']>0.00]
            monitoring_elements = pd.concat([monitoring_elements, df], ignore_index=True)
            del df
        print("N. LSs: ", len(monitoring_elements))
        monitoring_elements = monitoring_elements.sort_values(by=['run_number', 'ls_number']).reset_index()
        monitoring_elements = monitoring_elements.drop(columns=["index"])
        self.monitoring_elements = monitoring_elements
    
    def add_hltscale(self):
        with open("hltsclae.json") as f:
            json_data = json.load(f)
        self.monitoring_elements["lumi_scale"] = self.monitoring_elements["prescale_name"].map(json_data)
        self.monitoring_elements["lumi_scale"] = self.monitoring_elements["lumi_scale"].fillna(1)
        self.monitoring_elements["scaled_lumi"] = self.monitoring_elements["lumi_scale"]* self.monitoring_elements["recorded_lumi_per_lumisection"]
    
    
    def groupbylumi(self):
        lumisum = 0
        groupID = 0
        groups = []
        run = self.monitoring_elements["run_number"][0]

        for i, lumi in enumerate(self.monitoring_elements["scaled_lumi"]):
            if self.monitoring_elements["run_number"][i] != run:
                groupID += 1
                lumisum = 0
                run = self.monitoring_elements["run_number"][i]
                
            lumisum += lumi
            groups.append(groupID)

            if lumisum > self.lumimin:
                lumisum = 0
                groupID += 1

        self.monitoring_elements["group"] = groups
        
    def sum_consecutive_images(self):
        if self.lumimin is None:
            return
        self.groupbylumi()
        self.monitoring_elements = self.monitoring_elements.groupby("group").apply(sum_images, include_groups=False).reset_index()
        self.monitoring_elements = self.monitoring_elements[self.monitoring_elements["scaled_lumi"]>=self.lumimin]
        print("N. LSs after summing: ", len(self.monitoring_elements))
        
    def apply_custom_selection(self):
        ring_in_ids = [x[0] for x in self.chamber_params["ring_in"]]
        ring_out_ids = [x[0] for x in self.chamber_params["ring_out"]]
        total_good_flags = None

        for id in ring_in_ids:
            good_flag  = csc.main(self.monitoring_elements, self.me_name, ring_id=id, name=f"img_in{id}")
            if total_good_flags is None:
                total_good_flags = good_flag
            else:
                total_good_flags = total_good_flags + good_flag

        img_cols = [col for col in self.monitoring_elements.columns if col.startswith("img_in")]
        self.monitoring_elements["img_in"] = self.monitoring_elements.apply(lambda row: sum_row_images(row, img_cols), axis=1)
        self.monitoring_elements = self.monitoring_elements.drop(columns=img_cols)

        for id in ring_out_ids:
            good_flag  = csc.main(self.monitoring_elements, self.me_name, ring_id=id, name=f"img_out{id}")
            if total_good_flags is None:
                    total_good_flags = good_flag
            else:
                total_good_flags = total_good_flags + good_flag
        
        img_cols = [col for col in self.monitoring_elements.columns if col.startswith("img_out")]
        self.monitoring_elements["img_out"] = self.monitoring_elements.apply(lambda row: sum_row_images(row, img_cols), axis=1)
        self.monitoring_elements = self.monitoring_elements.drop(columns=img_cols)

        self.monitoring_elements["good_flag"] = total_good_flags
        
    def save_results(self):
        self.monitoring_elements["img"] = self.monitoring_elements["img"].apply(lambda x: x.tolist())
        self.monitoring_elements["img_in"] = self.monitoring_elements["img_in"].apply(lambda x: x.tolist())
        self.monitoring_elements["img_out"] = self.monitoring_elements["img_out"].apply(lambda x: x.tolist())
        self.monitoring_elements.to_parquet(self.output_path, index=False)
    
    def run(self):
        self.load_data()
        self.add_hltscale()
        self.sum_consecutive_images()
        self.apply_custom_selection()
        self.save_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True, help="job_id")
    
    args = parser.parse_args()
    job_id = args.job_id
    
    analyzer = MEPreprocessor(job_id=args.job_id)

    change_jobid = False
    job_id_new, _ = db.update_step(job_id, "S1", analyzer.output_path)
    if job_id_new != job_id:
        analyzer.update_jobid(job_id_new)
        db.update_step(job_id_new, "S1", analyzer.output_path)
        change_jobid = True

    analyzer.run()

    if change_jobid:
        print("Your new job id is: ", job_id_new)
