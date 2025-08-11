import os, json, subprocess, argparse
import numpy as np
import pandas as pd
import Utilities.database as db
import Utilities.GenerateAnomaly as anomaly
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MEGenerateAnomaly:
    def __init__(self, job_id, ring):
        self.ring = ring
        self.file_path = db.get_job_info(job_id)["S2"]
        self.me_name = db.get_job_info(job_id)["MEname"]
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id)
        self.output_name = self.output_path + os.sep + self.me_name+'_'+ring+'_step3.parquet'
        self.monitoring_elements = None
        self.df_anomalies = None
        self.model = torch.jit.load(f"{self.output_path}/trained_model_{ring}_{device}.pth")
        with open("chamber_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.chamber_params = config["MEs"][self.me_name]
        ring_info = self.chamber_params["ring_"+ring]
        if len(ring_info)==1: # Multi-ring not implemented yet
            self.ring_info = ring_info[0]
        else:
            raise RuntimeError("Multi-ring not implemented yet (hRHGlobalm1)")
        

    def update_jobid(self, job_id):
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id)
        self.output_name = self.output_path + os.sep + self.me_name+'_'+ring+'_step3.parquet'

    def load_data(self):
        self.monitoring_elements = pd.read_parquet(self.file_path, engine='pyarrow')
        self.monitoring_elements[ring] = self.monitoring_elements[ring].apply(lambda histo: np.vstack(histo).astype(np.float64))

    def genanomay(self):
        self.df_anomalies = anomaly.main(self.monitoring_elements, self.model, self.ring, self.ring_info[0], self.ring_info[1])
            
    def save(self):
        self.df_anomalies.to_parquet(self.output_name, index=False)

    def run(self):
        self.load_data()
        self.genanomay()
        self.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True, help="job_id")
    parser.add_argument("--ring", type=str, required=True, help="in or out")
    
    args = parser.parse_args()
    job_id = args.job_id
    ring = args.ring
    
    analyzer = MEGenerateAnomaly(job_id=job_id, ring=ring)

    change_jobid = False
    job_id_new, _ = db.update_step(job_id, "S3", analyzer.output_name)
    if job_id_new != job_id:
        analyzer.update_jobid(job_id_new)
        db.update_step(job_id_new, "S3", analyzer.output_name)
        change_jobid = True
    
    analyzer.run()

    if change_jobid:
        print("!!! Your new job id is: ", job_id_new)
