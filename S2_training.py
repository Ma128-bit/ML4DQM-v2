import os, json, subprocess, argparse, yaml, torch
import numpy as np
import pandas as pd
import Utilities.database as db
import Utilities.ResNet as ResNet

class METraining:
    def __init__(self, job_id, ring):
        self.ring = ring
        self.file_path = db.get_job_info(job_id)["S1"]
        self.me_name = db.get_job_info(job_id)["MEname"]
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id)
        self.output_name = self.output_path + os.sep + self.me_name+'_'+ring+'_step2.parquet'
        self.monitoring_elements = None
        self.model = None
        with open("chamber_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.training_params = config["training"]
        self.chamber_params = config["MEs"][self.me_name]
        ring_info = self.chamber_params["ring_"+ring]
        if len(ring_info)==1:
            self.ring_div = ring_info[0][1]
        else:
            raise RuntimeError("Multi-ring not implemented yet (hRHGlobalm1)")

    def update_jobid(self, job_id):
        self.output_path = os.path.join("outputs", self.me_name+"-"+job_id)
        self.output_name = self.output_path + os.sep + self.me_name+'_'+ring+'_step2.parquet'
    
    def load_data(self):
        self.monitoring_elements = pd.read_parquet(self.file_path, engine='pyarrow')

    def training(self):
        scripted_model = ResNet.train(self.monitoring_elements, self.output_path, "img_"+ring, **self.training_params)

        if torch.cuda.is_available():
            scripted_model.save(f"{self.output_path}/trained_model_{ring}_cuda.pth")
            scripted_model.to("cpu").save(f"{self.output_path}/trained_model_{ring}_cpu.pth")
            scripted_model.to("cuda")
        else:
            scripted_model.save(f"{self.output_path}/trained_model_{ring}_cpu.pth")

        self.model = scripted_model
        print("[S2] Training completed!")

    def predictions(self):
        self.monitoring_elements = ResNet.predictions(self.monitoring_elements, self.model, "img_"+ring, self.ring_div)
        print("[S2] Predictions completed!")

    def save(self):
        self.monitoring_elements.to_parquet(self.output_name, index=False)
        print("[S2] Saving completed!")

    def run(self):
        self.load_data()
        self.training()
        self.predictions()
        self.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True, help="job_id")
    parser.add_argument("--ring", type=str, required=True, help="in or out")
    
    args = parser.parse_args()
    job_id = args.job_id
    ring = args.ring
    
    analyzer = METraining(job_id=job_id, ring=ring)

    change_jobid = False
    job_id_new, _ = db.update_step(job_id, "S2", analyzer.output_name)
    if job_id_new != job_id:
        analyzer.update_jobid(job_id_new)
        db.update_step(job_id_new, "S2", analyzer.output_name)
        change_jobid = True


    analyzer.run()

    if change_jobid:
        print("!!! Your new job id is: ", job_id_new)
