from dotenv import load_dotenv
load_dotenv()
import sys, os, json, yaml, runregistry, ast
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
sys.path.append(os.path.abspath('../oms-api-client'))
from omsapi import OMSAPI
with open("config.yaml", 'r') as f:
    try:
        info = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Cannot read the file: {exc}")
omsapi = OMSAPI("https://cmsoms.cern.ch/agg/api", "v1", cert_verify=False)
omsapi.auth_oidc(info["APIClient"]["client_ID"], info["APIClient"]["Client_Secret"])
from concurrent.futures import ThreadPoolExecutor, as_completed

run_list = np.array(ast.literal_eval(sys.argv[1]))
file_path = sys.argv[2]

def getruninfo(runID):
    dict = {"run_number": int(runID), "class": "", "cscGOOD": 0, "cscSTANDBY":0, "cscBAD":0, "cscEMPTY":0}
    try:
        run = runregistry.get_run(run_number=int(runID))
        dict["class"]=run["class"]
    except:
        dict["class"]="BAD"
        return dict
    if 'csc-csc' in run["lumisections"]:
        data_dict = run["lumisections"]["csc-csc"]
        for key in data_dict.keys():
            if key == "GOOD":
                dict["cscGOOD"] = data_dict["GOOD"]
            if key == "STANDBY":
                dict["cscSTANDBY"] = data_dict["STANDBY"]
            if key == "BAD":
                dict["cscBAD"] = data_dict["BAD"]
            if key == "EMPTY":
                dict["cscEMPTY"] = data_dict["EMPTY"]
        del data_dict
    else:
        dict["class"]="BAD"
    del run
    return dict
    
def getLSinfo(runID):
    df_temp = []
    ls_query = omsapi.query("lumisections")
    ls_query.filter("run_number", runID)
    ls_query.sort("lumisection_number", asc=False).paginate(page=1, per_page=100000)
    response = ls_query.data().json()["data"];
    for i in range(len(response)):
        df_temp.append(response[i]["attributes"])
    del response
    del ls_query
    return pd.DataFrame(df_temp)

if __name__ == "__main__":
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        runs = df['run_number'].to_numpy()
        run_list = np.setdiff1d(run_list, runs)

    if (len(run_list)==0):
        exit() 
        
    # runregistry info:
    rr_df = pd.DataFrame(columns=["run_number", "class", "cscGOOD", "cscSTANDBY", "cscBAD", "cscEMPTY"])
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(getruninfo, run): run for run in run_list}
        results = [future.result() for future in as_completed(futures)]
        rr_df = pd.concat([pd.DataFrame([line]) for line in results], ignore_index=True)
    print(rr_df)

    # OMS info:
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(getLSinfo, run): run for run in run_list}
        lumi_dfs = [future.result() for future in as_completed(futures)]
        
    for i in range(len(lumi_dfs)):
        lumi_dfs[i]['castor_ready'] = lumi_dfs[i]['castor_ready'].fillna(False)
        lumi_dfs[i]['gem_ready'] = lumi_dfs[i]['gem_ready'].fillna(False)
        lumi_dfs[i]['zdc_ready'] = lumi_dfs[i]['zdc_ready'].fillna(False)
        lumi_dfs[i]['prescale_index'] = lumi_dfs[i]['prescale_index'].fillna(-1)
        lumi_dfs[i]['prescale_name'] = lumi_dfs[i]['prescale_name'].fillna("")

    #Merge and save
    if lumi_dfs:
        lumi_info = pd.concat(lumi_dfs)
        del lumi_dfs
        lumi_info = lumi_info.rename(columns={'lumisection_number': 'ls_number'})
        lumi_info["mean_lumi"]=(lumi_info["init_lumi"]+lumi_info["end_lumi"])/2
        lumi_info = pd.merge(lumi_info, rr_df, on=['run_number'], how='left') 
        if os.path.exists(file_path):
            lumi_info = pd.concat([df, lumi_info], axis=0, ignore_index=True)
            del df
            lumi_info.to_parquet(file_path)
        else:
            lumi_info.to_parquet(file_path)