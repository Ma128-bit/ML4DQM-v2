# Training DQM Tool for CSC

```
git clone https://github.com/Ma128-bit/ML4DQM.git
```

## Get Monitoring elements (ME)
`Requires a working conda installation`
```=shell
conda create --name MEProd python=3.9
conda activate MEProd
pip3 install -r requirements_getME.txt 
chmod +x submit.sh
```
Use `Submit_getMEwithDIALS.py` to submit (with **condor**) the code, based on [dials_api](https://github.com/cms-DQM/dials-py), that gets the MEs. List of arguments:

| Argument                   | Default    | Required | Description                                |
| -------------------------- | :--------: | :------: | ------------------------------------------ |
| `-w / --workspace`         | csc        | False    | DIALS-workspace                            |
| `-m / --menames`           |            | True     | One or a list of monitoring elements       |
| `-d / --dtnames`           |            | False    | One or a list of dataset elements (if None takes all the possible datasets) |
| `-t / --metype`            |            | True     | Type of monitoring elements h1d or h2d     |
| `-o / --outputdir`         | test       | False    | Output directory                           |
| `-c / --conda`             | MEProd     | False    | Conda environment name                     |
| `-p / --miniconda_path`    |            | True     | Path to the miniconda installation directory |
| `--min_run`                |            | True     | Minimum run (Not required if `--era`)        |
| `--max_run`                |            | True     | Maximum run (Not required if `--era`)        |
| `--max_splits`             | 16         | False    | Number of splits per ME                      |
| `--era`                    |            | False    | Automatically select the min and max run according to the chosen era (ex: Run2024D)|

Usage example:
```
python3 Submit_getMEwithDIALS.py -m CSC/CSCOfflineMonitor/recHits/hRHGlobalm4 -t h2d -p /lustrehome/mbuonsante/miniconda3 \
-c MEProd --era Run2024E --n_splits 20 --outputdir hRHGlobalm4E
```

To ensure that all the jobs have finished, use:
```=shell
grep "Done:" "outputdir"/log/*.out | wc -l
```
**Note:**

If you get the error:

`ImportError: cannot import name 'MutableMapping' from 'collections' `

Modify `classad/_expression.py` changing `from collections import MutableMapping` with `from collections.abc import MutableMapping`

## Main Workflow
It is split into 5 steps, from S0 to S4, listed below.
The first time you run the code, you need to perform:
```=shell
python3
import Utilities.database as db
pip3 install -r requirementsPrePro.txt 
db.init_db()
```

### S0: Fetch image info
**For CONDA users:**
```=shell
conda create --name PrePro python=3.9
conda activate PrePro
pip3 install -r requirementsPrePro.txt 
```

**For SWAN notebook users:**

There is no need to follow the steps above. You only need to install oms-api-client and runregistry_api_client (as below) and import them as:
```
import sys
sys.path.append('run registry site')
sys.path.append('./oms-api-client')
```
where `run registry site` is obtained usign: `pip show runregistry`

**For all users:**

Follow the "Authentication Prerequisites" instructions on [runregistry_api_client](https://github.com/cms-DQM/runregistry_api_client). Then follow [oms-api-client](https://gitlab.cern.ch/cmsoms/oms-api-client) instructions. (You can use the same application for both runregistry and oms)
Save the oms application credentials in a file named `config.yaml` with this structure:
```=yaml
APIClient:
    client_ID: 'id_example'
    Client_Secret: 'secret_example'
```

```=shell
conda activate PrePro
```

Run: `python3 S0_GetInfo.py`. List of arguments:

| Argument                   | Default     | Required | Description                                |
| -------------------------- | :--------:  | :------: | ------------------------------------------ |
| `-p / --path`              | csc         | True     | ME parquet file location                   |
| `-m / --mename`            | hRHGlobalm2 | True     | One monitoring element.                    |
| `--conda_env`              |             | True     | Path to conda env (like ~/miniconda3/envs/oms) |

Example: `python3 S0_GetInfo.py -m "hRHGlobalm2" -p "../ML4DQM/MEs/" --conda_env ~/miniconda3/envs/oms`

Returns a `job_id`!!

### S1: Pre-Processing and sum of consecutive LSs
```=shell
conda activate PrePro
python3 S1_PreProcessing.py --job_id XXXXXXX
```

### S2: Train Autoencoder
```=shell
conda create --name pytorch python=3.9
conda activate pytorch
pip3 install -r requirementsTraining.txt
```

```=shell
conda activate pytorch
python3 S2_training.py --job_id XXXXXXX --ring ["in" or "out"]
```

### S3: Generate fake anomalies
```=shell
conda activate pytorch
python3 S3_GenerateAnomaly.py --job_id XXXXXXX --ring ["in" or "out"]
```
### S4: Study model performance with the fake anomalies
```=shell
conda activate pytorch
python3 S4_performance.py --job_id XXXXXXX --ring ["in" or "out"]
```
Optional argument: ` --anomalysample ` path to a custom set of anomalies

**Important**: If you run the same step twice (for example, after making some changes), it will return a new job_id, and you will need to use this new job_id for the subsequent steps.



