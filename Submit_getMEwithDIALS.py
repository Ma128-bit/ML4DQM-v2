import argparse, os, htcondor, sys
import pandas as pd
from tqdm import tqdm
from cmsdials.auth.bearer import Credentials

parser = argparse.ArgumentParser(description="Get data")
parser.add_argument(
    '-w', '--workspace', 
    default='csc', 
    help="DIALS-workspace, see https://github.com/cms-DQM/dials-py?tab=readme-ov-file#workspace"
)
parser.add_argument(
    '-m', '--menames', 
    nargs='+',  
    type=str,   
    required=True,  
    help='One or a list of monitoring elements'
)
parser.add_argument(
    '-d', '--dtnames', 
    nargs='+',  
    type=str,   
    required=False,  
    help='One or a list of dataset elements (if None takes all the possible datasets)'
)
parser.add_argument(
    '-t', '--metype', 
    type=str,   
    required=True,  
    choices=['h1d', 'h2d'],
    help='Tipe of monitoring elements h1d or h2d'
)
parser.add_argument(
    '-o', '--outputdir', 
    default='.',
    type=str,   
    help='output directory '
)
parser.add_argument(
    '-c', '--conda', 
    default='MEProd',
    type=str,   
    help='Conda enviroment name (default = MEProd)'
)
parser.add_argument(
    '-p', '--miniconda_path', 
    type=str,   
    required=True,  
    help='miniconda directory path'
)
parser.add_argument(
    '--min_run', 
    type=int,   
    help='Min Run'
)
parser.add_argument(
    '--max_run', 
    type=int,   
    help='Max Run'
)
parser.add_argument(
    '--n_splits', 
    default=16,
    type=int,   
    help='Number of splits per ME (default=16)'
)
parser.add_argument(
    '--era', 
    type=str,   
    help='Era, use this format: Run2024A'
)
args = parser.parse_args()

if (args.min_run == None or args.max_run==None) and args.era == None:
    sys.exit("\n******** Input ERROR! ********\nWrong use of min/max run. Pass both or an era!") 

eradef = {
    "Run2024A": (378142, 378970),
    "Run2024B": (378971, 379411),
    "Run2024C": (379412, 380252),
    "Run2024D": (380253, 380947),
    "Run2024E": (380948, 381943),
    "Run2024F": (381944, 383779),
    "Run2024G": (383780, 385813),
    "Run2024H": (385814, 386408),
    "Run2024I": (386409, 387121),
    
    "Run2023A": (365739, 366364),
    "Run2023B": (366365, 367079),
    "Run2023C": (367080, 369802),
    "Run2023D": (369803, 372415),
    
    "Run2022A": (352319, 355064),
    "Run2022B": (355065, 355793),
    "Run2022C": (355794, 357486),
    "Run2022D": (357487, 359021),
    "Run2022E": (359022, 360331),
    "Run2022F": (360332, 362180),
    "Run2022G": (362350, 362760)
}
if args.era is not None:
    if args.era in eradef.keys():
        args.min_run = eradef[args.era][0]
        args.max_run = eradef[args.era][1]
    else:
        sys.exit("\n******** Input ERROR! ******** \nWrong era! \nUse this format: Run2024A. No v-X are allowed!") 
    
creds = Credentials.from_creds_file()

if __name__ == "__main__":
    try:
        os.mkdir(args.outputdir)
    except FileNotFoundError:
        print(f"The path {args.outputdir} doesn't exists.")

    try:
        os.mkdir(args.outputdir+"/log")
    except FileNotFoundError:
        print(f"The path {args.outputdir}/log doesn't exists.")
        
    current_directory = os.getcwd()
    
    print(args.menames)
    print(args.dtnames)
    inputs = []
    if args.dtnames is not None:
        for m in args.menames:
            for d in args.dtnames:
                diff = int((args.max_run - args.min_run)/args.n_splits) +1 
                for i in range(args.n_splits-1):
                    inputs.append((m, d, args.min_run+i*diff, args.min_run+(i+1)*diff-1))
                inputs.append((m, d, args.min_run+(args.n_splits-1)*diff, args.max_run))
    else:
        for m in args.menames:
            diff = int((args.max_run - args.min_run)/args.n_splits) +1 
            for i in range(args.n_splits-1):
                inputs.append((m, None, args.min_run+i*diff, args.min_run+(i+1)*diff-1))
            inputs.append((m, None, args.min_run+(args.n_splits-1)*diff, args.max_run))

    param = [args.miniconda_path, args.conda, args.metype, args.workspace, args.outputdir]
    param = [str(arg) for arg in param]
    itemdata = [{"argument": " ".join([str(arg) for arg in input]+param)} for input in inputs]

    print(itemdata)

    job = htcondor.Submit({
        "executable": current_directory+"/submit.sh",
        "arguments": "$(argument)",
        "output": current_directory+"/"+args.outputdir+"/log/test-$(ProcId).out",  
        "error": current_directory+"/"+args.outputdir+"/log/test-$(ProcId).err",
        "log": current_directory+"/"+args.outputdir+"/log/test.log",
        "request_cpus": "2",
        "request_memory": "10GB",
        "request_disk": "10GB"
    })
    schedd = htcondor.Schedd()
    #with schedd.transaction() as txn:
    #    submit_result = job.queue_with_itemdata(txn, itemdata = iter(itemdata))
    submit_result = schedd.submit(job, itemdata = iter(itemdata))
    

