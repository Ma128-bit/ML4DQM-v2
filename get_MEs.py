import argparse
import pandas as pd
from tqdm import tqdm
from cmsdials.auth.bearer import Credentials
from cmsdials import Dials
from cmsdials.filters import LumisectionHistogram1DFilters
from cmsdials.filters import LumisectionHistogram2DFilters
from cmsdials.filters import RunFilters
from cmsdials.filters import LumisectionFilters
from cmsdials.filters import MEFilters

parser = argparse.ArgumentParser(description="Get data")
parser.add_argument(
    '-w', '--workspace', 
    default='csc', 
    help="DIALS-workspace, see https://github.com/cms-DQM/dials-py?tab=readme-ov-file#workspace"
)
parser.add_argument(
    '-m', '--mename', 
    type=str,   
    required=True,  
    help='One monitoring element'
)
parser.add_argument(
    '-d', '--datasetname', 
    type=str,   
    required=False,  
    help='One dataset elements (if None takes all the possible datasets)'
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
    '--min_run', 
    type=int,   
    required=True,  
    help='Min Run'
)
parser.add_argument(
    '--max_run', 
    type=int,   
    required=True,  
    help='Max Run'
)

def get_data(args, dials):
    name = args.mename.replace("/","_")
    print("Start: ", name, args.min_run, args.max_run)
    if args.datasetname is not None and args.datasetname != "None":
        if args.metype == "h2d":
            data = dials.h2d.list_all(
                LumisectionHistogram2DFilters(
                    me=args.mename,
                    dataset= args.datasetname,
                    run_number__gte=args.min_run,
                    run_number__lte=args.max_run
                ),
                enable_progress=False,
            )
        if args.metype == "h1d":
            data = dials.h2d.list_all(
                LumisectionHistogram1DFilters(
                    me=args.mename,
                    dataset= args.datasetname,
                    run_number__gte=args.min_run,
                    run_number__lte=args.max_run
                ),
                enable_progress=False,
            )
    else:
        if args.metype == "h2d":
            data = dials.h2d.list_all(
                LumisectionHistogram2DFilters(
                    me=args.mename,
                    run_number__gte=args.min_run,
                    run_number__lte=args.max_run
                ),
                enable_progress=False,
            )
        if args.metype == "h1d":
            data = dials.h2d.list_all(
                LumisectionHistogram1DFilters(
                    me=args.mename,
                    run_number__gte=args.min_run,
                    run_number__lte=args.max_run
                ),
                enable_progress=False,
            )

    data = data.to_pandas()
    data.to_parquet(args.outputdir+"/"+name+f'_{args.datasetname}_{args.min_run}_{args.max_run}.parquet', index=False)
    print("Done: ", name, args.min_run, args.max_run)
    del data

if __name__ == "__main__":
    print("I'm in python main")
    args = parser.parse_args()
    is_cred = False
    while not is_cred:
        try:
            creds = Credentials.from_creds_file()
            is_cred = True
        except Exception as e:
            print("Problem in credential. Retry ...")
            #print(f"Error details: {e}")            
    dials = Dials(creds, workspace=args.workspace)
    get_data(args, dials)
