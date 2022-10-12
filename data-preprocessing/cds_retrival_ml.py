import os
from optparse import OptionParser
import cdsapi

parser = OptionParser()
parser.add_option("-y", dest="year", type="str", default="2022")
parser.add_option("-m", dest="month", type="str", default="06")
options, args = parser.parse_args()

filename = os.path.join("data", "netcdf-raw", "era5", f"ERA5_ml_{options.year}-{options.month}.grib")

if os.path.isfile(filename):
    print(f" Output file \"{filename}\" exists...\n\n")

else:
    print(f" Processing {filename}")
   
    args = {
        "class"  : "ea",	
        "expver"  : "1",
        "stream"  : "oper",
        "type"    : "an",
        "levtype" : "ml",
        "area"    : [49.75, 8.25, 45.25, 16.75,],
        "grid"    : "0.25/0.25",
        "levelist": [f"{x}" for x in range(64, 138)],
        "date"    : f"{options.year}-{options.month}-01/to/{options.year}-{options.month}-31",
        "time"    : [f"{x:02d}:00" for x in range(0, 24)],
        "param"   : "247/246/133/75/76/130/131/132/135",
        "use"     : "infrequent"
    }

    try:
        server = cdsapi.Client(timeout=1000, quiet=False, debug=True)
        cdstype = "reanalysis-era5-complete"
        server.retrieve(cdstype, args, filename)
        print(f"Everything seems to be fine for {filename}")

    except Exception as e:
        print(f"cdsapi returned error code != 0 for file: {filename}")
            
print("DONE.")