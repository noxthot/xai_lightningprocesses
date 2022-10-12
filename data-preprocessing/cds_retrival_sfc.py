import os
from optparse import OptionParser
from cdsapi import Client

parser = OptionParser()
parser.add_option("-y", dest="year", type="str", default="2022")
parser.add_option("-m", dest="month", type="str", default="06")
options, args = parser.parse_args()

filename  = os.path.join("data", "netcdf-raw", "era5sl", f"ERA5_sfc_{options.year}-{options.month}.nc")

if os.path.isfile(filename):
    print(f" Output file \"{filename}\" exists...")
else:
    print(f" Processing {filename}")

    args = {
        "product_type" : "reanalysis",
        "format"       : "netcdf",
        "area"         : [49.75, 8.25, 45.25, 16.75,],
        "variable"     : [
            '2m_temperature', 'convective_available_potential_energy',
            'convective_precipitation', 'instantaneous_surface_sensible_heat_flux',
            'medium_cloud_cover', 'total_column_supercooled_liquid_water',
        ],
        "year"         : options.year,
        "month"        : options.month,
        "day"          : [f"{x:02d}" for x in range(1, 32)],
        "time"         : [f"{x:02d}:00" for x in range(0, 24)]
    }
    try:
        server = Client()
        cdstype = "reanalysis-era5-single-levels"
        server.retrieve(cdstype, args, filename)
        print(f"Everything seems to be fine for {filename}")

    except Exception as e:
        print(f"cdsapi returned error code != 0 for file: {filename}")

print("DONE.")