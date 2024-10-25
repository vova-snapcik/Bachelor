from netCDF4 import Dataset
import numpy as np

# Input file path
input_file_path = "/work3/sjan/Database/TurbineAsSensor/U12Sx12Sy20Ti5Sh14AD14/WT/operationWT01.nc"

# Output file path
output_file_path = "/zhome/cf/0/188047/Bachelor/Data/Transformer 3.0/WT01U12_sensor.nc"

# Open the input NetCDF file
with Dataset(input_file_path, "r") as src:
    # Get the time variable and ensure we limit it to 1000 time steps
    time_var = src.variables['t']
    time_steps = min(len(time_var), 1000)

    # Create the output NetCDF file
    with Dataset(output_file_path, "w", format="NETCDF4") as dst:
        # Define the dimensions in the output file
        dst.createDimension("t", time_steps)

        # Copy the time variable
        t_out = dst.createVariable("t", time_var.dtype, ("t",))
        t_out[:] = time_var[:time_steps]

        # Copy the selected variables for the first 1000 time steps
        for var_name in ["Vhub", "Power", "Omega"]:
            var = src.variables[var_name]
            out_var = dst.createVariable(var_name, var.dtype, ("t",))
            out_var[:] = var[:time_steps]

print(f"File '{output_file_path}' created with variables 'Vhub', 'Power', 'Omega' containing 1000 time steps.")
