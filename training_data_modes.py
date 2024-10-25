from netCDF4 import Dataset
import numpy as np

# Input file path
input_file_path = "/work3/sjan/Database/TurbineAsSensor/CROM/projections/modes_50/modal_ts_U8Sx12Sy20Ti5Sh14AD14_inflowFlucWT01.nc"

# Output file path
output_file_path = "/zhome/cf/0/188047/Bachelor/Data/Transformer 3.0/WT01U8_modes.nc"

# Open the input NetCDF file
with Dataset(input_file_path, "r") as src:
    # Get the dimensions and variables we need
    time_var = src.variables['t']
    Cmt_var = src.variables['Cmt']

    # Define the subset size
    time_steps = min(len(time_var), 1000)
    modes_to_copy = min(Cmt_var.shape[2], 5)  # Adjusted to select first 5 modes from the third dimension

    # Create the output NetCDF file
    with Dataset(output_file_path, "w", format="NETCDF4") as dst:
        # Define the dimensions in the output file
        dst.createDimension("t", time_steps)
        dst.createDimension("mode", modes_to_copy)

        # Copy the time variable
        t_out = dst.createVariable("t", time_var.dtype, ("t",))
        t_out[:] = time_var[:time_steps]

        # Copy the Cmt variable for the first 5 modes and 1000 time steps, removing the 'WT' dimension
        Cmt_out = dst.createVariable("Cmt", Cmt_var.dtype, ("t", "mode"))
        Cmt_out[:, :] = Cmt_var[0, :time_steps, :modes_to_copy]

print(f"File '{output_file_path}' created with variable 'Cmt' containing first 1000 time steps and 50 modes.")
