import astra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mrcfile


tlt_filename = 'synthetic_data/sino_synthetic_fullangles_center.tlt'
mrc_filename = 'synthetic_data/sino_synthetic_fullangles_center.mrc'

# Load tilt angles from .tlt file
with open(tlt_filename, 'r') as file:
    theta = np.array([float(line.strip()) for line in file.readlines()])
angles = theta #* np.pi / 180
# Load the MRC file
with mrcfile.open(mrc_filename, 'r') as mrc:
    tilt_series = mrc.data # (Nangles, 1024 1024)

tilt_series = np.array(tilt_series)
tilt_series = np.swapaxes(tilt_series, 0, 1)



v_shape = (500,500,500)
vol_geom = astra.create_vol_geom(500, 500, 500)

proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 500, 500, angles)

# Create a simple hollow cube phantom
# cube = np.zeros((128,128,128))
# cube[17:113,17:113,17:113] = 1
# cube[33:97,33:97,33:97] = 0

# Create projection data from this
# proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

# Display a single projection image
# plt.figure()
# plt.imshow(proj_data[:,20,:])

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('CGLS3D_CUDA')
#cfg = astra.astra_dict('BP3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
proj_id = astra.data3d.create('-sino', proj_geom, tilt_series)
cfg['ProjectionDataId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 15)

# Get the result
rec = astra.data3d.get(rec_id)

np.save('astra', rec)

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)