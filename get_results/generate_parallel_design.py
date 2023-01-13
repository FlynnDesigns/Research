import numpy as np

# Output dir 
output_dir = 'A:\\Research\\Research\\get_results\\'

# Base array structure 
array = np.zeros((64, 64))
array[:, 0:2] = 1
array[:, 5:7] = 1
array[:, 9:11] = 1
array[:, 13:15] = 1
array[:, 17:19] = 1
array[:, 21:23] = 1
array[:, 25:27] = 1
array[:, 29:31] = 1
array[:, 33:35] = 1
array[:, 37:39] = 1
array[:, 41:43] = 1
array[:, 45:47] = 1
array[:, 49:51] = 1
array[:, 53:55] = 1
array[:, 57:59] = 1
array[:, 62:64] = 1

# Adding in the extra fluid region 
output = np.ones((74,66), dtype=np.uint8) * 255
output[5:69, 1:65] = array

# Rotating the coordinates so that they show correct in the simulation 
output[5:69, 1:65] = np.rot90(output[5:69, 1:65], 3)

# Transposing everything to get it centered correctly 
output = np.transpose(output)

# Writing the solid coordinates file for 0
with open(output_dir + 'parallel.txt', 'w') as f: 
    for k in range(66):
        for i in range(74):
            # Applying offset to get the coordinates to the right positon 
            x_val = k + 0.5 
            y_val = i + 0.5 

            # If the coordinates are the middle of the physical domain, write the coordinates
            if output[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                    f.write("%s " % (x_val))
                    f.write("%s\n" % (y_val))