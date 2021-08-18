#!Users/harmang/anaconda3/bin/python

import numpy as np
import nibabel as nb
import ants


thr = 3
sig = 2

def thresh(x, t):

    if t == 0:
        x[x <= t] = 0; x[x > t] = 1
    else:
        x[x < t] = 0; x[x >= t] = 1

    return x

for ii in ['reward', 'set_shifting', 'working_memory', 'inhibitory_control']:

    # Load synth + query items
    filename_synth = ii + '_synth.nii.gz' 
    filename_query = ii + '.nii.gz' 

    img_synth = ants.image_read('init_queries/' + filename_synth)
    img_query = ants.image_read('init_queries/' + filename_query)

        # Apply transform to query image
    img_query = ants.resample_image_to_target(img_query, img_synth, interp_type = 'nearestNeighbor')

    # Binarize 
    img_synth_arr = img_synth.numpy()
    img_synth_arr = thresh(img_synth_arr, 0)

    img_query_arr = img_query.numpy()
    img_query_arr = thresh(img_query_arr, thr)

    # Combined image
    img_comb = thresh(img_query_arr + img_synth_arr, 0)

    # Create new imgs to write 
    img_synth_out = img_synth.new_image_like(img_synth_arr)
    img_query_out = img_synth.new_image_like(img_query_arr)
    img_comb_out = img_synth.new_image_like(img_comb)

    # Apply some smoothin 
    img_synth_out = thresh(ants.smooth_image(img_synth_out, sigma = sig), .5) 
    img_query_out = thresh(ants.smooth_image(img_query_out, sigma = sig), .5)
    img_comb_out = thresh(ants.smooth_image(img_comb_out, sigma = sig), .5) 

    # Write images
    img_synth_out.to_file('bin/' + ii + '_synth_bin.nii.gz')
    img_query_out.to_file('bin/' + ii + '_query_bin.nii.gz')
    img_comb_out.to_file('bin/' + ii + '_comb_bin.nii.gz')


