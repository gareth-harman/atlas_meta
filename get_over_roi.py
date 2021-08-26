#!/Users/harmang/anaconda3/bin/python

import numpy as np
import nibabel as nb
import ants
import pickle as pkl


def ret_mask(x, inds):

   return np.isin(x, inds) 

# Load atlas
atlas_img = ants.image_read('atlases/BN_Atlas_246_2mm.nii')

# Convert to numpy 
atlas_img_arr = atlas_img.numpy()

# Create dict of parcel key and num voxels in said parcel
uniq, cnts = np.unique(atlas_img_arr, return_counts = True)
d_atlas = {uniq[ii]: cnts[ii] for ii in range(len(uniq))} 

d_good = {} # Store good ids for each

# Loop through all terms
for term in ['working_memory', 'inhibitory_control', 'set_shifting', 'reward']:
   
    # Load current image
    img = ants.image_read('bin/' + term + '_comb_bin.nii.gz')
    img_arr = img.numpy()

    # Multiple term mask by atlas
    img_masked = img_arr * atlas_img_arr

    # Get new counts
    m_uniq, m_cnts = np.unique(img_masked, return_counts = True)
    d_mask = {m_uniq[ii]: m_cnts[ii] for ii in range(len(m_uniq))}
    
    good_parcel_ids = []

    for ii in d_mask.keys():

        # Skip zero
        if ii == 0: continue

        if d_mask[ii] / d_atlas[ii] >= .5:
            good_parcel_ids.append(ii)
  
    # Store good parcels here
    d_good[term] = good_parcel_ids 

    # Set all elements not in these parcels to zero 
    out_mask_bad = np.isin(atlas_img_arr, good_parcel_ids, invert = True)
    out_mask_good = np.copy(atlas_img.numpy()) 
    out_mask_good[out_mask_bad] = 0

    # Create a binary version of this
    out_mask_good_bin = np.copy(out_mask_good)
    out_mask_good_bin[out_mask_good_bin <= 0] = 0
    
    # Set aside the original good one if we want
    out_mask_good_orig = np.copy(out_mask_good_bin)
    out_mask_good_bin[out_mask_good_bin > 0] = 1

    img_out = atlas_img.new_image_like(out_mask_good)

    img_out_bin = atlas_img.new_image_like(out_mask_good_bin)
    img_out_orig = atlas_img.new_image_like(out_mask_good_orig)

    img_out.to_file('good_parcels/' + term + '_parcels.nii.gz')
    img_out_bin.to_file('good_parcels/' + term + '_parcels_bin.nii.gz')

    if term == 'working_memory':
        img_out_orig.to_file('good_parcels/BN_atlas_WM_good.nii.gz')

with open('atlases/good_parcels.pkl', 'wb') as f:
    pkl.dump(d_good, f) 

