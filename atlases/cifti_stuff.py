#!/Users/harmang/anaconda3/bin/python


from nibabel import cifti2
import xmltodict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle as pkl


##################################################################
# Function defitions 
##################################################################

def load_BN_dict(fileName):

    with open(fileName, 'rb') as f:
        BN_dict = pkl.load(f)

    return BN_dict

def load_colors():

    ''' load color scheme '''
    with open('/Users/harmang/Desktop/git_home/labwork/cbcl_neuro_clust/colors.txt') as f:
        colors = [[float(y)/255 for y in x.strip().split(',')] for x in f.readlines()]

    return colors

def repl_header(h, goodKeys, ind):

    ''' Replace items in the header '''

    # Get axes
    ax_0 = h.get_axis(0)
    ax_1 = h.get_axis(1)

    # Store the actual dict
    d = {0: ('???', (1.0, 1.0, 1.0, 0.0))}

    # Get colors WSIR = GBYR
    #colors = load_colors() 
    colors = [[0.0, 1.0, 0.0, 1.0],
              [0.0, 0.0, 1.0, 1.0],
              [1.0, 1.0, 0.0, 1.0],
              [1.0, 0.0, 0.0, 1.0]]

    new_good_keys = {} 

    for ii in range(len(brainLabels)):
        if brainLabels[ii]['#text'] in goodKeys: 
            d[ii] = (brainLabels[ii]['#text'], tuple(colors[ind]))
            new_good_keys[brainLabels[ii]['#text']] = ii
        else:
            d[ii] = ('empt', tuple([0,0,0,0]))

    # # Replace objects for 
    # for ind, net in enumerate(gord_items.keys()):
    #     for ind_2, ii in enumerate(gord_items[net]): 
    #         d[ii + 1] = (net + '_' + str(ii + 1), tuple(colors[ind])) 

    dd = dict(OrderedDict(sorted(d.items())))

    return dd, ax_1, new_good_keys 

def cifti_NIGHTMARE(d, axis_1):

    ''' Cant even begin to describe what this is '''
 
    lable_table = cifti2.Cifti2LabelTable()
    mim = cifti2.Cifti2MatrixIndicesMap([0], 'CIFTI_INDEX_TYPE_LABELS')

    for key, value in d.items():
        lable_table[key] = (value[0],) + tuple(value[1])

    meta = None
    named_map = cifti2.Cifti2NamedMap('', cifti2.Cifti2MetaData(meta), lable_table)
    mim.append(named_map)
    
    rest = cifti2.cifti2_axes.ScalarAxis.from_index_mapping(mim)
    tables = [{key: (value.label, value.rgba) for key, value in nm.label_table.items()} for nm in mim.named_maps]
    sleepy_d = cifti2.cifti2_axes.LabelAxis(rest.name, tables, rest.meta)


    h_out = cifti2.cifti2_axes.to_header((sleepy_d, ax_1))

    return h_out

def load_dlabel(dlabelFile):

    ''' Function to load a dlabel file and return a new file to write out '''
    a = cifti2.load(dlabelFile)

    # Store header
    cHeader = a.header
    cHeaderXML = xmltodict.parse(cHeader.to_xml())

    # Extact label key and name
    cXML = cHeaderXML['CIFTI']['Matrix']['MatrixIndicesMap'][0]
    brainLabels = cXML['NamedMap']['LabelTable']['Label']

    # Store labels and keys
    labelKeys = [int(x['@Key']) for x in brainLabels]
    labelText = [x['#text'] for x in brainLabels]

    # Extract label and vertex keys
    vertInd = np.array(a.get_data())[0]

    # Get parcel by key and index
    labvert = {labelText[x]: np.where(vertInd == x)[0] for x in labelKeys if np.where(vertInd == x)[0].shape[0] != 0}

    return labelKeys, labelText, cHeader, labvert, brainLabels, vertInd.shape[0]


def place_data(goodKeys, labvert, vertIndShape):

    ''' Split this up to try and get consistent header '''

    # Setup shape
    ff = np.zeros((1, vertIndShape))

    for key, ii in goodKeys.items():
        if key in labvert.keys():
            ff[:, labvert[key]] = int(ii)

    

    # # Loop through cbcl_items 
    # for ind, net in enumerate(cbcl_items[spec_item].keys()):
        
    #     # Loop through networks in each item
    #     for ind_2, ii in enumerate(cbcl_items[spec_item][net]):
    #         ff[:, labvert[ii]] = int(ii.split('_')[-1]) 
    

    return ff

def match_good_BN(BN_dict, good_parcs, labVertKeys):

    better_dict = {}

    for key, item in good_parcs.items():

        to_store = [BN_dict[ii]['hemi'] + '_' + BN_dict[ii]['anat_label'] + '_' + BN_dict[ii]['hemi'] for ii in item]
        better_dict[key] = [ii for ii in to_store if ii in labVertKeys] 

    return better_dict
        
    

##################################################################
# Run
##################################################################

parDir = '/Users/harmang/Desktop/git_home/labwork/atlas_meta/atlases/'

# Open dictionary of parcellation items
with open(parDir + 'BN_labels.pkl', 'rb') as f:
    BN_dict = pkl.load(f)

with open(parDir + 'good_parcels.pkl', 'rb') as f:
    good_parcs = pkl.load(f)

# Open cifti dlabel
labelKeys, labelText, cHeader, labvert, brainLabels, vertIndShape = load_dlabel(parDir + 'BN_cifti/fsaverage_LR164k/fsaverage.BN_Atlas.164k_fs_LR.dlabel.nii')
good_labels = match_good_BN(BN_dict, good_parcs, labvert.keys())

for ind, term in enumerate(['working_memory', 'set_shifting', 'inhibitory_control', 'reward']):

    print('Parsing: {}'.format(term))
    dd, ax_1, new_good_keys = repl_header(cHeader, good_labels[term], ind)
    c_out = cifti_NIGHTMARE(dd, ax_1) 
    ff = place_data(new_good_keys, labvert, vertIndShape)

    cOut = cifti2.Cifti2Image(ff, c_out) 
    cifti2.save(cOut, term + '_.dlabel.nii')



if __name__ == "_i_main__":

    # Parent directory

    parDir= "/Users/harmang/Desktop/git_home/labwork/cbcl_neuro_clust/"
    ind_codes = ['all', 'male', 'female']
    ind_use = 1
    cbcl_items = extract_nets(parDir, ind_use)
    f = match_net_gord(parDir, cbcl_items)

    cHeader, labvert, ff = load_dlabel(f)
    d, ax_1 = repl_header(cHeader)
    newHeader = cifti_NIGHTMARE(d, ax_1)
        
    for c_item in f.keys():
        
        print('Parsing: {}'.format(c_item))

        outDat = place_data(f, c_item, labvert, ff)
        cOut = cifti2.Cifti2Image(outDat, newHeader) 
        cifti2.save(cOut, '/Users/harmang/Desktop/ciftis/' + ind_codes[ind_use] + '_' + c_item + '.dlabel.nii')


