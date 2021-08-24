#!/Users/harmang/anaconda3/bin/python

import pandas as pd
import json
import pickle as pkl


'''
Script to create a more useful brainnetome parcel label data structure
'''


def load_xl():

    ''' Load the subregions '''

    df = pd.read_excel('BNA_subregions.xlsx')

    for ii in df.columns:
        df[ii] = df[ii].fillna(method = 'ffill')

    return df


def load_json():

    ''' Load the json with associated function of each parcel '''

    with open('BDf_FDR05.json') as jsonFile:
        
        data = json.load(jsonFile)

    return data


def ret_dict(r, func_items, lr):

    ''' Return a dictionary for each parcel ID '''
    anat_cyto = r['Unnamed: 5'].split(',')

    d = {'hemi': lr,
         'lobe': r['Lobe'].replace(' ', ''),
         'gyrus': r['Gyrus'].replace(' ', ''),
         'bn_name': r['Left and Right Hemisphere'],
         'anat_label': anat_cyto[0],
         'cyto_label': anat_cyto[-1].replace(' ', ''),
         'functions': list(func_items.keys())}

    return d

# Load files and create empty dict
df = load_xl()
jf = load_json()
BN_dict = {}

# Iterate through all parcels
for ii in range(len(df)):
  
    # Do same for left and right
    BN_dict[df['Label ID.L'][ii]] = ret_dict(df.loc[ii], jf[str(df['Label ID.L'][ii])], 'L')
    BN_dict[df['Label ID.R'][ii]] = ret_dict(df.loc[ii], jf[str(df['Label ID.R'][ii])], 'R')

with open('BN_labels.pkl', 'wb') as f:
    pkl.dump(BN_dict, f)
    


