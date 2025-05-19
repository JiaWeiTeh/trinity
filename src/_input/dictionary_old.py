#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:21:52 2023

@author: Jia Wei Teh

Dictionary class. Handles the main bulk of parameter saving and calling.
"""

import numpy as np
import collections.abc
from functools import reduce
import os
import json
#--
# from numba import njit, jitclass

# for JSON file saving
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):             
            return bool(obj)
        return super(NpEncoder, self).default(obj)
        
    
    # TODO: perhaps in the future add in numba jitclass to optimise. 
    # this requires, however, set type for .value and .info. 
    # maybe force them to be np.arrays. idk now. 
    # see: https://numba.pydata.org/numba-doc/dev/user/jitclass.html


class DescribedItem:
    def __init__(self, value, info):
        self.value = value
        self.info = info
        
    def __str__(self):
        return (f"{self.value} ({self.info})")

class DescribedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialise snapshot counter.
        self.save_count = 1
        # save snapshot in intervals of 3 = 10^(snapshot_interval-1) = 1e2
        self.snapshot_interval = 3
        # which snapshots are stored. List of str.
        self.snapshots_stored = []
        # how much zero padding? 6 = 000001 
        # minimum 4 (allows up to 9999 saves)
        # internal, dont change if not sure.
        self._zero_padding = 4
        # tracks how many times JSON file is saved
        self.flush_count = 0
    
    @staticmethod
    def shorten_display(arr, nshow = 3):
        """
        # shorten an array so we don't print too much and clog the output.
        # nshow: number of elements you want to show beginning/end.
        """
        if len(arr) > 10:
            arr = list(arr[:nshow]) + ['...'] + list(arr[-nshow:]) 
        return arr
    
    def __str__(self):
        """
        # Customize the printed string for the dictionary here
        """
        custom_str = "\n-------------\ncontents\n========\n"
        sorted_items = sorted(self.items()) #alphabetically
        val_str = ""
        snapshots_stored = []
        for key, val in sorted_items:
            # do not display too much snapshot details
            # identify snapshot saves by type; they will not be DescribedItem
            if not isinstance(val, DescribedItem):
                # check if already in 
                if str(key[3:3+self._zero_padding]) not in snapshots_stored:
                    snapshots_stored.append(str(key[3:3+self._zero_padding]))
                # exit for loop.
                continue
            val_str = f"{key} : {val}\n"
            if isinstance(val.value, (collections.abc.Sequence, np.ndarray)):
                if hasattr(val.value, "__len__"):
                    shortened_val = self.shorten_display(val.value)
                    val_str = f"{key} : {shortened_val}\n" 
            custom_str += val_str
        custom_str += 'saved snapshot(s): ' + str(self.shorten_display(snapshots_stored)) + '\n'
        num_snapshotkeys = len([k for k in self.keys() if k.startswith('_sS')])
        custom_str += f'{num_snapshotkeys} snapshot key(s) hidden. Access them by adding _sS(number) as suffix to keys.\n'
        return custom_str + "-------------"
    
    def __setitem__(self, key, value):
        """
        # foolproof - when trying to store value to key that does not exist.
        # Also to remind adding DescribedItem.
        """
        # allow however snapshot (keys with 'sS') to bypass this rule.
        if not isinstance(value, DescribedItem) and 'sS' not in key:
            raise KeyError(f"'{value}' does not belong to the class DescribedItem().")
        super().__setitem__(key, value)
        
    @staticmethod
    def simplify(x_arr, y_arr, nmin = 100, grad_inc = 1):
        """
        this function simplifies/shortens arrays adaptively, i.e., less
        points where gradient is small and more points when curve is steep.
        
        Some problems with inaccurate nsimp due to inflection points and 
        change in gradients. Nonetheless it gets the job done.
        
        Parameters
        ----------
        x_arr : x array
        y_arr : y array
        nsimp : number of elements after simplification (not accurate)
    
        Returns
        -------
        x_arr, y_arr with length of desired nsimp.
    
        """
        
        # if number given is larger than array size.
        if nmin >= len(x_arr):
            return x_arr, y_arr
    
        # set minimum
        if nmin < 100:
            nmin = 100   
            
        if len(x_arr) == 0 or len(y_arr) == 0:
            return x_arr, y_arr
        
        # add 1e-15 to avoid zeros in division.
        grad = np.gradient(y_arr) + 1e-15
        
        #-- increase in gradient over 1 = 100% to capture (spikes) in both directions
        percentage_increase = np.diff(grad) / grad[:-1]
        
        important_percent = np.where(np.abs(percentage_increase) > grad_inc)[0]
        
        #-- gradient change
        important_gradchange = np.where(np.diff(np.sign(grad)) != 0)[0]
        
        #-- increase in absolute value
        maximum_distance = (max(y_arr)-min(y_arr))/(nmin)
        
        # first calculate the difference
        y_diff = np.abs(y_arr[1:] - y_arr[:-1])
        # use cumsum to find difference and factors
        y_diff_cumsum = np.cumsum(y_diff)
        y_diff_cumsum_factor = (y_diff_cumsum/maximum_distance).astype(int)
        # which values differ?
        # print(y_diff_cumsum_factor)
        idx_diff = np.where(y_diff_cumsum_factor[:-1] != y_diff_cumsum_factor[1:])[0] 
        
        #-- merge
        merged = reduce(np.union1d, (0, important_percent, important_gradchange, len(x_arr) - 1), idx_diff)
    
        new_y = y_arr[merged]
        new_x = x_arr[merged]
    
        return new_x, new_y
            
    def save_snapShot(self):
        """
        # saves all current keys and values and add suffix for snapshots
        
        """
        # clean and simplify first
        save_dict = self.clean()
        for key, val in save_dict.items():
            # pad to four digits so it's easier for str sorting
            # format: _sS001_key (snapshot 1)
            new_key = f'_sS{str(self.save_count).zfill(self._zero_padding)}_{key}'
            self[new_key] = val
            # update list
            if new_key[3:self._zero_padding+3] not in self.snapshots_stored:
                self.snapshots_stored.append(str(new_key[3:self._zero_padding+3]))

        if self.save_count % 10**(self.snapshot_interval-1) == 0:
            self.flush()
            print('Snapshot saved to JSON at t = ', self['t_now'].value)
        else:
            self.save_count += 1
            print('Snapshot saved at t = ', self['t_now'].value)
            
        return
    
    def flush(self):
        """
        After certain amount of snapshots (see self.snapshot_inverval), flush 
        output into a JSON file. 
        """
            
        save_dict = {}
        
        for snapshots in self.snapshots_stored:
            # for each snapshot create its own dictionary 
            snapshot_dict = {k:v for k, v in self.items() if k.startswith('_sS'+str(snapshots))}
            # save key to new dictionary
            save_dict[snapshots] = snapshot_dict
            
        # remove snapshots
        for k, v in list(self.items()):
            if k.startswith('_sS'):
                self.pop(k, None)
                
        path2trinity = os.environ['path2trinity']
        path2json = os.path.join(path2trinity, 'dictionary' + '.json')
        
        # Three cases: file exists; file exists but zero byte; file does not exist.
        
        # if file does not exist, then create it.
        # if file exists, but it shouldn't be (i.e., simulation just started), then remove it.
        if not os.path.exists(path2json) or (os.path.exists(path2json) and self.save_count <= 10**(self.snapshot_interval-1)):
            with open(path2json, 'w') as infile:
                json.dump({}, infile)
                infile.close()

        # Once empty file is created, then load it
        with open(path2json, 'r+') as infile:
            load_dict = json.load(infile)
            infile.close()
         
        # then update dictionary
        with open(path2json, 'w') as outfile:
            save_dict.update(load_dict)
            json.dump(save_dict, outfile, cls = NpEncoder)
            outfile.close()

        # update
        self.save_count += 1
        self.flush_count += 1
            
        return
    
    def get_snapshot(t_now):
        # is this worthwhile? because we flush every n interval. The snapshot
        # that is obtained here might not mean anything.
        # Idea: given a time, get the interpo;lation between the time before and after
        # such timeframe.
        return 
    
    # @njit
    def clean(self):
        """
        clean and simplify before saving.
        """
        # create new dictionary
        new_dict = {}
        # remove unnecessary details
        skip_key = ['SB99_data', 'SB99f', 'cStruc_cooling_CIE_interpolation', 
                    'cStruc_cooling_CIE_logLambda', 'cStruc_cooling_CIE_logLambda', 
                    'cStruc_cooling_CIE_logT', 
                    'cStruc_cooling_nonCIE', 
                    'cStruc_heating_nonCIE', 
                    'cStruc_net_nonCIE_interpolation']
        # start iteration
        for key, val in self.items():
            if key in skip_key:
                continue
            # only clean original dictionary; skip snapshots
            if isinstance(val, DescribedItem):
            # if not key.beginswith('_sS'):
                # no need to store the description
                val = val.value
                # if a single string, float or value, save it einfach
                if (type(val) is str) or (type(val) is float) or (type(val) is int):
                # if isinstance(val, (str, float, int)):
                    new_dict[key] = val
                else:
                    # if the key is linked to bubble or shell strcuture, simplify it.
                    # -- if bubble
                    if key == 'bubble_r_arr':
                        # dont use this
                        continue
                    
                    if key in ['bubble_T_arr', 'bubble_n_arr']:
                        # log these
                        x_arr = self['bubble_r_arr'].value
                        y_arr = np.log10(val)    
                        new_r, new_val = self.simplify(x_arr, y_arr)
                        # record
                        new_dict['log_' + key] = np.array(new_val)
                        new_dict[key+'_r_arr'] = np.array(new_r)
                        continue
                    
                    if key == 'bubble_dTdr_arr':
                        # - log these
                        # todo: what happens when its all zero? need to add an if/else
                        x_arr = self['bubble_r_arr'].value
                        y_arr = np.log10(-val)    
                        new_r, new_val = self.simplify(x_arr, y_arr)
                        # record
                        new_dict['log_' + key] = np.array(new_val)
                        new_dict[key+'_r_arr'] = np.array(new_r)
                        continue
                                        
                    if key == 'bubble_v_arr':
                        # get new points
                        x_arr = self['bubble_r_arr'].value
                        y_arr = val
                        new_r, new_val = self.simplify(x_arr, y_arr)
                        # record
                        new_dict[key] = np.array(new_val)
                        new_dict[key+'_r_arr'] = np.array(new_r)
                        continue
                    
                    # -- if shell
                    if key == 'shell_grav_r':
                        # dont use this
                        continue
                    if key in ['shell_grav_force_m']:
                        # get new points
                        new_r, new_val = self.simplify(self['shell_grav_r'].value, np.log10(val))
                        # record
                        new_dict[key] = np.array(new_val)
                        new_dict['shell_grav_r'] = np.array(new_r)
                        continue            
                    # otherwise make list of list.
                    # if (type(val) is list) or (type(val) is np.ndarray):
                    if isinstance(val, (collections.abc.Sequence, np.ndarray)):
                        new_dict[key] = np.array(val)
                    else:
                        new_dict[key] = val
        return new_dict
    
    

def updateDict(dictionary, keys, values):
    # sanity check
    if len(keys) != len(values):
        raise ValueError('Length of keys must match length of values.')
    for key, val in zip(keys, values):
        dictionary[key].value = val
        
        
        
        