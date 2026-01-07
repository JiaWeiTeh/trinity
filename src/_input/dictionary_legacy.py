#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# legacy dictionary, before update.

import numpy as np
import collections.abc
from functools import reduce
import os
import json
import sys
#--

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

    # TODO: add unit=None, with attribute .units. 
    # this can then work when print valeus.

class DescribedItem:
    def __init__(self, value = None, info = None, ori_units = None, exclude_from_snapshot = False, isPersistent = False):
        self.value = value
        self.info = info
        self.ori_units = ori_units
        self.exclude_from_snapshot = exclude_from_snapshot
        self.isPersistent = isPersistent
        
    def __str__(self):
        return (f"{self.value}\t({self.info})")
    
    def __repr__(self):
        return str(self.value)

    # Arithmetic operations
    def __add__(self, other): return self.value + other
    def __radd__(self, other): return other + self.value
    def __sub__(self, other): return self.value - other
    def __rsub__(self, other): return other - self.value
    def __mul__(self, other): return self.value * other
    def __rmul__(self, other): return other * self.value
    def __truediv__(self, other): return self.value / other
    def __rtruediv__(self, other): return other / self.value
    def __pow__(self, other): return self.value ** other
    def __rpow__(self, other): return other ** self.value

    def __neg__(self): return -self.value
    def __abs__(self): return abs(self.value)

    # Comparison
    def __eq__(self, other): return self.value == other
    def __lt__(self, other): return self.value < other
    def __le__(self, other): return self.value <= other
    def __gt__(self, other): return self.value > other
    def __ge__(self, other): return self.value >= other

    # NumPy compatibility
    def __float__(self): return float(self.value)
    def __int__(self): return int(self.value)
    def __array__(self, dtype=None): return np.array(self.value, dtype=dtype)
    

class DescribedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialise snapshot counter.
        self.save_count = 0
        # save snapshot in intervals of?
        self.snapshot_interval = 25
        # which snapshots are stored. List of str.
        # self.snapshots_stored = []
        self.previous_snapshot = {}
        # how much zero padding? 6 = 000001 
        # minimum 4 (allows up to 9999 saves)
        # internal, dont change if not sure.
        self._zero_padding = 4
        # tracks how many times JSON file is saved
        self.flush_count = 0
        # mark keys to exclude from snapshots
        self._excluded_keys = set()
        # mark keys to remain within dictionary even after .flush()
        self._persistent_keys = set()
        # store persistent data
        self._persistent_data = {}
        
    # Mark a key to be excluded from snapshots.
    def exclude_key(self, key): self._excluded_keys.add(key)

    # Unmark a key (include it back in snapshot).
    def include_key(self, key): self._excluded_keys.discard(key)
    
    # Mark a key to be excluded from flush and stays in self
    def mark_persistent(self, key): self._persistent_keys.add(key)

    # Unmark a key to be excluded from flush and stays in self
    def unmark_persistent(self, key): self._persistent_keys.discard(key)

    # Option to check if a key is excluded or persistent
    def is_excluded(self, key): return key in self._excluded_keys
    def is_persistent(self, key): return key in self._persistent_keys
    
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
                # if it has length but is not string.
                if hasattr(val.value, "__len__") and type(val.value) != str:
                    shortened_val = self.shorten_display(val.value)
                    val_str = f"{key} : {shortened_val}\n" 
            custom_str += val_str
        custom_str += 'saved snapshot(s): ' + str(self.save_count) + '\n'
        return custom_str + "-------------"
    
    def __setitem__(self, key, value):
        
        # """
        # # foolproof - when trying to store value to key that does not exist.
        # # Also to remind adding DescribedItem.
        # """
        if not isinstance(value, DescribedItem):
            raise TypeError(f"Value assigned to '{key}' must be a DescribedItem instance. You might be thinking of using dict['key'].value = val instead.")
        
        # default is False (third keyword)
        if getattr(value, "exclude_from_snapshot", False):
            self._excluded_keys.add(key)
        
        super().__setitem__(key, value)
        
        return 
        
    @staticmethod
    def simplify(x_arr, y_arr, nmin = 100, grad_inc = 1, keyname = ''):
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
        # print(f'simplifying {keyname}...')
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
    
        # print('Simplification complete')
        return new_x, new_y
            
    def save_snapshot(self):
        """
        # saves all current keys and values and add suffix for snapshots
        
        """
        
        # no need to save if the time is the same (to avoid duplicates) and if dictionary is empty
        if self.save_count >= 1 and self.previous_snapshot:
            if self['t_now'] == self.previous_snapshot[str(self.save_count-1)]['t_now'] or\
                self['R2'] == self.previous_snapshot[str(self.save_count-1)]['R2']:
                print(f"duplicate detected in save_snapshot at t = {self['t_now']}. Snapshot not saved.")
                return
        
        # clean and simplify first (remove .info etc)
        clean_dict = self.clean()
        
        # for k, v in clean_dict.items():
        #     print(type(v))
        
        self.previous_snapshot[str(self.save_count)] = clean_dict
        self.save_count += 1

        if self.save_count % self.snapshot_interval == 0:
            print('flushing dictinoary...')
            self.flush()
            print('All snapshots flushed to JSON at t = ', self['t_now'].value)
        else:
            print('Current snapshot saved at t = ', self['t_now'].value)

            
        return
    
    def flush(self):
        """
        After certain amount of snapshots (see self.snapshot_inverval), flush 
        output into a JSON file. 
        
        If you want to use this function as a standalone, make sure you've already done
        .save_snapshot() beforehand. 
        """
                
        path2output = self['path2output'].value
        path2json = os.path.join(path2output, 'dictionary' + '.json')
        
        load_list = None
        
        # Three cases: file exists; file exists but zero byte; file does not exist.
        # if file does not exist, then create it.
        # if file exists, but it shouldn't be (i.e., simulation just started), then remove it.
        if not os.path.exists(path2json) or (os.path.exists(path2json) and self.flush_count == 0):
            with open(path2json, 'w') as infile:
                infile.close()
            print('Initialising JSON file for saving purpose...')
                
        
        else: 
        #---
        # if it is an empty file, 
            try:
                with open(path2json, 'r') as infile:
                    load_list = json.load(infile)
                    infile.close()
                    
            except json.decoder.JSONDecodeError as e:
                print(f'Exception: {e} catched; file is probably empty.')
                
            except Exception as e:
                print(f'Something else went wrong in .flush(): {e}')
                sys.exit()
            
        # If it is None, return an empty list instead
        if load_list is None:
            load_list = {}
    
        # combine to record
        combined_list = {**load_list, **self.previous_snapshot}
        
        # then update dictionary
        with open(path2json, 'w') as outfile:
            print('Updating dictionary in .flush()')
            # json.dump(combined_list, outfile)
            json.dump(combined_list, outfile, cls = NpEncoder, indent = 2)
            outfile.close()

        # update
        self.flush_count += 1
        self.previous_snapshot = {}
        
        return
    
    def get_snapshot(t_now):
        # is this worthwhile? because we flush every n interval. The snapshot
        # that is obtained here might not mean anything.
        # Idea: given a time, get the interpo;lation between the time before and after
        # such timeframe.
        return 
    
    def clean(self):
        """
        clean and simplify before saving.
        """
        # create new dictionary
        new_dict = {}
        
        if self.save_count == 0:
            for key, val in self.items():
                if val.exclude_from_snapshot:
                    self._excluded_keys.add(key)
                if val.isPersistent:
                    self._persistent_keys.add(key)
            
        # start iteration
        for key, val in self.items():
            if key in self._excluded_keys:
                continue
            if key in self._persistent_keys:
                # bring the DescribedItem() forward.
                self._persistent_data[key] = val
            # only clean original dictionary; skip snapshots
            if isinstance(val, DescribedItem):
            # if not key.beginswith('_sS'):
                # no need to store the description
                val = val.value
                # if a single string, float or value, save it einfach
                # if (type(val) is str) or (type(val) is float) or (type(val) is int):
                if isinstance(val, (str, float, int)):
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
                        new_r, new_val = self.simplify(x_arr, y_arr, keyname = key)
                        # record
                        new_dict['log_' + key] = np.array(new_val)
                        new_dict[key+'_r_arr'] = np.array(new_r)
                        continue
                    
                    if key == 'bubble_dTdr_arr':
                        # - log these
                        # todo: what happens when its all zero? need to add an if/else
                        x_arr = self['bubble_r_arr'].value
                        y_arr = np.log10(-val)    
                        new_r, new_val = self.simplify(x_arr, y_arr, keyname = key)
                        # record
                        new_dict['log_' + key] = np.array(new_val)
                        new_dict[key+'_r_arr'] = np.array(new_r)
                        continue
                                        
                    if key == 'bubble_v_arr':
                        # get new points
                        x_arr = self['bubble_r_arr'].value
                        y_arr = val
                        new_r, new_val = self.simplify(x_arr, y_arr, keyname = key)
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
                        new_r, new_val = self.simplify(self['shell_grav_r'].value, np.log10(val), keyname = key)
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
    
    
# some lazy handles so that i can save just calling updateDict(params_name, values).
# not sure if this is good.
def updateDict(dictionary, keys, values):
    # sanity check
    if len(keys) != len(values):
        raise ValueError('Length of keys must match length of values.')
    for key, val in zip(keys, values):
        dictionary[key].value = val
        
        