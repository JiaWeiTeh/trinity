#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:04:04 2024

@author: Jia Wei Teh

A script that calculates time elapsed, for debugging and performance purposes.
"""

#python3
import humanfriendly
from time import time
from datetime import timedelta

class Timer:
    """
    Timer class that calculates time elapses and prints it out 
        in a human-friendly way. Based on .datetime and .humanfriendly.
    Uses: 1) from clock import timer
          2) _timer.begin('optional str here')
          3) _timer.end() 
          4) proift
    """
    
    # initialisation
    def __init__(self):
        # start time
        self.start = None
        # end time
        self.stop = None
    
    # converts time elapsed into string
    def secs2str(self):
        # calculate difference
        elapsed = timedelta(seconds = self.stop - self.start)
        # return in formatted string
        return humanfriendly.format_timespan(elapsed)

    # sets beginning of timer
    def begin(self, s = ''):
        print('~'*20)
        print('Timer begins. ' + s) #display message if requried
        print('~'*20)
        # record the beginning time
        self.start = time()

    # sets end of timer
    def end(self):
        # make sure start is evoked:
        if self.start == None:
            raise InvalidCall('_timer.end() called, but .begin() not detected.')
        # record the end time
        self.stop = time()
        # then, print out time elapsed.
        print('~'*20)
        print(f'Timer ends. Time elapsed: {self.secs2str()}.')
        print('~'*20)
        # reset
        self.start = None
        self.stop = None

class InvalidCall(Exception):
    """
    Raised when timer call is invalid. 

    For example: calling _timer.end() without explicitly calling _timer.begin().
    """
    
# lazy initialisation
_timer = Timer()




