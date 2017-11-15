# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:51:15 2017

@author: PulkitMaloo
"""
import os

fname = "test-strings.txt"

curr_path = os.getcwd()
ocr_path = os.path.join(curr_path, "ocr.py")


with open(fname) as fhand:
    for i in range(19):
        print "####################################"
        print "Correct:", fhand.readline(),
        # use this to run on server
        os.system('python ocr.py '+'courier-train.png bc.train test-'+str(i)+'-0.png')
        # use this to run on spyder
#        runfile(ocr_path, args='courier-train.png bc.train test-'+str(i)+'-0.png', wdir=curr_path)