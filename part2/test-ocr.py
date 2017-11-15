# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:51:15 2017

@author: PulkitMaloo
"""
fname = "test-strings.txt"
with open(fname) as fhand:
    for i in range(19):
        print "####################################"
        print "Correct:", fhand.readline(),
        eval("""runfile('C:/Users/PulkitMaloo/Box Sync/Courses/AI/Assignment_3/hankjang-maloop-shynaras-a3/part2/ocr.py', args='courier-train.png bc.train test-"""
                        + str(i) + """-0.png', wdir='C:/Users/PulkitMaloo/Box Sync/Courses/AI/Assignment_3/hankjang-maloop-shynaras-a3/part2')""")