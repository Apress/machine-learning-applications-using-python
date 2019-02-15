# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 15:07:13 2018

@author: PUNEETMATHUR
"""
#Baby Boomers Five Point Summary example
import numpy as np
bboomers = np.array([14230, 345, 1912, 472, 63, 861, 270, 713])
fivepoints= [np.min(bboomers), np.percentile(bboomers, 25, interpolation='midpoint'), np.median(bboomers),np.percentile(bboomers, 75, interpolation='midpoint'),np.max(bboomers)]
for fivepointsummary in fivepoints:
     print(fivepointsummary)
     
#Millenials Five Point Summary example
import numpy as np
millennials  = np.array([12519, 845, 912, 72, 93, 615, 70, 538])
fivepoints= [np.min(millennials ), np.percentile(millennials , 25, interpolation='midpoint'), np.median(millennials ),np.percentile(millennials , 75, interpolation='midpoint'),np.max(millennials)]
for fivepointsummary in fivepoints:
    print(fivepointsummary)

