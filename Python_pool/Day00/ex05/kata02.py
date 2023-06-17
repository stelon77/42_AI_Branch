#!/usr/bin/env python3
"""
outputs date format
"""

from datetime import datetime
t = (3, 30, 2019, 9, 25)

print('{:%m/%d/%Y %H:%M}'.format(datetime(t[2], t[3], t[4], t[0], t[1])))
