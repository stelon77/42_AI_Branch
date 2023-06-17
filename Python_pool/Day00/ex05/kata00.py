#!/usr/bin/env python3
"""
write the numbers included in tupple 't'
"""

t = (5, 35, 67)

if len(t) == 0:
    print("no number provided")
else:
    print("the", len(t), "numbers are :", ', '.join([str(i) for i in t]))
