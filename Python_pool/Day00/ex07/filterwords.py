#!/usr/bin/env python3
import string
from sys import argv

ret = ""
if len(argv) != 3:
    ret = "ERROR"
new_string = argv[1].translate(str.maketrans('', '', string.punctuation))
new_list = new_string.split()
try:
    nb = int(argv[2])
except ValueError:
    ret = "ERROR"
if ret != "":
    print(ret)
else:
    print([word for word in new_list if len(word) > nb])
