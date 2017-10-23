#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pandoc command to build everything:
#   pandoc -t latex --toc -V colorlinks=true -V geometry:margin=1in out.md

import glob
import pprint
import os
import re

# See: http://stackoverflow.com/a/4836734
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# Remove out.md first.
if os.path.isfile("out.md"):
    os.remove("out.md")

files = natural_sort(glob.glob("./*.md"))
pp = pprint.PrettyPrinter(indent=2)

print("concatenating following files to out.md:")
pp.pprint(files)
print("(we will exclude README.md if that's present)")

with open("out.md", "w") as f:
    for filename in files:
        if os.path.basename(filename) != "README.md":
            with open(filename) as read:
                f.write(read.read())
                f.write("\n\n")
