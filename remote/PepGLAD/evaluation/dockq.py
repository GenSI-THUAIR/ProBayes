#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re

from remote.PepGLAD.globals import DOCKQ_DIR
from remote.PepGLAD.evaluation.DockQ.DockQ import run_DockQ

def check_and_convert_to_absolute(path):
    if not os.path.isabs(path):  
        path = os.path.abspath(path) 
    return path

def dockq(mod_pdb: str, native_pdb: str, pep_chain: str):
    # print(mod_pdb, native_pdb)
    try:
        return run_DockQ([mod_pdb], [native_pdb], pep_chain)
    except:
        return 0
    # return run_DockQ([os.path.abspath(mod_pdb)], [os.path.abspath(native_pdb)], pep_chain)

    p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {native_pdb} -model_chain1 {pep_chain} -native_chain1 {pep_chain} -no_needle')
    text = p.read()
    p.close()
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    return score