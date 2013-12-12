#!/usr/bin/env python2.7

"""
Extract the Cls from the chain CHAIN_FILE into a numpy file named CHAIN_FILE.cls.npy

Usage:
    extract_cls.py CHAIN_FILE
"""

import sys
from cosmoslik_plugins.samplers.metropolis_hastings import load_chain
import numpy as np

burnin = 1000
thin = 10

chain = load_chain(sys.argv[1]).burnin(burnin).join().thin(thin)
# chain = load_chain('data1/run1_lcdm.chain').burnin(burnin).join().thin(thin)
cls = np.array([x['cl_TT'] for x in chain['cl_TT']])
parnames = np.array(chain.keys())
parvls = np.transpose(np.array([chain[pn] for pn in parnames if pn!="cl_TT" ]))
parnames_iterable = np.array([pn for pn in parnames if pn!="cl_TT" ])


np.save(sys.argv[1]+'.cls',cls)
np.save(sys.argv[1]+'.parnames',parnames_iterable)
np.save(sys.argv[1]+'.parvls',parvls)
