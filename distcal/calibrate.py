"""
Functions to perform distributed calibration
"""
import logging
import os
import sys
import glob
import numpy
import lofar.parmdb
import lofar.parameterset
import pyrap.tables
import loadbalance
from .libs import *


def calibrate(MSFile, parset, skymodel, ncores=6, solint=1, parmdb='instrument',
    resume=False, clobber=False, timecorr=False, block=None, ionfactor=1.0):
    """
    Runs BBS in distributed manner with or without time-correlated solve
    """
    logfilename = MSFile + '.distcal.log'
    init_logger(logfilename)
    log = logging.getLogger("DistCal.calibrate")

    # Start iPython engines
    lb = loadbalance.LoadBalance(ppn=ncores, logfile=None,
        loglevel=logging.DEBUG)
    lb.sync_import('from distcal.libs import *')

    band = Band(MSFile, timecorr, block, solint, ionfactor, len(lb.rc), resume,
        parset, skymodel, parmdb)
    chunk_list, chunk_list_full = makeChunks(band)

    if chunk_list is None or chunk_list_full is None:
        return
    else:
        if len(chunk_list) > 0:
            for i, chunk in enumerate(chunk_list):
                chunk.start_delay = i * 10.0 # start delay in seconds to avoid too much disk IO
            lb.map(runChunk, chunk_list)

        collectSols(band, chunk_list_full)
