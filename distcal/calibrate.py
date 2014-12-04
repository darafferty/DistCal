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


def calibrate(MSFile, parset, skymodel, ncores=6, solint=1, parmdb=None,
    resume=False, clobber=False, timecorr=False, block=None, ionfactor=1.0,
    solver='BBS'):
    """
    Runs BBS in distributed manner with or without time-correlated solve
    """
    logfilename = MSFile + '.distcal.log'
    init_logger(logfilename)
    log = logging.getLogger("DistCal.calibrate")

    # Start iPython engines
    lb = loadbalance.LoadBalance(ppn=ncores, logfile=None)
    lb.sync_import('from distcal.libs import *')
    nengines = len(lb.rc)

    band = Band(MSFile, timecorr, block, solint, ionfactor, nengines, resume,
        parset, skymodel, parmdb, clobber, solver)
    chunk_list, chunk_list_full = makeChunks(band)

    if chunk_list is None or chunk_list_full is None:
        return
    else:
        if len(chunk_list) > 0:
            # Give all chunks a random start delay to avoid too much disk IO
            delays = numpy.random.random_sample(len(chunk_list)) * 10.0 # seconds
            for chunk, delay in zip(chunk_list, delays):
                chunk.start_delay = delay
            lb.map(runChunk, chunk_list)

        collectSols(band, chunk_list_full)
