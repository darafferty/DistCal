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

def calibrate(MSfile, parset, skymodel, ncores=6, solint=1, parmdb='instrument',
    resume=False, clobber=False, timecorr=False, block=None, ionfactor=None):
    """
    Runs BBS in distributed manner with or without time-correlated solve
    """
    logfilename = MSFile + '.distcal.log'
    init_logger(logfilename)
    log = logging.getLogger("DistCal.runMS")

    # Start iPython engines
    lb = loadbalance.LoadBalance(ppn=options.ncores, logfile=None,
        loglevel=logging.DEBUG)
    lb.sync_import('from distcal.libs import *')

    band = Band(MSfile, outdir, timecorr, block, ionfactor, ncores, resume)
    chunk_list, chunk_list_full = makeChunks(band)

    if chunk_list is None or chunk_list_full is None:
        return
    else:
        if len(chunk_list) > 0:
            for i, chunk in enumerate(chunk_list):
                chunk.start_delay = i * 10.0 # start delay in seconds to avoid too much disk IO
            lb.map(runChunk, chunk_list)

        # Copy over the solutions to the final output parmdb
        try:
            log.info('Copying distributed solutions to final parmdb...')
            instrument_out = '{0}/{1}'.format(MSfile, parmdb)
            os.system("rm %s -rf" % instrument_out)
            pdb_out = lofar.parmdb.parmdb(instrument_out, create=True)
            for j, chunk_obj in enumerate(chunk_list_orig):
                chunk_instrument = chunk_obj.output_instrument
                try:
                    pdb_part = lofar.parmdb.parmdb(chunk_instrument)
                except:
                    continue
                log.info('  copying part{0}'.format(j))
                for parmname in pdb_part.getNames():
                    if j == 0 or 'Phase' in parmname:
                        v = pdb_part.getValuesGrid(parmname)
                        try:
                            pdb_out.addValues(v)
                        except:
                            continue
        except Exception as e:
            log.error(str(e))
