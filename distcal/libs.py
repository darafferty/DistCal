"""
Functions and classes needed for distributed calibration
"""
import logging
import os
import commands
import subprocess
import glob
import shutil
import sys
import numpy as np
import pyrap.tables as pt
import scipy.signal
import lofar.parameterset
import lofar.parmdb
import lofar.expion.parmdbmain
import multiprocessing
import multiprocessing.pool
from numpy import sum, sqrt, min, max, any
from numpy import argmax, argmin, mean, abs
from numpy import int32 as Nint
from numpy import float32 as Nfloat
import copy
import socket
import time


def init_logger(logfilename, debug=False):
    if debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    # Remove any existing handlers
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])

    # File handler
    fh = logging.FileHandler(logfilename, 'a')
    fmt = MultiLineFormatter('%(asctime)s:: %(name)-6s:: %(levelname)-8s: '
        '%(message)s', datefmt='%a %d-%m-%Y %H:%M:%S')
    fh.setFormatter(fmt)
    logging.root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    fmt = logging.Formatter('\033[31;1m%(levelname)s\033[0m: %(message)s')
    ch.setFormatter(fmt)
    logging.root.addHandler(ch)


class MultiLineFormatter(logging.Formatter):
    """Simple logging formatter that splits a string over multiple lines"""
    def format(self, record):
        str = logging.Formatter.format(self, record)
        header, footer = str.split(record.message)
        str = str.replace('\n', '\n' + ' '*len(header))
        return str


def makeChunks(band):
    """
    Returns list of chunk objects input band object
    """
    logfilename = band.msname + '.distcal.log'
    init_logger(logfilename)
    log = logging.getLogger("DistCal.makeChunks")

    # Wrap everything in a try-except block to be sure any exception is caught
    try:
        dataset = band.msname
        blockl = band.sol_block
        solint = band.solint
        ionfactor = band.ionfactor
        parset = band.parset
        if solint < 1:
            solint = 1

        # Get time per sample and number of times
        t = pt.table(dataset, readonly=True, ack=False)
        for t2 in t.iter(["ANTENNA1","ANTENNA2"]):
            if (t2.getcell('ANTENNA1',0)) < (t2.getcell('ANTENNA2',0)):
                timepersample = t2[1]['TIME']-t2[0]['TIME'] # sec
                trows = t2.nrows()
        t.close()

        # Calculate various intervals
        if band.timecorr:
            fwhm_min, fwhm_max = modify_weights(msname, ionfactor, dryrun=True) # s
            if blockl is None:
                # Set blockl to enclose the max FWHM and be divisible by 2 and by solint
                blockl = int(np.ceil(fwhm_max / timepersample / 2.0 / solint) * 2 * solint)
        else:
            if blockl is None:
                # Set blockl to get one chunk per core
                blockl = int(np.ceil(trows / float(solint) / band.ncores))

        tdiff = solint * timepersample / 3600. # difference between chunk start times in hours
        tlen = timepersample * np.float(blockl) / 3600. # length of block in hours
        if not band.timecorr:
            nsols = int(np.ceil(trows / float(solint) / blockl)) # number of chunks
        else:
            nsols = int(np.ceil(trows / float(solint))) # number of solutions

        if band.timecorr:
            log.info('Performing distributed time-correlated calibration for {0}...\n'
                '      Time per sample: {1} (s)\n'
                '      Samples in total: {2}\n'
                '      Block size: {3} (samples)\n'
                '                  {4} (s)\n'
                '      Solution interval: {5} (samples)\n'
                '      Number of solutions: {6}\n'
                '      Ionfactor: {7}\n'
                '      FWHM range: {8} - {9} (s)'.format(msname, timepersample,
                trows, blockl, tlen*3600.0, solint, nsols, ionfactor, fwhm_min, fwhm_max))
        else:
            log.info('Performing distributed calibration for {0}...\n'
                '      Time per sample: {1} (s)\n'
                '      Samples in total: {2}\n'
                '      Block size: {3} (samples)\n'
                '                  {4} (s)\n'
                '      Solution interval: {5} (samples)\n'
                '      Number of solutions: {6}\n'
                '      Ionfactor: {7}'.format(msname, timepersample,
                trows, blockl, tlen*3600.0, solint, nsols, ionfactor))


        # Update cellsize and chunk size of parset
        if band.timecorr:
            parset = update_parset(parset)

        # Set up the chunks
        chunk_list = []
        if not band.timecorr:
            chunk_mid_start = 0
            chunk_mid_end = nsols
            tdiff = tlen
        else:
            tlen_mod = tlen / 2.0 # hours
            chunk_mid_start = blockl / 2 / solint
            chunk_mid_end = nsols - blockl / 2 / solint
        for c in range(nsols):
            chunk_obj = Chunk(dataset)
            chunk_obj.chunk = c
            chunk_obj.outdir = '{1}_temp'.format(os.path.basename(chunk_obj.dataset))
            if not os.path.exists(chunk_obj.outdir):
                os.mkdir(chunk_obj.outdir)
            if c < chunk_mid_start:
                chunk_obj.trim_start = True
                chunk_obj.t0 = 0.0 # hours
                chunk_obj.t1 = np.float(chunk_obj.t0) + tlen_mod # hours
                tlen_mod += tdiff # add one solution interval (in hours)
            elif c > chunk_mid_end:
                tlen_mod -= tdiff # subtract one solution interval (in hours)
                chunk_obj.trim_start = False
                chunk_obj.t0 = tdiff*float(chunk_obj.chunk - chunk_mid_start) # hours
                chunk_obj.t1 = np.float(chunk_obj.t0) + tlen_mod # hours
            else:
                chunk_obj.trim_start = False
                chunk_obj.t0 = tdiff*float(chunk_obj.chunk - chunk_mid_start) # hours
                chunk_obj.t1 = np.float(chunk_obj.t0) + tlen # hours
            if c == nsols-1:
                chunk_obj.t1 += tlen # make sure last chunks gets everything that remains
            chunk_obj.ionfactor = ionfactor
            chunk_obj.parset = parset
            chunk_obj.skymodel = skymodel
            chunk_obj.logname_root = logname_root + '_part' + str(c)
            chunk_obj.solnum = chunk_obj.chunk
            range_start = chunk_obj.solnum*int(np.ceil(blockl/solint))
            range_end = range_start + int(np.ceil(blockl/solint))
            chunk_obj.solrange = range(range_start, range_end)
            chunk_obj.output = chunk_obj.outdir + '/part' + str(chunk_obj.chunk) + os.path.basename(chunk_obj.dataset)
            chunk_obj.input_instrument = instrument_orig
            chunk_obj.output_instrument = '{0}/parmdbs/part{1}{2}_instrument'.format(chunk_obj.outdir,
                    chunk_obj.chunk, os.path.basename(chunk_obj.dataset))
            chunk_obj.state_file = '{0}/state/part{1}{2}.done'.format(chunk_obj.outdir,
                    chunk_obj.chunk, os.path.basename(chunk_obj.dataset))
            chunk_obj.ntot = blockl
            chunk_obj.start_delay = 0.0
            chunk_list.append(chunk_obj)

        chunk_list_orig = chunk_list[:]
        if band.resume:
            # Determine which chunks need to be calibrated
            for chunk_obj in chunk_list_orig:
                if os.path.exists(chunk_obj.state_file):
                    chunk_list.remove(chunk_obj)
            if len(chunk_list) > 0:
                log.debug('Chunks remaining to be calibrated:')
                for chunk_obj in chunk_list:
                    log.debug('  Solution #{0}'.format(chunk_obj.chunk))
            else:
                log.info('Calibration complete for {0}.'.format(msname))

        return chunk_list, chunk_list_orig
    except Exception as e:
        log.error(str(e))


def runChunk(chunk):
    """
    Calibrate a chunk
    """
    logfilename = band.msname + '.distcal.log'
    init_logger(logfilename)
    log = logging.getLogger("DistCal.runChunks")
    time.sleep(chunk.start_delay)

    # Wrap everything in a try-except block to be sure any exception is caught
    try:
        # Split the dataset into parts
        split_ms(chunk.dataset, chunk.output, chunk.t0, chunk.t1)

        # Copy over instrument db to chunk in case it's needed
        subprocess.call('cp -r {0} {1}/instrument'.
            format(chunk.input_instrument, chunk.output), shell=True)

        # Calibrate
        calibrateChunk(chunk)

        # Clean up, copying instrument parmdb for later collection
        subprocess.call('cp -r {0}/instrument {1}'.
            format(chunk.output, chunk.output_instrument), shell=True)
        shutil.rmtree(chunk.output)

        # Record successful completion
        success_file = chunk.state_file
        cmd = 'touch {0}'.format(success_file)
        subprocess.call(cmd, shell=True)
    except Exception as e:
        log.error(str(e))


def calibrateChunk(chunk):
    """Calibrates a single MS chunk using a time-correlated solve"""
    if chunk.ionfactor is not None:
        # Modify weights
        fwhm_min, fwhm_max = modify_weights(chunk.output, chunk.ionfactor,
            ntot=chunk.ntot, trim_start=chunk.trim_start)

    # Run BBS
    subprocess.call("calibrate-stand-alone {0} {1} {2} > {3}/logs/"
        "{4}_peeling_calibrate_timecorr.log 2>&1".format(chunk.output, chunk.parset,
        chunk.skymodel, chunk.outdir, chunk.logname_root), shell=True)


def update_parset(parset):
    """
    Update the parset to set cellsize and chunksize = 0
    where a value of 0 forces all time/freq/cell intervals to be considered
    """
    updated_parset = parset + '_timecorr'
    f = open(parset, 'r')
    newlines = f.readlines()
    f.close()
    for i in range(0, len(newlines)):
	if 'ChunkSize' in newlines[i] or 'CellSize.Time' in newlines[i]:
	    vars = newlines[i].split()
	    newlines[i] = vars[0]+' '+vars[1]+' 0\n'
    f = open(parset,'w')
    f.writelines(newlines)
    f.close()
    return updated_parset


def split_ms(msin, msout, start_out, end_out):
    """Splits an MS between start and end times in hours relative to first time"""
    if os.path.exists(msout):
        os.system('rm -rf {0}'.format(msout))
    if os.path.exists(msout):
        os.system('rm -rf {0}'.format(msout))

    t = pt.table(msin, ack=False)

    starttime = t[0]['TIME']
    t1 = t.query('TIME > ' + str(starttime+start_out*3600) + ' && '
      'TIME < ' + str(starttime+end_out*3600), sortlist='TIME,ANTENNA1,ANTENNA2')

    t1.copy(msout, True)
    t1.close()
    t.close()


def modify_weights(msname, ionfactor, dryrun=False, ntot=None, trim_start=True):
    """Modifies the WEIGHTS column of the input MS"""
    logfilename = band.msname + '.distcal.log'
    init_logger(logfilename)
    log = logging.getLogger("DistCal.modWeights")

    t = pt.table(msname, readonly=False, ack=False)
    freqtab = pt.table(msname + '/SPECTRAL_WINDOW', ack=False)
    freq = freqtab.getcol('REF_FREQUENCY')
    freqtab.close()
    wav = 3e8 / freq
    fwhm_list = []

    for t2 in t.iter(["ANTENNA1", "ANTENNA2"]):
        if (t2.getcell('ANTENNA1', 0)) < (t2.getcell('ANTENNA2', 0)):
            weightscol = t2.getcol('WEIGHT_SPECTRUM')
            uvw = t2.getcol('UVW')
            uvw_dist = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2 + uvw[:, 2]**2)
            weightscol_modified = np.copy(weightscol)
            timepersample = t2[1]['TIME'] - t2[0]['TIME']
            dist = np.mean(uvw_dist) / 1e3
            stddev = ionfactor * np.sqrt((25e3 / dist)) * (freq / 60e6) # in sec
            fwhm = 2.3548 * stddev
            fwhm_list.append(fwhm[0])
            nslots = len(weightscol[:, 0, 0])
            if ntot is None:
                ntot = nslots
            elif ntot < nslots:
                log.debug('Number of samples for Gaussian is {0}, but number '
                    'in chunk is {1}. Setting number for Gaussian to {1}.'.format(ntot, nslots))
                ntot = nslots
            gauss = scipy.signal.gaussian(ntot, stddev/timepersample)

            if not dryrun:
                for pol in range(0, len(weightscol[0, 0, :])):
                    for chan in range(0, len(weightscol[0, :, 0])):
                        weights = weightscol[:, chan, pol]
                        if trim_start:
                            weightscol_modified[:, chan, pol] = weights * gauss[ntot - len(weights):]
                        else:
                            weightscol_modified[:, chan, pol] = weights * gauss[:len(weights)]
                t2.putcol('WEIGHT_SPECTRUM', weightscol_modified)
    t.close()
    return (min(fwhm_list), max(fwhm_list))


class Band(object):
    """The Band object contains parameters needed for each band (MS)."""
    def __init__(self, MSfile, timecorr, block, solint, ionfactor, ncores,
        resume, parset):
        self.file = MSfile
        self.msname = self.file.split('/')[-1]
        sw = pt.table(self.file + '/SPECTRAL_WINDOW', ack=False)
        self.freq = sw.col('REF_FREQUENCY')[0]
        sw.close()
        obs = pt.table(self.file + '/FIELD', ack=False)
        self.ra = np.degrees(float(obs.col('REFERENCE_DIR')[0][0][0]))
        if self.ra < 0.:
            self.ra=360.+(self.ra)
        self.dec = np.degrees(float(obs.col('REFERENCE_DIR')[0][0][1]))
        obs.close()
        ant = pt.table(self.file + '/ANTENNA', ack=False)
        diam = float(ant.col('DISH_DIAMETER')[0])
        ant.close()
        self.fwhm_deg = 1.1*((3.0e8/self.freq)/diam)*180./np.pi
        self.name = str(self.freq)
        self.timecorr = timecorr
        self.sol_block = block
        self.ionfactor = ionfactor
        self.ncores = ncores
        self.resume = resume
        self.solint = solint
        self.parset = parset


class Chunk(object):
    """The Chunk object contains parameters for time-correlated calibration
    (most of which are set later during calibration).
    """
    def __init__(self, MSfile):
        self.dataset = MSfile
