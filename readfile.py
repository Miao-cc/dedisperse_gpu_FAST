#!/usr/bin/env python3
import time
import numpy as np
from decimal import Decimal
import astropy.time as aptime
import astropy.io.fits as pyfits
from astropy import coordinates, units

secperday = 3600 * 24


def getTime():
    return time.asctime(time.localtime(time.time()))


class readFile():
    def __init__(self, file):
        self.filename = file

    def readFAST(self):
        """
        Help:
        self.filename
        self.device
        self.dm
        self.fits
        self.data1
        self.chan_freqs
        self.dat_scl
        self.nline
        self.nsblk
        self.tbin
        self.npol
        self.nsuboffs
        self.tsamp
        self.chan_bw
        self.freq
        self.nchan
        self.obsbw
        self.telescope
        self.backend
        self.nbits
        self.poln_order
        self.beam
        self.STT_IMJD
        self.STT_SMJD
        self.STT_OFFS
        self.tstart

        self.ra_deg
        self.dec_deg
        """

        filename = self.filename
        self.device=0
        self.dm=None
        self.fits = pyfits.open(filename, mode="readonly", memmap=True)
        hdu0 = self.fits[0]
        hdu1 = self.fits[1]
        data0 = hdu0.data
        data1 = hdu1.data
        header0 = hdu0.header
        header1 = hdu1.header

        # get the data 
        #self.data1 = hdu1.data
        self.chan_freqs = self.fits['SUBINT'].data[0]['DAT_FREQ']
        self.dat_scl= np.array(data1['DAT_SCL'])
        self.nline = header1['NAXIS2']
        self.nsblk = header1['NSBLK']
        #self.tbin = header1['TBIN']
        self.npol = header1['NPOL']
        self.nsuboffs = header1['NSUBOFFS']
        self.tsamp = header1['TBIN']
        self.chan_bw = header1['CHAN_BW']
        self.freq = header0['OBSFREQ']
        self.nchan = header0['OBSNCHAN']
        self.obsbw = header0['OBSBW']
        self.telescope = header0["TELESCOP"].strip()
        self.backend = header0["BACKEND"].strip()
        self.nbits = header1["NBITS"]
        self.poln_order = header1["POL_TYPE"]
        self.beam = header0['IBEAM']
        self.STT_IMJD = header0['STT_IMJD']
        self.STT_SMJD = header0['STT_SMJD']
        self.STT_OFFS = header0['STT_OFFS']
        self.tstart = "%.13f" % (Decimal(self.STT_IMJD)+(Decimal(self.STT_SMJD)+Decimal(self.STT_OFFS))/secperday)

        loc = coordinates.SkyCoord(header0["RA"], header0["DEC"], unit=(units.hourangle, units.deg))
        self.ra_deg = loc.ra.value
        self.dec_deg = loc.dec.value
        return self

    def getdata(self, *args, swap = False):
        hdu1 = self.fits[1]
        data = np.array(hdu1.data['DATA']).squeeze()
        # subint , spectra, pol, nchan
        #(128, 1024, 4, 4096, 1)
        if not swap:
            self.data = np.swapaxes(data.reshape((-1, self.npol, self.nchan)), 1, 2)
        else:
            self.data = None
            subintStart, subintEnd= args[0]
            if subintEnd >=  data.shape[0]:
                subintEnd = data.shape[0]
            self.data = np.swapaxes(data[subintStart:subintEnd+1, :, :, :].reshape((-1, self.npol, self.nchan)), 1, 2)
        return self.data

    def closefile(self):
        self.fits.close()
