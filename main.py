#!/usr/bin/evn python
import time
import math
import logging
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
from optparse import OptionParser
## cal my own module
from readfile import getTime
from readfile import readFile
from machineInfo import getInfo
from gpu_dedisp import gpudedis
from gpu_dedisp import gpu_dmt_timeseries


def gpuMemCheck(machineInfo):
    gpumemTotal = machineInfo.gpumemTotal
    gpumemFree = machineInfo.gpumemFree
    gpumemUsed = machineInfo.gpumemUsed
    print("Total GPU memory %s GB" % gpumemTotal)
    print("Free GPU memory: %s GB" % gpumemFree)
    print("Used GPU memory: %s GB" % gpumemUsed)

    #bytes to GB
    div_gb_factor = (1024.0 ** 3)
    float32Size = np.array([], dtype=np.float32).itemsize
    print("Numpy float32 item size: ", float32Size)

    # set 98 percent GPU memory
    gpuMaxSize = int(gpumemFree * 0.98 * div_gb_factor)
    print("Max GPU acceptable size is : ", round(gpuMaxSize /div_gb_factor, 3) , " GB")
    return gpuMaxSize


#logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
#logging.basicConfig(level=logging.INFO, format=logging_format)

parser = OptionParser("")

parser.add_option("--fits", type="string", dest="fits", default="",
                  help="Fits file name.")
parser.add_option("--mindm", type="float", dest="mindm", default=0.0,
                  help="Minimum DM to dedisperse.")
parser.add_option("--maxdm", type="float", dest="maxdm", default=20000.000,
                  help="Maximum DM to dedisperse.")
parser.add_option("--dmstep", type="float", dest="dmstep", default=100.0,
                  help="DM step size.")
parser.add_option("--device", type="int", dest="device", default=0,
                  help="GPU device")
(options, args) = parser.parse_args()

machine = getInfo()
machine.get_gpu_info(options.device)

print("Machine Info: ")
print("Current CPU usage: %s %%" % machine.cpu_percent)
print("Total CPU number: %s %%" % machine.cpu_count)
print("Total memory %s GB" % machine.memTotal)
print("Available memory: %s GB" % machine.memAval)
print("Used memory: %s GB" % machine.memUsed)
print("GPU device index %s " % options.device)



print("Start processiong file: ", options.fits)
print(getTime(), "- Reading File : ", options.fits)
fp = gpudedis(options.fits)
print(getTime(), "- Reading Header")
fits = fp.readFAST()
print(getTime(), "- Getting Data")
fits.getdata()
data = fits.data
#[subintStart, subintEnd]
#fits.getdata([70, 90], swap=True)
#data = fits.data
print(getTime(), "- End Reading")
print(getTime(), "- Close File")
fits.closefile()
print("-"*50)
print("Pol num: %s Pol order: %s" %(fits.npol, fits.poln_order))
print("Data shape: ", data.shape, data.dtype)
print("-"*50)

dm_list = np.arange(options.mindm, options.maxdm + options.dmstep, options.dmstep)
print(getTime(), "- DM list length: %s " % len(dm_list))
if (dm_list[-1] > options.maxdm):
    dm_list = np.delete(dm_list, -1)

dedisp_times = np.zeros((fits.nchan,len(dm_list)),dtype=np.int64)  # np.int64 is 8bytes
print(getTime(), "- Max frequence: %s MHz" % fits.chan_freqs[-1])

for idx, dms in enumerate(dm_list):
    dedisp_times[:,idx] = np.round(-1*4148808.0*dms*(1 / (fits.chan_freqs[-1])**2 - 1/(fits.chan_freqs)**2)/1000/fits.tsamp).astype('int64')

max_delay = dedisp_times.max() - dedisp_times.min()

gulp=1E12
if gulp < max_delay:
    logging.error('Gulp smaller than max dispersion delay')
    gulp = int(2*max_delay)

psr_data = data[:,:,0:2].sum(axis=2).astype(np.uint16)
print(psr_data.dtype, data.dtype, psr_data.size * psr_data.itemsize)

###
# gpuMaxSize out put in bytes
# We will copy psr_data, dedisp_times into the GPU and create a out put data array dedisp_data 
# psr_data : samples in time and samples in frequence, uint8 + uint8 = uint16 (fits.nsblk * fits.nline * fits.nchan * np.uint16) 
# dedisp_times : delay time in each channel for each DM (fits.nchan * len(dm_list) * np.int64)
# dedisp_data : dedispersioned results ((fits.nsblk * fits.nline - max_delay) * len(dm_list) * np.float32)
# should make sure psr_data + dedisp_time + dedisp_data < gpu free memory*0.95
gpuMaxSize = gpuMemCheck(machine)   

memAllocate_psr_data = psr_data.size * psr_data.itemsize  # in bytes
memAllocate_dedisp_time = fits.nchan * len(dm_list) * dedisp_times.itemsize  # in bytes
memAllocate_dedisp_data = len(dm_list) * (psr_data.shape[0] - max_delay) * np.array([], dtype=np.float32).itemsize
memAllocate = memAllocate_psr_data + memAllocate_dedisp_time + memAllocate_dedisp_data

if gpuMaxSize*0.99 > memAllocate:
    print(getTime(), "GPU memory allocate passed.")
    print(getTime(), "Total GPU %s free memory : %s GB" %(options.device, round(gpuMaxSize/1024.**3,3)))
    print(getTime(), "Allocate GPU memory : %s Gb" % round(memAllocate/1024.**3, 3))

###

# for function gpu_dmt_timeseries
print(getTime(), "- Start dedisp")
print(getTime(), "- In data shape: ", psr_data.shape)
dedisp_data = gpu_dmt_timeseries(dedisp_times, psr_data, max_delay, device=options.device)
print(getTime(), "- Out data shape: ", dedisp_data.shape)
print(getTime(), "- End dedisp")

# plot the out put
extent = [0*fits.tsamp, dedisp_data.shape[1]*fits.tsamp, options.mindm, options.maxdm]
plt.imshow(dedisp_data,aspect='auto', cmap='hot', extent=extent)
plt.show()
