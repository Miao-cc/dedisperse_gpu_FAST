#!/usr/bin/env python
import math
import numpy as np
from numba import cuda
from readfile import readFile

from utils.gpu import gpu_dedisperse, gpu_dmt

class gpudedis(readFile):
    def dedisperse(self, dms=None, target="CPU"):
        """
        Dedisperse a chunk of data. Saves the dedispersed chunk in `self.dedispersed`.
    
        Note:
            Our method rolls the data around while dedispersing it.
    
        Args:
            dms (float): The DM to dedisperse the data at.
            target (str): 'CPU' to run the code on the CPU or 'GPU' to run it on a GPU.
    
        """
    
        if dms is None:
            dms = self.dm
        if self.data is not None:
            if target == "CPU":
                nt, nf = self.data.shape
                assert nf == len(self.chan_freqs)
                delay_time = (
                    4148808.0
                    * dms
                    * (1 / (self.chan_freqs[0]) ** 2 - 1 / (self.chan_freqs) ** 2)
                    / 1000
                )
                delay_bins = np.round(delay_time / self.tsamp).astype("int64")
                self.dedispersed = np.zeros(self.data.shape, dtype=np.float32)
                for ii in range(nf):
                    self.dedispersed[:, ii] = np.concatenate(
                        [
                            self.data[-delay_bins[ii] :, ii],
                            self.data[: -delay_bins[ii], ii],
                        ]
                    )
            elif target == "GPU":
                print("Starting GPU")
                gpu_dedisperse(self, device=self.device)
        else:
            self.dedispersed = None
        return self
    
    def dedispersets(self, dms=None):
        """
        Create a dedispersed time series
    
        Note:
            Our method rolls the data around while dedispersing it.
    
        Args:
            dms (float): The DM to dedisperse the data at.
    
        Returns:
            numpy.ndarray: Dedispersed time series.
    
        """
        if dms is None:
            dms = self.dm
        if self.data is not None:
            nt, nf = self.data.shape
            assert nf == len(self.chan_freqs)
            print(nt,nf)
            delay_time = (
                4148808.0
                * dms
                * (1 / (self.chan_freqs[0]) ** 2 - 1 / (self.chan_freqs) ** 2)
                / 1000
            )
            delay_bins = np.round(delay_time / self.tsamp).astype("int64")
            ts = np.zeros(nt, dtype=np.float32)
            for ii in range(nf):
                ts += np.concatenate(
                    [self.data[-delay_bins[ii] :, ii], self.data[: -delay_bins[ii], ii]]
                )
            return ts

def gpu_dmt_timeseries(dedisp_times, psr_data, max_delay, device=0):
    """
    :param cand: Candidate object
    :param device: GPU id
    :return:
    """
    cuda.select_device(device)
    dm_time = np.zeros((dedisp_times.shape[1], int(psr_data.shape[0]-max_delay)), dtype=np.float32)

    @cuda.jit(fastmath=True)
    def gpu_dmt(cand_data_in, all_delays, cand_data_out):
        ii, jj, kk = cuda.grid(3)
        if ii < cand_data_in.shape[0] and jj < cand_data_out.shape[1] and kk < all_delays.shape[1]:
            cuda.atomic.add(cand_data_out, (kk, jj), cand_data_in[ii, (jj + all_delays[ii,kk]) ]) 

    #with cuda.pinned(dedisp_times, dm_time, psr_data):
    all_delays = cuda.to_device(dedisp_times)
    dmt_return = cuda.device_array(dm_time.shape, dtype=np.float32)

    cand_data_in = cuda.to_device(np.array(psr_data.T, dtype=psr_data.dtype))

    threadsperblock = (4, 8, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])
    blockspergrid_z = math.ceil(dedisp_times.shape[1] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    gpu_dmt[blockspergrid, threadsperblock](cand_data_in, all_delays,  dmt_return)
    dm_time = dmt_return.copy_to_host()
    #print(all_delays.shape)
    cuda.close()
    return dm_time
