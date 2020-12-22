#!/usr/bin/evn python
import math
import numpy as np
from numba import cuda

def gpu_dmt(dedisp_times, psr_data, max_delay, device=0):
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
