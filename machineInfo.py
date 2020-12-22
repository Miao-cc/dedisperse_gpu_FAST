import pynvml
import psutil


class getInfo():
    """
    a = getInfo()
    print(a.cpu_percent)
    print(a.cpu_count)
    print(a.memTotal)
    print(a.memAval)
    print(a.memUsed)
    """
    def __init__(self):
        self.get_cpu_info()
        self.get_memory_info()
    
    def get_cpu_info(self):
        self.cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_count = psutil.cpu_count()
    
    def get_memory_info(self):
        div_gb_factor = (1024.0 ** 3)
        mem = psutil.virtual_memory()
        self.memTotal = mem.total/div_gb_factor
        self.memAval  = mem.available/div_gb_factor
        self.memUsed  = mem.used/div_gb_factor

    def get_gpu_info(self, device=0):
        div_gb_factor = (1024.0 ** 3)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.gpumemTotal = meminfo.total / div_gb_factor
        self.gpumemFree = meminfo.free / div_gb_factor
        self.gpumemUsed = meminfo.used / div_gb_factor
