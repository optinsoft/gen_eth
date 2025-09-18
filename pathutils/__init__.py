from decouple import config
import os

CL_PATH = config('CL_PATH', default='')
if len(CL_PATH) > 0:
    os.environ['PATH'] += ';'+CL_PATH
CUDA_DLL_PATH = config('CUDA_DLL_PATH', default='')
if len(CUDA_DLL_PATH) > 0:
    os.add_dll_directory(CUDA_DLL_PATH)
