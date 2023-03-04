import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
from decouple import config
import os
from functools import reduce

CL_PATH = config('CL_PATH', default='')
if len(CL_PATH) > 0:
    os.environ['PATH'] += ';'+CL_PATH

from eth_account import Account

# import secrets
# priv = secrets.token_hex(32)
priv = "726cc9f00e9a12eda1ad2bb1e778d40034da2e171082cf663d57b3b77c054da7"
private_key = "0x" + priv
print ("Private Key:", private_key)
acct = Account.from_key(private_key)
print("Address:", acct.address) # 0x02bCb427D68353E91d31047102153fd086a74242

kernel_code = '''

typedef unsigned int u32;
typedef unsigned long u64;

#define IS_CUDA

'''
def load_code(path: str) -> str:
    with open(path, 'r') as text_file:
        code_text = text_file.read()
    lines = code_text.splitlines()
    result = reduce(lambda t, l: 
                    t + "\n" + l if len(l) > 0 and not l.startswith('#include ') else t, 
                    lines, '')
    return result

# kernel_code += load_code('./OpenCL/inc_common.h')
kernel_code += load_code('./OpenCL/inc_vendor.h')
kernel_code += load_code('./OpenCL/inc_ecc_secp256k1.h')
kernel_code += load_code('./OpenCL/inc_ecc_secp256k1.cl')

kernel_code += '''

__global__ void generateKeysKernel(GLOBAL_AS u32 *r, GLOBAL_AS const u32 *k)
{
    u32 basepoint_g[PUBLIC_KEY_LENGTH_WITH_PARITY];
    u32 u32r_local[PUBLIC_KEY_LENGTH_WITH_PARITY];
    u32 u32k_local[PRIVATE_KEY_LENGTH];
    secp256k1_t basepoint_precalculated;

    basepoint_g[0] = SECP256K1_G0;
    basepoint_g[1] = SECP256K1_G1;
    basepoint_g[2] = SECP256K1_G2;
    basepoint_g[3] = SECP256K1_G3;
    basepoint_g[4] = SECP256K1_G4;
    basepoint_g[5] = SECP256K1_G5;
    basepoint_g[6] = SECP256K1_G6;
    basepoint_g[7] = SECP256K1_G7;
    basepoint_g[8] = SECP256K1_G_PARITY;

    // global to local
    u32k_local[0] = k[0];
    u32k_local[1] = k[1];
    u32k_local[2] = k[2];
    u32k_local[3] = k[3];
    u32k_local[4] = k[4];
    u32k_local[5] = k[5];
    u32k_local[6] = k[6];
    u32k_local[7] = k[7];

    parse_public(&basepoint_precalculated, basepoint_g);
    point_mul(u32r_local, u32k_local, &basepoint_precalculated);

    // local to global
    r[0] = u32r_local[0];
    r[1] = u32r_local[1];
    r[2] = u32r_local[2];
    r[3] = u32r_local[3];
    r[4] = u32r_local[4];
    r[5] = u32r_local[5];
    r[6] = u32r_local[6];
    r[7] = u32r_local[7];
    r[8] = u32r_local[8];
}
'''

with open('./kernel.cl', 'w') as f:
    f.write(kernel_code)

mod = SourceModule(kernel_code)
generateKeysKernel = mod.get_function('generateKeysKernel')
