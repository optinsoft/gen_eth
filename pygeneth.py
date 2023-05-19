"""
@author: Vitaly <vitaly@optinsoft.net> | github.com/optinsoft
"""
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
from decouple import config
import os
from functools import reduce
import ecdsa
from Crypto.Hash import keccak

def randomUInt32() -> int:
    return int.from_bytes(np.random.bytes(4), byteorder='little', signed=False)

'''
test private key: 0x68e23530deb6d5011ab56d8ad9f7b4a3b424f1112f08606357497495929f72dc
test public key:  0x5d99d81d9e731e0d7eebd1c858b1155da7981b1f0a16d322a361f8b589ad2e3bde53dc614e3a84164dab3f5899abde3b09553dca10c9716fa623a5942b9ea420
test keccak256:   0x4c84817f57c18372837905af33f4b63eb1c5a9966a31cebc302f563685695506
test eth address: 0x33f4b63eb1c5a9966a31cebc302f563685695506
'''

def testUInt32(idx: int) -> int:
    r = [0x68e23530, 0xdeb6d501, 0x1ab56d8a, 0xd9f7b4a3, 0xb424f111, 0x2f086063, 0x57497495, 0x929f72dc][idx]
    return r

def randomUInt32Array(count: int) -> list[int]:
    return [randomUInt32() for i in range(count)]

def randomWithTestUInt32Array(count: int, idx: int) -> list[int]:
    return [testUInt32(idx) if i == 0 else randomUInt32() for i in range(count)]

def constUInt32Array(count: int, v: int) -> list[int]:
    return [v for i in range(count)]

def public_key_to_address(public_key, i, print_keccak):
    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(public_key)
    keccak_digest = keccak_hash.digest()
    if print_keccak:
        print(f'keccak[{i}] (verification):     0x{keccak_digest.hex()}')
    address = '0x' + keccak_digest[-20:].hex()
    return address

def key_to_hex(k: list[int]) -> str:
    return reduce(lambda s, t: str(s) + t.to_bytes(4, byteorder='big').hex(), k[1:], k[0].to_bytes(4, byteorder='big').hex())

def main_genPubKey(keyCount: int, verify: bool):
    CL_PATH = config('CL_PATH', default='')
    if len(CL_PATH) > 0:
        os.environ['PATH'] += ';'+CL_PATH

    kernel_code = '''

    '''
    def load_code(path: str) -> str:
        with open(path, 'r') as text_file:
            code_text = text_file.read()
        lines = code_text.splitlines()
        result = reduce(lambda t, l: 
                        t + "\n" + l if len(l) > 0 and not l.startswith('#include ') else t, 
                        lines, '')
        return result
    dirSecp256k1 = './secp256k1/'    
    kernel_code += load_code(dirSecp256k1 + 'inc_vendor.h')
    kernel_code += load_code(dirSecp256k1 + 'inc_types.h')
    kernel_code += load_code(dirSecp256k1 + 'inc_ecc_secp256k1.h')
    kernel_code += load_code(dirSecp256k1 + 'inc_ecc_secp256k1.cl')
    dirKeccak = './keccak/'
    kernel_code += load_code(dirKeccak + 'keccak256.h')
    kernel_code += load_code(dirKeccak + 'keccak256.cl')
    dirKernels = './kernels/'
    kernel_code += load_code(dirKernels + 'gen_pub_key.cl')

    # with open('./kernel.cl', 'w') as f:
    #     f.write(kernel_code)

    k = [np.array(randomUInt32Array(keyCount), dtype=np.uint32) for i in range(8)]
    xy = [np.array(constUInt32Array(keyCount, 0), dtype=np.uint32) for i in range(16)]
    h = [np.array(constUInt32Array(keyCount, 0), dtype=np.uint32) for i in range(8)]

    k_gpu = [gpuarray.to_gpu(k[i]) for i in range(8)]
    xy_gpu = [gpuarray.to_gpu(xy[i]) for i in range(16)]
    h_gpu = [gpuarray.to_gpu(h[i]) for i in range(8)]

    mod = SourceModule(kernel_code)
    genPubKey = mod.get_function('genPubKey')

    genPubKey(xy_gpu[0], xy_gpu[1], xy_gpu[2], xy_gpu[3], xy_gpu[4], xy_gpu[5], xy_gpu[6], xy_gpu[7],
        xy_gpu[8], xy_gpu[9], xy_gpu[10], xy_gpu[11], xy_gpu[12], xy_gpu[13], xy_gpu[14], xy_gpu[15],
        h_gpu[0], h_gpu[1], h_gpu[2], h_gpu[3], h_gpu[4], h_gpu[5], h_gpu[6], h_gpu[7],
        k_gpu[0], k_gpu[1], k_gpu[2], k_gpu[3], k_gpu[4], k_gpu[5], k_gpu[6], k_gpu[7],
        block=(keyCount, 1, 1))

    for i in range(keyCount):
        # print(f'--- [{i}] ---')
        _k = [k_gpu[j][i].get().item() for j in range(8)]
        priv = key_to_hex(_k)
        print(f"priv[{i}]:                      0x{priv}")
        xy = [xy_gpu[j][i].get().item() for j in range(16)]
        pub = key_to_hex(xy)
        print(f"pub[{i}]:                       0x{pub}")
        _h = [h_gpu[j][i].get().item() for j in range(8)]
        keccak = key_to_hex(_h)
        print(f"keccak[{i}]:                    0x{keccak}")      
        if verify:  
            pk_bytes = bytes.fromhex(priv) 
            public_key = ecdsa.SigningKey.from_string(pk_bytes, curve=ecdsa.SECP256k1).verifying_key.to_string()    
            print(f"public Key[{i}] (verification): 0x{public_key.hex()}")
            address = public_key_to_address(public_key, i, True)
            # print(f"Address[{i}]:    {address}")

def main_genEthAddress(keyCount: int, verify: bool):
    CL_PATH = config('CL_PATH', default='')
    if len(CL_PATH) > 0:
        os.environ['PATH'] += ';'+CL_PATH

    kernel_code = '''

    '''
    def load_code(path: str) -> str:
        with open(path, 'r') as text_file:
            code_text = text_file.read()
        lines = code_text.splitlines()
        result = reduce(lambda t, l: 
                        t + "\n" + l if len(l) > 0 and not l.startswith('#include ') else t, 
                        lines, '')
        return result
    dirSecp256k1 = './secp256k1/'    
    kernel_code += load_code(dirSecp256k1 + 'inc_vendor.h')
    kernel_code += load_code(dirSecp256k1 + 'inc_types.h')
    kernel_code += load_code(dirSecp256k1 + 'inc_ecc_secp256k1.h')
    kernel_code += load_code(dirSecp256k1 + 'inc_ecc_secp256k1.cl')
    dirKeccak = './keccak/'
    kernel_code += load_code(dirKeccak + 'keccak256.h')
    kernel_code += load_code(dirKeccak + 'keccak256.cl')
    dirKernels = './kernels/'
    kernel_code += load_code(dirKernels + 'gen_eth_addr.cl')

    # with open('./kernel.cl', 'w') as f:
    #     f.write(kernel_code)

    k = [np.array(randomUInt32Array(keyCount), dtype=np.uint32) for i in range(8)]
    a = [np.array(constUInt32Array(keyCount, 0), dtype=np.uint32) for i in range(5)]

    k_gpu = [gpuarray.to_gpu(k[i]) for i in range(8)]
    a_gpu = [gpuarray.to_gpu(a[i]) for i in range(5)]

    mod = SourceModule(kernel_code)
    genEthAddress = mod.get_function('genEthAddress')

    genEthAddress(
        a_gpu[0], a_gpu[1], a_gpu[2], a_gpu[3], a_gpu[4],
        k_gpu[0], k_gpu[1], k_gpu[2], k_gpu[3], k_gpu[4], k_gpu[5], k_gpu[6], k_gpu[7],
        block=(keyCount, 1, 1))

    for i in range(keyCount):
        # print(f'--- [{i}] ---')
        _k = [k_gpu[j][i].get().item() for j in range(8)]
        priv = key_to_hex(_k)
        if verify:
            print(f"priv[{i}]:                       0x{priv}")
        _a = [a_gpu[j][i].get().item() for j in range(5)]
        eth_address = key_to_hex(_a)
        if verify:
            print(f"eth address[{i}]:                0x{eth_address}")
            pk_bytes = bytes.fromhex(priv) 
            public_key = ecdsa.SigningKey.from_string(pk_bytes, curve=ecdsa.SECP256k1).verifying_key.to_string()    
            address = public_key_to_address(public_key, i, False)
            print(f"eth address[{i}] (verification): {address}")
        else:
            print(f"0x{priv},0x{eth_address}")

if __name__ == "__main__":
    # main_genPubKey(keyCount=32, verify=True)
    main_genEthAddress(keyCount=32, verify=True)