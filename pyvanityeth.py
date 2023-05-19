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
import argparse
import time

def randomUInt32() -> int:
    return int.from_bytes(os.urandom(4), byteorder='little', signed=False)

def randomUInt32Array(count: int) -> list[int]:
    return [randomUInt32() for i in range(count)]

def constUInt32Array(count: int, v: int) -> list[int]:
    return [v for i in range(count)]

def prefixUInt32(prefixBytes: bytes) -> int:
    pl = len(prefixBytes)
    p = [prefixBytes[i] if i < pl else 0 for i in range(4)]
    return int.from_bytes(p, byteorder='big', signed=False)

def prefixUInt32Array(prefixBytes: bytes) -> list[int]:
    return [prefixUInt32(prefixBytes[i*4:i*4+4]) for i in range(5)]

def public_key_to_address(public_key, i, print_keccak):
    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(public_key)
    keccak_digest = keccak_hash.digest()
    if print_keccak:
        print(f'Keccak[{i}] (verification):     0x{keccak_digest.hex()}')
    address = '0x' + keccak_digest[-20:].hex()
    return address

def key_to_hex(k: list[int]) -> str:
    return reduce(lambda s, t: str(s) + t.to_bytes(4, byteorder='big').hex(), k[1:], k[0].to_bytes(4, byteorder='big').hex())

def main_vanityEthAddress(prefixBytes: bytes, keyBlockCount: int, maxBlocks: int, blockIterations: int, verify: bool, verbose: bool) -> int:
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

    if verbose:
        print("Building kernel...")

    mod = SourceModule(kernel_code)
    genEthAddressWithPrefix = mod.get_function('genEthAddressWithPrefix')

    prefix = prefixBytes.hex()

    if verbose:
        print(f'Searching vanity address with prefix "{prefix}"...')

    start_time = time.time()

    a = [np.array(constUInt32Array(keyBlockCount, 0), dtype=np.uint32) for i in range(5)]
    a_gpu = [gpuarray.to_gpu(a[i]) for i in range(5)]
    ap_gpu = gpuarray.to_gpu(np.array(constUInt32Array(keyBlockCount, 0), dtype=np.uint32))

    p =  np.array(prefixUInt32Array(prefixBytes), dtype=np.uint32)
    p_gpu = gpuarray.to_gpu(p)
    p_len = np.int32(len(prefixBytes))
    n_iterations = np.int32(blockIterations)

    for n in range(maxBlocks):
        k = [np.array(randomUInt32Array(keyBlockCount), dtype=np.uint32) for i in range(8)]
        k_gpu = [gpuarray.to_gpu(k[i]) for i in range(8)]

        genEthAddressWithPrefix(
            a_gpu[0], a_gpu[1], a_gpu[2], a_gpu[3], a_gpu[4], ap_gpu,
            k_gpu[0], k_gpu[1], k_gpu[2], k_gpu[3], k_gpu[4], k_gpu[5], k_gpu[6], k_gpu[7],
            p_gpu, p_len, n_iterations,
            block=(keyBlockCount, 1, 1))
        
        for i in range(keyBlockCount):
            # print(f'--- [{i}] ---')
            _ap = ap_gpu[i].get().item()
            if _ap != 0:
                _a = [a_gpu[j][i].get().item() for j in range(5)]
                eth_address = '0x'+key_to_hex(_a)
                if eth_address.startswith('0x'+prefix):
                    if verbose:
                        end_time = time.time()  # end time
                        elapsed_time = end_time - start_time
                        print(f"Vanity address found in block # {n+1} iteration # {_ap}, {elapsed_time:.2f} seconds")
                        count = (n + 1) * keyBlockCount * (blockIterations if blockIterations > 0 else 1)
                        print(f"Generated {count} ethereum addresses, {count/elapsed_time:.2f} addresses/second")
                    _k = [k_gpu[j][i].get().item() for j in range(8)]
                    priv = key_to_hex(_k)
                    if verify and verbose:
                        print(f"private key[{i}]:                0x{priv}")
                        print(f"eth address[{i}]:                {eth_address}")
                    if verify:
                        pk_bytes = bytes.fromhex(priv) 
                        public_key = ecdsa.SigningKey.from_string(pk_bytes, curve=ecdsa.SECP256k1).verifying_key.to_string()    
                        address = public_key_to_address(public_key, i, False)
                        if verbose:
                            print(f"eth address[{i}] (verification): {address}")
                        if address != eth_address:
                            print(f"Verification failed: _as[{i}]={_ap}, eth_address[{i}]={eth_address}, verification={address}")
                    return 1
                else:
                    print(f"Unexpected result: _ap[{i}]={_ap}, eth_address[{i}]={eth_address}")
    if verbose:
        end_time = time.time()  # end time
        elapsed_time = end_time - start_time
        print(f"Not found, {elapsed_time:.2f} seconds")
        count = maxBlocks * keyBlockCount * (blockIterations if blockIterations > 0 else 1)
        print(f"Generated {count} ethereum addresses, {count/elapsed_time:.2f} addresses/second")
    return 0

def hexPrefix(s: str) -> bytes:
    if s.startswith('0x'):
        return bytes.fromhex(s[2:])
    return bytes.fromhex(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyvanityeth.py")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('--verify', action='store_true', help='verify found ethereum address')
    parser.add_argument("--prefix", required=True, type=hexPrefix, help="vanity ethereum address PREFIX (without leading 0x)")
    parser.add_argument("--blocks", required=False, type=int, default=1000, help="try find vanity ethereum address within BLOCKS blocks (default: 1000)")
    parser.add_argument("--blockSize", required=False, type=int, default=128, help="generate block of BLOCKSIZE ethereum addresses by using GPU (default: 128)")
    parser.add_argument("--blockIterations", required=False, type=int, default=1, help="attempts to find vanity  ethereum address within each block")
    args = parser.parse_args()
    main_vanityEthAddress(args.prefix, args.blockSize, args.blocks, args.blockIterations, args.verify, args.verbose)
