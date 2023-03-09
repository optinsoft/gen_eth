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
        print(f'Keccak[{i}] (verification):     0x{keccak_digest.hex()}')
    address = '0x' + keccak_digest[-20:].hex()
    return address

def key_to_hex(k: list[int]) -> str:
    return reduce(lambda s, t: str(s) + t.to_bytes(4, byteorder='big').hex(), k[1:], k[0].to_bytes(4, byteorder='big').hex())

def main_vanityEthAddress(prefix: str, keyBlockCount: int, maxBlocks: int, verify: bool, verbose: bool) -> int:
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
    genEthAddress = mod.get_function('genEthAddress')

    if verbose:
        print("Searching vanity address...")

    start_time = time.time()

    for n in range(maxBlocks):
        k = [np.array(randomUInt32Array(keyBlockCount), dtype=np.uint32) for i in range(8)]
        a = [np.array(constUInt32Array(keyBlockCount, 0), dtype=np.uint32) for i in range(5)]

        k_gpu = [gpuarray.to_gpu(k[i]) for i in range(8)]
        a_gpu = [gpuarray.to_gpu(a[i]) for i in range(5)]

        genEthAddress(
            a_gpu[0], a_gpu[1], a_gpu[2], a_gpu[3], a_gpu[4],
            k_gpu[0], k_gpu[1], k_gpu[2], k_gpu[3], k_gpu[4], k_gpu[5], k_gpu[6], k_gpu[7],
            block=(keyBlockCount, 1, 1))
        
        for i in range(keyBlockCount):
            # print(f'--- [{i}] ---')
            _a = [a_gpu[j][i].get().tolist() for j in range(5)]
            eth_address = key_to_hex(_a)
            if eth_address.startswith(prefix):
                if verbose:
                    end_time = time.time()  # end time
                    elapsed_time = end_time - start_time
                    print(f"Vanity address found in block {n+1}, {elapsed_time:.2f} seconds")
                    count = n * keyBlockCount
                    print(f"Generated {count} ethereum addresses, {count/elapsed_time:.2f} addresses/second")
                _k = [k_gpu[j][i].get().tolist() for j in range(8)]
                priv = key_to_hex(_k)
                if verify:
                    print(f"priv[{i}]:                       0x{priv}")
                if verify:
                    print(f"eth address[{i}]:                0x{eth_address}")
                    pk_bytes = bytes.fromhex(priv) 
                    public_key = ecdsa.SigningKey.from_string(pk_bytes, curve=ecdsa.SECP256k1).verifying_key.to_string()    
                    address = public_key_to_address(public_key, i, False)
                    print(f"eth address[{i}] (verification): {address}")
                else:
                    print(f"0x{priv},0x{eth_address}")
                return 1
    if verbose:
        end_time = time.time()  # end time
        elapsed_time = end_time - start_time
        print(f"Not found, {elapsed_time:.2f} seconds")
        count = maxBlocks * keyBlockCount
        print(f"Generated {count} ethereum addresses, {count/elapsed_time:.2f} addresses/second")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyvanityeth.py")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('--verify', action='store_true', help='verify found ethereum address')
    parser.add_argument("--prefix", required=True, type=str, help="vanity ethereum address PREFIX (without leading 0x)")
    parser.add_argument("--blocks", required=False, type=int, default=1000, help="try find vanity ethereum address within BLOCKS blocks (default: 1000)")
    parser.add_argument("--blockSize", required=False, type=int, default=128, help="generate block of BLOCKSIZE ethereum addresses by using GPU (default: 128)")
    args = parser.parse_args()
    main_vanityEthAddress(args.prefix, args.blockSize, args.blocks, args.verify, args.verbose)
