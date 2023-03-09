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

def randomUInt32Array(count: int, idx: int) -> list[int]:
    return [testUInt32(idx) if i == 0 else randomUInt32() for i in range(count)]

def constUInt32Array(count: int, v: int) -> list[int]:
    return [v for i in range(count)]

def public_key_to_address(public_key, i, print_keccak):
    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(public_key)
    keccak_digest = keccak_hash.digest()
    if print_keccak:
        print(f'Keccak[{i}]:     0x{keccak_digest.hex()}')
    address = '0x' + keccak_digest[-20:].hex()
    return address

def key_to_hex(k: list[int]) -> str:
    return reduce(lambda s, t: str(s) + t.to_bytes(4, byteorder='big').hex(), k[1:], k[0].to_bytes(4, byteorder='big').hex())

def main_genPubKey():
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

    kernel_code += '''

    // little endian to big endian
    DECLSPEC u32 l2be(u32 x) {
        return (x & 0xff) << 24 | (x & 0xff00) << 8 | (x & 0xff0000) >> 8 | (x & 0xff000000) >> 24;
    }

    __global__ void genPubKey(
        GLOBAL_AS u32 *r0, GLOBAL_AS u32 *r1, GLOBAL_AS u32 *r2, GLOBAL_AS u32 *r3, 
        GLOBAL_AS u32 *r4, GLOBAL_AS u32 *r5, GLOBAL_AS u32 *r6, GLOBAL_AS u32 *r7,
        GLOBAL_AS u32 *r8, GLOBAL_AS u32 *r9, GLOBAL_AS u32 *r10, GLOBAL_AS u32 *r11, 
        GLOBAL_AS u32 *r12, GLOBAL_AS u32 *r13, GLOBAL_AS u32 *r14, GLOBAL_AS u32 *r15,
        GLOBAL_AS u32* h0, GLOBAL_AS u32* h1, GLOBAL_AS u32* h2, GLOBAL_AS u32* h3,
        GLOBAL_AS u32* h4, GLOBAL_AS u32* h5, GLOBAL_AS u32* h6, GLOBAL_AS u32* h7,
        GLOBAL_AS const u32 *k0, GLOBAL_AS const u32 *k1, GLOBAL_AS const u32 *k2, GLOBAL_AS const u32 *k3,
        GLOBAL_AS const u32 *k4, GLOBAL_AS const u32 *k5, GLOBAL_AS const u32 *k6, GLOBAL_AS const u32 *k7)
    {
        u32 g_local[PUBLIC_KEY_LENGTH_WITH_PARITY];
        u32 k_local[PRIVATE_KEY_LENGTH];
        secp256k1_t g_xy_local;
        u32 return_value;

        int i = threadIdx.x;

        g_local[0] = SECP256K1_G_STRING0;
        g_local[1] = SECP256K1_G_STRING1;
        g_local[2] = SECP256K1_G_STRING2;
        g_local[3] = SECP256K1_G_STRING3;
        g_local[4] = SECP256K1_G_STRING4;
        g_local[5] = SECP256K1_G_STRING5;
        g_local[6] = SECP256K1_G_STRING6;
        g_local[7] = SECP256K1_G_STRING7;
        g_local[8] = SECP256K1_G_STRING8;

        // global to local
        k_local[7] = k0[i];
        k_local[6] = k1[i];
        k_local[5] = k2[i];
        k_local[4] = k3[i];
        k_local[3] = k4[i];
        k_local[2] = k5[i];
        k_local[1] = k6[i];
        k_local[0] = k7[i];

        return_value = parse_public(&g_xy_local, g_local);
        if (return_value != 0) {
            return;
        }

        u32 x[8];
        u32 y[8];
        point_mul_xy (x, y, k_local,  &g_xy_local);

        // local to global
        r7[i] = x[0]; 
        r6[i] = x[1];
        r5[i] = x[2];
        r4[i] = x[3];
        r3[i] = x[4];
        r2[i] = x[5];
        r1[i] = x[6];
        r0[i] = x[7];
        r15[i] = y[0];
        r14[i] = y[1];
        r13[i] = y[2];
        r12[i] = y[3];
        r11[i] = y[4];
        r10[i] = y[5];
        r9[i] = y[6];
        r8[i] = y[7];

        // keccak256
        u64 keccak_state[KECCAK256_STATE_LEN] = {};
        u32 w[16];

        w[7]  = l2be(x[0]);
        w[6]  = l2be(x[1]);
        w[5]  = l2be(x[2]);
        w[4] = l2be(x[3]);
        w[3]  = l2be(x[4]);
        w[2]  = l2be(x[5]);
        w[1]  = l2be(x[6]);
        w[0]  = l2be(x[7]);
        w[15] = l2be(y[0]);
        w[14] = l2be(y[1]);
        w[13] = l2be(y[2]);
        w[12] = l2be(y[3]);
        w[11] = l2be(y[4]);
        w[10] = l2be(y[5]);
        w[9]  = l2be(y[6]);
        w[8]  = l2be(y[7]);

        keccak256_update_state(keccak_state, (u8*)w, 64);

        h0[i] = l2be((u32)keccak_state[0]);
        h1[i] = l2be((u32)(keccak_state[0] >> 32));
        h2[i] = l2be((u32)keccak_state[1]);
        h3[i] = l2be((u32)(keccak_state[1] >> 32));
        h4[i] = l2be((u32)keccak_state[2]);
        h5[i] = l2be((u32)(keccak_state[2] >> 32));
        h6[i] = l2be((u32)keccak_state[3]);
        h7[i] = l2be((u32)(keccak_state[3] >> 32));
    }
    '''

    with open('./kernel.cl', 'w') as f:
        f.write(kernel_code)

    keyCount = 32

    k = [np.array(randomUInt32Array(keyCount, i), dtype=np.uint32) for i in range(8)]
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
        print(f'--- [{i}] ---')
        _k = [k_gpu[j][i].get().tolist() for j in range(8)]
        priv = key_to_hex(_k)
        print(f"priv[{i}]: 0x{priv}")
        xy = [xy_gpu[j][i].get().tolist() for j in range(16)]
        pub = key_to_hex(xy)
        print(f"pub[{i}]:  0x{pub}")
        _h = [h_gpu[j][i].get().tolist() for j in range(8)]
        keccak = key_to_hex(_h)
        print(f"keccak[{i}: 0x{keccak}")        
        pk_bytes = bytes.fromhex(priv) 
        public_key = ecdsa.SigningKey.from_string(pk_bytes, curve=ecdsa.SECP256k1).verifying_key.to_string()    
        print(f"Public Key[{i}]: 0x{public_key.hex()}")
        address = public_key_to_address(public_key, i, True)
        print(f"Address[{i}]:    {address}")

def main_genEthAddress():
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

    kernel_code += '''

    // little endian to big endian
    DECLSPEC u32 l2be(u32 x) {
        return (x & 0xff) << 24 | (x & 0xff00) << 8 | (x & 0xff0000) >> 8 | (x & 0xff000000) >> 24;
    }

    __global__ void genEthAddress(
        GLOBAL_AS u32 *r0, GLOBAL_AS u32 *r1, GLOBAL_AS u32 *r2, GLOBAL_AS u32 *r3, GLOBAL_AS u32 *r4, 
        GLOBAL_AS const u32 *k0, GLOBAL_AS const u32 *k1, GLOBAL_AS const u32 *k2, GLOBAL_AS const u32 *k3,
        GLOBAL_AS const u32 *k4, GLOBAL_AS const u32 *k5, GLOBAL_AS const u32 *k6, GLOBAL_AS const u32 *k7)
    {
        u32 g_local[PUBLIC_KEY_LENGTH_WITH_PARITY];
        u32 k_local[PRIVATE_KEY_LENGTH];
        secp256k1_t g_xy_local;
        u32 return_value;

        int i = threadIdx.x;

        g_local[0] = SECP256K1_G_STRING0;
        g_local[1] = SECP256K1_G_STRING1;
        g_local[2] = SECP256K1_G_STRING2;
        g_local[3] = SECP256K1_G_STRING3;
        g_local[4] = SECP256K1_G_STRING4;
        g_local[5] = SECP256K1_G_STRING5;
        g_local[6] = SECP256K1_G_STRING6;
        g_local[7] = SECP256K1_G_STRING7;
        g_local[8] = SECP256K1_G_STRING8;

        // global to local
        k_local[7] = k0[i];
        k_local[6] = k1[i];
        k_local[5] = k2[i];
        k_local[4] = k3[i];
        k_local[3] = k4[i];
        k_local[2] = k5[i];
        k_local[1] = k6[i];
        k_local[0] = k7[i];

        return_value = parse_public(&g_xy_local, g_local);
        if (return_value != 0) {
            return;
        }

        u32 x[8];
        u32 y[8];
        point_mul_xy (x, y, k_local,  &g_xy_local);

        // keccak256
        u64 keccak_state[KECCAK256_STATE_LEN] = {};
        u32 w[16];

        w[7]  = l2be(x[0]);
        w[6]  = l2be(x[1]);
        w[5]  = l2be(x[2]);
        w[4] = l2be(x[3]);
        w[3]  = l2be(x[4]);
        w[2]  = l2be(x[5]);
        w[1]  = l2be(x[6]);
        w[0]  = l2be(x[7]);
        w[15] = l2be(y[0]);
        w[14] = l2be(y[1]);
        w[13] = l2be(y[2]);
        w[12] = l2be(y[3]);
        w[11] = l2be(y[4]);
        w[10] = l2be(y[5]);
        w[9]  = l2be(y[6]);
        w[8]  = l2be(y[7]);

        keccak256_update_state(keccak_state, (u8*)w, 64);

        r0[i] = l2be((u32)(keccak_state[1] >> 32));
        r1[i] = l2be((u32)keccak_state[2]);
        r2[i] = l2be((u32)(keccak_state[2] >> 32));
        r3[i] = l2be((u32)keccak_state[3]);
        r4[i] = l2be((u32)(keccak_state[3] >> 32));
    }
    '''

    with open('./kernel.cl', 'w') as f:
        f.write(kernel_code)

    keyCount = 32

    k = [np.array(randomUInt32Array(keyCount, i), dtype=np.uint32) for i in range(8)]
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
        print(f'--- [{i}] ---')
        _k = [k_gpu[j][i].get().tolist() for j in range(8)]
        priv = key_to_hex(_k)
        print(f"priv[{i}]:                       0x{priv}")
        _a = [a_gpu[j][i].get().tolist() for j in range(5)]
        eth_address = key_to_hex(_a)
        print(f"eth address[{i}:                 0x{eth_address}")
        pk_bytes = bytes.fromhex(priv) 
        public_key = ecdsa.SigningKey.from_string(pk_bytes, curve=ecdsa.SECP256k1).verifying_key.to_string()    
        address = public_key_to_address(public_key, i, False)
        print(f"eth address[{i}] (verification): {address}")

if __name__ == "__main__":
    main_genEthAddress()