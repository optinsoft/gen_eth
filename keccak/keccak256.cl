#include "keccak256.h"

#define NUM_ROUNDS				24

DECLSPEC u64 rotl64(u64 x, int i) {
    return ((0U + x) << i) | (x >> ((64 - i) & 63));
}

DECLSPEC void keccak256_absorb(PRIVATE_AS u64* state, PRIVATE_AS const int* rotation) { // u64 state[5 * 5]
    u8 r = 1;  // LFSR
    for (int i = 0; i < NUM_ROUNDS; i++) {
        // Theta step
        u64 c[5] = {};
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++)
                c[x] ^= state[x + y + (y << 2)]; // x * 5 + y
        }
        for (int x = 0; x < 5; x++) {
            u64 d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++)
                state[x + y + (y << 2)] ^= d;
        }
        // Rho and pi steps
        u64 b[5][5];
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++)
                b[y][(x * 2 + y * 3) % 5] = rotl64(state[x + y + (y << 2)], rotation[(x << 2) + x + y]);
        }
        // Chi step
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++)
                state[x + y + (y << 2)] = b[x][y] ^ (~b[(x + 1) % 5][y] & b[(x + 2) % 5][y]);
        }
        // Iota step
        for (int j = 0; j < 7; j++) {
            state[0] ^= (u64)(r & 1) << ((1 << j) - 1);
            r = (u8)((r << 1) ^ ((r >> 7) * 0x171));
        }
    }
}

DECLSPEC void keccak256_update_state(PRIVATE_AS u64* state, PRIVATE_AS const u8* msg, PRIVATE_AS const u32 len) // u64 state[5 * 5]
{
    const int rotation[25] = {
        0, 36,  3, 41, 18,
        1, 44, 10, 45,  2,
        62,  6, 43, 15, 61,
        28, 55, 25, 21, 56,
        27, 20, 39,  8, 14
    };
    u32 blockOff = 0;
    for (u32 i = 0; i < len; i++) {
        u32 j = blockOff >> 3;
        u32 xj = j % 5;
        u32 yj = j / 5;
        state[xj + yj + (yj << 2)] ^= (u64)(msg[i]) << ((blockOff & 7) << 3);
        blockOff++;
        if (blockOff == KECCAK256_BLOCKSIZE) {
            keccak256_absorb(state, rotation);
            blockOff = 0;
        }
    }
    // Final block and padding
    {
        int i = blockOff >> 3;
        u32 xi = i % 5;
        u32 yi = i / 5;
        state[xi + yi + (yi << 2)] ^= UINT64_C(0x01) << ((blockOff & 7) << 3);
        blockOff = KECCAK256_BLOCKSIZE - 1;
        int j = blockOff >> 3;
        u32 xj = j % 5;
        u32 yj = j / 5;
        state[xj + yj + (yj << 2)] ^= UINT64_C(0x80) << ((blockOff & 7) << 3);
        keccak256_absorb(state, rotation);
    }
}

DECLSPEC void keccak256_get_hash(GLOBAL_AS u8* r, GLOBAL_AS const u8* msg, GLOBAL_AS const u32 len)
{
    u64 state[25] = {};
    keccak256_update_state(state, (u8*)msg, len);
    // Uint64 array to bytes in little endian
    for (int i = 0; i < KECCAK256_HASH_LEN; i++) {
        int j = i >> 3;
        u32 xj = j % 5;
        u32 yj = j / 5;
        r[i] = (u8)(state[xj + yj + (yj << 2)] >> ((i & 7) << 3));
    }
}
