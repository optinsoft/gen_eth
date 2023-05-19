// @author: Vitaly <vitaly@optinsoft.net> | github.com/optinsoft

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