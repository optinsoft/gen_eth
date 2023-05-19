// @author: Vitaly <vitaly@optinsoft.net> | github.com/optinsoft

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
    w[4]  = l2be(x[3]);
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

__global__ void genEthAddressWithPrefix(
    GLOBAL_AS u32 *r0, GLOBAL_AS u32 *r1, GLOBAL_AS u32 *r2, GLOBAL_AS u32 *r3, GLOBAL_AS u32 *r4, GLOBAL_AS u32 *rp,
    GLOBAL_AS u32 *k0, GLOBAL_AS u32 *k1, GLOBAL_AS u32 *k2, GLOBAL_AS u32 *k3,
    GLOBAL_AS u32 *k4, GLOBAL_AS u32 *k5, GLOBAL_AS u32 *k6, GLOBAL_AS u32 *k7,
    GLOBAL_AS const u32 p[5], GLOBAL_AS const u32 plen, GLOBAL_AS const u32 n)
{
    u32 g_local[PUBLIC_KEY_LENGTH_WITH_PARITY];
    u32 k_local[PRIVATE_KEY_LENGTH];
    secp256k1_t g_xy_local;
    u32 return_value;
    u32 p_local[5];
    u32 m_local[5];
    u32 r_local[5];
    u32 rp_local = 0;

    int i = threadIdx.x;

    // global to local
    k_local[7] = k0[i];
    k_local[6] = k1[i];
    k_local[5] = k2[i];
    k_local[4] = k3[i];
    k_local[3] = k4[i];
    k_local[2] = k5[i];
    k_local[1] = k6[i];
    k_local[0] = k7[i];

    u32 l = plen;
    m_local[0] = (l >= 4) ? 0xffffffff : 0xffffffff << ((4-l) << 3);
    l = (l >= 4) ? l-4 : 0; 
    p_local[0] = p[0] & m_local[0];
    m_local[1] = (l >= 4) ? 0xffffffff : 0xffffffff << ((4-l) << 3);
    l = (l >= 4) ? l-4 : 0; 
    p_local[1] = p[1] & m_local[1];
    m_local[2] = (l >= 4) ? 0xffffffff : 0xffffffff << ((4-l) << 3);
    l = (l >= 4) ? l-4 : 0; 
    p_local[2] = p[2] & m_local[2];
    m_local[3] = (l >= 4) ? 0xffffffff : 0xffffffff << ((4-l) << 3);
    l = (l >= 4) ? l-4 : 0; 
    p_local[3] = p[3] & m_local[3];
    m_local[4] = (l >= 4) ? 0xffffffff : 0xffffffff << ((4-l) << 3);
    l = (l >= 4) ? l-4 : 0; 
    p_local[4] = p[4] & m_local[4];

    u32 n_local = n > 0 ? n : 1;

    u32 x[8];
    u32 y[8];
    u32 w[16];
    u32 ni = 0;

    while (1) {

        g_local[0] = SECP256K1_G_STRING0;
        g_local[1] = SECP256K1_G_STRING1;
        g_local[2] = SECP256K1_G_STRING2;
        g_local[3] = SECP256K1_G_STRING3;
        g_local[4] = SECP256K1_G_STRING4;
        g_local[5] = SECP256K1_G_STRING5;
        g_local[6] = SECP256K1_G_STRING6;
        g_local[7] = SECP256K1_G_STRING7;
        g_local[8] = SECP256K1_G_STRING8;

        return_value = parse_public(&g_xy_local, g_local);
        if (return_value != 0) {
            return;
        }

        point_mul_xy (x, y, k_local,  &g_xy_local);

        // keccak256
        u64 keccak_state[KECCAK256_STATE_LEN] = {0};

        w[7]  = l2be(x[0]);
        w[6]  = l2be(x[1]);
        w[5]  = l2be(x[2]);
        w[4]  = l2be(x[3]);
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

        ni++;

        r_local[0] = l2be((u32)(keccak_state[1] >> 32));
        r_local[1] = l2be((u32)keccak_state[2]);
        r_local[2] = l2be((u32)(keccak_state[2] >> 32));
        r_local[3] = l2be((u32)keccak_state[3]);
        r_local[4] = l2be((u32)(keccak_state[3] >> 32));
        rp_local = (((r_local[0] & m_local[0]) == p_local[0]) &&  
                    ((r_local[1] & m_local[1]) == p_local[1]) &&
                    ((r_local[2] & m_local[2]) == p_local[2]) &&
                    ((r_local[3] & m_local[3]) == p_local[3]) &&
                    ((r_local[4] & m_local[4]) == p_local[4])) 
                    ? ni : 0;

        if (ni >= n_local || rp_local) break;

        k_local[(ni & 7)] += 479001599;
    }

    //save results
    r0[i] = r_local[0];
    r1[i] = r_local[1];
    r2[i] = r_local[2];
    r3[i] = r_local[3];
    r4[i] = r_local[4];
    rp[i] = rp_local;

    k0[i] = k_local[7];
    k1[i] = k_local[6];
    k2[i] = k_local[5];
    k3[i] = k_local[4];
    k4[i] = k_local[3];
    k5[i] = k_local[2];
    k6[i] = k_local[1];
    k7[i] = k_local[0];
}