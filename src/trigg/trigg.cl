__constant static uint c_K[64] __attribute__((aligned(8))) =
{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
} ;

__constant static uint threads = 600047615;

__constant static int Z_PREP[8]  = {12,13,14,15,16,17,12,13}; /* Confirmed */
__constant static int Z_ING[32]  = {18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,23,24,31,32,33,34}; /* Confirmed */
__constant static int Z_INF[16]  = {44,45,46,47,48,50,51,52,53,54,55,56,57,58,59,60}; /* Confirmed */
__constant static int Z_ADJ[64]  = {61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,112,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128}; /* Confirmed */
__constant static int Z_AMB[16]  = {77,94,95,96,126,214,217,218,220,222,223,224,225,226,227,228}; /* Confirmed */
__constant static int Z_TIMED[8] = {84,243,249,250,251,252,253,255}; /* Confirmed */
__constant static int Z_NS[64] = {129,130,131,132,133,134,135,136,137,138,145,149,154,155,156,157,177,178,179,180,182,183,184,185,186,187,188,189,190,191,192,193,194,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,241,244,245,246,247,248,249,250,251,252,253,254,255}; /* Confirmed */
__constant static int Z_NPL[32] = {139,140,141,142,143,144,146,147,148,150,151,153,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,181}; /* Confirmed */
__constant static int Z_MASS[32] = {214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,242,214,215,216,219}; /* Confirmed */
__constant static int Z_INGINF[32] = {18,19,20,21,22,25,26,27,28,29,30,36,37,38,39,40,41,42,44,46,47,48,49,51,52,53,54,55,56,57,58,59}; /* Confirmed */
__constant static int Z_TIME[16] = {82,83,84,85,86,87,88,243,249,250,251,252,253,254,255,253}; /* Confirmed */
__constant static int Z_INGADJ[64]  = {18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,23,24,31,32,33,34,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92};/* Confirmed */


// For gcn3 to use v_perm_b32, we need Rocm driver. Use rotation instead.
//inline uint swab32(uint in)
//{
//    return ((((in) << 24) & 0xff000000u) | (((in) << 8) & 0x00ff0000u) | \
//            (((in) >> 8) & 0x0000ff00u) | (((in) >> 24) & 0x000000ffu));
//}
#define swab32(a) (as_uint(as_uchar4(a).wzyx))

#define ROTR32(x, n) (rotate( as_uint(x), as_uint(-n) ))
#define xor3b(a,b,c) ((a) ^ (b) ^ (c))

inline uint EP1(const uint e)
{
    return xor3b(ROTR32(e,6),ROTR32(e,11),ROTR32(e,25));
}

inline uint CH(const uint e, const uint f, const uint g)
{
    return bitselect(g, f, e);
}

inline uint EP0(const uint a)
{
    return xor3b(ROTR32(a,2),ROTR32(a,13),ROTR32(a,22));
}

inline uint MAJ(const uint a, const uint b, const uint c)
{
    return bitselect(c, a, c ^ b);
}

inline uint SIG0(const uint x)
{
    return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

inline uint SIG1(const uint x)
{
    return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

static void sha2_step2(uint a, uint b, uint c, uint *d, uint e, uint f, uint g, uint *h,
                       uint* in, uint pc, const uint Kshared)
{
    uint t1,t2;

    uint inx0 = in[pc];
    uint inx1 = in[(pc-2) & 0xF];
    uint inx2 = in[(pc-7) & 0xF];
    uint inx3 = in[(pc-15) & 0xF];

    uint sig1 = SIG1(inx1);
    uint sig0 = SIG0(inx3);
    uint ch = CH(e, f, g);
    uint ep1 = EP1(e);
    uint ep0 = EP0(a);
    uint maj = MAJ(a,b,c);

    in[pc] = sig1 + inx2 + sig0 + inx0;

    t1 = *h + ep1 + ch + Kshared + in[pc];
    t2 = ep0 + maj;
    *d =  *d + t1;
    *h = t1 + t2;
}

static void sha2_step1(uint a, uint b, uint c, uint *d, uint e, uint f, uint g, uint *h,
                       uint in, const uint Kshared)
{
    uint t1,t2;
    uint ep1 = EP1(e);
    uint ch = CH(e, f, g);
    uint ep0 = EP0(a);
    uint maj = MAJ(a,b,c);

    t1 = *h + ep1 + ch + Kshared + in; // Rotation of 'in' already done in trigg
    t2 = ep0 + maj;
    *d = *d + t1;
    *h = t1 + t2;
}

static void sha_transform(uint* in, uint* state, __constant uint* const Kshared)
{
    uint a = state[0];
    uint b = state[1];
    uint c = state[2];
    uint d = state[3];
    uint e = state[4];
    uint f = state[5];
    uint g = state[6];
    uint h = state[7];

    sha2_step1(a,b,c,&d,e,f,g,&h,in[ 0], Kshared[ 0]);
    sha2_step1(h,a,b,&c,d,e,f,&g,in[ 1], Kshared[ 1]);
    sha2_step1(g,h,a,&b,c,d,e,&f,in[ 2], Kshared[ 2]);
    sha2_step1(f,g,h,&a,b,c,d,&e,in[ 3], Kshared[ 3]);
    sha2_step1(e,f,g,&h,a,b,c,&d,in[ 4], Kshared[ 4]);
    sha2_step1(d,e,f,&g,h,a,b,&c,in[ 5], Kshared[ 5]);
    sha2_step1(c,d,e,&f,g,h,a,&b,in[ 6], Kshared[ 6]);
    sha2_step1(b,c,d,&e,f,g,h,&a,in[ 7], Kshared[ 7]);
    sha2_step1(a,b,c,&d,e,f,g,&h,in[ 8], Kshared[ 8]);
    sha2_step1(h,a,b,&c,d,e,f,&g,in[ 9], Kshared[ 9]);
    sha2_step1(g,h,a,&b,c,d,e,&f,in[10], Kshared[10]);
    sha2_step1(f,g,h,&a,b,c,d,&e,in[11], Kshared[11]);
    sha2_step1(e,f,g,&h,a,b,c,&d,in[12], Kshared[12]);
    sha2_step1(d,e,f,&g,h,a,b,&c,in[13], Kshared[13]);
    sha2_step1(c,d,e,&f,g,h,a,&b,in[14], Kshared[14]);
    sha2_step1(b,c,d,&e,f,g,h,&a,in[15], Kshared[15]);

#pragma unroll
    for (int i=0; i<3; i++)
    {
        sha2_step2(a,b,c,&d,e,f,g,&h,in,0, Kshared[16+16*i]);
        sha2_step2(h,a,b,&c,d,e,f,&g,in,1, Kshared[17+16*i]);
        sha2_step2(g,h,a,&b,c,d,e,&f,in,2, Kshared[18+16*i]);
        sha2_step2(f,g,h,&a,b,c,d,&e,in,3, Kshared[19+16*i]);
        sha2_step2(e,f,g,&h,a,b,c,&d,in,4, Kshared[20+16*i]);
        sha2_step2(d,e,f,&g,h,a,b,&c,in,5, Kshared[21+16*i]);
        sha2_step2(c,d,e,&f,g,h,a,&b,in,6, Kshared[22+16*i]);
        sha2_step2(b,c,d,&e,f,g,h,&a,in,7, Kshared[23+16*i]);
        sha2_step2(a,b,c,&d,e,f,g,&h,in,8, Kshared[24+16*i]);
        sha2_step2(h,a,b,&c,d,e,f,&g,in,9, Kshared[25+16*i]);
        sha2_step2(g,h,a,&b,c,d,e,&f,in,10,Kshared[26+16*i]);
        sha2_step2(f,g,h,&a,b,c,d,&e,in,11,Kshared[27+16*i]);
        sha2_step2(e,f,g,&h,a,b,c,&d,in,12,Kshared[28+16*i]);
        sha2_step2(d,e,f,&g,h,a,b,&c,in,13,Kshared[29+16*i]);
        sha2_step2(c,d,e,&f,g,h,a,&b,in,14,Kshared[30+16*i]);
        sha2_step2(b,c,d,&e,f,g,h,&a,in,15,Kshared[31+16*i]);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

static int gpu_trigg_eval(uint *h, uint d)
{
    uint *bp,n;
    d = d & 0xff;
    for (bp = h, n = d >> 5; n; n--) {
        if (*bp++ != 0) return 0;
    }
    return clz(*bp) >= (d & 31);
}

__kernel void hello(__global char* string)
{
    string[0] = 'H';
    string[1] = 'e';
    string[2] = 'l';
    string[3] = 'l';
    string[4] = 'o';
    string[5] = ',';
    string[6] = ' ';
    string[7] = 'W';
    string[8] = 'o';
    string[9] = 'r';
    string[10] = 'l';
    string[11] = 'd';
    string[12] = '!';
    string[13] = '\0';
}

__kernel void trigg(__global int *g_found, __global uint8 *g_seed, __constant uint* c_input32,
                    __constant uint *c_midstate256, __constant uint *c_blockNumber8, uint c_difficulty)
{
    const uint thread = get_global_id(0);
    uint8 seed[16] = {0};
    uint input[16], state[8];

    if (thread <= threads) {

        if(0 < thread <= 131071) { /* Total Permutations, this frame: 131,072 */
            seed[ 0] = Z_PREP[(thread & 7)];
            seed[ 1] = Z_TIMED[(thread >> 3) & 7];
            seed[ 2] = 1;
            seed[ 3] = 5;
            seed[ 4] = Z_NS[(thread >> 6) & 63];
            seed[ 5] = 1;
            seed[ 6] = Z_ING[(thread >> 12) & 31];
        }
        if(131071 < thread <= 262143) { /* Total Permutations, this frame: 131,072 */
            seed[ 0] = Z_TIME[(thread & 15)];
            seed[ 1] = Z_MASS[(thread >> 4) & 31];
            seed[ 2] = 1;
            seed[ 3] = Z_INF[(thread >> 9) & 15];
            seed[ 4] = 9;
            seed[ 5] = 2;
            seed[ 6] = 1;
            seed[ 7] = Z_AMB[(thread >> 13) & 15];
        }
        if(262143 < thread <= 4456447) { /* Total Permutations, this frame: 4,194,304 */
            seed[ 0] = Z_PREP[(thread & 7)];
            seed[ 1] = Z_TIMED[(thread >> 3) & 7];
            seed[ 2] = 1;
            seed[ 3] = Z_ADJ[(thread >> 6) & 63];
            seed[ 4] = Z_NPL[(thread >> 12) & 31];
            seed[ 5] = 1;
            seed[ 6] = Z_INGINF[(thread >> 17) & 31];
        }
        if(4456447 < thread <= 12845055) { /* Total Permutations, this frame: 8,388,608 */
            seed[ 0] = 5;
            seed[ 1] = Z_NS[(thread & 63)];
            seed[ 2] = 1;
            seed[ 3] = Z_PREP[(thread >> 6) & 7];
            seed[ 4] = Z_TIMED[(thread >> 9) & 7];
            seed[ 5] = Z_MASS[(thread >> 12) & 31];
            seed[ 6] = 3;
            seed[ 7] = 1;
            seed[ 8] = Z_ADJ[(thread >> 17) & 63];
        }
        if(12845055 < thread <= 29622271) { /* Total Permutations, this frame: 16,777,216 */
            seed[ 0] = Z_PREP[thread & 7];
            seed[ 1] = Z_ADJ[(thread >> 3) & 63];
            seed[ 2] = Z_MASS[(thread >> 9) & 31];
            seed[ 3] = 1;
            seed[ 4] = Z_NPL[(thread >> 14) & 31];
            seed[ 5] = 1;
            seed[ 6] = Z_INGINF[(thread >> 19) & 31];
        }
        if(29622271 < thread <= 46399487) { /* Total Permutations, this frame: 16,777,216 */
            seed[ 0] = Z_PREP[(thread & 7)];
            seed[ 1] = Z_MASS[(thread >> 3) & 31];
            seed[ 2] = 1;
            seed[ 3] = Z_ADJ[(thread >> 8) & 63];
            seed[ 4] = Z_NPL[(thread >> 14) & 31];
            seed[ 5] = 1;
            seed[ 6] = Z_INGINF[(thread >> 19) & 31];
        }
        if(46399487 < thread <= 63176703) { /* Total Permutations, this frame: 16,777,216 */
            seed[ 0] = Z_TIME[(thread & 15)];
            seed[ 1] = Z_AMB[(thread >> 4) & 15];
            seed[ 2] = 1;
            seed[ 3] = Z_ADJ[(thread >> 8) & 63];
            seed[ 4] = Z_MASS[(thread >> 14) & 31];
            seed[ 5] = 1;
            seed[ 6] = Z_ING[(thread >> 19) & 31];
        }
        if(63176703 < thread <= 600047615 ) { /* Total Permutations, this frame: 536,870,912 */
            seed[ 0] = Z_TIME[(thread & 15)];
            seed[ 1] = Z_AMB[(thread >> 4) & 15];
            seed[ 2] = 1;
            seed[ 3] = Z_PREP[(thread >> 8) & 7];
            seed[ 4] = 5;
            seed[ 5] = Z_ADJ[(thread >> 11) & 63];
            seed[ 6] = Z_NS[(thread >> 17) & 63];
            seed[ 7] = 3;
            seed[ 8] = 1;
            seed[ 9] = Z_INGADJ[(thread >> 23) & 63];
        }

#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            input[i] = c_input32[i];
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            input[8 + i] = swab32(((uint *) seed)[i]);
        }

        input[12] = swab32(c_blockNumber8[0]);
        input[13] = swab32(c_blockNumber8[1]);
        input[14] = 0x80000000;
        input[15] = 0;

#pragma unroll
        for (int i = 0; i < 8; i += 2)
        {
            *((uint2 *)(&state[i])) = *((__constant uint2 *)(&c_midstate256[i]));
        }

        sha_transform(input, state, c_K);

#pragma unroll
        for (int i = 0; i < 15; i++)
        {
            input[i] = 0;
        }
        input[15] = 0x9c0;

        sha_transform(input, state, c_K);

        if (gpu_trigg_eval(state, c_difficulty))
        {
            *g_found = 1;
#pragma unroll
            for (int i = 0; i < 16; i++)
            {
                g_seed[i] = seed[i];
            }
        }
    }
}
