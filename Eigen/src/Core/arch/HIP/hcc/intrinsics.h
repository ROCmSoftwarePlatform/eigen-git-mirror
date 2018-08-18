/* 
** Alternates for CUDA intrinsics
*/
#ifndef INTRINSICS_H
#define INTRINSICS_H

#ifdef __HIPCC__        // For HC backend
    #define WARP_SIZE 64
#else                  // For NVCC backend
    #define WARP_SIZE 32
#endif

#define __HIP_FP16_DECL_PREFIX__ __device__

/*-----------------------HIPRT NUMBERS-----------------------*/
__HIP_FP16_DECL_PREFIX__ inline float __hip_int_as_float(int a) {
  union {
    int a;
    float b;
  }u;
  u.a = a;
  return u.b;
}

// HILO INT 2 DOUBLE
// Combine two 32 bit integer into a 64 bit double
__HIP_FP16_DECL_PREFIX__  inline double __hip_hiloint2double(int hi, int lo) {
   union {
      long longType;
      double doubleType;
   }u;

   long mostSignificantBits = (long)hi & 0xFFFFFFFF;
   long leastSignificantBits = (long)lo & 0xFFFFFFFF;
   /* Store the hi as 32 MSB and lo as 32 LSB of double */
   u.longType = (mostSignificantBits << 32) | leastSignificantBits;
   /* Return the equivalent double type */
   return u.doubleType;
}

__HIP_FP16_DECL_PREFIX__ inline double __hip_longlong_as_double(const long long x) {
   union {
      long long a;
      double b;
   }u;

   u.a = x;
   return u.b;
}

__HIP_FP16_DECL_PREFIX__ inline long long __hip_double_as_longlong(const double x) {
   union {
      long long a;
      double b;
   }u;

   u.b = x;
   return u.a;
}

// Single Precision Macros
#define HIPRT_INF_F        __hip_int_as_float(0x7f800000)
#define HIPRT_NAN_F        __hip_int_as_float(0x7fffffff)
#define HIPRT_MAX_NORMAL_F __hip_int_as_float(0x7f7fffff)
#define HIPRT_MIN_DENORM_F __hip_int_as_float(0x00000001)
#define HIPRT_NEG_ZERO_F   __hip_int_as_float(0x80000000)
#define HIPRT_ZERO_F       0.0f
#define HIPRT_ONE_F        1.0f


// Double Precision Macros
#define HIPRT_INF          __hip_hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN          __hip_hiloint2double(0xfff80000, 0x00000000)

/*-----------------------HIPRT NUMBERS-----------------------*/





/*--------------------BIT MANIPULATION INTRINSICS--------------------*/

__HIP_FP16_DECL_PREFIX__ inline int __hip_clz(int x)
{
    int count = 0;
    int input = x;
    for (int i = 0; i < 32; i++)
    {
        if (input % 2 == 0) count++;
        else count = 0;
        input = input / 2;
    }
    return count;
}

__HIP_FP16_DECL_PREFIX__ inline int __hip_clzll(long long x)
{
    int count = 0;
    long long input = x;
    for (int i = 0; i < 64; i++)
    {
        if (input % 2 == 0) count++;
        else count = 0;
        input = input / 2;
    }
    return count;
}

__HIP_FP16_DECL_PREFIX__ inline unsigned int __hip_umulhi(unsigned int x, unsigned int y)
{
    unsigned long out = ((unsigned long)x) * ((unsigned long)y);
    unsigned int res = (unsigned int)(out >> 32);
    return res;
}

__HIP_FP16_DECL_PREFIX__ inline unsigned long long __hip_umul64hi(unsigned long long x, unsigned long long y)
{
    unsigned long long lo = 0x00000000FFFFFFFF;
    unsigned long long hi = 0xFFFFFFFF00000000;

    // Seperate 32-bit LSBs & MSBs of 64-bit inputs
    unsigned long long in1_lo = x & lo;
    unsigned long long in1_hi = (x & hi) >> 32;
    unsigned long long in2_lo = y & lo;
    unsigned long long in2_hi = (y & hi) >> 32;

    // Multiply each part of input and store
    unsigned long long out[4];
    out[0] = in1_lo * in2_lo;
    out[1] = in1_lo * in2_hi;
    out[2] = in1_hi * in2_lo;
    out[3] = in1_hi * in2_hi;

    unsigned long long carry;
    unsigned long long res;
    unsigned long long part[4];

    // Store the result of x*y in a vector that can hold 128 bit result
    part[0] = out[0] & lo;
    res = ((out[0] & hi) >> 32) + (out[1] & lo) + (out[2] & lo);
    part[1] = res & lo;
    carry = (res & hi) >> 32;
    res = carry + ((out[1] & hi) >> 32) + ((out[2] & hi) >> 32) + (out[3] & lo);
    part[2] = res & lo;
    carry = (res & hi) >> 32;
    part[3] = carry + ((out[3] & hi) >> 32);

    // Get the 64-bit MSB's of x*y
    res = (((part[3] << 32) & hi) | (part[2] & lo));

    return res;
}

/*--------------------BIT MANIPULATION INTRINSICS--------------------*/


/*------------------DUMMY SUPPORT FOR UNSUPPORTED INTRINSICS------------------*/

//TODO: Replace them once supported by HC
#ifdef __HIPCC__        // For HC backend
    #define __hip_threadfence() hc_barrier(CLK_LOCAL_MEM_FENCE)
    #define __hip_threadfence_block() hc_barrier(CLK_LOCAL_MEM_FENCE)

    template <typename T>
    __HIP_FP16_DECL_PREFIX__ T __hip_ldg(const T* ptr) { return *ptr; }

    #define __hip_pld(ADDR) __builtin_prefetch(ADDR)
#endif

/*------------------DUMMY SUPPORT FOR UNSUPPORTED INTRINSICS------------------*/


#endif

