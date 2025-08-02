#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <quadmath.h>
#include <stdio.h>

void print_float128( __float128 value )
{
    char buffer[128];
    quadmath_snprintf( buffer, sizeof( buffer ), "%.34Qf", value );
    printf( "Float128: Input = %s, Raw bits = ", buffer );
    unsigned char *bytes = (unsigned char *)&value;
    for ( int i = sizeof( __float128 ) - 1; i >= 0; i-- )
    {
        printf( "%02x", bytes[i] );
    }
    printf( "\n" );
}

void print_fp8_e4m3( float input )
{
    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8( input, __NV_SATFINITE, __NV_E4M3 );

    __half half = __nv_cvt_fp8_to_halfraw( fp8, __NV_E4M3 );
    float output = __half2float( half );

    printf( "FP8 E4M3: Input = %.6f, Raw bits = 0x%02x, Round trip: %.6f\n", input,
            unsigned( *(unsigned char *)&fp8 ) & 0xFF, output );
}

void print_fp8_e5m2( float input )
{
    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8( input, __NV_SATFINITE, __NV_E5M2 );

    __half half = __nv_cvt_fp8_to_halfraw( fp8, __NV_E5M2 );
    float output = __half2float( half );

    printf( "FP8 E5M2: Input = %.6f, Raw bits = 0x%02x, Round trip: %.6f\n", input,
            unsigned( *(unsigned char *)&fp8 ) & 0xFF, output );
}

void print_bfloat16( float input )
{
    __nv_bfloat16 bf16 = __float2bfloat16( input );
    float output = __bfloat162float( bf16 );
    printf( "BF16: Input = %.6f, Raw bits = 0x%04x, Round trip: %.6f\n", input,
            unsigned( *(unsigned short *)&bf16 ) & 0xFFFF, output );
}

void print_fp16( float input )
{
    __half fp16 = __float2half( input );
    float output = __half2float( fp16 );
    printf( "FP16: Input = %.6f, Raw bits = 0x%04x, Round trip: %.6f\n", input,
            unsigned( *(unsigned short *)&fp16 ) & 0xFFFF, output );
}

int main()
{
    __float128 q = 42.125Q;    // Requires -fext-numeric-literals
    float f = 42.125f;
    print_float128( q );
    print_fp8_e4m3( f );
    print_fp8_e5m2( f );
    print_bfloat16( f );
    print_fp16( f );
    return 0;
}
