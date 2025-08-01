#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main()
{
    long double ld = 1.0;
    __float128 f128 = 1.0;
    __uint128_t uld;
    __uint128_t u128;

    printf( "%zu\n", sizeof( ld ) );
    printf( "%zu\n", sizeof( __float128 ) );

    char buffer[128];
    quadmath_snprintf( buffer, sizeof( buffer ), "%Qf", f128 );

    printf( "%Lf\n", ld );
    printf( "%s\n", buffer );

    memcpy( &uld, &ld, sizeof(uld) );
    memcpy( &u128, &f128, sizeof(u128) );

    printf( "long double: %08lX%08lX\n", uint64_t(uld >> 64), uint64_t(uld) );
    printf( "__float128: %08lX%08lX\n", uint64_t(u128 >> 64), uint64_t(u128) );

    return 0;
}
