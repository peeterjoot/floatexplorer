#include <stdio.h>

int main()
{
    float f{};
    double d{};
    long double ld{};
    printf( "%zu\n", sizeof(f) );
    printf( "%zu\n", sizeof(d) );
    printf( "%zu\n", sizeof(ld) );
    printf( "%zu\n", sizeof(__uint128_t) );

#ifdef __HAVE_FLOAT128
    printf( "float128\n" );
#endif

    return 0;
}

