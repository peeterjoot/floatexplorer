//
// This is a little program that unpacks a 32-bit floating point value
//
//
#include <string.h>

#include <bitset>
#include <format>
#include <iostream>
#include <string>

int main( int argc, char** argv )
{
    float f = 0;
    static_assert( sizeof(f) == 4 );

    if ( argc == 2 )
    {
        f = std::stof( argv[1] );
    }

    unsigned x;
    memcpy( &x, &f, 4 );
    char h[10];
    snprintf( h, sizeof( h ), "%08X", x );

    std::bitset<32> b = x;
    std::string bs = b.to_string();
    int exponent = ( x >> 23 ) & 0xFF;

    // special case: zero has all zero exponent bits, as well as mantissa bits:
    if ( exponent )
    {
        exponent -= 127;
    }
    unsigned sign = x >> 31;

    auto mstring = bs.substr( 9, 23 );
    auto estring = bs.substr( 1, 8 );

    std::cout << std::format(
        "hex:      {}\n"
        "bits:     {}\n"
        "sign:     {}\n"
        "exponent:  {}                        (127 {}{})\n"
        "mantissa:          {}\n",
        h,
	b.to_string(),
	sign,
	estring, exponent >= 0 ? "+" : "", exponent,
	mstring );

    if ( x )
    {
        std::cout << std::format( "number:         {}1.{} x 2^({})\n", (sign ? "-" : " "), mstring, exponent );
    }

    return 0;
}
