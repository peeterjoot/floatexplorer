//
// This is a little program that unpacks a 32-bit floating point value (letting
// std::bitset do the bitarray printing)
//
#include <bitset>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <limits>

void print_float32_representation( float f )
{
    static_assert( sizeof( f ) == 4 );
    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

    std::uint32_t x;
    std::memcpy( &x, &f, 4 );

    std::bitset<32> b = x;
    std::string bs = b.to_string();
    std::uint32_t mantissa = x & ( ( 1 << 23 ) - 1 );
    std::uint32_t exponent_with_bias = ( x >> 23 ) & 0xFF;
    std::int32_t exponent;

    if ( exponent_with_bias && exponent_with_bias != 255 )
    {
        exponent = (std::int32_t)exponent_with_bias - 127;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -126;    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint32_t sign = x >> 31;

    auto mstring = bs.substr( 9, 23 );
    auto estring = bs.substr( 1, 8 );

    if (exponent_with_bias == 255) {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:08X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:          {}\n",
            f, x, b.to_string(), sign, estring, mstring);
    } else {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:08X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}                        ({}{}{})\n"
            "mantissa:          {}\n",
            f, x, b.to_string(), sign,
            estring, exponent_with_bias ? "127 " : "0", exponent >= 0 ? "+" : "", exponent,
            mstring );
    }

    if ( exponent_with_bias == 255 )
    {
        if ( mantissa == 0 )
        {
            std::cout << std::format( "number:   {}\n\n",
                                      sign ? "-inf" : "+inf" );
        }
        else
        {
            std::cout << "number:   NaN\n\n";
        }
    }
    else if ( !exponent_with_bias && mantissa )
    {
        // Denormal: exponent is -126, no implicit leading 1
        std::cout << std::format( "number:         {}0.{} x 2^({})\n\n",
                                  ( sign ? "-" : " " ), mstring, -126 );
    }
    else
    {
        std::cout << std::format( "number:         {}{}.{} x 2^({})\n\n",
                                  ( sign ? "-" : " " ), x ? 1 : 0, mstring,
                                  exponent );
    }
}

int main( int argc, char** argv )
{
    if ( argc == 1 )
    {
        std::cout << "Usage example:\n\n./floatexplorer 1 -2 6 1.5 0.125 -inf\n\nNo parameters will test special cases:\n";

        float tests[] = { 0.0f, std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(),
                          1.17549435e-38f, // Smallest normal
                          3.4028235e38f }; // Largest normal

        for ( float test : tests )
        {
            std::cout << "\nTest value: " << test << "\n";
            print_float32_representation( test );
        }

        float f;
        // Test denormals
        std::cout << "\nSmallest denormal:\n";
        std::uint32_t denormal_bits = 0x00000001;
        std::memcpy( &f, &denormal_bits, sizeof( float ) );
        print_float32_representation( f );

        std::cout << "\nLargest denormal:\n";
        denormal_bits = 0x007FFFFF;
        std::memcpy( &f, &denormal_bits, sizeof( float ) );
        print_float32_representation( f );

        return 0;
    }

    for ( int i = 1; i < argc; i++ )
    {
        try
        {
            float f = std::stof( argv[i] );

            print_float32_representation( f );
        }
        catch ( std::exception& e )
        {
            std::cerr << std::format(
                "Failed to convert input '{}' to floating point. error: {}\n",
                argv[i], e.what() );
        }
    }

    return 0;
}

// vim: et ts=4 sw=4
