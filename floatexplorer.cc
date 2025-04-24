//
// This is a little program that unpacks a 32-bit floating point value (letting
// std::bitset do the bitarray printing)
//
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <limits>
#include <getopt.h>

#define FLOAT32_MANTISSA_BITS           23
#define FLOAT32_EXPONENT_BITS           8
#define FLOAT32_EXPONENT_MASK           ( (std::uint32_t(1) << FLOAT32_EXPONENT_BITS) - 1 )
#define FLOAT32_EXPONENT_BIAS           ( (std::uint32_t(1) << ( FLOAT32_EXPONENT_BITS - 1 ) ) -1 )
#define FLOAT32_EXPONENT_BIAS_STRING    "127 "
void print_float32_representation( float f )
{
    static_assert( sizeof( f ) == sizeof( std::uint32_t ) );
    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

    std::uint32_t x;
    std::memcpy( &x, &f, sizeof(f) );

    std::bitset<32> b = x;
    std::string bs = b.to_string();
    std::uint32_t mantissa = x & ( ( std::uint32_t(1) << FLOAT32_MANTISSA_BITS ) - 1 );
    std::uint32_t exponent_with_bias = ( x >> FLOAT32_MANTISSA_BITS ) & FLOAT32_EXPONENT_MASK;
    std::int32_t exponent;

    if ( exponent_with_bias && exponent_with_bias != FLOAT32_EXPONENT_MASK )
    {
        exponent = (std::int32_t)exponent_with_bias - FLOAT32_EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -(FLOAT32_EXPONENT_BIAS-1);    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint32_t sign = x >> (FLOAT32_EXPONENT_BITS + FLOAT32_MANTISSA_BITS);

    auto mstring = bs.substr( 1 + FLOAT32_EXPONENT_BITS, FLOAT32_MANTISSA_BITS );
    auto estring = bs.substr( 1, FLOAT32_EXPONENT_BITS );

    if (exponent_with_bias == FLOAT32_EXPONENT_MASK) {
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
            estring, exponent_with_bias ? FLOAT32_EXPONENT_BIAS_STRING : "0", exponent >= 0 ? "+" : "", exponent,
            mstring );
    }

    if ( exponent_with_bias == FLOAT32_EXPONENT_MASK )
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
                                  ( sign ? "-" : " " ), mstring, -(FLOAT32_EXPONENT_BIAS-1) );
    }
    else
    {
        std::cout << std::format( "number:         {}{}.{} x 2^({})\n\n",
                                  ( sign ? "-" : " " ), x ? 1 : 0, mstring,
                                  exponent );
    }
}

#define FLOAT64_MANTISSA_BITS           52
#define FLOAT64_EXPONENT_BITS           11
#define FLOAT64_EXPONENT_MASK           ( (std::uint64_t(1) << FLOAT64_EXPONENT_BITS) - 1 )
#define FLOAT64_EXPONENT_BIAS           ( (std::uint64_t(1) << ( FLOAT64_EXPONENT_BITS - 1 ) ) -1 )
#define FLOAT64_EXPONENT_BIAS_STRING    "1023 "
void print_float64_representation( double f )
{
    static_assert( sizeof( f ) == sizeof( std::uint64_t ) );
    static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 required");

    std::uint64_t x;
    std::memcpy( &x, &f, sizeof(f) );

    std::bitset<64> b = x;
    std::string bs = b.to_string();
    std::uint64_t mantissa = x & ( ( std::uint64_t(1) << FLOAT64_MANTISSA_BITS ) - 1 );
    std::uint64_t exponent_with_bias = ( x >> FLOAT64_MANTISSA_BITS ) & FLOAT64_EXPONENT_MASK;
    std::int64_t exponent;

    if ( exponent_with_bias && exponent_with_bias != FLOAT64_EXPONENT_MASK )
    {
        exponent = (std::int64_t)exponent_with_bias - FLOAT64_EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -(FLOAT64_EXPONENT_BIAS-1);    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint64_t sign = x >> (FLOAT64_EXPONENT_BITS + FLOAT64_MANTISSA_BITS);

    auto mstring = bs.substr( 1 + FLOAT64_EXPONENT_BITS, FLOAT64_MANTISSA_BITS );
    auto estring = bs.substr( 1, FLOAT64_EXPONENT_BITS );

    if (exponent_with_bias == FLOAT64_EXPONENT_MASK) {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:016X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:          {}\n",
            f, x, b.to_string(), sign, estring, mstring);
    } else {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:016X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}                                                     ({}{}{})\n"
            "mantissa:             {}\n",
            f, x, b.to_string(), sign,
            estring, exponent_with_bias ? FLOAT64_EXPONENT_BIAS_STRING : "0", exponent >= 0 ? "+" : "", exponent,
            mstring );
    }

    if ( exponent_with_bias == FLOAT64_EXPONENT_MASK )
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
        std::cout << std::format( "number:            {}0.{} x 2^({})\n\n",
                                  ( sign ? "-" : " " ), mstring, -(FLOAT64_EXPONENT_BIAS-1) );
    }
    else
    {
        std::cout << std::format( "number:            {}{}.{} x 2^({})\n\n",
                                  ( sign ? "-" : " " ), x ? 1 : 0, mstring,
                                  exponent );
    }
}

void printHelpAndExit()
{
    std::cout <<
        "floatexplorer [--float] [--double] [--special] number [number]*\n\n"
        "Examples:\n"
        "floatexplorer 1 -2 6 1.5 0.125 -inf # --float is the default\n"
        "floatexplorer --double 1 -2 6 1.5 0.125 -inf\n"
        "floatexplorer --float --double 1 # both representations\n";

    std::exit(0);
}

int main( int argc, char** argv )
{
    int c;
    bool dofloat{};
    bool dodouble{};
    bool specialcases{};
    const struct option long_options[] = { { "help", 0, NULL, 'h' },
                                           { "float", 0, NULL, 'f' },
                                           { "double", 0, NULL, 'd' },
                                           { "special", 0, NULL, 's' },
                                           { NULL, 0, NULL, 0 } };

    while ( -1 !=
            ( c = getopt_long( argc, argv, "hfds", long_options, NULL ) ) )
    {
        switch ( c )
        {
            case 'f':
            {
                dofloat = true;
                break;
            }
            case 'd':
            {
                dodouble = true;
                break;
            }
            case 's':
            {
                specialcases = true;
                break;
            }
            case 'h':
            default:
            {
                printHelpAndExit();
            }
        }
    }

    if ( !dofloat && !dodouble )
    {
        dofloat = true;
    }

    if ( specialcases )
    {
        if ( dofloat )
        {
            float tests[] = { 0.0f,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN(),
                              1.17549435e-38f,    // Smallest normal
                              3.4028235e38f };    // Largest normal

            for ( auto test : tests )
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
            denormal_bits =
                ( std::uint32_t( 1 ) << FLOAT32_MANTISSA_BITS ) - 1;
            std::memcpy( &f, &denormal_bits, sizeof( float ) );
            print_float32_representation( f );
        }

        if ( dodouble )
        {
            double tests[] = {
                0.0d,
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN(),
                2.2250738585072014e-308, // Smallest normal double
                1.7976931348623157e308    // Largest normal double
            };

            for ( auto test : tests )
            {
                std::cout << "\nTest value: " << test << "\n";
                print_float64_representation( test );
            }

            double f;
            // Test denormals
            std::cout << "\nSmallest denormal:\n";
            std::uint64_t denormal_bits = 0x00000001;
            std::memcpy( &f, &denormal_bits, sizeof( double ) );
            print_float64_representation( f );

            std::cout << "\nLargest denormal:\n";
            denormal_bits =
                ( std::uint64_t( 1 ) << FLOAT64_MANTISSA_BITS ) - 1;
            std::memcpy( &f, &denormal_bits, sizeof( double ) );
            print_float64_representation( f );
        }
    }
    else if ( argc == optind )
    {
        printHelpAndExit();
    }

    for ( int i = optind; i < argc; i++ )
    {
        try
        {
            if ( dofloat )
            {
                float f = std::stof( argv[i] );

                print_float32_representation( f );
            }

            if ( dodouble )
            {
                double f = std::stod( argv[i] );

                print_float64_representation( f );
            }
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
