//
// This is a little program that unpacks a 32-bit or 64-bit IEEE floating point value (letting
// std::bitset do the bitarray printing)
//
#include <getopt.h>
#include <strings.h>

#include <bitset>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define FLOAT32_MANTISSA_BITS 23
#define FLOAT32_EXPONENT_BITS 8
#define FLOAT32_EXPONENT_MASK ( ( std::uint32_t( 1 ) << FLOAT32_EXPONENT_BITS ) - 1 )
#define FLOAT32_EXPONENT_BIAS ( ( std::uint32_t( 1 ) << ( FLOAT32_EXPONENT_BITS - 1 ) ) - 1 )

using float32 = float;

void print_float32_representation( float32 f )
{
    static_assert( sizeof( f ) == sizeof( std::uint32_t ) );
    static_assert( std::numeric_limits<float32>::is_iec559, "IEEE 754 required" );

    std::uint32_t x;
    std::memcpy( &x, &f, sizeof( f ) );

    std::bitset<32> b = x;
    std::string bs = b.to_string();
    std::uint32_t mantissa = x & ( ( std::uint32_t( 1 ) << FLOAT32_MANTISSA_BITS ) - 1 );
    std::uint32_t exponent_with_bias = ( x >> FLOAT32_MANTISSA_BITS ) & FLOAT32_EXPONENT_MASK;
    std::int32_t exponent;

    if ( exponent_with_bias && exponent_with_bias != FLOAT32_EXPONENT_MASK )
    {
        exponent = (std::int32_t)exponent_with_bias - FLOAT32_EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -( FLOAT32_EXPONENT_BIAS - 1 );    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint32_t sign = x >> ( FLOAT32_EXPONENT_BITS + FLOAT32_MANTISSA_BITS );

    auto mstring = bs.substr( 1 + FLOAT32_EXPONENT_BITS, FLOAT32_MANTISSA_BITS );
    auto estring = bs.substr( 1, FLOAT32_EXPONENT_BITS );

    if ( exponent_with_bias == FLOAT32_EXPONENT_MASK )
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:08X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:          {}\n",
            f, x, b.to_string(), sign, estring, mstring );
    }
    else
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:08X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}                        ({}{}{}{})\n"
            "mantissa:          {}\n",
            f, x, b.to_string(), sign, estring, exponent_with_bias ? FLOAT32_EXPONENT_BIAS : 0,
            exponent_with_bias ? " " : "", exponent >= 0 ? "+" : "", exponent, mstring );
    }

    if ( exponent_with_bias == FLOAT32_EXPONENT_MASK )
    {
        if ( mantissa == 0 )
        {
            std::cout << std::format( "number:   {}\n\n", sign ? "-inf" : "+inf" );
        }
        else
        {
            std::cout << "number:   NaN\n\n";
        }
    }
    else if ( !exponent_with_bias && mantissa )
    {
        // Denormal: exponent is -126, no implicit leading 1
        std::cout << std::format( "number:         {}0.{} x 2^({})\n\n", ( sign ? "-" : " " ), mstring,
                                  -( FLOAT32_EXPONENT_BIAS - 1 ) );
    }
    else
    {
        std::cout << std::format( "number:         {}{}.{} x 2^({})\n\n", ( sign ? "-" : " " ), x ? 1 : 0, mstring,
                                  exponent );
    }
}

#define FLOAT64_MANTISSA_BITS 52
#define FLOAT64_EXPONENT_BITS 11
#define FLOAT64_EXPONENT_MASK ( ( std::uint64_t( 1 ) << FLOAT64_EXPONENT_BITS ) - 1 )
#define FLOAT64_EXPONENT_BIAS ( ( std::uint64_t( 1 ) << ( FLOAT64_EXPONENT_BITS - 1 ) ) - 1 )

using float64 = double;

void print_float64_representation( float64 f )
{
    static_assert( sizeof( f ) == sizeof( std::uint64_t ) );
    static_assert( std::numeric_limits<float64>::is_iec559, "IEEE 754 required" );

    std::uint64_t x;
    std::memcpy( &x, &f, sizeof( f ) );

    std::bitset<64> b = x;
    std::string bs = b.to_string();
    std::uint64_t mantissa = x & ( ( std::uint64_t( 1 ) << FLOAT64_MANTISSA_BITS ) - 1 );
    std::uint64_t exponent_with_bias = ( x >> FLOAT64_MANTISSA_BITS ) & FLOAT64_EXPONENT_MASK;
    std::int64_t exponent;

    if ( exponent_with_bias && exponent_with_bias != FLOAT64_EXPONENT_MASK )
    {
        exponent = (std::int64_t)exponent_with_bias - FLOAT64_EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -( FLOAT64_EXPONENT_BIAS - 1 );    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint64_t sign = x >> ( FLOAT64_EXPONENT_BITS + FLOAT64_MANTISSA_BITS );

    auto mstring = bs.substr( 1 + FLOAT64_EXPONENT_BITS, FLOAT64_MANTISSA_BITS );
    auto estring = bs.substr( 1, FLOAT64_EXPONENT_BITS );

    if ( exponent_with_bias == FLOAT64_EXPONENT_MASK )
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:016X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:          {}\n",
            f, x, b.to_string(), sign, estring, mstring );
    }
    else
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:016X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}                                                     "
            "({}{}{}{})\n"
            "mantissa:             {}\n",
            f, x, b.to_string(), sign, estring, exponent_with_bias ? FLOAT64_EXPONENT_BIAS : 0,
            exponent_with_bias ? " " : "", exponent >= 0 ? "+" : "", exponent, mstring );
    }

    if ( exponent_with_bias == FLOAT64_EXPONENT_MASK )
    {
        if ( mantissa == 0 )
        {
            std::cout << std::format( "number:   {}\n\n", sign ? "-inf" : "+inf" );
        }
        else
        {
            std::cout << "number:   NaN\n\n";
        }
    }
    else if ( !exponent_with_bias && mantissa )
    {
        // Denormal: exponent is -126, no implicit leading 1
        std::cout << std::format( "number:            {}0.{} x 2^({})\n\n", ( sign ? "-" : " " ), mstring,
                                  -( FLOAT64_EXPONENT_BIAS - 1 ) );
    }
    else
    {
        std::cout << std::format( "number:            {}{}.{} x 2^({})\n\n", ( sign ? "-" : " " ), x ? 1 : 0, mstring,
                                  exponent );
    }
}

enum class option_values : int
{
    float_ = '4',
    float32_ = '4',
    float64_ = '8',
    double_ = '8',
    float80_ = 'A',     // 10
    float128_ = 'G',    // 16
    longdouble_ = 'l',
    help_ = 'h',
    special_ = 's',
};

#if defined __x86_64
using float80 = long double;
#define FLOAT80_SPECIFIER "%La"    // fixme
#define FLOAT80_HELP "[--float80 | --longdouble] "
#define FLOAT80_OPTIONS { "f80", 0, NULL, (int)option_values::float80_ },
#define LONGDOUBLE_OPTIONS { "longdouble", 0, NULL, (int)option_values::longdouble_ },

#define LONG_DOUBLE_IS_FLOAT80
#endif

#ifdef __HAVE_FLOAT128
using float128 = long double;
#define FLOAT128_SPECIFIER "%La"
#define FLOAT128_HELP "[--float128 | --longdouble] "
#define FLOAT128_OPTIONS { "f128", 0, NULL, 'l' },
#define LONGDOUBLE_OPTIONS { "longdouble", 0, NULL, (int)option_values::longdouble_ },

#define LONG_DOUBLE_IS_FLOAT128
#elif defined __GNUC__
#include <quadmath.h>
using float128 = __float128;
#define FLOAT128_SPECIFIER "%Qf"
#define FLOAT128_HELP "[--float128] "
#define FLOAT128_OPTIONS { "f128", 0, NULL, (int)option_values::float128_ },
#else
#define FLOAT128_HELP ""
#define NO_FLOAT128
#define FLOAT128_OPTIONS
#endif

#if !defined LONGDOUBLE_OPTIONS
#error platform implementatin of long double is unsupported.
#endif

#ifndef NO_FLOAT128
// Number of bits in the mantissa (excluding the implicit bit)
#define FLOAT128_MANTISSA_BITS_HIGH ( 112 - 64 )

// Number of bits in the exponent
#define FLOAT128_EXPONENT_BITS 15

// Mask to extract the exponent: (2^15 - 1)
#define FLOAT128_EXPONENT_MASK_HIGH ( ( std::uint64_t( 1 ) << FLOAT128_EXPONENT_BITS ) - 1 )

// Exponent bias: (2^(15-1) - 1) = 16383
#define FLOAT128_EXPONENT_BIAS ( ( std::uint64_t( 1 ) << ( FLOAT128_EXPONENT_BITS - 1 ) ) - 1 )

std::string float128_tostring( float128 f )
{
    char buffer[128];
#ifdef __HAVE_FLOAT128
    snprintf( buffer, sizeof( buffer ), FLOAT128_SPECIFIER, f );
#else
    quadmath_snprintf( buffer, sizeof( buffer ), FLOAT128_SPECIFIER, f );
#endif
    return buffer;
}

void print_float128_representation( float128 f )
{
    static_assert( sizeof( f ) == sizeof( __uint128_t ) );
#ifdef __HAVE_FLOAT128
    static_assert( std::numeric_limits<float128>::is_iec559, "IEEE 754 required" );
#endif

    __uint128_t x;
    std::memcpy( &x, &f, sizeof( f ) );
    uint64_t high = x >> 64;
    uint64_t mantissa_low = std::uint64_t( x );

    std::bitset<64> b_high = high;
    std::bitset<64> b_low = mantissa_low;
    std::string bs_high = b_high.to_string();
    std::uint64_t mantissa_high = high & ( ( std::uint64_t( 1 ) << FLOAT128_MANTISSA_BITS_HIGH ) - 1 );
    std::uint64_t exponent_with_bias = ( high >> FLOAT128_MANTISSA_BITS_HIGH ) & FLOAT128_EXPONENT_MASK_HIGH;
    std::int64_t exponent;

    if ( exponent_with_bias && ( exponent_with_bias != FLOAT128_EXPONENT_MASK_HIGH ) )
    {
        exponent = (std::int64_t)exponent_with_bias - FLOAT128_EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && ( mantissa_high || mantissa_low ) )
    {
        exponent = -( FLOAT128_EXPONENT_BIAS - 1 );    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint64_t sign = high >> ( FLOAT128_EXPONENT_BITS + FLOAT128_MANTISSA_BITS_HIGH );

    auto mstring = bs_high.substr( 1 + FLOAT128_EXPONENT_BITS, FLOAT128_MANTISSA_BITS_HIGH ) + b_low.to_string();
    auto estring = bs_high.substr( 1, FLOAT128_EXPONENT_BITS );

    if ( exponent_with_bias == FLOAT128_EXPONENT_MASK_HIGH )
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:016X}{:016X}\n"
            "bits:     {}{}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:                 {}\n",
            float128_tostring( f ), high, mantissa_low, b_high.to_string(), b_low.to_string(), sign, estring, mstring );
    }
    else
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:016X}{:016X}\n"
            "bits:     {}{}\n"
            "sign:     {}\n"
            "exponent:  {}                                                     "
            "({}{}{}{})\n"
            "mantissa:                 {}\n",
            float128_tostring( f ), high, mantissa_low, b_high.to_string(), b_low.to_string(), sign, estring,
            exponent_with_bias ? FLOAT128_EXPONENT_BIAS : 0, exponent_with_bias ? " " : "", exponent >= 0 ? "+" : "",
            exponent, mstring );
    }

    if ( exponent_with_bias == FLOAT128_EXPONENT_MASK_HIGH )
    {
        if ( mantissa_high == 0 && mantissa_low == 0 )
        {
            std::cout << std::format( "number:       {}\n\n", sign ? "-inf" : "+inf" );
        }
        else
        {
            std::cout << "number:       NaN\n\n";
        }
    }
    else if ( !exponent_with_bias && ( mantissa_high && mantissa_low ) )
    {
        // Denormal: exponent is −16494, no implicit leading 1
        std::cout << std::format( "number:                {}0.{} x 2^({})\n\n", ( sign ? "-" : " " ), mstring,
                                  -( FLOAT128_EXPONENT_BIAS - 1 ) );
    }
    else
    {
        std::cout << std::format( "number:                {}{}.{} x 2^({})\n\n", ( sign ? "-" : " " ), x ? 1 : 0,
                                  mstring, exponent );
    }
}
#endif

// a rough equivalent of std::stoull(str, e, 16)
__uint128_t stou128x( const char* str, char** endptr = nullptr )
{
    // Validate input
    if ( !str || *str == '\0' )
    {
        throw std::invalid_argument( "Empty or null string" );
    }

    // Skip leading whitespace
    const char* start = str;
    while ( std::isspace( *str ) )
    {
        ++str;
    }

    // Handle optional 0x or 0X prefix
    if ( str[0] == '0' && ( str[1] == 'x' || str[1] == 'X' ) )
    {
        str += 2;
    }

    // Count valid hex digits
    const char* digit_start = str;
    size_t digit_count = 0;
    while ( *str && std::isxdigit( *str ) )
    {
        ++digit_count;
        ++str;
    }

    // If no valid digits, throw invalid_argument
    if ( digit_count == 0 )
    {
        throw std::invalid_argument( "No valid hexadecimal digits" );
    }

    // Check if input exceeds 128 bits (32 hex digits)
    if ( digit_count > 32 )
    {
        throw std::out_of_range( "Value exceeds __uint128_t range" );
    }

    // Set endptr if provided
    if ( endptr )
    {
        *endptr = const_cast<char*>( str );
    }

    // Split the string into high and low parts (up to 16 digits each)
    std::string high_str, low_str;
    size_t high_digits = ( digit_count > 16 ) ? digit_count - 16 : 0;
    high_str = std::string( digit_start, high_digits );
    low_str = std::string( digit_start + high_digits, digit_count - high_digits );

    // Parse high and low parts using std::stoull
    uint64_t high = 0, low = 0;
    try
    {
        if ( !high_str.empty() )
        {
            high = std::stoull( high_str, nullptr, 16 );
        }
        if ( !low_str.empty() )
        {
            low = std::stoull( low_str, nullptr, 16 );
        }
    }
    catch ( const std::invalid_argument& )
    {
        throw std::invalid_argument( "Invalid hexadecimal string" );
    }
    catch ( const std::out_of_range& )
    {
        throw std::out_of_range( "Value exceeds uint64_t range in high or low part" );
    }

    // Combine into __uint128_t
    return ( static_cast<__uint128_t>( high ) << 64 ) | static_cast<__uint128_t>( low );
}

void printHelpAndExit()
{
    std::cout << "floatexplorer [--float] [--double] " FLOAT128_HELP
                 "[--special] number [number]*\n\n"
                 "Examples:\n"
                 "floatexplorer 1 -2 6 1.5 0.125 -inf # --float is the default\n"
                 "floatexplorer --double 1 -2 6 1.5 0.125 -inf\n"
                 "floatexplorer --float --double 1 # both representations\n";

    std::exit( 0 );
}

int main( int argc, char** argv )
{
    int c;
    bool dofloat32{};
    bool dofloat64{};
    bool dofloat80{};
    bool dofloat128{};
    bool specialcases{};
    const struct option long_options[] = {
        { "help", 0, NULL, (int)option_values::help_ },
        { "float", 0, NULL, (int)option_values::float_ },
        { "double", 0, NULL, (int)option_values::float64_ },
        { "f32", 0, NULL, (int)option_values::float32_ },
        { "f64", 0, NULL, (int)option_values::float64_ },
        FLOAT128_OPTIONS LONGDOUBLE_OPTIONS
        { "special", 0, NULL, (int)option_values::special_ },
        { NULL, 0, NULL, 0 } };

    while ( -1 != ( c = getopt_long( argc, argv, "hfds", long_options, NULL ) ) )
    {
        switch ( option_values(c) )
        {
            case option_values::float_:
            {
                dofloat32 = true;
                break;
            }
            case option_values::double_:
            {
                dofloat64 = true;
                break;
            }
            case option_values::float80_:
            {
                dofloat80 = true;
                break;
            }
            case option_values::float128_:
            {
                dofloat128 = true;
                break;
            }
            case option_values::special_:
            {
                specialcases = true;
                break;
            }
            case option_values::help_:
            default:
            {
                printHelpAndExit();
            }
        }
    }

    if ( !dofloat32 && !dofloat64 && !dofloat128 )
    {
        dofloat32 = true;
    }

    if ( specialcases )
    {
        if ( dofloat32 )
        {
            float32 tests[] = { 0.0f,
                                std::numeric_limits<float32>::infinity(),
                                -std::numeric_limits<float32>::infinity(),
                                std::numeric_limits<float32>::quiet_NaN(),
                                1.17549435e-38f,    // Smallest normal
                                3.4028235e38f };    // Largest normal

            for ( auto test : tests )
            {
                std::cout << "\nTest value: " << test << "\n";
                print_float32_representation( test );
            }

            float32 f;
            // Test denormals
            std::cout << "\nSmallest denormal:\n";
            std::uint32_t denormal_bits = 0x00000001;
            std::memcpy( &f, &denormal_bits, sizeof( float32 ) );
            print_float32_representation( f );

            std::cout << "\nLargest denormal:\n";
            denormal_bits = ( std::uint32_t( 1 ) << FLOAT32_MANTISSA_BITS ) - 1;
            std::memcpy( &f, &denormal_bits, sizeof( float32 ) );
            print_float32_representation( f );
        }

        if ( dofloat64 )
        {
            float64 tests[] = {
                0.0,
                std::numeric_limits<float64>::infinity(),
                -std::numeric_limits<float64>::infinity(),
                std::numeric_limits<float64>::quiet_NaN(),
                2.2250738585072014e-308,    // Smallest normal double
                1.7976931348623157e308      // Largest normal double
            };

            for ( auto test : tests )
            {
                std::cout << "\nTest value: " << test << "\n";
                print_float64_representation( test );
            }

            float64 f;
            // Test denormals
            std::cout << "\nSmallest denormal:\n";
            std::uint64_t denormal_bits = 0x00000001;
            std::memcpy( &f, &denormal_bits, sizeof( float64 ) );
            print_float64_representation( f );

            std::cout << "\nLargest denormal:\n";
            denormal_bits = ( std::uint64_t( 1 ) << FLOAT64_MANTISSA_BITS ) - 1;
            std::memcpy( &f, &denormal_bits, sizeof( float64 ) );
            print_float64_representation( f );
        }

#if defined LONG_DOUBLE_IS_FLOAT80
        if ( dofloat80 )
        {
            float80 tests[] = {
                0.0,
                std::numeric_limits<float80>::infinity(),
                -std::numeric_limits<float80>::infinity(),
                std::numeric_limits<float80>::quiet_NaN(),
                0x1.0p-16382L, // Smallest normal float128
                0x1.fffffffffffffffp+16383L // Largest normal float128
            };

            for ( auto test : tests )
            {
                std::cout << "\nTest value: " << test << "\n";
                print_float80_representation( test );
            }

            float80 f;
            // Test denormals
            std::cout << "\nSmallest denormal:\n";
            std::uint80_t denormal_bits = 0x00000001;
            std::memcpy( &f, &denormal_bits, sizeof( float80 ) );
            print_float80_representation( f );

            std::cout << "\nLargest denormal:\n";
            denormal_bits = ( std::uint80_t( 1 ) << FLOAT80_MANTISSA_BITS ) - 1;
            std::memcpy( &f, &denormal_bits, sizeof( float80 ) );
            print_float80_representation( f );
        }
#endif

#ifndef NO_FLOAT128
        if ( dofloat128 )
        {
            float128 tests[] = {
                float128( 0.0 ),
#ifdef __HAVE_FLOAT128
                std::numeric_limits<float128>::infinity(),
                -std::numeric_limits<float128>::infinity(),
                std::numeric_limits<float128>::quiet_NaN(),
                0x1.0p-16382L,                              // Smallest normal float128
                0x1.ffffffffffffffffffffffffffffp+16383L    // Largest normal float128
#else
                HUGE_VALQ,
                -HUGE_VALQ,
                nanq( "" ),
                0x1.0p-16382Q,                              // Smallest normal float128
                0x1.ffffffffffffffffffffffffffffp+16383Q    // Largest normal float128
#endif
            };

            for ( auto test : tests )
            {
                std::cout << std::format( "\nTest value: {}\n", float128_tostring( test ) );

                print_float128_representation( test );
            }

            float128 f;
            // Test denormals
            std::cout << "\nSmallest denormal:\n";
            __uint128_t denormal_bits = 0x00000001;
            std::memcpy( &f, &denormal_bits, sizeof( float128 ) );
            print_float128_representation( f );

            std::cout << "\nLargest denormal:\n";
            denormal_bits = ( std::uint64_t( 1 ) << FLOAT128_MANTISSA_BITS_HIGH ) - 1;
            denormal_bits = ( denormal_bits << 64 ) | std::uint64_t( -1 );
            std::memcpy( &f, &denormal_bits, sizeof( float128 ) );
            print_float128_representation( f );
        }
#endif
    }
    else if ( argc == optind )
    {
        printHelpAndExit();
    }

    for ( int i = optind; i < argc; i++ )
    {
        try
        {
            if ( dofloat32 )
            {
                float32 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    uint32_t u32;
                    unsigned long t;
                    static_assert( sizeof( t ) >= sizeof( u32 ) );
                    t = std::stoul( argv[i], nullptr, 16 );
                    if ( t <= UINT_MAX )
                    {
                        u32 = t;
                        memcpy( &f, &u32, sizeof( u32 ) );
                    }
                    else
                    {
                        std::cerr << std::format( "Hex Input {} exceeds UINT_MAX, incompatible with --float.\n", t );

                        throw std::runtime_error( "Bad input." );
                    }
                }
                else
                {
                    f = std::stof( argv[i] );
                }

                print_float32_representation( f );
            }

            if ( dofloat64 )
            {
                float64 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    uint64_t u64;
                    static_assert( sizeof( long long ) == sizeof( u64 ) );
                    u64 = std::stoull( argv[i], nullptr, 16 );
                    memcpy( &f, &u64, sizeof( u64 ) );
                }
                else
                {
                    f = std::stod( argv[i] );
                }

                print_float64_representation( f );
            }

#ifdef LONG_DOUBLE_IS_FLOAT80
            if ( dofloat80 )
            {
                float80 f{};
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    __uint128_t u128 = stou128x( argv[i] );
                    memset( &f, 0, sizeof(f) );
                    memcpy( &f, &u80, 10 );
                }
                else
                {
                    f = std::stold( argv[i] );
                }

                print_float80_representation( f );
            }
#endif

#ifndef NO_FLOAT128
            if ( dofloat128 )
            {
                float128 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    __uint128_t u128 = stou128x( argv[i] );
                    memcpy( &f, &u128, sizeof( u128 ) );
                }
                else
                {
                    f = std::stold( argv[i] );
                }

                print_float128_representation( f );
            }
#endif
        }
        catch ( std::exception& e )
        {
            std::cerr << std::format( "Failed to convert input '{}' to floating point. error: {}\n", argv[i],
                                      e.what() );
        }
    }

    return 0;
}

// vim: et ts=4 sw=4
