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

#if defined HAVE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

// Type        Exponent-Size   Exponent-Bias Format    (Sign.Exponent.Mantissa)
// FP8 E4M3    4 bits          7                       1.4.3
// FP8 E5M2    5 bits          15                      1.5.2
// BF16        8 bits          127                     1.8.7
// FP16        5 bits          15                      1.5.10
union float_e4m3
{
    using UNSIGNED_TYPE = std::uint8_t;
    using SIGNED_TYPE = std::int8_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 4;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 3;

    UNSIGNED_TYPE u;
#if defined HAVE_CUDA
    __nv_fp8_storage_t s;
#endif

    std::string tostring() const
    {
#if defined HAVE_CUDA
        __half half = __nv_cvt_fp8_to_halfraw( s, __NV_E4M3 );
        float output = __half2float( half );

        return std::format( "{}", output );
#else
        return "<cvt-unsupported>";
#endif
    }

    std::string tohex() const
    {
        return std::format( "{:02X}", u );
    }

    std::string espace() const
    {
        return "";
    }

    std::string espace2() const
    {
        return "";
    }
};

union float_e5m2
{
    using UNSIGNED_TYPE = std::uint8_t;
    using SIGNED_TYPE = std::int8_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 5;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 2;

    UNSIGNED_TYPE u;

#if defined HAVE_CUDA
    __nv_fp8_storage_t s;
#endif

    std::string tostring() const
    {
#if defined HAVE_CUDA
        __half half = __nv_cvt_fp8_to_halfraw( s, __NV_E5M2 );
        float output = __half2float( half );

        return std::format( "{}", output );
#else
        return "<cvt-unsupported>";
#endif
    }

    std::string tohex() const
    {
        return std::format( "{:02X}", u );
    }

    std::string espace() const
    {
        return " ";
    }

    std::string espace2() const
    {
        return "";
    }
};

union float_bf16
{
    using UNSIGNED_TYPE = std::uint16_t;
    using SIGNED_TYPE = std::int16_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 8;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 7;

    UNSIGNED_TYPE u;

#if defined HAVE_CUDA
    __nv_bfloat16 s;
#endif

    std::string tostring() const
    {
#if defined HAVE_CUDA
        float output = __bfloat162float( s );

        return std::format( "{}", output );
#else
        return "<cvt-unsupported>";
#endif
    }

    std::string tohex() const
    {
        return std::format( "{:04X}", u );
    }

    std::string espace() const
    {
        return "    ";
    }

    std::string espace2() const
    {
        return "";
    }
};

union float_fp16
{
    using UNSIGNED_TYPE = std::uint16_t;
    using SIGNED_TYPE = std::int16_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 5;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 10;

    UNSIGNED_TYPE u;

#if defined HAVE_CUDA
    __half s;
#endif

    std::string tostring() const
    {
#if defined HAVE_CUDA
        float output = __half2float( s );

        return std::format( "{}", output );
#else
        return "<cvt-unsupported>";
#endif
    }

    std::string tohex() const
    {
        return std::format( "{:04X}", u );
    }

    std::string espace() const
    {
        return " ";
    }

    std::string espace2() const
    {
        return "";
    }
};

union float_ieee32
{
    using UNSIGNED_TYPE = std::uint32_t;
    using SIGNED_TYPE = std::int32_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 8;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 23;

    UNSIGNED_TYPE u;

    float s;

    std::string tostring() const
    {
        return std::format( "{}", s );
    }

    std::string tohex() const
    {
        return std::format( "{:08X}", u );
    }

    std::string espace() const
    {
        return "    ";
    }

    std::string espace2() const
    {
        return "";
    }
};

union float_ieee64
{
    using UNSIGNED_TYPE = std::uint64_t;
    using SIGNED_TYPE = std::int64_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 11;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 52;

    UNSIGNED_TYPE u;
    double s;

    std::string tostring() const
    {
        return std::format( "{}", s );
    }

    std::string tohex() const
    {
        return std::format( "{:016X}", u );
    }

    std::string espace() const
    {
        return "       ";
    }

    std::string espace2() const
    {
        return "                             ";
    }
};

template <class T>
void print_float_representation( T f )
{
    static constexpr typename T::UNSIGNED_TYPE EXPONENT_MASK =
        ( ( typename T::UNSIGNED_TYPE( 1 ) << T::EXPONENT_BITS ) - 1 );
    static constexpr
        typename T::UNSIGNED_TYPE EXPONENT_BIAS( ( typename T::UNSIGNED_TYPE( 1 ) << ( T::EXPONENT_BITS - 1 ) ) - 1 );

    std::bitset<8 * sizeof( T )> b = f.u;
    std::string bs = b.to_string();
    typename T::UNSIGNED_TYPE mantissa = f.u & ( ( typename T::UNSIGNED_TYPE( 1 ) << T::MANTISSA_BITS ) - 1 );
    typename T::UNSIGNED_TYPE exponent_with_bias = ( f.u >> T::MANTISSA_BITS ) & EXPONENT_MASK;
    typename T::SIGNED_TYPE exponent;

    if ( exponent_with_bias && exponent_with_bias != EXPONENT_MASK )
    {
        exponent = (typename T::SIGNED_TYPE)exponent_with_bias - EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -( EXPONENT_BIAS - 1 );    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    typename T::UNSIGNED_TYPE sign = f.u >> ( T::EXPONENT_BITS + T::MANTISSA_BITS );

    auto mstring = bs.substr( 1 + T::EXPONENT_BITS, T::MANTISSA_BITS );
    auto estring = bs.substr( 1, T::EXPONENT_BITS );

    std::string fs = f.tostring();
    std::string es = f.espace();

    if ( exponent_with_bias == EXPONENT_MASK )
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:      {}{}\n",
            fs, f.tohex(), b.to_string(), sign, estring, es, mstring );
    }
    else
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}                        {}({}{}{}{})\n"
            "mantissa:      {}{}\n",
            fs, f.tohex(), b.to_string(), sign, estring, f.espace2(), exponent_with_bias ? EXPONENT_BIAS : 0,
            exponent_with_bias ? " " : "", exponent >= 0 ? "+" : "", exponent, es, mstring );
    }

    if ( exponent_with_bias == EXPONENT_MASK )
    {
        if ( mantissa == 0 )
        {
            std::cout << std::format( "number:    {}\n\n", sign ? "-inf" : "+inf" );
        }
        else
        {
            std::cout << "number:  NaN\n\n";
        }
    }
    else if ( !exponent_with_bias && mantissa )
    {
        // Denormal: exponent is -126, no implicit leading 1
        std::cout << std::format( "number:     {}{}0.{} x 2^({})\n\n", es, ( sign ? "-" : " " ), mstring,
                                  -( EXPONENT_BIAS - 1 ) );
    }
    else
    {
        std::cout << std::format( "number:     {}{}{}.{} x 2^({})\n\n", es, ( sign ? "-" : " " ), f.u ? 1 : 0, mstring,
                                  exponent );
    }
}

void print_float_e4m3_representation( float_e4m3 f )
{
    print_float_representation<float_e4m3>( f );
}

void print_float_e5m2_representation( float_e5m2 f )
{
    print_float_representation<float_e5m2>( f );
}

void print_float_bf16_representation( float_bf16 f )
{
    print_float_representation<float_bf16>( f );
}

void print_float_fp16_representation( float_fp16 f )
{
    print_float_representation<float_fp16>( f );
}

using float32 = float;

void print_float32_representation( float32 f )
{
    static_assert( std::numeric_limits<float32>::is_iec559, "IEEE 754 required" );
    float_ieee32 x;
    x.s = f;

    print_float_representation<float_ieee32>( x );
}

using float64 = double;

void print_float64_representation( float64 f )
{
    static_assert( sizeof( f ) == sizeof( std::uint64_t ) );
    static_assert( std::numeric_limits<float64>::is_iec559, "IEEE 754 required" );

    float_ieee64 x;
    x.s = f;

    print_float_representation<float_ieee64>( x );
}

enum class option_values : int
{
    e5m2_ = '2',
    e4m3_ = '3',
    fp16_ = 'f',
    bf16_ = 'b',
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
#define FLOAT80_HELP "[--f80 | --longdouble] "
#define FLOAT80_OPTIONS { "f80", 0, NULL, (int)option_values::float80_ },
#define LONGDOUBLE_OPTIONS { "longdouble", 0, NULL, (int)option_values::longdouble_ },

#define LONG_DOUBLE_IS_FLOAT80
#else
#define FLOAT80_HELP ""
#define FLOAT80_OPTIONS
#endif

// This is not portable.  There are surely other platforms where long double is an ieee 128 bit floating point:
#if defined __aarch64__ && defined __linux__
using float128 = long double;
#define FLOAT128_SPECIFIER "%La"
#define FLOAT128_HELP "[--f128 | --longdouble] "
#define FLOAT128_OPTIONS { "f128", 0, NULL, 'l' },
#define LONGDOUBLE_OPTIONS { "longdouble", 0, NULL, (int)option_values::longdouble_ },

#define LONG_DOUBLE_IS_FLOAT128
#elif defined __APPLE__
#define LONG_DOUBLE_IS_FLOAT64
#define LONGDOUBLE_OPTIONS { "longdouble", 0, NULL, (int)option_values::double_ },
#endif

#if !defined LONG_DOUBLE_IS_FLOAT128
#if defined __GNUC__ && defined USE_QUADMATH
#include <quadmath.h>
using float128 = __float128;
#define FLOAT128_SPECIFIER "%Qf"
#define FLOAT128_HELP "[--f128] "
#define FLOAT128_OPTIONS { "f128", 0, NULL, (int)option_values::float128_ },
#else
#define FLOAT128_HELP ""
#define NO_FLOAT128
#define FLOAT128_OPTIONS
#endif
#endif

#if !defined LONGDOUBLE_OPTIONS
#error Implementation of long double on this platform is not unsupported.
#endif

#if defined LONG_DOUBLE_IS_FLOAT80
#define FLOAT80_MANTISSA_BITS 64

// Number of bits in the exponent
#define FLOAT80_EXPONENT_BITS 15

// Mask to extract the exponent: (2^15 - 1)
#define FLOAT80_EXPONENT_MASK_HIGH ( ( std::uint64_t( 1 ) << FLOAT80_EXPONENT_BITS ) - 1 )

// Exponent bias: (2^(15-1) - 1) = 16383
#define FLOAT80_EXPONENT_BIAS ( ( std::uint64_t( 1 ) << ( FLOAT80_EXPONENT_BITS - 1 ) ) - 1 )

void print_float80_representation( float80 f )
{
    static_assert( sizeof( f ) == sizeof( __uint128_t ) );
    // #if defined __HAVE_FLOAT80
    //     static_assert( std::numeric_limits<float80>::is_iec559, "IEEE 754 required" );
    // #endif

    __uint128_t x;
    std::memset( &x, 0, sizeof( f ) );
    std::memcpy( &x, &f, 10 );
    std::uint64_t high = x >> 64;
    std::uint64_t mantissa = std::uint64_t( x );

    std::bitset<16> high_bits = high & 0xFFFF;
    std::bitset<64> low_bits = mantissa;
    std::string bs = high_bits.to_string() + low_bits.to_string();

    std::uint64_t exponent_with_bias = high & FLOAT80_EXPONENT_MASK_HIGH;
    std::int64_t exponent;

    if ( exponent_with_bias && ( exponent_with_bias != FLOAT80_EXPONENT_MASK_HIGH ) )
    {
        exponent = (std::int64_t)exponent_with_bias - FLOAT80_EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa )
    {
        exponent = -( FLOAT80_EXPONENT_BIAS - 1 );    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    std::uint64_t sign = high >> FLOAT80_EXPONENT_BITS;

    auto mstring = bs.substr( 80 - FLOAT80_MANTISSA_BITS, FLOAT80_MANTISSA_BITS );
    auto estring = bs.substr( 1, FLOAT80_EXPONENT_BITS );

    if ( exponent_with_bias == FLOAT80_EXPONENT_MASK_HIGH )
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:04X}{:016X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}\n"
            "mantissa:                 {}\n",
            f, high, mantissa, bs, sign, estring, mstring );
    }
    else
    {
        std::cout << std::format(
            "value:    {}\n"
            "hex:      {:04X}{:016X}\n"
            "bits:     {}\n"
            "sign:     {}\n"
            "exponent:  {}                                                     "
            "({}{}{}{})\n"
            "mantissa:                 {}\n",
            f, high, mantissa, bs, sign, estring, exponent_with_bias ? FLOAT80_EXPONENT_BIAS : 0,
            exponent_with_bias ? " " : "", exponent >= 0 ? "+" : "", exponent, mstring );
    }

    if ( exponent_with_bias == FLOAT80_EXPONENT_MASK_HIGH )
    {
        if ( mantissa == 0 )
        {
            std::cout << std::format( "number:       {}\n\n", sign ? "-inf" : "+inf" );
        }
        else
        {
            std::cout << "number:       NaN\n\n";
        }
    }
    else if ( !exponent_with_bias && mantissa )
    {
        // Denormal: exponent is −16494, no implicit leading 1
        std::cout << std::format( "number:                {}0.{} x 2^({})\n\n", ( sign ? "-" : " " ), mstring,
                                  -( FLOAT80_EXPONENT_BIAS - 1 ) );
    }
    else
    {
        bool isnormal = exponent_with_bias && exponent_with_bias != FLOAT80_EXPONENT_MASK_HIGH;

        std::cout << std::format( "number:                {}0.{} x 2^({})\n\n", ( sign ? "-" : " " ), mstring,
                                  isnormal ? exponent + 1 : exponent );
    }
}
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
#if defined USE_QUADMATH
    quadmath_snprintf( buffer, sizeof( buffer ), FLOAT128_SPECIFIER, f );
#elif defined LONG_DOUBLE_IS_FLOAT128
    snprintf( buffer, sizeof( buffer ), FLOAT128_SPECIFIER, f );
#else
#error conversion from float128 to string is not supported.
#endif
    return buffer;
}

void print_float128_representation( float128 f )
{
    static_assert( sizeof( f ) == sizeof( __uint128_t ) );
#if defined LONG_DOUBLE_IS_FLOAT128
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
    std::cout << "floatexplorer [--e5m2] [--e4m3] [--float] [--double] " FLOAT80_HELP FLOAT128_HELP
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
    bool dofloat_e5m2{};
    bool dofloat_e4m3{};
    bool dofloat_bf16{};
    bool dofloat_fp16{};
    bool dofloat32{};
    bool dofloat64{};
    bool dofloat80{};
    bool dofloat128{};
    bool specialcases{};
    const struct option long_options[] = {
        { "help", 0, NULL, (int)option_values::help_ },
        { "e5m2", 0, NULL, (int)option_values::e5m2_ },
        { "e4m3", 0, NULL, (int)option_values::e4m3_ },
        { "bf16", 0, NULL, (int)option_values::bf16_ },
        { "fp16", 0, NULL, (int)option_values::fp16_ },
        { "float", 0, NULL, (int)option_values::float_ },
        { "double", 0, NULL, (int)option_values::float64_ },
        { "f32", 0, NULL, (int)option_values::float32_ },
        { "f64", 0, NULL, (int)option_values::float64_ },
        FLOAT80_OPTIONS FLOAT128_OPTIONS LONGDOUBLE_OPTIONS{ "special", 0, NULL, (int)option_values::special_ },
        { NULL, 0, NULL, 0 } };

    while ( -1 != ( c = getopt_long( argc, argv, "h", long_options, NULL ) ) )
    {
        switch ( option_values( c ) )
        {
            case option_values::e5m2_:
            {
                dofloat_e5m2 = true;
                break;
            }
            case option_values::e4m3_:
            {
                dofloat_e4m3 = true;
                break;
            }
            case option_values::bf16_:
            {
                dofloat_bf16 = true;
                break;
            }
            case option_values::fp16_:
            {
                dofloat_fp16 = true;
                break;
            }
            case option_values::float_:
            {
                dofloat32 = true;
                break;
            }
#if defined LONG_DOUBLE_IS_FLOAT64
            case option_values::longdouble_:
#endif
            case option_values::double_:
            {
                dofloat64 = true;
                break;
            }
#if defined LONG_DOUBLE_IS_FLOAT80
            case option_values::longdouble_:
#endif
            case option_values::float80_:
            {
                dofloat80 = true;
                break;
            }
#if defined LONG_DOUBLE_IS_FLOAT128
            case option_values::longdouble_:
#endif
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

    if ( !dofloat_e4m3 && !dofloat_e5m2 && !dofloat_bf16 && !dofloat_fp16 && !dofloat32 && !dofloat64 && !dofloat80 && !dofloat128 )
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
            denormal_bits = ( std::uint32_t( 1 ) << float_ieee32::MANTISSA_BITS ) - 1;
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
            denormal_bits = ( std::uint64_t( 1 ) << float_ieee64::MANTISSA_BITS ) - 1;
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
                0x1.0p-16382L,                 // Smallest normal float128
                0x1.fffffffffffffffp+16383L    // Largest normal float128
            };

            for ( auto test : tests )
            {
                std::cout << "\nTest value: " << test << "\n";
                print_float80_representation( test );
            }

            float80 f;
            // Test denormals
            std::cout << "\nSmallest denormal:\n";
            __uint128_t denormal_bits = 0x00000001;
            std::memset( &f, 0, sizeof( f ) );
            std::memcpy( &f, &denormal_bits, 10 );
            print_float80_representation( f );

            std::cout << "\nLargest denormal:\n";
            denormal_bits = ( __uint128_t( 1 ) << FLOAT80_MANTISSA_BITS ) - 1;
            std::memset( &f, 0, sizeof( f ) );
            std::memcpy( &f, &denormal_bits, 10 );
            print_float80_representation( f );
        }
#endif

#ifndef NO_FLOAT128
        if ( dofloat128 )
        {
            float128 tests[] = {
                float128( 0.0 ),
#if defined LONG_DOUBLE_IS_FLOAT128
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
            denormal_bits = ( static_cast<__uint128_t>( 1 ) << 112 ) - 1;
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
            if ( dofloat_bf16 )
            {
                float_bf16 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    unsigned long t;
                    t = std::stoul( argv[i], nullptr, 16 );
                    if ( t <= UINT16_MAX )
                    {
                        f.u = t;
                    }
                    else
                    {
                        std::cerr << std::format( "Hex Input {} exceeds UINT16_MAX, incompatible with --bf16.\n", t );

                        throw std::runtime_error( "Bad input." );
                    }
                }
                else
                {
#if defined HAVE_CUDA
                    float tf = std::stof( argv[i] );
                    __nv_bfloat16 bf16 = __float2bfloat16( tf );
                    f.s = bf16;
#else
                    throw std::runtime_error( "Conversion from float to bf16 is unsupported." );
#endif
                }

                print_float_bf16_representation( f );
            }

            if ( dofloat_fp16 )
            {
                float_fp16 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    unsigned long t;
                    t = std::stoul( argv[i], nullptr, 16 );
                    if ( t <= UINT16_MAX )
                    {
                        f.u = t;
                    }
                    else
                    {
                        std::cerr << std::format( "Hex Input {} exceeds UINT16_MAX, incompatible with --fp16.\n", t );

                        throw std::runtime_error( "Bad input." );
                    }
                }
                else
                {
#if defined HAVE_CUDA
                    float tf = std::stof( argv[i] );
                    __half fp16 = __float2half( tf );
                    f.s = fp16;
#else
                    throw std::runtime_error( "Conversion from float to fp16 is unsupported." );
#endif
                }

                print_float_fp16_representation( f );
            }

            if ( dofloat_e4m3 )
            {
                float_e4m3 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    unsigned long t;
                    t = std::stoul( argv[i], nullptr, 16 );
                    if ( t <= UINT8_MAX )
                    {
                        f.u = t;
                    }
                    else
                    {
                        std::cerr << std::format( "Hex Input {} exceeds UINT8_MAX, incompatible with --e4m3.\n", t );

                        throw std::runtime_error( "Bad input." );
                    }
                }
                else
                {
#if defined HAVE_CUDA
                    float tf = std::stof( argv[i] );
                    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8( tf, __NV_SATFINITE, __NV_E4M3 );
                    f.s = fp8;
#else
                    throw std::runtime_error( "Conversion from float to e4m3 is unsupported." );
#endif
                }

                print_float_e4m3_representation( f );
            }

            if ( dofloat_e5m2 )
            {
                float_e5m2 f;
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    unsigned long t;
                    t = std::stoul( argv[i], nullptr, 16 );
                    if ( t <= UINT8_MAX )
                    {
                        f.u = t;
                    }
                    else
                    {
                        std::cerr << std::format( "Hex Input {} exceeds UINT8_MAX, incompatible with --e5m2.\n", t );

                        throw std::runtime_error( "Bad input." );
                    }
                }
                else
                {
#if defined HAVE_CUDA
                    float tf = std::stof( argv[i] );
                    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8( tf, __NV_SATFINITE, __NV_E5M2 );
                    f.s = fp8;
#else
                    throw std::runtime_error( "Conversion from float to e5m2 is unsupported." );
#endif
                }

                print_float_e5m2_representation( f );
            }

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

#if defined LONG_DOUBLE_IS_FLOAT80
            if ( dofloat80 )
            {
                float80 f{};
                if ( strncasecmp( argv[i], "0x", 2 ) == 0 )
                {
                    __uint128_t u128 = stou128x( argv[i] );
                    std::memset( &f, 0, sizeof( f ) );
                    std::memcpy( &f, &u128, 10 );
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
