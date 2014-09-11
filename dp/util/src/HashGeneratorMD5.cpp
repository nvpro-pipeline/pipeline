// Copyright NVIDIA Corporation 2002-2008
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <dp/util/HashGeneratorMD5.h>
#include <dp/util/DPAssert.h>
#include <string.h>
#include <sstream>

using std::string;

namespace dp
{
  namespace util
  {

    // if sizeof( unsigned int ) is not 4 then this code wont work!!
    // NOTE that the CTASSERT_BYTESIZE macro cant accept 'unsigned int' so we just pass 'int'
    DP_STATIC_ASSERT_BYTESIZE( int, 4 );

    // initialization constants
    #define MD5_INIT_STATE_0 0x67452301
    #define MD5_INIT_STATE_1 0xefcdab89
    #define MD5_INIT_STATE_2 0x98badcfe
    #define MD5_INIT_STATE_3 0x10325476

    // constants for the transformation process
    #define MD5_S11  7
    #define MD5_S12 12
    #define MD5_S13 17
    #define MD5_S14 22
    #define MD5_S21  5
    #define MD5_S22  9
    #define MD5_S23 14
    #define MD5_S24 20
    #define MD5_S31  4
    #define MD5_S32 11
    #define MD5_S33 16
    #define MD5_S34 23
    #define MD5_S41  6
    #define MD5_S42 10
    #define MD5_S43 15
    #define MD5_S44 21

    // transformation constant - part 1
    #define MD5_T01  0xd76aa478 //transformation constant 1 
    #define MD5_T02  0xe8c7b756 //transformation constant 2
    #define MD5_T03  0x242070db //transformation constant 3
    #define MD5_T04  0xc1bdceee //transformation constant 4
    #define MD5_T05  0xf57c0faf //transformation constant 5
    #define MD5_T06  0x4787c62a //transformation constant 6
    #define MD5_T07  0xa8304613 //transformation constant 7
    #define MD5_T08  0xfd469501 //transformation constant 8
    #define MD5_T09  0x698098d8 //transformation constant 9
    #define MD5_T10  0x8b44f7af //transformation constant 10
    #define MD5_T11  0xffff5bb1 //transformation constant 11
    #define MD5_T12  0x895cd7be //transformation constant 12
    #define MD5_T13  0x6b901122 //transformation constant 13
    #define MD5_T14  0xfd987193 //transformation constant 14
    #define MD5_T15  0xa679438e //transformation constant 15
    #define MD5_T16  0x49b40821 //transformation constant 16

    // transformation constant - part 2
    #define MD5_T17  0xf61e2562 //transformation constant 17
    #define MD5_T18  0xc040b340 //transformation constant 18
    #define MD5_T19  0x265e5a51 //transformation constant 19
    #define MD5_T20  0xe9b6c7aa //transformation constant 20
    #define MD5_T21  0xd62f105d //transformation constant 21
    #define MD5_T22  0x02441453 //transformation constant 22
    #define MD5_T23  0xd8a1e681 //transformation constant 23
    #define MD5_T24  0xe7d3fbc8 //transformation constant 24
    #define MD5_T25  0x21e1cde6 //transformation constant 25
    #define MD5_T26  0xc33707d6 //transformation constant 26
    #define MD5_T27  0xf4d50d87 //transformation constant 27
    #define MD5_T28  0x455a14ed //transformation constant 28
    #define MD5_T29  0xa9e3e905 //transformation constant 29
    #define MD5_T30  0xfcefa3f8 //transformation constant 30
    #define MD5_T31  0x676f02d9 //transformation constant 31
    #define MD5_T32  0x8d2a4c8a //transformation constant 32

    // transformation constant - part 3
    #define MD5_T33  0xfffa3942 //transformation constant 33
    #define MD5_T34  0x8771f681 //transformation constant 34
    #define MD5_T35  0x6d9d6122 //transformation constant 35
    #define MD5_T36  0xfde5380c //transformation constant 36
    #define MD5_T37  0xa4beea44 //transformation constant 37
    #define MD5_T38  0x4bdecfa9 //transformation constant 38
    #define MD5_T39  0xf6bb4b60 //transformation constant 39
    #define MD5_T40  0xbebfbc70 //transformation constant 40
    #define MD5_T41  0x289b7ec6 //transformation constant 41
    #define MD5_T42  0xeaa127fa //transformation constant 42
    #define MD5_T43  0xd4ef3085 //transformation constant 43
    #define MD5_T44  0x04881d05 //transformation constant 44
    #define MD5_T45  0xd9d4d039 //transformation constant 45
    #define MD5_T46  0xe6db99e5 //transformation constant 46
    #define MD5_T47  0x1fa27cf8 //transformation constant 47
    #define MD5_T48  0xc4ac5665 //transformation constant 48

    // transformation constant - part 4
    #define MD5_T49  0xf4292244 //transformation constant 49
    #define MD5_T50  0x432aff97 //transformation constant 50
    #define MD5_T51  0xab9423a7 //transformation constant 51
    #define MD5_T52  0xfc93a039 //transformation constant 52
    #define MD5_T53  0x655b59c3 //transformation constant 53
    #define MD5_T54  0x8f0ccc92 //transformation constant 54
    #define MD5_T55  0xffeff47d //transformation constant 55
    #define MD5_T56  0x85845dd1 //transformation constant 56
    #define MD5_T57  0x6fa87e4f //transformation constant 57
    #define MD5_T58  0xfe2ce6e0 //transformation constant 58
    #define MD5_T59  0xa3014314 //transformation constant 59
    #define MD5_T60  0x4e0811a1 //transformation constant 60
    #define MD5_T61  0xf7537e82 //transformation constant 61
    #define MD5_T62  0xbd3af235 //transformation constant 62
    #define MD5_T63  0x2ad7d2bb //transformation constant 63
    #define MD5_T64  0xeb86d391 //transformation constant 64

    // zero data (besides the first uchar) are used to end the hash calculation
    static unsigned char PADDING[64] =
    {
      0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    static inline unsigned int rotateLeft( unsigned int x, int n )
    {
      // shift and return
      return( (x << n) | (x >> (32-n)) );
    }

    HashGeneratorMD5::HashGeneratorMD5()
    {
      init();
    }

    unsigned int HashGeneratorMD5::getSizeOfHash() const
    {
      return( 16 );
    }

    void HashGeneratorMD5::finalize( void * hash )
    {
      DP_ASSERT( hash );
      doFinalize();
      memcpy( hash, &m_hash[0], 16 );
      init();   // re-init for next hash calculation
    }

    string HashGeneratorMD5::finalize()
    {
      doFinalize();

      // get the current hash in bytes
      unsigned char hashBytes[16];
      memcpy( hashBytes, m_hash, 16 );

      // convert the hash to a string
      std::stringstream str;
      for ( int i=0 ; i < 16 ; i++ ) 
      {
        if ( hashBytes[i] == 0 ) 
        {
          str << "00";
        }
        else if ( hashBytes[i] <= 15 )      
        {
          str << "0" << std::hex << static_cast<int>(hashBytes[i]);
        }
        else 
        {
          str << std::hex << static_cast<int>(hashBytes[i]);
        }
        DP_ASSERT( str.str().length() == 2*(i+1) );
      }
      DP_ASSERT( str.str().length() == 32 );

      init();   // re-init for next hash calculation

      return( str.str() );
    }

    void HashGeneratorMD5::doFinalize()
    {
      // get the number of bits as uchars
      unsigned char bits[8];
      memcpy( bits, m_bitCount, 8 );

      // fill with padding to 56 mod 64
      unsigned int index = (unsigned int)((m_bitCount[0] >> 3) & 0x3f);
      unsigned int padLength = (index < 56) ? (56 - index) : (120 - index);
      update( PADDING, padLength );

      // add length
      update( bits, 8 );
    }

    void HashGeneratorMD5::update( const unsigned char * input, unsigned int byteCount )
    {
      // determine number of bytes mod 64
      unsigned int index = (unsigned int)((m_bitCount[0] >> 3) & 0x3F);
     
      // adjust the number of bits
      unsigned int bitCount = byteCount << 3;
      m_bitCount[0] += bitCount;
      if ( m_bitCount[0] < bitCount )   // overflow?
      {
        m_bitCount[1]++;
      }
      m_bitCount[1] += (byteCount >> 29);   // if there are really lots of bytes...

      // transform as often as possible
      unsigned int i = 0;              
      unsigned int partLength = 64 - index;
      if ( partLength <= byteCount )
      {
        memmove( &m_inputBuffer[index], input, partLength );
        
        transform( m_inputBuffer );
        for ( i = partLength ; i + 63 < byteCount ; i += 64 ) 
        {
          transform( &input[i] );
        }
        index = 0;
      } 
      else 
      {
        i = 0;
      }

      // get the rest of the input buffer
      memmove( &m_inputBuffer[index], &input[i], byteCount-i );
    }

    void HashGeneratorMD5::init()
    {
      memset( m_inputBuffer, 0, 64 );
      m_bitCount[0] = m_bitCount[1] = 0;

      // set magic hash initialization constants
      m_hash[0] = MD5_INIT_STATE_0;
      m_hash[1] = MD5_INIT_STATE_1;
      m_hash[2] = MD5_INIT_STATE_2;
      m_hash[3] = MD5_INIT_STATE_3;
    }

    void HashGeneratorMD5::step1( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                           , unsigned int x, unsigned int s, unsigned int t)
    {
      unsigned int f = (b & c) | (~b & d);
      a += f + x + t;
      a = rotateLeft( a, s );
      a += b;
    }

    void HashGeneratorMD5::step2( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                           , unsigned int x, unsigned int s, unsigned int t)
    {
      unsigned int g = (b & d) | (c & ~d);
      a += g + x + t;
      a = rotateLeft( a, s );
      a += b;
    }

    void HashGeneratorMD5::step3( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                           , unsigned int x, unsigned int s, unsigned int t)
    {
      unsigned int h = (b ^ c ^ d);
      a += h + x + t;
      a = rotateLeft( a, s );
      a += b;
    }

    void HashGeneratorMD5::step4( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                           , unsigned int x, unsigned int s, unsigned int t)
    {
      unsigned int i = (c ^ (b | ~d));
      a += i + x + t;
      a = rotateLeft( a, s );
      a += b;
    }

    void HashGeneratorMD5::transform( const unsigned char block[64] )
    {
      // initialize the local data with the current hash
      unsigned int a = m_hash[0];
      unsigned int b = m_hash[1];
      unsigned int c = m_hash[2];
      unsigned int d = m_hash[3];

      // get the bytes into ulongs
      const unsigned int *x = reinterpret_cast<const unsigned int *>(block);

      // step 1 transformations
      step1( a, b, c, d, x[ 0], MD5_S11, MD5_T01 ); 
      step1( d, a, b, c, x[ 1], MD5_S12, MD5_T02 ); 
      step1( c, d, a, b, x[ 2], MD5_S13, MD5_T03 ); 
      step1( b, c, d, a, x[ 3], MD5_S14, MD5_T04 ); 
      step1( a, b, c, d, x[ 4], MD5_S11, MD5_T05 ); 
      step1( d, a, b, c, x[ 5], MD5_S12, MD5_T06 ); 
      step1( c, d, a, b, x[ 6], MD5_S13, MD5_T07 ); 
      step1( b, c, d, a, x[ 7], MD5_S14, MD5_T08 ); 
      step1( a, b, c, d, x[ 8], MD5_S11, MD5_T09 ); 
      step1( d, a, b, c, x[ 9], MD5_S12, MD5_T10 ); 
      step1( c, d, a, b, x[10], MD5_S13, MD5_T11 ); 
      step1( b, c, d, a, x[11], MD5_S14, MD5_T12 ); 
      step1( a, b, c, d, x[12], MD5_S11, MD5_T13 ); 
      step1( d, a, b, c, x[13], MD5_S12, MD5_T14 ); 
      step1( c, d, a, b, x[14], MD5_S13, MD5_T15 ); 
      step1( b, c, d, a, x[15], MD5_S14, MD5_T16 ); 

      // step 2 transformations
      step2( a, b, c, d, x[ 1], MD5_S21, MD5_T17 ); 
      step2( d, a, b, c, x[ 6], MD5_S22, MD5_T18 ); 
      step2( c, d, a, b, x[11], MD5_S23, MD5_T19 ); 
      step2( b, c, d, a, x[ 0], MD5_S24, MD5_T20 ); 
      step2( a, b, c, d, x[ 5], MD5_S21, MD5_T21 ); 
      step2( d, a, b, c, x[10], MD5_S22, MD5_T22 ); 
      step2( c, d, a, b, x[15], MD5_S23, MD5_T23 ); 
      step2( b, c, d, a, x[ 4], MD5_S24, MD5_T24 ); 
      step2( a, b, c, d, x[ 9], MD5_S21, MD5_T25 ); 
      step2( d, a, b, c, x[14], MD5_S22, MD5_T26 ); 
      step2( c, d, a, b, x[ 3], MD5_S23, MD5_T27 ); 
      step2( b, c, d, a, x[ 8], MD5_S24, MD5_T28 ); 
      step2( a, b, c, d, x[13], MD5_S21, MD5_T29 ); 
      step2( d, a, b, c, x[ 2], MD5_S22, MD5_T30 ); 
      step2( c, d, a, b, x[ 7], MD5_S23, MD5_T31 ); 
      step2( b, c, d, a, x[12], MD5_S24, MD5_T32 ); 

      // step 3 transformations
      step3( a, b, c, d, x[ 5], MD5_S31, MD5_T33 ); 
      step3( d, a, b, c, x[ 8], MD5_S32, MD5_T34 ); 
      step3( c, d, a, b, x[11], MD5_S33, MD5_T35 ); 
      step3( b, c, d, a, x[14], MD5_S34, MD5_T36 ); 
      step3( a, b, c, d, x[ 1], MD5_S31, MD5_T37 ); 
      step3( d, a, b, c, x[ 4], MD5_S32, MD5_T38 ); 
      step3( c, d, a, b, x[ 7], MD5_S33, MD5_T39 ); 
      step3( b, c, d, a, x[10], MD5_S34, MD5_T40 ); 
      step3( a, b, c, d, x[13], MD5_S31, MD5_T41 ); 
      step3( d, a, b, c, x[ 0], MD5_S32, MD5_T42 ); 
      step3( c, d, a, b, x[ 3], MD5_S33, MD5_T43 ); 
      step3( b, c, d, a, x[ 6], MD5_S34, MD5_T44 ); 
      step3( a, b, c, d, x[ 9], MD5_S31, MD5_T45 ); 
      step3( d, a, b, c, x[12], MD5_S32, MD5_T46 ); 
      step3( c, d, a, b, x[15], MD5_S33, MD5_T47 ); 
      step3( b, c, d, a, x[ 2], MD5_S34, MD5_T48 ); 

      // step 4 transformations
      step4( a, b, c, d, x[ 0], MD5_S41, MD5_T49 ); 
      step4( d, a, b, c, x[ 7], MD5_S42, MD5_T50 ); 
      step4( c, d, a, b, x[14], MD5_S43, MD5_T51 ); 
      step4( b, c, d, a, x[ 5], MD5_S44, MD5_T52 ); 
      step4( a, b, c, d, x[12], MD5_S41, MD5_T53 ); 
      step4( d, a, b, c, x[ 3], MD5_S42, MD5_T54 ); 
      step4( c, d, a, b, x[10], MD5_S43, MD5_T55 ); 
      step4( b, c, d, a, x[ 1], MD5_S44, MD5_T56 ); 
      step4( a, b, c, d, x[ 8], MD5_S41, MD5_T57 ); 
      step4( d, a, b, c, x[15], MD5_S42, MD5_T58 ); 
      step4( c, d, a, b, x[ 6], MD5_S43, MD5_T59 ); 
      step4( b, c, d, a, x[13], MD5_S44, MD5_T60 ); 
      step4( a, b, c, d, x[ 4], MD5_S41, MD5_T61 ); 
      step4( d, a, b, c, x[11], MD5_S42, MD5_T62 ); 
      step4( c, d, a, b, x[ 2], MD5_S43, MD5_T63 ); 
      step4( b, c, d, a, x[ 9], MD5_S44, MD5_T64 ); 

      // add the changed values to the current hash
      m_hash[0] += a;
      m_hash[1] += b;
      m_hash[2] += c;
      m_hash[3] += d;
    }
  } // namespace util
} // namespace dp
