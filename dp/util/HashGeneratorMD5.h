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


#pragma once
/** \file */

#include <dp/util/Config.h>
#include <dp/util/HashGenerator.h>
#include <dp/util/DPAssert.h>

#include <string>

namespace dp
{
  namespace util
  {
    /*! \brief HashGenerator class implementing the MD5 algorithm. */
    class HashGeneratorMD5 : public dp::util::HashGenerator
    {
      public:
        DP_UTIL_API HashGeneratorMD5();
     
        // import non-virtual update signature
        using dp::util::HashGenerator::update;

        // update the hash with the input
        DP_UTIL_API void update( const unsigned char * input, unsigned int byteCount );

        // get the size of the hash value
        DP_UTIL_API virtual unsigned int getSizeOfHash() const;

        // do the final hash calculation and get the hash value
        DP_UTIL_API virtual void finalize( void * hash );

        // do the final hash calculation and get the hash value as a string
        DP_UTIL_API virtual std::string finalize();

      private:
        void doFinalize();
        void init();
        inline void step1( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                         , unsigned int x, unsigned int s, unsigned int t );
        inline void step2( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                         , unsigned int x, unsigned int s, unsigned int t );
        inline void step3( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                         , unsigned int x, unsigned int s, unsigned int t );
        inline void step4( unsigned int & a, unsigned int b, unsigned int c, unsigned int d
                         , unsigned int x, unsigned int s, unsigned int t );
        void transform( const unsigned char block[64] );
     
      private:
        unsigned char m_inputBuffer[64];  // input buffer
        unsigned int m_bitCount[2];      // number of bits, modulo 2^64 (lsb first)
        unsigned int m_hash[4];          // MD5 hash
    };
  } // namespace util
} // namespace dp

