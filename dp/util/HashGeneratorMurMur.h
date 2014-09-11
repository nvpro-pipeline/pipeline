// Copyright NVIDIA Corporation 2012
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


// Implementation of MurmurHash by Austin Appleby.
#pragma once

#include <dp/util/HashGenerator.h>

namespace dp
{
  namespace util
  {

    class HashGeneratorMurMur : public HashGenerator
    {
      public:
        DP_UTIL_API HashGeneratorMurMur( unsigned int seed = 0);
        DP_UTIL_API virtual ~HashGeneratorMurMur();
     
        // import non-virtual update signature
        using HashGenerator::update;

        // update the hash with the input
        DP_UTIL_API void update( const unsigned char * input, unsigned int byteCount );

        // get the size of the hash value
        DP_UTIL_API virtual unsigned int getSizeOfHash() const;

        // do the final hash calculation and get the hash value
        DP_UTIL_API virtual void finalize( void * hash );

        DP_UTIL_API std::string finalize();
    private:
      DP_UTIL_API void updateTail( const unsigned char * & data, unsigned int & len );

      static const unsigned int m_m = 0x5bd1e995;
      static const int m_r = 24;

      static inline void mmix(unsigned int &h, unsigned int & k)
      {
        k *= m_m;
        k ^= k >> m_r;
        k *= m_m;
        h *= m_m;
        h ^= k;
      }

      unsigned int m_hash;
      unsigned int m_tail;
      unsigned int m_count;
      unsigned int m_size;
      unsigned int m_seed;
    };

  } // namespace util
} // namespace dp
