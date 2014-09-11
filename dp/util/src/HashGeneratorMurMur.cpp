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

#include <dp/util/HashGeneratorMurMur.h>

namespace dp
{
  namespace util
  {

    HashGeneratorMurMur::HashGeneratorMurMur( unsigned int seed )
    : m_seed( seed )
    , m_hash( 0 )
    , m_count( 0 )
    , m_size( 0 )
    , m_tail( 0 )
    {
    };
    
    HashGeneratorMurMur::~HashGeneratorMurMur()
    {
    }

    void HashGeneratorMurMur::updateTail( const unsigned char * & data, unsigned int & len )
    {
      while( len && ((len<4) || m_count) )
      {
        m_tail |= (*data++) << (m_count * 8);

        m_count++;
        len--;

        if(m_count == 4)
        {
          mmix(m_hash,m_tail);
          m_tail = 0;
          m_count = 0;
        }
      }
    }

    void HashGeneratorMurMur::update ( const unsigned char * input, unsigned int byteCount )
    {
      m_size += byteCount;

      updateTail( input, byteCount );

      while( byteCount >= 4 )
      {
        unsigned int k = *(unsigned int*)input;

        mmix(m_hash,k);

        input += 4;
        byteCount -= 4;
      }

      updateTail( input, byteCount );
    }

    void HashGeneratorMurMur::finalize( void* hash )
    {
      mmix(m_hash,m_tail);
      mmix(m_hash,m_size);

      m_hash ^= m_hash >> 13;
      m_hash *= m_m;
      m_hash ^= m_hash >> 15;

      (*reinterpret_cast<unsigned int*>(hash)) = m_hash;

      m_hash = 0;
      m_count = 0;
      m_size = 0;
      m_tail = 0;
    }

    std::string HashGeneratorMurMur::finalize()
    {
      char result[5];
      finalize( result );
      result[4] = 0;
      return result;
    }
    
    unsigned int HashGeneratorMurMur::getSizeOfHash() const
    {
      return 4;
    }

  } // namespace util
} // namespace dp
