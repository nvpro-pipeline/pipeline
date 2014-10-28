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


#pragma once
/** \file */

#include <dp/util/Config.h>
#include <dp/util/SharedPtr.h>
#include <string>

namespace dp
{
  namespace util
  {

    typedef unsigned int HashKey;

    /*! \brief HashGenerator is the interface defining class for creating a hash out of some arbitrary data.
     *  \par Namespace: dp::util */
    class HashGenerator
    {
      public:
        /*! \brief Add data to the hash generator.
         *  \param input A pointer to the constant data to use hash.
         *  \param byteCount The number of bytes to process at \a input.
         *  \remarks To generate a hash, this function can be called an arbitrary amount of times, each time
         *  adding data to the hashing process. After all data to hash is added, a finalize determines the
         *  hash of all that data.
         *  \sa finalize */
        DP_UTIL_API virtual void update( const unsigned char * input, unsigned int byteCount ) = 0;

        /*! \brief Add data to the hash generator.
         *  \param input A pointer to the constant data to use hash.
         *  \param elementSize size of one element in the input array
         *  \param stride stride in bytes between two elements in the input array
         *  \param elementCount The number of elements to process at \a input.
         *  \remarks To generate a hash, this function can be called an arbitrary amount of times, each time
         *  adding data to the hashing process. After all data to hash is added, a finalize determines the
         *  hash of all that data.
         *  \sa finalize */
        DP_UTIL_API void update( const unsigned char * input, unsigned int elementSize, unsigned int stride, unsigned int elementCount );

        template <typename T> void update( SharedPtr<T> const& ptr );

        /*! \brief Get the size of the hash
         *  \return The size of the hash.
         *  \remarks A HashGenerator generates a hash of a specific size. This size should be independent of the
         *  data to hash. The function finalize( void * ) assumes, that \a hash points to some memory of the
         *  the required size, to get a copy of the hash.
         *  \sa update, finalize */
        DP_UTIL_API virtual unsigned int getSizeOfHash() const = 0;

        /*! \brief Finalize the hash of the data added via update.
         *  \param hash A pointer to some memory to get the hash.
         *  \remarks To generate a hash, the \c update can be called an arbitrary amount of times, each time adding
         *  data to the hashing process. After all data to hash is added, this function determines the hash value of
         *  all that data. That value is copied to \a hash, while it is assumed, that there is at least as much
         *  memory reserved at that position as the \c getSizeOfHash tells.
         *  \sa getSizeOfHash, update */
        DP_UTIL_API virtual void finalize( void * hash ) = 0;

        /*! \brief Finalize the hash of the data added via update, and get the hash as a string.
         *  \return The hash value converted to a string.
         *  \remarks To generate a hash, the \c update can be called an arbitrary amount of times, each time adding
         *  data to the hashing process. After all data to hash is added, this function determines the hash value of
         *  all that data. That value is converted to a string.
         *  \sa update */
        DP_UTIL_API virtual std::string finalize() = 0;
    };

    inline void HashGenerator::update( const unsigned char * input, unsigned int elementSize, unsigned int stride, unsigned int elementCount )
    {
      if (stride != elementSize)
      {
        for ( unsigned int element = 0;element < elementCount;++element)
        {
          update( input, elementSize );
          input += stride;
        }
      }
      else
      {
        update( input, elementCount * elementSize );
      }
    }

    template <typename T>
    inline void HashGenerator::update( SharedPtr<T> const& ptr )
    {
      update( reinterpret_cast<const unsigned char *>( ptr.getWeakPtr() ), sizeof(const T *) );
    }

  } // namespace util
} // namespace dp

