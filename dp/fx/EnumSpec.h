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

#include <dp/fx/Config.h>
#include <dp/util/SharedPtr.h>
#include <memory>
#include <string>
#include <vector>

namespace dp
{
  namespace fx
  {
    DEFINE_PTR_TYPES( EnumSpec );

    class EnumSpec
    {
      public:
        typedef int StorageType;

      public:
        DP_FX_API static EnumSpecSharedPtr create( const std::string & type, const std::vector<std::string> & values );
        DP_FX_API virtual ~EnumSpec();

      public:
        DP_FX_API const std::string & getType() const;
        DP_FX_API unsigned int getValueCount() const;
        DP_FX_API const std::string & getValueName( unsigned int idx ) const;
        DP_FX_API StorageType getValue( const std::string & name ) const;

        DP_FX_API bool isEquivalent( const EnumSpecSharedPtr & p, bool ignoreNames, bool deepCompare ) const;

      protected:
        DP_FX_API EnumSpec( const std::string & type, const std::vector<std::string> & value );
        DP_FX_API EnumSpec( const EnumSpec & rhs );
        DP_FX_API EnumSpec & operator=( const EnumSpec & rhs );

      private:
        std::string               m_type;
        std::vector<std::string>  m_values;
    };

  } // namespace fx
} // namespace dp
