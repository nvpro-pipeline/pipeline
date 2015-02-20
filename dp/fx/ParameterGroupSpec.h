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
#include <dp/fx/ParameterSpec.h>
#include <dp/util/HashGenerator.h>

namespace dp
{
  namespace fx
  {

    // The spec for a group of parameters
    DEFINE_PTR_TYPES( ParameterGroupSpec );

    class ParameterGroupSpec
    {
      private:
        typedef std::vector<std::pair<ParameterSpec,unsigned int> >  ParameterSpecsContainer;

      public:
        typedef ParameterSpecsContainer::const_iterator iterator;

      public:
        DP_FX_API static ParameterGroupSpecSharedPtr create( const std::string & name
                                                       , const std::vector<ParameterSpec> & specs );
        DP_FX_API virtual ~ParameterGroupSpec();

      public:
        const std::string & getName() const;
        unsigned int getNumberOfParameterSpecs() const;
        unsigned int getDataSize() const;
        iterator beginParameterSpecs() const;
        iterator endParameterSpecs() const;
        DP_FX_API iterator findParameterSpec( const std::string & name ) const;

        dp::util::HashKey getHashKey() const;
        bool isEquivalent( const ParameterGroupSpecSharedPtr & p, bool ignoreNames, bool deepCompare ) const;

      protected:
        DP_FX_API ParameterGroupSpec( const std::string & name, const std::vector<ParameterSpec> & specs );
        DP_FX_API ParameterGroupSpec( const ParameterGroupSpec &rhs );
        DP_FX_API ParameterSpec & operator=( const ParameterSpec & rhs );

      private:
        unsigned int            m_dataSize;
        dp::util::HashKey       m_hashKey;
        std::string             m_name;
        ParameterSpecsContainer m_specs;
    };


    inline ParameterGroupSpec::~ParameterGroupSpec()
    {
    }

    inline const std::string & ParameterGroupSpec::getName() const
    {
      return( m_name );
    }

    inline unsigned int ParameterGroupSpec::getNumberOfParameterSpecs() const
    {
      return( dp::checked_cast<unsigned int>(m_specs.size()) );
    }

    inline unsigned int ParameterGroupSpec::getDataSize() const
    {
      return( m_dataSize );
    }

    inline ParameterGroupSpec::iterator ParameterGroupSpec::beginParameterSpecs() const
    {
      return( m_specs.begin() );
    }

    inline ParameterGroupSpec::iterator ParameterGroupSpec::endParameterSpecs() const
    {
      return( m_specs.end() );
    }

    inline dp::util::HashKey ParameterGroupSpec::getHashKey() const
    {
      return( m_hashKey );
    }

    inline bool ParameterGroupSpec::isEquivalent( const ParameterGroupSpecSharedPtr & p, bool ignoreNames, bool /*deepCompare*/ ) const
    {
      return(   ( p.getWeakPtr() == this )
            ||  (   ( ignoreNames ? true : m_name == p->m_name )
                &&  ( m_dataSize == p->m_dataSize )
                &&  ( m_specs == p->m_specs ) ) );
    }

  } // namespace fx
} // namespace dp
