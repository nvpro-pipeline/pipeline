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

#include <dp/fx/ParameterGroupSpec.h>
#include <map>

namespace dp
{
  namespace fx
  {
    // The spec for an effect, holding a container of SmartParameterGroupSpecs
    DEFINE_PTR_TYPES( EffectSpec );

    class EffectSpec
    {
      public:
        typedef std::vector<ParameterGroupSpecSharedPtr>  ParameterGroupSpecsContainer;
        typedef ParameterGroupSpecsContainer::const_iterator iterator;

        typedef enum
        {
          EST_UNKNOWN,
          EST_LIGHT,
          EST_SYSTEM,
          EST_PIPELINE
        } Type;

        typedef std::map<Type, EffectSpecSharedPtr> DomainEffectSpecs;

      public:
        DP_FX_API static EffectSpecSharedPtr create( const std::string & name, Type type
                                                   , const ParameterGroupSpecsContainer & groupSpecs
                                                   , bool transparent = false );
        virtual ~EffectSpec();

      public:
                  const std::string & getName() const;
                  Type getType() const;
                  bool getTransparent() const;
                  unsigned int getNumberOfParameterGroupSpecs() const;
                  iterator beginParameterGroupSpecs() const;
                  iterator endParameterGroupSpecs() const;
        DP_FX_API iterator findParameterGroupSpec( const ParameterGroupSpecSharedPtr & groupSpec ) const;
        DP_FX_API iterator findParameterGroupSpec( const std::string & name ) const;

                  dp::util::HashKey getHashKey() const;
        DP_FX_API bool isEquivalent( const EffectSpecSharedPtr & p, bool ignoreNames, bool deepCompare ) const;

      protected:
        // new pipeline version
        DP_FX_API EffectSpec( std::string const & name, Type type, ParameterGroupSpecsContainer const & specs, bool transparent );

      private:
        std::string                   m_name;
        Type                          m_type;
        ParameterGroupSpecsContainer  m_specs;
        bool                          m_transparent;
        dp::util::HashKey             m_hashKey;

    };

    inline EffectSpec::~EffectSpec()
    {
    }

    inline const std::string & EffectSpec::getName() const
    {
      return( m_name );
    }

    inline EffectSpec::Type EffectSpec::getType() const
    {
      return( m_type );
    }

    inline bool EffectSpec::getTransparent() const
    {
      return( m_transparent );
    }

    inline unsigned int EffectSpec::getNumberOfParameterGroupSpecs() const
    {
      return( dp::checked_cast<unsigned int>( m_specs.size() ) );
    }

    inline EffectSpec::iterator EffectSpec::beginParameterGroupSpecs() const
    {
      return( m_specs.begin() );
    }

    inline EffectSpec::iterator EffectSpec::endParameterGroupSpecs() const
    {
      return( m_specs.end() );
    }

    inline dp::util::HashKey EffectSpec::getHashKey() const
    {
      return( m_hashKey );
    }

  } // namespace fx
} // namespace dp
