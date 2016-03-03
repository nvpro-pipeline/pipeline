// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/fx/EffectSpec.h>
#include <dp/util/HashGeneratorMurMur.h>
#include <set>

using namespace dp::util;
using std::list;
using std::string;
using std::vector;

namespace dp
{
  namespace fx
  {

    EffectSpecSharedPtr EffectSpec::create( const string & name, Type type, const ParameterGroupSpecsContainer & groupSpecs, bool transparent )
    {
      return( std::shared_ptr<EffectSpec>( new EffectSpec( name, type, groupSpecs, transparent ) ) );
    }

    bool specSorter( const ParameterGroupSpecSharedPtr & spec0, const ParameterGroupSpecSharedPtr & spec1 )
    {
      return( spec0->getName() < spec1->getName() );
    }

    EffectSpec::EffectSpec( const std::string & name, Type type, const ParameterGroupSpecsContainer & specs, bool transparent )
      : m_name( name )
      , m_type( type )
      , m_transparent( transparent )
      , m_specs( specs )
    {
#if !defined(NDEBUG)
      std::set<string> nameSet;
      for ( iterator it = m_specs.begin() ; it != m_specs.end() ; ++it )
      {
        DP_ASSERT( nameSet.insert( (*it)->getName() ).second );
      }
#endif
      sort( m_specs.begin(), m_specs.end(), specSorter );

      HashGeneratorMurMur hg;
      // don't hash the name
      hg.update( reinterpret_cast<const unsigned char *>(&m_type), sizeof(m_type) );
      hg.update( reinterpret_cast<const unsigned char *>(&m_transparent), sizeof(m_transparent) );
      for ( iterator it = m_specs.begin() ; it != m_specs.end() ; ++it )
      {
        HashKey hk = (*it)->getHashKey();
        hg.update( reinterpret_cast<const unsigned char *>(&hk), sizeof(hk) );
      }
      hg.finalize( (unsigned int *)&m_hashKey );
    }

    EffectSpec::iterator EffectSpec::findParameterGroupSpec( const ParameterGroupSpecSharedPtr & groupSpec ) const
    {
      for ( iterator it = m_specs.begin() ; it != m_specs.end() ; ++it )
      {
        if ( *it == groupSpec )
        {
          return( it );
        }
      }
      return( m_specs.end() );
    }

    EffectSpec::iterator EffectSpec::findParameterGroupSpec( const std::string & name ) const
    {
      for ( iterator it = m_specs.begin() ; it != m_specs.end() ; ++it )
      {
        if ( (*it)->getName() == name )
        {
          return( it );
        }
      }
      return( m_specs.end() );
    }

    bool EffectSpec::isEquivalent( const EffectSpecSharedPtr & p, bool ignoreNames, bool deepCompare ) const
    {
      if ( p.get() == this )
      {
        return( true );
      }

      bool equi =   ( ignoreNames ? true : m_name == p->m_name )
                &&  ( m_type == p->m_type )
                &&  ( m_transparent == p->m_transparent )
                &&  ( m_specs.size() == p->m_specs.size() );
      if ( equi )
      {
        for ( iterator it = m_specs.begin(), jt = p->m_specs.begin()
            ; equi && it != m_specs.end() ; ++it, ++jt )
        {
          DP_ASSERT( jt != p->m_specs.end() );
          equi = deepCompare ? (*it)->isEquivalent( *jt, ignoreNames, true ) : ( *it == *jt );
        }
      }
      return( equi );
    }

  } // namespace fx
} // namespace dp
