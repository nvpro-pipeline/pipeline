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


#include <dp/fx/ParameterGroupSpec.h>
#include <dp/util/HashGeneratorMurMur.h>
#include <set>

using namespace dp::util;
using std::make_pair;
using std::string;
using std::vector;

namespace dp
{
  namespace fx
  {

    ParameterGroupSpecSharedPtr ParameterGroupSpec::create( const string & name
                                                          , const vector<ParameterSpec> & specs )
    {
      return( std::shared_ptr<ParameterGroupSpec>( new ParameterGroupSpec( name, specs ) ) );
    }

    bool specSorter( vector<ParameterSpec>::const_iterator it0
                   , vector<ParameterSpec>::const_iterator it1 )
    {
      unsigned int ps0 = getParameterSize( it0->getType() );
      unsigned int ps1 = getParameterSize( it1->getType() );
      return( ( ps1 < ps0 ) || ( ( ps0 == ps1 ) && ( it0->getName() < it1->getName() ) ) );
    }

    ParameterGroupSpec::ParameterGroupSpec( const string & name
                                          , const vector<ParameterSpec> & specs )
      : m_dataSize(0)
      , m_name(name)
    {
#if !defined(NDEBUG)
      // check that no parameter name occurs more than once
      std::set<string> nameSet;
      for ( vector<ParameterSpec>::const_iterator it = specs.begin() ; it != specs.end() ; ++it )
      {
        DP_ASSERT( nameSet.insert( it->getName() ).second );
      }
#endif

      // order the specs by their base parameter size first, and name second
      vector<vector<ParameterSpec>::const_iterator> orderedSpecs;
      orderedSpecs.reserve( specs.size() );
      for ( vector<ParameterSpec>::const_iterator it = specs.begin() ; it != specs.end() ; ++it )
      {
        orderedSpecs.push_back( it );
      }
      sort( orderedSpecs.begin(), orderedSpecs.end(), specSorter );

      // get the pairs of ParameterSpec and offset
      m_specs.reserve( specs.size() );
      for ( vector<vector<ParameterSpec>::const_iterator>::const_iterator it = orderedSpecs.begin() ; it != orderedSpecs.end() ; ++it )
      {
        DP_ASSERT( m_dataSize % getParameterAlignment( (*it)->getType() ) == 0 );
        m_specs.push_back( make_pair( **it, m_dataSize ) );
        m_dataSize += (*it)->getSizeInByte();
      }

      HashGeneratorMurMur hg;
      // don't hash the name
      for ( iterator it = m_specs.begin() ; it != m_specs.end() ; ++it )
      {
        if ( ! it->first.getName().empty() )
        {
          hg.update( reinterpret_cast<const unsigned char *>(&it->first.getName()[0]), checked_cast<unsigned int>(it->first.getName().size()) );
        }
        unsigned int tmp = it->first.getType();
        hg.update( reinterpret_cast<const unsigned char *>(&tmp), sizeof(tmp) );
        tmp = it->first.getArraySize();
        hg.update( reinterpret_cast<const unsigned char *>(&tmp), sizeof(tmp) );
      }
      hg.finalize( (unsigned int *)&m_hashKey );
    }

    ParameterGroupSpec::iterator ParameterGroupSpec::findParameterSpec( const std::string & name ) const
    {
      for ( iterator it = m_specs.begin() ; it != m_specs.end() ; ++it )
      {
        if ( it->first.getName() == name )
        {
          return( it );
        }
      }
      return( m_specs.end() );
    }

  } // namespace fx
} // namespace cp
