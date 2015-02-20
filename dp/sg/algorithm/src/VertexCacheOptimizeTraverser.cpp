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


#include <dp/sg/algorithm/VertexCacheOptimizeTraverser.h>

using namespace dp::util;
using namespace dp::sg::core;

using std::pair;
using std::set;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      VertexCacheOptimizeTraverser::VertexCacheOptimizeTraverser( void )
      {
      }

      VertexCacheOptimizeTraverser::~VertexCacheOptimizeTraverser( void )
      {
      }

      void VertexCacheOptimizeTraverser::postApply( const NodeSharedPtr & root )
      {
        ExclusiveTraverser::postApply( root );
        m_objects.clear();
      }

      void VertexCacheOptimizeTraverser::handlePrimitive( Primitive * p )
      {
      pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
      if ( pitb.second )
      {
          if ( ( p->getPrimitiveType() == PRIMITIVE_TRIANGLES ) && p->isIndexed() )
          {
            unsigned int count = p->getElementCount();
            DP_ASSERT( count % 3 == 0 );
            m_newIndices.resize( count );
            {
              IndexSet::ConstIterator<unsigned int> idx( p->getIndexSet(), p->getElementOffset() );
              for ( unsigned int i=0 ; i<count ; i++ )
              {
                m_newIndices[i] = dp::checked_cast<int>( idx[i] );
              }
            }

            VertexCacheOptimizer::Result r = m_vco.Optimize( &m_newIndices[0], count / 3 );

            if ( !m_vco.Failed( r ) )
            {
              IndexSetSharedPtr indexSet = IndexSet::create();
              indexSet->setData( (const unsigned int *)&m_newIndices[0], count );
              p->setIndexSet( indexSet );
              p->setElementRange( 0, ~0 );
            }

            // TODO: reordering vertices to make reading them as sequential as possible
          }
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
