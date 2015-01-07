// Copyright NVIDIA Corporation 2002-2005
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


#include  <dp/sg/core/GeoNode.h>
#include  <dp/sg/core/Primitive.h>
#include  <dp/sg/algorithm/TriangulateTraverser.h>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::make_pair;
using std::vector;
using std::pair;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      TriangulateTraverser::TriangulateTraverser(void)
      {
      }

      TriangulateTraverser::~TriangulateTraverser(void)
      {
      }

      template<typename IndicesPtrType>
      void convertQuadsToTriangles( IndicesPtrType indices, unsigned int elementCount
                                  , const VertexAttributeSetSharedPtr & vasSP, std::vector<unsigned int> & newIndices
                                  , unsigned int pri = ~0 )
      {
        //  for each quad-face create two triangles
        Buffer::ConstIterator<Vec3f>::Type vertices = vasSP->getVertices();
        newIndices.reserve( 6 * ( elementCount / 4 ) );   // might reserve more than needed, due to pri

        for ( unsigned int i=3 ; i<elementCount ; i+=4 )
        {
          if ( indices[i-3] == pri )
          {
            i -= 3;
          }
          else if ( indices[i-2] == pri )
          {
            i -= 2;
          }
          else if ( indices[i-1] == pri )
          {
            i -= 1;
          }
          else if ( indices[i] != pri )
          {
            //  determine the shorter diagonal
            if (  lengthSquared( vertices[indices[i-1]] - vertices[indices[i-3]] )
                < lengthSquared( vertices[indices[i-0]] - vertices[indices[i-2]] ) )
            {
              //  cut diagonal 0-2 => 0,1,2 & 2,3,0
              newIndices.push_back( indices[i-3] );
              newIndices.push_back( indices[i-2] );
              newIndices.push_back( indices[i-1] );
              newIndices.push_back( indices[i-1] );
              newIndices.push_back( indices[i-0] );
              newIndices.push_back( indices[i-3] );
            }
            else
            {
              //  cut diagonal 1-3 => 1,2,3 & 3,0,1
              newIndices.push_back( indices[i-2] );
              newIndices.push_back( indices[i-1] );
              newIndices.push_back( indices[i-0] );
              newIndices.push_back( indices[i-0] );
              newIndices.push_back( indices[i-3] );
              newIndices.push_back( indices[i-2] );
            }
          }
        }
      }

      void TriangulateTraverser::handleGeoNode( GeoNode * p )
      {
        DP_ASSERT( !m_triangulatedPrimitive );
        ExclusiveTraverser::handleGeoNode( p );
        if ( m_triangulatedPrimitive )
        {
          p->setPrimitive( m_triangulatedPrimitive );
          m_triangulatedPrimitive.reset();
          setTreeModified();
        }
      }

      void TriangulateTraverser::handlePrimitive( Primitive * p )
      {
        if( !optimizationAllowed( p->getSharedPtr<Primitive>() ) )
        {
          return;
        }

        DP_ASSERT( !m_triangulatedPrimitive );
        unsigned int primitiveType = p->getPrimitiveType();
        if ( ( primitiveType == PRIMITIVE_QUAD_STRIP ) || ( primitiveType == PRIMITIVE_QUADS ) )
        {
          // create a new Primitive QuadStrip -> TriStrip or Quads -> Tris
          m_triangulatedPrimitive = Primitive::create( primitiveType == PRIMITIVE_QUAD_STRIP ? PRIMITIVE_TRIANGLE_STRIP : PRIMITIVE_TRIANGLES );
          *static_cast<Object*>(m_triangulatedPrimitive.getWeakPtr()) = *p;    // copy all but the Primitive itself
          m_triangulatedPrimitive->setElementRange( p->getElementOffset(), p->getElementCount() );
          m_triangulatedPrimitive->setInstanceCount( p->getInstanceCount() );
          m_triangulatedPrimitive->setVertexAttributeSet( p->getVertexAttributeSet() );
          m_triangulatedPrimitive->setIndexSet( p->getIndexSet() );

          // from quad strip to tri strip is just copying the Primitive into a new tri strip; that is, we're alread done
          // otherwise...
          if ( primitiveType == PRIMITIVE_QUADS )
          {
            m_triangulatedPrimitive->makeIndexed();
            vector<unsigned int> newIndices;
            convertQuadsToTriangles( IndexSet::ConstIterator<unsigned int>( m_triangulatedPrimitive->getIndexSet(), m_triangulatedPrimitive->getElementOffset() )
                                   , m_triangulatedPrimitive->getElementCount(), m_triangulatedPrimitive->getVertexAttributeSet(), newIndices
                                   , m_triangulatedPrimitive->getIndexSet()->getPrimitiveRestartIndex() );
            IndexSetSharedPtr triangulatedIndexSet = IndexSet::create();
            *static_cast<Object*>(triangulatedIndexSet.getWeakPtr()) = *( m_triangulatedPrimitive->getIndexSet().getWeakPtr() );
            triangulatedIndexSet->setData( &newIndices[0], checked_cast<unsigned int>(newIndices.size()) );
            triangulatedIndexSet->setPrimitiveRestartIndex( ~0 );
          }
        }
      }

      bool TriangulateTraverser::optimizationAllowed( ObjectSharedPtr const& obj )
      {
        DP_ASSERT( obj != NULL );

        return (obj->getHints( Object::DP_SG_HINT_DYNAMIC ) == 0);
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
