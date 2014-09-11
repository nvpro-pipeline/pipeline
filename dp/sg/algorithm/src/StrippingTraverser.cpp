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


#pragma push_macro("free")
#undef free
#include <valarray>
#pragma pop_macro("free")

#include <dp/sg/core/FaceConnections.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/algorithm/StrippingTraverser.h>

using namespace dp::util;
using namespace dp::sg::core;

using std::valarray;
using std::map;
using std::list;
using std::set;
using std::vector;
using std::pair;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      StrippingTraverser::StrippingTraverser(void)
      {
      }

      StrippingTraverser::~StrippingTraverser(void)
      {
      }

      void StrippingTraverser::handleGeoNode( GeoNode * p )
      {
        DP_ASSERT( !m_strip );
        ExclusiveTraverser::handleGeoNode( p );
        if ( m_strip )
        {
          p->setPrimitive( m_strip );
          m_strip.reset();
          setTreeModified();
        }
      }

      void StrippingTraverser::handlePrimitive( Primitive * p )
      {
        ExclusiveTraverser::handlePrimitive( p );
        if ( optimizationAllowed( p->getSharedPtr<Primitive>() ) )
        {
          switch( p->getPrimitiveType() )
          {
            case PRIMITIVE_TRIANGLES :
            case PRIMITIVE_QUADS :
              changeToStrips( p );
              break;
            case PRIMITIVE_LINES :                // might to be added, as needed
            case PRIMITIVE_TRIANGLES_ADJACENCY :  // might to be added, as needed
            case PRIMITIVE_LINE_STRIP_ADJACENCY : // might to be added, as needed
            default:
              break;
          }
        }
      }

      void StrippingTraverser::changeToStrips( Primitive * p )
      {
        DP_ASSERT(    ( p->getPrimitiveType() == PRIMITIVE_TRIANGLES )
                    ||  ( p->getPrimitiveType() == PRIMITIVE_QUADS ) );

        unsigned int vpp = p->getNumberOfVerticesPerPrimitive();
        DP_ASSERT( ( vpp == 3 ) || ( vpp == 4 ) );

        DP_ASSERT( !m_strip );
        m_strip = Primitive::create( ( vpp == 3 ) ? PRIMITIVE_TRIANGLE_STRIP : PRIMITIVE_QUAD_STRIP );

        *((OwnedObject<Object>*)m_strip.getWeakPtr()) = *p;    // copy all but the Primitive itself
        //m_strip->setName( p->getName() );
        //m_strip->setAnnotation( p->getAnnotation() );
        //m_strip->setHints( p->getHints() );
        //for ( Object::CallbackContainer occ = p->beginAppTraverserCallbacks() ; occ != p->endAppTraverserCallbacks() ; ++occ )
        //{
        //  m_strip->addAppTraverserCallback( *occ );
        //}
        //m_strip->setTraversalMask( p->getTraversalMask() );
        //DP_ASSERT( !p->getSkin() );
        m_strip->setInstanceCount( p->getInstanceCount() );
        m_strip->setVertexAttributeSet( p->getVertexAttributeSet() );
        m_strip->setIndexSet( p->getIndexSet() );

        m_strip->makeIndexed();

        vector<unsigned int> strippedIndices;
        vector<unsigned int> nonStrippedIndices;
        FaceConnections fc( p );
        {
          // get the indices (using the offset of p)
          IndexSet::ConstIterator<unsigned int> indices( m_strip->getIndexSet(), p->getElementOffset() );

          for ( unsigned int fi = fc.getNextFaceIndex() ; fi != ~0 ; fi = fc.getNextFaceIndex() )
          {
            list<unsigned int> stripFaces;
            unsigned int length = ( vpp == 3 ) ? fc.findLongestTriStrip( indices, fi, strippedIndices, stripFaces )
                                                : fc.findLongestQuadStrip( indices, fi, strippedIndices, stripFaces );
            fc.disconnectFaces( stripFaces );
            strippedIndices.push_back( ~0 );
          }

          fc.getAndClearZeroConnectionIndices( indices, nonStrippedIndices );
        }

        for ( size_t i=vpp-1 ; i<nonStrippedIndices.size() ; i+=vpp )
        {
          if ( vpp == 3 )
          {
            strippedIndices.push_back( nonStrippedIndices[i-2] );
            strippedIndices.push_back( nonStrippedIndices[i-1] );
            strippedIndices.push_back( nonStrippedIndices[i] );
          }
          else
          {
            strippedIndices.push_back( nonStrippedIndices[i-3] );
            strippedIndices.push_back( nonStrippedIndices[i-2] );
            strippedIndices.push_back( nonStrippedIndices[i] );
            strippedIndices.push_back( nonStrippedIndices[i-1] );
          }
          strippedIndices.push_back( ~0 );
        }
        strippedIndices.pop_back();      // remove the last pri again

        IndexSetSharedPtr strippedIndexSet = IndexSet::create();
        *((OwnedObject<Primitive>*)strippedIndexSet.getWeakPtr()) = *(m_strip->getIndexSet().getWeakPtr());
        strippedIndexSet->setData( &strippedIndices[0], checked_cast<unsigned int>(strippedIndices.size()) );
        strippedIndexSet->setPrimitiveRestartIndex( ~0 );

        m_strip->setIndexSet( strippedIndexSet );
      }

      bool StrippingTraverser::optimizationAllowed( ObjectSharedPtr const& obj )
      {
        DP_ASSERT( obj != NULL );

        return (obj->getHints( Object::DP_SG_HINT_DYNAMIC ) == 0);
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
