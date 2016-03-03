// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/algorithm/DeindexTraverser.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/IndexSet.h>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using namespace std;


namespace
{

  template <unsigned int n, typename T>
  void copyVertexDataNT( VertexAttributeSetSharedPtr const& dst, VertexAttributeSetSharedPtr const& src, VertexAttributeSet::AttributeID attrib,
                         IndexSet::ConstIterator<unsigned int> indices, unsigned int count, unsigned int pri )
  {
    vector<Vecnt<n,T> > a;
    a.reserve( count );  // This includes possible primitive restart indices. The final size might be smaller.

    typename Buffer::ConstIterator<Vecnt<n,T> >::Type data = src->getVertexData<Vecnt<n,T> >(attrib);

    for ( unsigned int i = 0; i < count; i++ )
    {
      unsigned int idx = indices[i];

      // Skip primitive restart index. Mind that there has been a destripping traverser applied before.
      if ( idx != pri )
      {
        a.push_back( data[idx] );
      }
    }

    dst->setVertexData( attrib, n, src->getTypeOfVertexData( attrib ),
                        (void *) &a[0], 0, dp::checked_cast<unsigned int>( a.size() ) );
  }


  template <unsigned int n>
  void copyVertexDataN( VertexAttributeSetSharedPtr const& dst, VertexAttributeSetSharedPtr const& src, VertexAttributeSet::AttributeID attrib,
                        IndexSet::ConstIterator<unsigned int> indices, unsigned int count, unsigned int pri )
  {
    switch( src->getTypeOfVertexData( attrib ) )
    {
      case dp::DataType::INT_8 :
        copyVertexDataNT<n,char>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::UNSIGNED_INT_8 :
        copyVertexDataNT<n,unsigned char>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::INT_16 :
        copyVertexDataNT<n,short>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::UNSIGNED_INT_16 :
        copyVertexDataNT<n,unsigned short>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::INT_32 :
        copyVertexDataNT<n,int>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::UNSIGNED_INT_32 :
        copyVertexDataNT<n,unsigned int>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::FLOAT_32 :
        copyVertexDataNT<n,float>( dst, src, attrib, indices, count, pri );
        break;
      case dp::DataType::FLOAT_64 :
        copyVertexDataNT<n,double>( dst, src, attrib, indices, count, pri );
        break;
      default :
        DP_ASSERT( !"Unexpected getTypeOfVertexData() result." );
        break;
    }
  }

}

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

    DeindexTraverser::DeindexTraverser(void)
    {
    }

    DeindexTraverser::~DeindexTraverser(void)
    {
    }

    void DeindexTraverser::doApply( const NodeSharedPtr & root )
    {
      // We can't convert a stripped primitive into a single array primitive, so we destripify before traversal.
      DestrippingTraverser destrip;
      destrip.apply( root );

      ExclusiveTraverser::doApply(root);
    }

    void DeindexTraverser::handlePrimitive( Primitive *p )
    {
      ExclusiveTraverser::handlePrimitive( p );

      // If the primitive is not indexed
      // do not remove them.
      if ( !p->isIndexed() )
      {
        return;
      }

      VertexAttributeSetSharedPtr vash = VertexAttributeSet::create();
      VertexAttributeSetSharedPtr const& ovas = p->getVertexAttributeSet();

      unsigned int offset = p->getElementOffset();
      unsigned int count  = p->getElementCount();
      unsigned int pri    = p->getIndexSet()->getPrimitiveRestartIndex();

      IndexSet::ConstIterator<unsigned int> indices( p->getIndexSet(), offset );

      for ( unsigned int i = 0; i < static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT); i++ )
      {
        VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(i);
        // Handle all provided VertexAttributes, not only the enabled ones.
        switch ( ovas->getSizeOfVertexData( attribute ) )
        {
          case 0: // VertexAttribute not specified for this attrib.
            break;

          case 1:
            copyVertexDataN<1>( vash, ovas, attribute, indices, count, pri );
            break;

          case 2:
            copyVertexDataN<2>( vash, ovas, attribute, indices, count, pri );
            break;

          case 3:
            copyVertexDataN<3>( vash, ovas, attribute, indices, count, pri );
            break;

          case 4:
            copyVertexDataN<4>( vash, ovas, attribute, indices, count, pri );
            break;

          default:
            DP_ASSERT( !"Unexpected getSizeOfVertexData() result." );
            break;
        }
        // Copy the enable state.
        vash->setEnabled( attribute, ovas->isEnabled( attribute ) );
      }

      p->setIndexSet( dp::sg::core::IndexSetSharedPtr() );        // Remove the IndexSet from this primitive.
      p->setVertexAttributeSet( vash ); // Replace with the new de-indexed VertexAttributeSet.
      p->setElementRange( 0, ~0 );      // Use the whole VertexAttributeSet.
      setTreeModified();
    }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
