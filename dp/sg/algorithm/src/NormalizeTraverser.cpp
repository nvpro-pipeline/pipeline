// Copyright NVIDIA Corporation 2002-2011
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


#include <vector>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/algorithm/NormalizeTraverser.h>

using namespace dp::sg::core;

using std::pair;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      DEFINE_STATIC_PROPERTY( NormalizeTraverser, VertexAttributeIndex );

      BEGIN_REFLECTION_INFO( NormalizeTraverser )
        DERIVE_STATIC_PROPERTIES( NormalizeTraverser, ExclusiveTraverser );
  
        INIT_STATIC_PROPERTY_RW( NormalizeTraverser, VertexAttributeIndex, unsigned int, SEMANTIC_VALUE, value, value );
      END_REFLECTION_INFO

      NormalizeTraverser::NormalizeTraverser(void)
      : m_attrib(VertexAttributeSet::NVSG_NORMAL)
      {
      }

      NormalizeTraverser::~NormalizeTraverser(void)
      {
      }

      void  NormalizeTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT( root );

        ExclusiveTraverser::doApply( root );

        m_objects.clear();
      }

      void  NormalizeTraverser::handleVertexAttributeSet( VertexAttributeSet * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          if ( p->getSizeOfVertexData(m_attrib) )
          {
            VertexAttribute va;
            p->swapVertexData(m_attrib, va);
            normalize( va );
            p->swapVertexData(m_attrib, va);
            setTreeModified();
          }
        }
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
