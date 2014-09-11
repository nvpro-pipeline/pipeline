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


#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/algorithm/ModelViewTraverser.h>

using namespace dp::math;
using namespace dp::sg::core;

using std::make_pair;
using std::pair;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      SharedModelViewTraverser::SharedModelViewTraverser()
      {
        m_transformStack.setWorldToView( cIdentity44f, cIdentity44f );
      }

      SharedModelViewTraverser::~SharedModelViewTraverser(void)
      {
        DP_ASSERT( m_transformStack.getStackDepth() == 1 );
      }

      void  SharedModelViewTraverser::handleBillboard( const Billboard *p )
      {
        // multiply trafo on top of current matrices
        Trafo trafo = p->getTrafo( m_camera, m_transformStack.getWorldToModel() );
        m_transformStack.pushModelToWorld( trafo.getMatrix(), trafo.getInverse() );

        //  call the (overloadable) preTraverse() between stack adjustment and traversal
        if ( preTraverseTransform( &trafo ) )
        {
          SharedTraverser::handleBillboard( p );

          //  call the (overloadable) postTraverse() between stack adjustment and traversal
          postTraverseTransform( &trafo );
        }

        // pop off view matrices after proceeding
        m_transformStack.popModelToWorld();
      }

      void  SharedModelViewTraverser::handleTransform( const Transform *p )
      {
        // multiply trafo on top of current matrices
        const Trafo & trafo = p->getTrafo();
        m_transformStack.pushModelToWorld( trafo.getMatrix(), trafo.getInverse() );

        //  call the (overloadable) preTraverse() between stack adjustment and traversal
        if ( preTraverseTransform( &trafo ) )
        {
          SharedTraverser::handleTransform( p );

          //  call the (overloadable) postTraverse() between stack adjustment and traversal
          postTraverseTransform( &trafo );
        }

        // pop off view matrices after proceeding
        m_transformStack.popModelToWorld();
      }

      bool SharedModelViewTraverser::preTraverseTransform( const Trafo *p )
      {
        return( true );
      }

      void  SharedModelViewTraverser::postTraverseTransform( const Trafo *p )
      {
      }

      void  SharedModelViewTraverser::traverseCamera( const Camera *p )
      {
        DP_ASSERT( m_transformStack.getStackDepth() == 1 );
        m_transformStack.setWorldToView( p->getWorldToViewMatrix(), p->getViewToWorldMatrix() );
        m_transformStack.setViewToClip( p->getProjection(), p->getInverseProjection() );

        SharedTraverser::traverseCamera( p );
      }

      ExclusiveModelViewTraverser::ExclusiveModelViewTraverser()
      {
        m_transformStack.setWorldToView( cIdentity44f, cIdentity44f );
      }

      ExclusiveModelViewTraverser::~ExclusiveModelViewTraverser(void)
      {
        DP_ASSERT( m_transformStack.getStackDepth() == 1 );
      }

      void  ExclusiveModelViewTraverser::handleBillboard( Billboard *p )
      {
        // multiply trafo on top of current matrices
        Trafo trafo = p->getTrafo( m_camera, m_transformStack.getWorldToModel() );
        m_transformStack.pushModelToWorld( trafo.getMatrix(), trafo.getInverse() );

        //  call the (overloadable) preTraverse() between stack adjustment and traversal
        if ( preTraverseTransform( &trafo ) )
        {
          ExclusiveTraverser::handleBillboard( p );

          //  call the (overloadable) postTraverse() between stack adjustment and traversal
          postTraverseTransform( &trafo );
        }

        // pop off view matrices after proceeding
        m_transformStack.popModelToWorld();
      }

      void  ExclusiveModelViewTraverser::handleTransform( Transform *p )
      {
        // multiply trafo on top of current matrices
        const Trafo & trafo = p->getTrafo();
        m_transformStack.pushModelToWorld( trafo.getMatrix(), trafo.getInverse() );

        //  call the (overloadable) preTraverse() between stack adjustment and traversal
        if ( preTraverseTransform( &trafo ) )
        {
          ExclusiveTraverser::handleTransform( p );

          //  call the (overloadable) postTraverse() between stack adjustment and traversal
          postTraverseTransform( &trafo );
        }

        // pop off view matrices after proceeding
        m_transformStack.popModelToWorld();
      }

      bool ExclusiveModelViewTraverser::preTraverseTransform( const Trafo *p )
      {
        return( true );
      }

      void  ExclusiveModelViewTraverser::postTraverseTransform( const Trafo *p )
      {
      }

      void  ExclusiveModelViewTraverser::traverseCamera( Camera *p )
      {
        DP_ASSERT( m_transformStack.getStackDepth() == 1 );
        m_transformStack.setWorldToView( p->getWorldToViewMatrix(), p->getViewToWorldMatrix() );
        m_transformStack.setViewToClip( p->getProjection(), p->getInverseProjection() );

        ExclusiveTraverser::traverseCamera( p );
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
