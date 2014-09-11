// Copyright NVIDIA Corporation 2010-2013
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

#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/Camera.h>
#include <vector>

#if defined(DP_ARCH_X86_64)
#define SSE
#endif

#if defined(DP_ARCH_ARM_32)
#define NEON
#endif

#if defined(SSE)
#include <dp/math/sse/Vecnt.h>
#include <dp/math/sse/Matmnt.h>
static bool useSSE = true;
#else
static bool useSSE = false;
#endif

#if defined(NEON)
#include <dp/math/neon/Vecnt.h>
#include <dp/math/neon/Matmnt.h>
static bool useNEON = true;
#else
static bool useNEON = false;
#endif


namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      class UpdateTransformVisitor
      {
      public:
        UpdateTransformVisitor( TransformTree& tree, SceneTree& sceneTree, const dp::sg::core::CameraSharedPtr &camera, std::vector<TransformTreeIndex>& changedTransforms ) 
          : m_transformTree(tree)
          , m_sceneTree(sceneTree)
          , m_camera(camera)
          , m_changedTransforms(changedTransforms)
        {
        }

        struct Data {};

        bool preTraverse( TransformTreeIndex index, const Data& data )
        {
          TransformTreeNode& current = m_transformTree[index];

          m_changedTransforms.push_back( index );

          // need a parent to propagate info downwards
          if( current.m_parentIndex != ~0 )
          {
            const TransformTreeNode& curParent = m_transformTree[ current.m_parentIndex ];

            // update local transform if needed
            if( current.m_billboard != nullptr )
            {
              // billboards always need an update
              dp::math::Mat44f mat = curParent.m_worldMatrix;
              mat.invert();
              DP_ASSERT( current.m_billboard != nullptr );
              dp::math::Trafo t = current.m_billboard->getTrafo( m_camera, mat );
              current.m_localMatrix = t.getMatrix();

              dp::math::Vec3f const& s( t.getScaling() );
              current.setLocalBits( TransformTreeNode::ISMIRRORTRANSFORM, s[0]*s[1]*s[2] < 0.0f );

              // don't set the node as dirty as no check is done below
            }

            // don't propagate and check dirty bits as there is only one default bit

            // calculate world matrix for current nodeIndex
#if defined(SSE)
            if ( useSSE )
            {
              dp::math::sse::Mat44f& currentLocalSSE = *reinterpret_cast<dp::math::sse::Mat44f*>(&current.m_localMatrix);
              dp::math::sse::Mat44f& currentWorldSSE = *reinterpret_cast<dp::math::sse::Mat44f*>(&current.m_worldMatrix);
              dp::math::sse::Mat44f const& parentWorldSSE = *reinterpret_cast<dp::math::sse::Mat44f const*>(&curParent.m_worldMatrix);
              currentWorldSSE = currentLocalSSE * parentWorldSSE;
            }
            else
#elif defined(NEON)
              if ( useNEON )
              {
                dp::math::neon::Mat44f& currentLocalNEON = *reinterpret_cast<dp::math::neon::Mat44f*>(&current.m_localMatrix);
                dp::math::neon::Mat44f& currentWorldNEON = *reinterpret_cast<dp::math::neon::Mat44f*>(&current.m_worldMatrix);
                dp::math::neon::Mat44f const& parentWorldNEON = *reinterpret_cast<dp::math::neon::Mat44f const*>(&curParent.m_worldMatrix);
                currentWorldNEON = currentLocalNEON * parentWorldNEON;
              }
              else
#endif
            {
              current.m_worldMatrix = current.m_localMatrix * curParent.m_worldMatrix;
            }

            // world matrix is a mirror transform iff exactly one source matrix is a mirror transform
            current.setWorldBits( TransformTreeNode::ISMIRRORTRANSFORM, 
              !!(  current.m_localBits   & TransformTreeNode::ISMIRRORTRANSFORM 
              ^ curParent.m_worldBits & TransformTreeNode::ISMIRRORTRANSFORM ) );

            // transform has changed, mark the bounding volume of the corresponding node in the object tree as dirty
            DP_ASSERT( current.m_objectTreeIndex != ~0 );
          }
          else
          {
            // update root transform's matrix and bits
            current.m_worldMatrix = current.m_localMatrix;
            current.m_worldBits   = current.m_localBits;

            // no need to mark OT node dirty, this is the root sentinel transform
          }

          m_sceneTree.notifyTransformUpdated( index, current );

          return true;
        };

        void postTraverse( TransformTreeIndex index, const Data& data )
        {
          //m_transformTree[index].m_dirtyBits &= ~(ObjectTreeNode::DEFAULT_DIRTY | ObjectTreeNode::GEONODE_DIRTY);
          m_transformTree[index].m_dirtyBits = 0;
        };

      protected:
        TransformTree& m_transformTree;
        SceneTree& m_sceneTree;
        dp::sg::core::CameraSharedPtr m_camera;
        std::vector<TransformTreeIndex>& m_changedTransforms;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
