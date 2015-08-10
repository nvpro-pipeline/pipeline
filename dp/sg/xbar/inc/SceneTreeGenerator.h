// Copyright NVIDIA Corporation 2010-2012
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

#include <dp/sg/xbar/inc/GeneratorState.h>
#include <dp/sg/xbar/xbar.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/algorithm/ModelViewTraverser.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      class SceneTreeGenerator : public dp::sg::algorithm::SharedTraverser
      {
      public:
        DP_SG_XBAR_API SceneTreeGenerator( SceneTreeSharedPtr const& sceneTree );
        DP_SG_XBAR_API ~SceneTreeGenerator() { /* NOP */ }

        DP_SG_XBAR_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

        // functions for SceneTree update
        DP_SG_XBAR_API void setCurrentTransformTreeData( TransformTreeIndex parentIndex, TransformTreeIndex siblingIndex );
        DP_SG_XBAR_API void setCurrentObjectTreeData( ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex );  

        DP_SG_XBAR_API void addClipPlane( const dp::sg::core::ClipPlaneWeakPtr& clipPlane );

      protected:  
        DP_SG_XBAR_API virtual bool preTraverseGroup( const dp::sg::core::Group *p );
        DP_SG_XBAR_API virtual void postTraverseGroup( const dp::sg::core::Group *p );

        DP_SG_XBAR_API virtual void handleTransform( const dp::sg::core::Transform * p );

        DP_SG_XBAR_API virtual void handleBillboard( const dp::sg::core::Billboard *p );

        DP_SG_XBAR_API virtual void handleLOD( const dp::sg::core::LOD *p );
        DP_SG_XBAR_API virtual void handleSwitch( const dp::sg::core::Switch *p );  

        DP_SG_XBAR_API virtual void handleGeoNode( const dp::sg::core::GeoNode *p );

        DP_SG_XBAR_API virtual void handleLightSource( const dp::sg::core::LightSource * p );

      private:
        SceneTreeWeakPtr       m_sceneTree;
        GeneratorStateSharedPtr m_generatorState;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
