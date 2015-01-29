// Copyright NVIDIA Corporation 2011
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

#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>

namespace dp
{
  namespace sg
  {
    namespace renderer
    {
      namespace rix
      {
        namespace gl
        {

          class ShaderManagerTransformsUniform : public ShaderManagerTransforms
          {
          public:
            ShaderManagerTransformsUniform( dp::sg::xbar::SceneTree* sceneTree, const ResourceManagerSharedPtr& resourceManager );
            virtual ~ShaderManagerTransformsUniform();

            virtual void updateTransforms();
            virtual dp::rix::core::ContainerSharedHandle getTransformContainer( dp::sg::xbar::TransformTreeIndex transformIndex );

            virtual const dp::rix::core::ContainerDescriptorSharedHandle& getDescriptor() { return m_descriptorTransforms; }
          protected:
            dp::rix::core::ContainerDescriptorSharedHandle m_descriptorTransforms;

            //typedef std::map<dp::sg::xbar::TransformTreeIndex, dp::rix::core::ContainerSharedHandle> TransformContainerData;
            typedef std::vector<dp::rix::core::ContainerSharedHandle> TransformContainerData;

            //TransformContainerData                      m_containerTransforms;
            //TransformContainerData                      m_containerDirtyTransforms; // TODO make a vector with observer interface
            TransformContainerData                        m_containerDatas;
            std::vector<dp::sg::xbar::TransformTreeIndex> m_newContainerTransforms;

            dp::sg::xbar::SceneTree * m_sceneTree;
            ResourceManagerSharedPtr  m_resourceManager;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
