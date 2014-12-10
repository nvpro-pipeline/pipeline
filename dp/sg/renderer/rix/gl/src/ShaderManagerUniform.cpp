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


#include "dp/sg/renderer/rix/gl/inc/ShaderManagerUniform.h"
#include <dp/fx/EffectSpec.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/util/Array.h>

using namespace dp::fx;
using namespace dp::math;
using namespace dp::sg::xbar;
using dp::util::array;

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

          /************************************************************************/
          /* ShaderManagerTransformsUniform                                       */
          /************************************************************************/
          ShaderManagerTransformsUniform::ShaderManagerTransformsUniform( SceneTree *sceneTree, const ResourceManagerSharedPtr& resourceManager )
            : m_sceneTree( sceneTree )
            , m_resourceManager( resourceManager )
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            // create world descriptor & container
            dp::rix::core::ProgramParameter programParametersWorld[] = {
              dp::rix::core::ProgramParameter( "sys_WorldMatrices", dp::rix::core::CPT_MAT4X4, 2 )
            };

            m_descriptorTransforms = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( programParametersWorld, sizeof array(programParametersWorld) ) );
          }

          ShaderManagerTransformsUniform::~ShaderManagerTransformsUniform()
          {
          }

          dp::rix::core::ContainerSharedHandle ShaderManagerTransformsUniform::getTransformContainer( dp::sg::xbar::TransformTreeIndex transformIndex )
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            // keep same size as transform tree for a 1-1 mapping of transforms
            if ( m_containerDatas.size() <= transformIndex )
            {
              m_containerDatas.resize( m_sceneTree->getTransformTree().m_tree.size());
            }
            assert( transformIndex < m_containerDatas.size() && "passed invalid transform index");

            if ( !m_containerDatas[transformIndex] )
            {
              m_containerDatas[transformIndex] = renderer->containerCreate( m_descriptorTransforms );
              m_newContainerTransforms.push_back(transformIndex);
            }

            return m_containerDatas[transformIndex];
          }


          inline void updateTransformNode( dp::rix::core::Renderer* renderer, dp::rix::core::ContainerSharedHandle& container, dp::rix::core::ContainerEntry entry, const TransformTreeNode &node )
          {
            // compute world matrices
            Mat44f matrices[2];

            matrices[0] = node.m_worldMatrix;
            matrices[1] = matrices[0];
            matrices[1].invert();
            matrices[1] = ~matrices[1];

            renderer->containerSetData( container, entry, dp::rix::core::ContainerDataRaw( 0, matrices[0].getPtr(), 2 * sizeof(Mat44f) ) );
          }

          void ShaderManagerTransformsUniform::updateTransforms()
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            dp::rix::core::ContainerEntry entry = renderer->containerDescriptorGetEntry( m_descriptorTransforms, "sys_WorldMatrices" );

            std::vector<TransformTreeIndex>::iterator it = m_newContainerTransforms.begin();
            while ( it != m_newContainerTransforms.end() )
            {
              updateTransformNode( renderer, m_containerDatas[*it], entry, m_sceneTree->getTransformTreeNode( *it ) );
              ++it;
            }
            m_newContainerTransforms.clear();

            const std::vector<TransformTreeIndex>& transforms = m_sceneTree->getChangedTransforms();
            std::vector<TransformTreeIndex>::const_iterator it2 = transforms.begin();
            while ( it2 != transforms.end() )
            {
              // update only used transforms
              if ( m_containerDatas[*it2] )
              {
                updateTransformNode( renderer, m_containerDatas[*it2], entry, m_sceneTree->getTransformTreeNode( *it2 ) );
              }
              ++it2;
            }
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
