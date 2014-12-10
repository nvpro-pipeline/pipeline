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

#pragma once

#include <dp/sg/renderer/rix/gl/inc/DrawableManagerDefault.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceEffectDataRiXFx.h>
#include <dp/rix/fx/Manager.h>
#include <boost/scoped_ptr.hpp>

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

          class ShaderManagerTransformsRiXFx;

          class ShaderManagerRiXFx : public ShaderManager
          {
          public:
            ShaderManagerRiXFx( dp::sg::xbar::SceneTree *sceneTree, dp::fx::Manager managerType, const ResourceManagerSharedPtr& resourceManager, TransparencyManagerSharedPtr const & transparencyManager );

            virtual void updateCameraState( const dp::math::Mat44f& worldToProj, const dp::math::Mat44f& viewToWorld );
            virtual void updateTransforms();
            virtual void updateLights( dp::sg::ui::ViewStateSharedPtr const& vs );

            virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const;

            virtual ShaderManagerRenderGroupSharedPtr registerRenderGroup( dp::rix::core::RenderGroupSharedHandle const & renderGroup );

          protected:
            virtual void addSystemContainers( ShaderManagerInstanceSharedPtr const & shaderObject );
            virtual void addSystemContainers( ShaderManagerRenderGroupSharedPtr const & renderGroup );

            virtual void updateEnvironment( ResourceSamplerSharedPtr environmentSampler );
            virtual void updateFragmentParameter( std::string const & name, dp::rix::core::ContainerDataRaw const & data );

            /*********************************/
            /* Global SceniX Specific values */
            /*********************************/
            ShaderManagerInstanceSharedPtr registerGeometryInstance( const dp::sg::core::EffectDataSharedPtr &effectData
                                                                   , dp::sg::xbar::ObjectTreeIndex objectTreeIndex
                                                                   , dp::rix::core::GeometryInstanceSharedHandle &geometryInstance
                                                                   , RenderPassType rpt );

            ShaderManagerLights                             m_shaderManagerLights;
            boost::scoped_ptr<ShaderManagerTransformsRiXFx> m_shaderManagerTransforms;
            dp::rix::fx::ManagerSharedPtr                   m_rixFxManager;
            dp::rix::fx::Manager::SystemSpecs               m_systemSpecs;

            // camera state
            dp::fx::EffectSpecSharedPtr           m_effectSpecCamera;
            dp::fx::ParameterGroupSpec::iterator  m_itViewProjMatrix;
            dp::fx::ParameterGroupSpec::iterator  m_itViewMatrixI;
            dp::rix::fx::GroupDataSharedHandle    m_groupDataCamera;

          private:
            dp::rix::core::ContainerDescriptorSharedHandle m_descriptorEnvironment;
            dp::rix::core::ContainerDescriptorSharedHandle m_descriptorFragment;
            dp::rix::core::ContainerSharedHandle           m_containerEnvironment;
            dp::rix::core::ContainerSharedHandle           m_containerFragment;
            dp::fx::EffectSpecSharedPtr                    m_fragmentSystemSpec;

            std::map<std::string, dp::rix::core::ProgramHandle> m_mapEffectsToPrograms;

            dp::rix::fx::SourceFragments m_additionalCodeSnippets[RGL_COUNT][RGP_COUNT];    // code snippets for opaque|transparent and color|depth pass
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

