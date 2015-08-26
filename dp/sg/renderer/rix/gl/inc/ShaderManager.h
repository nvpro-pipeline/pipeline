// Copyright NVIDIA Corporation 2011-2015
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

#include <dp/sg/renderer/rix/gl/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceSampler.h>
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>
#include <dp/fx/EffectDefs.h>
#include <dp/fx/EffectSpec.h>
#include <vector>

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

          // LightState 
          #define MAXLIGHTS 128

          enum RenderPassType
          {
            RPT_DEPTH,
            RPT_FORWARD
          };

          DEFINE_PTR_TYPES( ShaderManagerInstance );

          class ShaderManagerInstance
          {
          public:
            ShaderManagerInstanceSharedPtr create();
            virtual ~ShaderManagerInstance();

          protected:
            ShaderManagerInstance();

          public:
            dp::rix::core::GeometryInstanceSharedHandle      geometryInstance;
            dp::sg::core::GeoNodeWeakPtr          geoNode;
            dp::sg::xbar::ObjectTreeIndex         objectTreeIndex;
            bool                                  isTransparent;
          };


          DEFINE_PTR_TYPES( ShaderManagerRenderGroup );

          class ShaderManagerRenderGroup
          {
          public:
            static ShaderManagerRenderGroupSharedPtr create();
            virtual ~ShaderManagerRenderGroup();

          protected:
            ShaderManagerRenderGroup();

          public:
            dp::rix::core::RenderGroupSharedHandle renderGroup;
          };

          class ShaderManager
          {
          public:
            ShaderManager( dp::sg::xbar::SceneTreeSharedPtr const& sceneTree, const ResourceManagerSharedPtr& resourceManager, TransparencyManagerSharedPtr const & transparencyManager );
            virtual ~ShaderManager() {};

            void setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler );
            const dp::sg::core::SamplerSharedPtr & getEnvironmentSampler() const;

            TransparencyManagerSharedPtr const & getTransparencyManager() const;

            virtual void updateFragmentParameter( std::string const & name, dp::rix::core::ContainerDataRaw const & data ) = 0;
            void update( dp::sg::ui::ViewStateSharedPtr const& vs );

            virtual void updateTransforms() = 0;

            ShaderManagerInstanceSharedPtr registerGeometryInstance( dp::sg::core::GeoNodeSharedPtr const & geoNode,
                                                                     dp::sg::xbar::ObjectTreeIndex objectTreeIndex,
                                                                     dp::rix::core::GeometryInstanceSharedHandle & geometryInstance,
                                                                     RenderPassType rpt = RPT_FORWARD );

            virtual ShaderManagerRenderGroupSharedPtr registerRenderGroup( dp::rix::core::RenderGroupSharedHandle const & renderGroup ) = 0;

            const dp::sg::core::EffectDataSharedPtr& getDefaultEffectData() const { return m_defaultEffectData; }

            virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const = 0;

          protected:
            virtual void updateEnvironment( ResourceSamplerSharedPtr environmentSampler ) = 0;
            virtual void updateLights( dp::sg::ui::ViewStateSharedPtr const& vs ) = 0;
            virtual void updateCameraState( dp::math::Mat44f const & worldToProj, dp::math::Mat44f const & viewToWorld ) = 0;

            virtual void addSystemContainers( ShaderManagerInstanceSharedPtr const & shaderObject ) = 0;
            virtual void addSystemContainers( ShaderManagerRenderGroupSharedPtr const & renderGroup ) = 0;

            virtual ShaderManagerInstanceSharedPtr registerGeometryInstance( const dp::sg::core::EffectDataSharedPtr &effectData,
                                                                             dp::sg::xbar::ObjectTreeIndex objectTreeIndex,
                                                                             dp::rix::core::GeometryInstanceSharedHandle &geometryInstance,
                                                                             RenderPassType rpt );

            dp::sg::xbar::SceneTreeWeakPtr m_sceneTree;
            dp::rix::core::Renderer* m_renderer;
            ResourceManagerSharedPtr m_resourceManager;

            dp::sg::core::EffectDataSharedPtr m_defaultEffectData;

            dp::rix::core::ProgramHandle m_programFixedFunction; // TODO does this belong to ResourceEffectSpec somehow?

          private:
            bool                            m_environmentNeedsUpdate;
            dp::sg::core::SamplerSharedPtr  m_environmentSampler;
            ResourceSamplerSharedPtr        m_environmentResourceSampler;
            TransparencyManagerSharedPtr    m_transparencyManager;
          };

          class ShaderManagerLights
          {
          public:
            struct LightInformation
            {
              LightInformation()
              {
              }

              LightInformation( const LightInformation &rhs )
                : m_container( rhs.m_container )
                , m_buffer( rhs.m_buffer )
              {
              }

              LightInformation& operator=( const LightInformation& rhs )
              {
                if( this != &rhs )
                {
                  m_container = rhs.m_container;
                  m_buffer = rhs.m_buffer;
                }
                return *this;
              }

              dp::rix::core::ContainerSharedHandle m_container;
              dp::rix::core::BufferSharedHandle    m_buffer;
            };

          public:
            ShaderManagerLights( dp::sg::xbar::SceneTreeSharedPtr const& sceneTree, const ResourceManagerSharedPtr& resourceManager );
            virtual ~ShaderManagerLights();

            void updateLights( dp::sg::ui::ViewStateSharedPtr const& vs );
            LightInformation getLightInformation();
            const dp::rix::core::ContainerDescriptorSharedHandle& getDescriptor() const { return m_descriptorLight; }

          protected:
            dp::sg::xbar::SceneTreeWeakPtr                    m_sceneTree;
            dp::rix::core::ContainerDescriptorSharedHandle    m_descriptorLight;
            ResourceManagerSharedPtr                          m_resourceManager;
            LightInformation                                  m_lightInformation;
          };

          class ShaderManagerTransforms
          {
          public:
            virtual ~ShaderManagerTransforms();

            virtual void updateTransforms() = 0;
            virtual dp::rix::core::ContainerSharedHandle getTransformContainer(dp::sg::xbar::TransformIndex transformIndex) = 0;

            virtual const dp::rix::core::ContainerDescriptorSharedHandle& getDescriptor() = 0;
          protected:
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

