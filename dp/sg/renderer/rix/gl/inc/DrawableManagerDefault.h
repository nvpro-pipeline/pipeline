// Copyright (c) 2011-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/xbar/DrawableManager.h>
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourcePrimitive.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>
#include <dp/sg/xbar/culling/Culling.h>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_array.hpp>
#include <dp/math/Boxnt.h>

#include <vector>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      class SceneTree;
    }
  }
}

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

          enum RenderGroupLayer
          {
            RGL_OPAQUE,
            RGL_TRANSPARENT,
            RGL_COUNT
          };

          enum RenderGroupPass
          {
            RGP_FORWARD,
            RGP_DEPTH,
            RGP_COUNT
          };

          class ShaderManager;

          class DrawableManagerDefault : public dp::sg::xbar::DrawableManager
          {
          public:
            DEFINE_PTR_TYPES( DefaultHandleData );

            class DefaultHandleData : public HandleData
            {
            public:
              static DefaultHandleDataSharedPtr create( dp::sg::xbar::ObjectTreeIndex index )
              {
                return( std::shared_ptr<DefaultHandleData>( new DefaultHandleData( index ) ) );
              }

            public:
              dp::sg::xbar::ObjectTreeIndex m_index;

            protected:
              DefaultHandleData( dp::sg::xbar::ObjectTreeIndex index )
                : m_index( index )
              {
              }
            };

            /** \brief Payload for handle **/
            DEFINE_PTR_TYPES( Payload );
            class Payload : public dp::util::Payload
            {
            public:
              static PayloadSharedPtr create()
              {
                return( std::shared_ptr<Payload>( new Payload() ) );
              }

              virtual ~Payload()
              {
              }

              void setHandle( DrawableManager::Handle handle )
              {
                m_handle = handle;
              }

              DrawableManager::Handle getHandle() const
              {
                return m_handle;
              }

            protected:
              Payload()
                : m_handle( nullptr )
              {
              }

            private:
              DrawableManager::Handle m_handle;
            };


            class Instance
            {
            public:
              Instance();

              dp::sg::core::GeoNodeSharedPtr               m_geoNode;

              // Don't forget to modify the copy constructor when adding new variables!
              dp::rix::core::GeometryInstanceSharedHandle  m_geometryInstance;
              dp::rix::core::GeometryInstanceSharedHandle  m_geometryInstanceDepthPass;

              // TODO MTA it would be more efficient to have an weakptr here. this will increase complexity
              // in the observer since the effect data could die before the instance.
              dp::sg::core::PipelineDataSharedPtr          m_currentPipelineData;

              ResourcePrimitiveSharedPtr                  m_resourcePrimitive;
              ShaderManagerInstanceSharedPtr              m_smartShaderObject;
              ShaderManagerInstanceSharedPtr              m_smartShaderObjectDepthPass;

              bool                          m_isVisible;
              bool                          m_isActive;
              bool                          m_isTraversalActive;
              uint32_t                      m_activeTraversalMask;
              dp::sg::xbar::ObjectTreeIndex m_objectTreeIndex;
              DefaultHandleDataSharedPtr    m_handle;

              // depth sorting
              float                            m_squaredDistance;
              bool                             m_transparent;
              dp::rix::core::RenderGroupHandle m_currentRenderGroup;

              // culling information
              dp::math::Vec4f                  m_boundingBoxLower;
              dp::math::Vec4f                  m_boundingBoxExtent;
              dp::sg::xbar::TransformIndex     m_transformIndex;

              PayloadSharedPtr  m_payload;
              bool              m_effectDataAttached;

              void updateRendererVisibility( dp::rix::core::Renderer* renderer );
            };

            DrawableManagerDefault( const ResourceManagerSharedPtr & resourceManager
                                  , TransparencyManagerSharedPtr const & transparencyManager
                                  , dp::fx::Manager shaderManagerType = dp::fx::Manager::SHADERBUFFER
                                  , dp::culling::Mode cullingMode = dp::culling::Mode::AUTO );
            virtual ~DrawableManagerDefault();

            virtual Handle addDrawableInstance( dp::sg::core::GeoNodeWeakPtr geoNode, dp::sg::xbar::ObjectTreeIndex objectTreeIndex );
            virtual void removeDrawableInstance( Handle handle );
            virtual void updateDrawableInstance( Handle handle );
            virtual void setDrawableInstanceActive( Handle handle, bool visible );
            virtual void setDrawableInstanceTraversalMask( Handle handle, uint32_t traversalMask );

            virtual void setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler );
            virtual const dp::sg::core::SamplerSharedPtr & getEnvironmentSampler() const;

            ShaderManager * getShaderManager() const;

            virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const;

            virtual void update( dp::sg::ui::ViewStateSharedPtr const& viewState);
            virtual void update( dp::math::Vec2ui const & viewportSize );

            virtual void cull( const dp::sg::core::CameraSharedPtr &camera );

            dp::rix::core::RenderGroupSharedHandle getRenderGroup() { return m_renderGroups[RGL_OPAQUE][RGP_FORWARD]; }
            dp::rix::core::RenderGroupSharedHandle getRenderGroupDepthPass() { return m_renderGroups[RGL_OPAQUE][RGP_DEPTH]; }
            dp::rix::core::RenderGroupSharedHandle getRenderGroupTransparent() { return m_renderGroups[RGL_TRANSPARENT][RGP_FORWARD]; }
            dp::rix::core::RenderGroupSharedHandle getRenderGroupTransparentDepthPass() { return m_renderGroups[RGL_TRANSPARENT][RGP_DEPTH]; }
            std::vector<dp::rix::core::GeometryInstanceSharedHandle>& getSortedTransparentGIs( const dp::math::Vec3f& cameraPosition );
            bool containsTransparentGIs();

            dp::math::Box3f getBoundingBox() const;

            void setCullingEnabled( bool enabled );
            bool isCullingEnabled( ) const;

          protected:
            class EffectDataObserver : public dp::util::Observer
            {
            public:
              EffectDataObserver( DrawableManagerDefault *drawableManager );
              ~EffectDataObserver();

              virtual void onNotify( const dp::util::Event& event, dp::util::Payload* payload );
              virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

            private:
              DrawableManagerDefault* m_drawableManager;
            };

            void onTransformChanged( dp::sg::xbar::TransformTree::EventTransform const& event );
            void detachEffectDataObserver();
            virtual void onSceneTreeChanged();

            friend class TransformObserver;

            void cullManager( const dp::sg::core::CameraSharedPtr &camera );
            void setActiveTraversalMask( unsigned int nodeMask );

            dp::fx::Manager                         m_shaderManagerType;
            ResourceManagerSharedPtr                m_resourceManager;
            std::vector<Instance>                   m_instances;

            //  m_renderGroup(Instance)s[Opaque|Transparent][Forward|DepthPass]
            dp::rix::core::RenderGroupSharedHandle  m_renderGroups[RGL_COUNT][RGP_COUNT];
            ShaderManagerRenderGroupSharedPtr       m_renderGroupInstances[RGL_COUNT][RGP_COUNT];

            std::vector<DefaultHandleDataSharedPtr>                   m_transparentDIs;
            std::vector<dp::rix::core::GeometryInstanceSharedHandle>  m_depthSortedTransparentGIs;

            boost::scoped_ptr<ShaderManager>        m_shaderManager;
            unsigned int                            m_activeTraversalMask;
            boost::scoped_ptr<EffectDataObserver>   m_effectDataObserver;

            dp::culling::Mode                       m_cullingMode;
            dp::sg::xbar::culling::CullingSharedPtr m_cullingManager;
            dp::sg::xbar::culling::ResultSharedPtr  m_cullingResult;
            bool                                    m_cullingEnabled;

          private:
            dp::sg::core::SamplerSharedPtr  m_environmentSampler;
            dp::math::Vec2ui                m_viewportSize;
            TransparencyManagerSharedPtr    m_transparencyManager;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
