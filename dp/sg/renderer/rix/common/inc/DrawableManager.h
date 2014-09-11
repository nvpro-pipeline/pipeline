// Copyright NVIDIA Corporation 2011-2012
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
#include <dp/culling/Manager.h>
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
      class RenderList;
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

          class ShaderManagerInstance;
          typedef dp::util::SmartPtr<ShaderManagerInstance> SmartShaderManagerInstance;

          class ShaderManager;

          class DrawableManagerDefault : public dp::sg::xbar::DrawableManager
          {
          public:
            class DefaultHandleData;
            typedef dp::util::SmartPtr<DefaultHandleData> DefaultHandleDataHandle;

            class DefaultHandleData : public HandleData
            {
            public:
              DefaultHandleData( dp::sg::xbar::ObjectTreeIndex index )
                : m_index( index )
              {
              }

              dp::sg::xbar::ObjectTreeIndex m_index;
            };
            
            /** \brief Payload for handle **/
            class Payload : public dp::util::Payload
            {
            public:
              Payload()
                : m_handle( nullptr )
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

            private:
              DrawableManager::Handle m_handle;
            };


            class Instance
            {
            public:
              Instance();

              // Don't forget to modify the copy constructor when adding new variables!
              dp::rix::core::SmartGeometryInstanceHandle  m_geometryInstance;
              dp::rix::core::SmartGeometryInstanceHandle  m_geometryInstanceDepthPass;

              // TODO MTA it would be more efficient to have an weakptr here. this will increase complexity
              // in the observer since the effect data could die before the instance.
              dp::sg::core::EffectDataSharedPtr           m_currentEffectGeometry;
              dp::sg::core::EffectDataSharedPtr           m_currentEffectSurface;

              SmartResourcePrimitive                      m_resourcePrimitive;
              SmartShaderManagerInstance                  m_smartShaderObject;

              bool                          m_isVisible;
              bool                          m_isActive;
              bool                          m_isTraversalActive;
              dp::util::Uint32              m_activeTraversalMask;
              dp::sg::xbar::ObjectTreeIndex m_objectTreeIndex;
              DefaultHandleDataHandle       m_handle;

              // depth sorting
              float                            m_squaredDistance;
              bool                             m_transparent;
              dp::rix::core::RenderGroupHandle m_currentRenderGroup;

              // culling information
              dp::math::Vec4f                  m_boundingBoxLower;
              dp::math::Vec4f                  m_boundingBoxExtent;
              dp::sg::xbar::TransformTreeIndex m_transformIndex;

              dp::culling::ObjectHandle m_cullingObject;

              boost::shared_ptr<Payload> m_payload;
              bool                       m_effectDataAttached;

              void updateRendererVisibility( dp::rix::core::Renderer* renderer );
            };

            DrawableManagerDefault( const SmartResourceManager & resourceManager, SmartTransparencyManager const & transparencyManager
                                  , dp::fx::Manager shaderManagerType = dp::fx::MANAGER_SHADERBUFFER
                                  , dp::culling::Mode cullingMode = dp::culling::MODE_AUTO );
            virtual ~DrawableManagerDefault();

            virtual void setRenderList( dp::sg::xbar::RenderList* renderList );
            virtual Handle addDrawableInstance( dp::sg::core::GeoNodeWeakPtr geoNode, dp::sg::xbar::ObjectTreeIndex objectTreeIndex );
            virtual void removeDrawableInstance( Handle handle );
            virtual void updateDrawableInstance( Handle handle );
            virtual void setDrawableInstanceActive( Handle handle, bool visible );
            virtual void setDrawableInstanceTraversalMask( Handle handle, dp::util::Uint32 traversalMask );

            virtual void setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler );
            virtual const dp::sg::core::SamplerSharedPtr & getEnvironmentSampler() const;

            ShaderManager * getShaderManager() const;

            virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode ) const;

            virtual void update( const nvsg::ViewStateSharedPtr &viewState);
            virtual void update( dp::math::Vec2ui const & viewportSize );

            virtual void cull( const dp::sg::core::CameraSharedPtr &camera );

            dp::rix::core::SmartRenderGroupHandle getRenderGroup() { return m_renderGroup; }
            dp::rix::core::SmartRenderGroupHandle getRenderGroupDepthPass() { return m_renderGroupDepthPass; }
            dp::rix::core::SmartRenderGroupHandle getRenderGroupTransparent() { return m_renderGroupTransparent; }
            std::vector<dp::rix::core::GeometryInstanceHandle>& getSortedTransparentGIs( const dp::math::Vec3f& cameraPosition );
            bool containsTransparentGIs();

            dp::sg::xbar::RenderList* getRenderList( ) const;

            dp::math::Box3f getBoundingBox() const;

            void setCullingEnabled( bool enabled );
            bool isCullingEnabled( ) const;

          protected:
            class TransformObserver : public dp::util::Observer
            {
            public:
              TransformObserver( DrawableManagerDefault* drawableManager );
              ~TransformObserver();

              virtual void onNotify( const dp::util::Event& event, dp::util::Payload* payload );
              virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

            private:
              DrawableManagerDefault* m_drawableManager;
            };

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

            void onTransformChanged( dp::sg::xbar::RenderList::EventTransform const& event );
            void detachEffectDataObserver();

            friend class TransformObserver;

            void cullManager( const dp::sg::core::CameraSharedPtr &camera );
            void setActiveTraversalMask( unsigned int nodeMask );

            dp::fx::Manager                       m_shaderManagerType;
            dp::sg::xbar::RenderList*             m_renderList;
            SmartResourceManager                  m_resourceManager;
            std::vector<Instance>                 m_instances;
            dp::rix::core::SmartRenderGroupHandle m_renderGroup;
            dp::rix::core::SmartRenderGroupHandle m_renderGroupDepthPass;
            dp::rix::core::SmartRenderGroupHandle m_renderGroupTransparent;

            // HACK for RTT demo
            std::vector<DefaultHandleDataHandle>               m_transparentDIs; 
            std::vector<dp::rix::core::GeometryInstanceHandle> m_depthSortedTransparentGIs;

            boost::scoped_ptr<ShaderManager>        m_shaderManager;
            unsigned int                            m_activeTraversalMask;
            boost::scoped_ptr<TransformObserver>    m_transformObserver;
            boost::scoped_ptr<EffectDataObserver>   m_effectDataObserver;

            dp::culling::Mode                       m_cullingMode;
            boost::scoped_ptr<dp::culling::Manager> m_cullingManager;
            dp::culling::GroupHandle                m_cullingGroup;
            dp::culling::ResultHandle               m_cullingResult;
            bool                                    m_cullingEnabled;

          private:
            dp::sg::core::SamplerSharedPtr  m_environmentSampler;
            dp::math::Vec2ui                m_viewportSize;
            SmartTransparencyManager        m_transparencyManager;
          };

          inline dp::sg::xbar::RenderList* DrawableManagerDefault::getRenderList( ) const
          {
            return m_renderList;
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
