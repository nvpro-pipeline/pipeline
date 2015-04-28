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


#include <dp/sg/ui/ViewState.h>
#include <dp/sg/ui/RendererOptions.h>
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/ui/RendererOptions.h>

using namespace dp::math;
using namespace dp::sg::core;

using std::max;
using std::make_pair;
using std::pair;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      DEFINE_STATIC_PROPERTY( ViewState, AutoClipPlanes );
      DEFINE_STATIC_PROPERTY( ViewState, TargetDistance );
      DEFINE_STATIC_PROPERTY( ViewState, StereoAutomaticEyeDistanceFactor );
      DEFINE_STATIC_PROPERTY( ViewState, StereoEyeDistance );
      DEFINE_STATIC_PROPERTY( ViewState, LODRangeScale );
      DEFINE_STATIC_PROPERTY( ViewState, TraversalMask );
      DEFINE_STATIC_PROPERTY( ViewState, StereoAutomaticEyeDistanceAdjustment );
      DEFINE_STATIC_PROPERTY( ViewState, StereoReversedEyes );


      BEGIN_REFLECTION_INFO( ViewState )
        DERIVE_STATIC_PROPERTIES( ViewState, Reflection );

        INIT_STATIC_PROPERTY_RW( ViewState, AutoClipPlanes,                       bool,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW( ViewState, TargetDistance,                       float,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW( ViewState, StereoAutomaticEyeDistanceFactor,     float,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW( ViewState, StereoEyeDistance,                    float,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW( ViewState, LODRangeScale,                        float,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW( ViewState, TraversalMask,                        unsigned int,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW_BOOL( ViewState, StereoAutomaticEyeDistanceAdjustment, bool,    SEMANTIC_VALUE,  value,   value);
        INIT_STATIC_PROPERTY_RW_BOOL( ViewState, StereoReversedEyes,                   bool,    SEMANTIC_VALUE,  value,   value);
      END_REFLECTION_INFO

      ViewStateSharedPtr ViewState::create()
      {
        return( std::shared_ptr<ViewState>( new ViewState() ) );
      }

      dp::sg::core::HandledObjectSharedPtr ViewState::clone() const
      {
        return( std::shared_ptr<ViewState>( new ViewState( *this ) ) );
      }

      ViewState::ViewState(void)
      : m_autoClipPlanes(true)
      , m_camera(NULL)
      , m_targetDistance(0.0f)
      , m_stereoAutomaticEyeDistanceAdjustment(true)
      , m_stereoAutomaticEyeDistanceFactor(0.03f)
      , m_stereoEyeDistance(0.0f)
      , m_reversedEyes(false)
      , m_scaleLODRange(1.f)
      , m_traversalMask(~0)
      {
      }

      ViewState::ViewState(const ViewState& rhs)
      : m_autoClipPlanes(rhs.m_autoClipPlanes)
      , m_targetDistance(rhs.m_targetDistance)
      , m_stereoAutomaticEyeDistanceAdjustment(rhs.m_stereoAutomaticEyeDistanceAdjustment)
      , m_stereoAutomaticEyeDistanceFactor(rhs.m_stereoAutomaticEyeDistanceFactor)
      , m_stereoEyeDistance(rhs.m_stereoEyeDistance)
      , m_reversedEyes(rhs.m_reversedEyes)
      , m_scaleLODRange(rhs.m_scaleLODRange)
      , m_traversalMask(rhs.m_traversalMask)
      , m_rendererOptions(rhs.m_rendererOptions)
      {
        if ( rhs.m_camera )
        {
          m_camera = rhs.m_camera.clone();
        }
        if ( rhs.m_sceneTree )
        {
          m_sceneTree = rhs.getSceneTree();
        }
      }

      ViewState &ViewState::operator=(const ViewState& rhs)
      {
        m_autoClipPlanes = rhs.m_autoClipPlanes;
        m_camera = rhs.m_camera;
        m_targetDistance = rhs.m_targetDistance;
        m_stereoAutomaticEyeDistanceAdjustment = rhs.m_stereoAutomaticEyeDistanceAdjustment;
        m_stereoAutomaticEyeDistanceFactor = rhs.m_stereoAutomaticEyeDistanceFactor;
        m_stereoEyeDistance = rhs.m_stereoEyeDistance;
        m_reversedEyes = rhs.m_reversedEyes;
        m_scaleLODRange = rhs.m_scaleLODRange;
        m_sceneTree = rhs.m_sceneTree;
        m_rendererOptions = rhs.m_rendererOptions;
        m_traversalMask = rhs.m_traversalMask;

        return *this;
      }

      ViewState::~ViewState(void)
      {
      }

      const dp::sg::core::CameraSharedPtr & ViewState::getCamera() const
      {
        return( m_camera );
      }

      void ViewState::setCamera( const dp::sg::core::CameraSharedPtr & pCamera )
      {
        if ( m_camera != pCamera )
        {
          m_camera = pCamera;
          if ( m_camera )
          {
            if ( fabsf( m_targetDistance ) < FLT_EPSILON )
            {
              m_targetDistance = m_camera->getFocusDistance();
            }
          }
          notify( dp::util::Event( ) );
        }
      }

      const SceneSharedPtr & ViewState::getScene( ) const
      {
        return m_sceneTree ? m_sceneTree->getScene() : dp::sg::core::SceneSharedPtr::null;
      }

      void ViewState::setSceneTree(dp::sg::xbar::SceneTreeSharedPtr const& sceneTree)
      {
        if (sceneTree != m_sceneTree)
        {
          m_sceneTree = sceneTree;
          notify( dp::util::Event( ) );
        }
      }

      dp::sg::xbar::SceneTreeSharedPtr const& ViewState::getSceneTree() const
      {
        return m_sceneTree;
      }

      void ViewState::setRendererOptions(const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions)
      {
        m_rendererOptions = rendererOptions;
      }

      const dp::sg::ui::RendererOptionsSharedPtr &ViewState::getRendererOptions( ) const
      {
        return m_rendererOptions;
      }

      bool setupDefaultViewState( dp::sg::ui::ViewStateSharedPtr const& viewState )
      {
        if (!viewState)
        {
          return false;
        }

        dp::sg::core::CameraSharedPtr   camera   = viewState->getCamera();
        dp::sg::core::SceneSharedPtr    scene    = viewState->getScene();;
        dp::sg::ui::RendererOptionsSharedPtr  options  = viewState->getRendererOptions();

        // the viewstate does not have an active camera, set one
        if ( !camera )
        {
          if( !scene )
          {
            return false;
          }

          // if there are cameras in the scene choose the first one
          if ( scene->getNumberOfCameras() )
          {
            camera = *scene->beginCameras();
          }
          else
          {
            // otherwise create a new one
            camera = PerspectiveCamera::create();
            {
              camera->setName( "ViewCamera" );

              // Make scene fit into the viewport.
              if (scene->getRootNode())
              {
                Sphere3f sphere( scene->getBoundingSphere() );
                if ( isPositive(sphere) )
                {
                  camera.staticCast<dp::sg::core::PerspectiveCamera>()->zoom( sphere, float(dp::math::PI_QUARTER) );
                }
              }
            }
          }
          viewState->setCamera( camera );
        }

        // a viewstate needs a valid renderer options object
        if( !options )
        {
          viewState->setRendererOptions( dp::sg::ui::RendererOptions::create() );
        }    

        return true;
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
