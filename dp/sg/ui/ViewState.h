// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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
/** \file */

#include <dp/sg/ui/Config.h>
#include <dp/sg/ui/RendererOptions.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/HandledObject.h>
#include <dp/sg/xbar/SceneTree.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      DEFINE_PTR_TYPES( ViewState );

      /*! \brief The ViewState class encapsulates view-specific state.
       *  \par Namespace: dp::sg::core
       *  \remarks The ViewState encapsulates the view-specific information into one object.
       *  The ViewState defines how the user looks at the scene. It contains the camera and specific viewing
       *  settings like stereo view and does the bookkeeping of animation information.
       *  \n\n
       *  dp::sg::ui::ViewState and dp::sg::core::Scene and dp::sg::ui::RendererOptions are the three major components that need
       *  to come together to produce a rendered image.
       *  The scene contains basically the tree, the ViewState defines how to look at
       *  the tree, and the RendererOptions contains the parameters for the renderer which renders
       *  the final image.
       */
      class ViewState : public dp::sg::core::HandledObject
      {
        public:
          DP_SG_UI_API static ViewStateSharedPtr create();

          DP_SG_UI_API virtual dp::sg::core::HandledObjectSharedPtr clone() const;

          DP_SG_UI_API virtual ~ViewState();

        public:
          /*! \brief Make a flat copy of another ViewState.
           *  \param rhs Source ViewState. */
          DP_SG_UI_API ViewState& operator=( const ViewState& rhs );

          /*! \brief Sets the Camera for this view.
           *  \param camera The Camera to set.
           *  \remarks The given camera is used for the rendering.
           *  \sa getCamera */
          DP_SG_UI_API void setCamera(const dp::sg::core::CameraSharedPtr & camera);

          /*! \brief Sets the RendererOptions for this view
              \param rendererOptions The RendererOptions to set
          */
          DP_SG_UI_API void setRendererOptions(const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions);

          /*! \brief Gets the RendererOptions for this view
              \param rendererOptions The RendererOptions to set
          */
          DP_SG_UI_API const dp::sg::ui::RendererOptionsSharedPtr &getRendererOptions( ) const;

          /*! \brief Returns the current Camera.
           *  \return The Camera which is used in this view.
           *  If there is no Camera associated with this view, the function returns NULL.
           *  \sa setCamera */
          DP_SG_UI_API const dp::sg::core::CameraSharedPtr & getCamera() const;

          /*! \brief Get the distance to the target point.
           *  \return The distance from the current view point to the target point.
           *  \remarks The target point is    as the reference point for rotations of the camera.
           *  \sa setTargetDistance*/
          DP_SG_UI_API float getTargetDistance() const;

          /*! \brief Set the distance to the target point.
           *  \param dist distance between the view point (camera position) and the target point.
           *  \remarks The target point is used as the reference point for rotations of the camera.
           *  \sa getTargetDistance*/
          DP_SG_UI_API void setTargetDistance(float dist);

          /*! \brief Test on automatic eye distance calculation.
           *  \return This function returns true when the distance is automatically calculated.
           *  \remarks This function can be used to test if the automatic eye distance
           *  calculation for stereo is turned on. The automatic eye distance is calculated by
           *  multiplying the adjustment factor by the focus distance of the camera.
           *  \sa setStereoAutomaticEyeDistanceAdjustment, getStereoAutomaticEyeDistanceFactor,
           *  setStereoAutomaticEyeDistanceFactor */
          DP_SG_UI_API bool isStereoAutomaticEyeDistanceAdjustment() const;

          /*! \brief Enable/Disable automatic eye distance adjustment.
           *  \param state Pass in true to enable automatic eye distance calculation.
           *  \remarks The automatic eye distance is calculated by multiplying
           *  the adjustment factor by the focus distance of the camera.
           *  \sa isStereoAutomaticEyeDistanceAdjustment, getStereoAutomaticEyeDistanceFactor,
           *  setStereoAutomaticEyeDistanceFactor */
          DP_SG_UI_API void setStereoAutomaticEyeDistanceAdjustment(bool state);

          /*! \brief Get the automatic eye distance adjustment factor.
           *  \return This function returns the eye distance adjustment factor.
           *  \remarks The automatic eye distance is calculated by multiplying
           *  the adjustment factor by the focus distance of the camera.\n
           *  \par Example
           *  \sa setStereoAutomaticEyeDistanceAdjustment, getStereoAutomaticEyeDistanceFactor,
           *  setStereoAutomaticEyeDistanceFactor */
          DP_SG_UI_API float getStereoAutomaticEyeDistanceFactor() const;

          /*! \brief Set the automatic eye distance adjustment factor.
           *  \param factor Distance factor.
           *  \remarks The automatic eye distance is calculated by multiplying
           *  the adjustment factor by the focus distance of the camera.\n
           *  The default value is 0.03 (3%). This value represents the following setup:\n
           *  A person with an eye distance of about six cm sitting in front of the monitor,
           *  where the monitor is about one meter away. This setup leads to very natural stereo
           *  impression.\n
           *  \sa isStereoAutomaticEyeDistanceAdjustment, getStereoAutomaticEyeDistanceFactor,
           *  setStereoAutomaticEyeDistanceFactor */
          DP_SG_UI_API void setStereoAutomaticEyeDistanceFactor(float factor);

          /*! \brief Get the eye distance for stereo rendering.
           *  \return This function returns the eye distance. If the camera of this ViewState is not
           *  valid, the behavior is undefined.
           *  \remarks The eye distance can be automatically calculated or manually set by the
           *  application. Make sure that a valid camera is defined when asking for the eye distance
           *  since the the automatic eye distance calculation is based on the focus distance of
           *  the camera of this ViewState.
           *  \sa setStereoEyeDistance,
           *  isStereoAutomaticEyeDistanceAdjustment, isStereoAutomaticEyeDistanceAdjustment,
           *  getStereoAutomaticEyeDistanceFactor,  setStereoAutomaticEyeDistanceFactor */
          DP_SG_UI_API float getStereoEyeDistance() const;

          /*! \brief Set the eye distance for stereo rendering.
           *  \param distance Distance between the left and the right eye.
           *  \remarks This function manually sets the eye distance for stereo rendering.
           *  \sa setStereoEyeDistance,
           *  isStereoAutomaticEyeDistanceAdjustment, isStereoAutomaticEyeDistanceAdjustment,
           *  getStereoAutomaticEyeDistanceFactor,  setStereoAutomaticEyeDistanceFactor */
          DP_SG_UI_API void setStereoEyeDistance(float distance);

          /*! \brief Function to reverse the left and the right eye for stereo rendering.
           *  \param state \c true puts the image for the left eye onto the right eye framebuffer and vice versa.
           *  The default state is \c false.
           *  \sa isStereoReversedEyes */
          DP_SG_UI_API void setStereoReversedEyes( bool state );

          /*! \brief Test on reversed eyes in stereo rendering.
           *  \return This function returns true when the left and the right eye are reversed.
           *  \remarks If the eyes are reversed you will see the image for the left eye on the
           *  right eye and vice versa.
           *  \sa setStereoReversedEyes */
          DP_SG_UI_API bool isStereoReversedEyes() const;

          /*! \brief Set the LOD range scale factor.
           *  \param factor The factor to scale the LOD scale ranges. The default value is 1.f,
           *  so the unscaled ranges are used.
           *  \remarks This function sets a scaling factor for LOD ranges of the LOD nodes in the tree.\n
           *  The scale factor can be used to globally scale the ranges without changing the LOD node
           *  ranges directly. This can be used, for example, for scenes that were initially created for viewing
           *  on small monitors. You can use this scaling to fine-tune these scenes for large projection
           *  walls by scaling the LOD levels to switch later to a lower resolution representation.
           *  \sa LOD::getLODToUse, getLODRangeScale */
          DP_SG_UI_API void setLODRangeScale(float factor);

          //! Get the LOD range scale factor
          /** Default value is 1.0, so the unscaled ranges are used. */
          /*! \brief Get the LOD range scale factor.
           *  \return The LOD range scale factor for all the LOD nodes in the tree.
           *  By default this factor is 1.f, so unscaled ranges of the LOD node will be used.
           *  \remarks The scale factor can be used to globally scale the ranges without changing the LOD node
           *  ranges directly. This can be used, for example, for scenes that were initially created for viewing
           *  on small monitors. You can use this scaling to fine-tune these scenes for large projection
           *  walls by scaling the LOD levels to switch later to a lower resolution representation.
           *  \sa LOD::getLODToUse, setLODRangeScale */
          DP_SG_UI_API float getLODRangeScale() const;

          /*! \brief Set a TraversalMask to be used with this ViewState.
           *  \param mask The mask to be used.
           *  \remarks The traversal mask is used in conjuction with Traverser-derived and Renderer-derived objects to
           *  determine whether nodes in the scene (and therefore possibly the entire subgraph) are traversed, and/or rendered.  Traversers
           *  and renderers will use the traversal mask stored in the ViewState along with their TraversalMaskOverride's.  See
           *  Traverser::setTraversalMask or SceneRenderer::setTraversalMask for more information.
           *  \note The default traversal mask is ~0 so that all objects will be traversed/rendered.  Setting the traversal mask to 0
           *  will cause no nodes to be traversed/rendered.
           *  \sa getTraversalMask, Object::setTraversalMask, Traverser::setTraversalMask, SceneRenderer::setTraversalMask */
          DP_SG_UI_API void setTraversalMask( unsigned int mask );

          /*! \brief Returns the current traversal mask
           *  \return The current traversal mask.
           *  \sa setTraversalMask */
          unsigned int getTraversalMask() const;

          /*! \brief Returns the current Scene of the ViewState
           *  \return The Scene of this ViewState.
           *  \deprecated
          */
          DP_SG_UI_API dp::sg::core::SceneSharedPtr getScene() const;

          /*! \brief Set the dp::sg::xbar::SceneTree of the ViewState
          *   \param sceneTree The new SceneTree.
              \remarks This function also updates the Scene of the ViewState.
          **/
          DP_SG_UI_API void setSceneTree(dp::sg::xbar::SceneTreeSharedPtr const& sceneTree);

          /*! \brief Returns the current SceneTree of the ViewState.
           *  \return The SceneTree of this ViewState.
          */
          DP_SG_UI_API dp::sg::xbar::SceneTreeSharedPtr const& getSceneTree() const;

          /*! \brief Query if automatic clip plane adjustment is on.
           *  \return \c true, if automatic clip plane adjustment is on, otherwise \c false.
           *  \sa setAutoClipPlanes, getCamera */
          DP_SG_UI_API bool getAutoClipPlanes() const;

          /*! \brief Enable/Disable automatic clip plane adjustment.
           *  \param enable \c true enables automatic clip plane adjustment, \c false disables it.
           *  \remarks When automatic clip plane adjustment is \c true, on calling getCamera the clip planes of this
           *  ViewState's camera are adjusted to ensure the whole depth of the scene is visible. If the user has
           *  additional information about the scene that might help clipping, this should be turned off.
           *  The default value is \c true.
           *  \sa getAutoClipPlanes, getCamera */
          DP_SG_UI_API void setAutoClipPlanes( bool enable );

        // reflected properties
        public:
          REFLECTION_INFO_API( DP_SG_UI_API, ViewState );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( AutoClipPlanes );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( TargetDistance );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( StereoAutomaticEyeDistanceFactor );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( StereoEyeDistance );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( LODRangeScale );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( TraversalMask );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( StereoAutomaticEyeDistanceAdjustment );
              DP_SG_UI_API DECLARE_STATIC_PROPERTY( StereoReversedEyes );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          /*! \brief Default-constructs a ViewState.
           *  \remarks After instantiation, the ViewState has the following properties:
           *   - no animation running
           *   - no camera defined
           *   - no stereo
           *   - no LOD range scaling
           *   - no cull information */
          DP_SG_UI_API ViewState();

          /*! \brief Copy-constructs a ViewState from another ViewState.
           *  \param rhs Source ViewState. */
          DP_SG_UI_API ViewState( const ViewState& rhs );

        private:
          bool                                 m_autoClipPlanes;  //!< automatically adjust the clip planes for the camera
          dp::sg::core::CameraSharedPtr        m_camera;          //!< this camera renders the tree
          dp::sg::ui::RendererOptionsSharedPtr m_rendererOptions; //!< RenderOptions object for SceneRenderers
          unsigned int                         m_traversalMask;   //!< Current Traversal mask

          float                         m_targetDistance;
          bool                          m_stereoAutomaticEyeDistanceAdjustment;
          float                         m_stereoAutomaticEyeDistanceFactor;
          float                         m_stereoEyeDistance;
          bool                          m_reversedEyes;
          float                         m_scaleLODRange;

          dp::sg::xbar::SceneTreeSharedPtr m_sceneTree; //!< SceneTree to render
      };

      inline float ViewState::getTargetDistance() const
      {
        return( m_targetDistance );
      }

      inline void ViewState::setTargetDistance( float dist )
      {
        if ( dist != m_targetDistance )
        {
          m_targetDistance = dist;
          notify( PropertyEvent( this, PID_TargetDistance ) );
        }
      }

      inline bool ViewState::isStereoAutomaticEyeDistanceAdjustment() const
      {
        return( m_stereoAutomaticEyeDistanceAdjustment );
      }

      inline void ViewState::setStereoAutomaticEyeDistanceAdjustment( bool state )
      {
        if ( m_stereoAutomaticEyeDistanceAdjustment != state )
        {
          m_stereoAutomaticEyeDistanceAdjustment = state;
          notify( PropertyEvent( this, PID_StereoAutomaticEyeDistanceAdjustment ) );
        }
      }

      inline float ViewState::getStereoAutomaticEyeDistanceFactor() const
      {
        return( m_stereoAutomaticEyeDistanceFactor );
      }

      inline void ViewState::setStereoAutomaticEyeDistanceFactor( float factor )
      {
        if ( factor != m_stereoAutomaticEyeDistanceFactor )
        {
          m_stereoAutomaticEyeDistanceFactor = factor;
          notify( PropertyEvent( this, PID_StereoAutomaticEyeDistanceFactor ) );
        }
      }

      inline float ViewState::getStereoEyeDistance() const
      {
        if ( m_stereoAutomaticEyeDistanceAdjustment && m_camera )
        {
          return(m_stereoAutomaticEyeDistanceFactor * m_camera->getFocusDistance());
        }
        else
        {
          return(m_stereoEyeDistance);
        }
      }

      inline void ViewState::setStereoEyeDistance( float distance )
      {
        if ( distance != m_stereoEyeDistance )
        {
          m_stereoEyeDistance = distance;
          notify( PropertyEvent( this, PID_StereoEyeDistance ) );
        }
      }

      inline void ViewState::setStereoReversedEyes( bool state )
      {
        if ( m_reversedEyes != state )
        {
          m_reversedEyes = state;
          notify( PropertyEvent( this, PID_StereoReversedEyes ) );
        }
      }

      inline bool ViewState::isStereoReversedEyes() const
      {
        return( m_reversedEyes );
      }

      inline void ViewState::setLODRangeScale(float factor)
      {
        if ( factor != m_scaleLODRange )
        {
          m_scaleLODRange = factor;
          notify( PropertyEvent( this, PID_LODRangeScale ) );
        }
      }

      inline float ViewState::getLODRangeScale() const
      {
        return m_scaleLODRange;
      }

      inline unsigned int ViewState::getTraversalMask() const
      {
        return( m_traversalMask );
      }

      inline void ViewState::setTraversalMask( unsigned int mask )
      {
        if ( m_traversalMask != mask )
        {
          m_traversalMask = mask;
          notify( PropertyEvent( this, PID_TraversalMask ) );
        }
      }

      inline bool ViewState::getAutoClipPlanes() const
      {
        return( m_autoClipPlanes );
      }

      inline void ViewState::setAutoClipPlanes( bool enable )
      {
        if ( enable != m_autoClipPlanes )
        {
          m_autoClipPlanes = enable;
          notify( PropertyEvent( this, PID_AutoClipPlanes ) );
        }
      }

      /** Initialize the given viewState with defaults. Choose the first camera of the scene.
        * If there's no camera in the scene create a new perspective camera and perform zoomAll.
        * Create a RendererOptions object if none is in the viewState. **/
      DP_SG_UI_API bool setupDefaultViewState( dp::sg::ui::ViewStateSharedPtr const& viewState );

    } // namespace ui
  } // namespace sg
} // namespace dp

namespace dp
{
  namespace util
  {
    /*! \brief Specialization of the TypedPropertyEnum template for type dp::sg::ui::ViewStateSharedPtr. */
    template <> struct TypedPropertyEnum< dp::sg::ui::ViewStateSharedPtr > {
      enum { type = Property::Type::VIEWSTATE };
    };
  } // namespace util
} // namespace dp
