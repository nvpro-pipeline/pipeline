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


#pragma once
/** @file */

#include <dp/sg/core/nvsgapi.h>
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/Camera.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Class describing a perspective or parallel Camera.
       *  \par Namespace: dp::sg::core
       *  \remarks A FrustumCamera overwrites the projection-specific functions of a Camera.
       *  \sa Camera, ParallalCamera, PerspectiveCamera, MatrixCamera */
      class FrustumCamera : public Camera
      {
        public:
          /*! Destructs a FrustumCamera. */
          DP_SG_CORE_API virtual ~FrustumCamera();

          /*! \brief Calculate the distances to the near and far clipping planes.
           *  \param sphere A reference to the constant sphere, where the clipping planes are to be
           *  adjusted to.
           *  \remarks Calculates the optimal near and far clipping distances to prevent clipping of
           *  the dp::math::Sphere3f \a sphere without changing the position or orientation of the
           *  Camera.\n
           *  If the \a sphere is completely "behind" the Camera, nothing is changed. Otherwise, the
           *  near and far distances are adjusted. The far distance is adjusted to be at the far end
           *  of the sphere. If the \a sphere is partly "behind" the Camera, the near distance is set
           *  to be at the far distance divided by 16384. This gives an acceptable depth resolution,
           *  but might clip objects that are close to, and in front of, the Camera that would otherwise be
           *  visible. If the \a sphere is completely in front of the Camera, the near distance is
           *  adjusted to be at the near end of the sphere.
           *  Behavior is undefined if sphere is invalid. This can be checked using !dp::math::isValid
           *  \sa getFarDistance, getNearDistance, isAutoClipPlanes */
          DP_SG_CORE_API void calcNearFarDistances( const dp::math::Sphere3f &sphere );

          /*! \brief Get the aspect ratio of the Camera window.
           *  \return The aspect ratio of the Camera window
           *  \remarks The aspect ratio is the width over the height of the Camera window.
           *  \sa getWindowSize, setAspectRatio */
          DP_SG_CORE_API float getAspectRatio( void ) const;

          /*! \brief Get the distance to the far clipping plane.
           *  \return The distance from the Camera to the far clipping plane
           *  \remarks The distance to the far clipping plane determines the back plane of the view
           *  frustum of this Camera. All objects behind this plane are clipped.\n
           *  The far clipping distance has to be greater than the near clipping distance. The ratio of
           *  far-to-near gives the depth resolution of the Camera. Therefore, near should be as large
           *  as possible, while far should be as small as possible.\n
           *  \sa calcNearFarDistances, getNearDistance, setFarDistance */
          DP_SG_CORE_API float getFarDistance( void ) const;

          /*! \brief Get the distance to the near clipping plane.
           *  \return The distance from the Camera to the near clipping plane
           *  \remarks The distance to the near clipping plane determines the front plane of the view
           *  frustum of this Camera. All objects in front of this plane are clipped.\n
           *  The near clipping distance has to be positive and less than the far clipping distance.
           *  The ratio of far-to-near gives the depth resolution of the Camera. Therefore, near
           *  should be as large as possible, while far should be as small as possible.\n
           *  \sa calcNearFarDistances, getFarDistance, setNearDistance */
          DP_SG_CORE_API float getNearDistance( void ) const;

          /*! \brief Get the window offset.
           *  \return A reference to the constant window offset.
           *  \remarks The window offset is a value in world space that describes how to offset the
           *  view frustum for projection purposes. By default, the offset is (0.0,0.0). That is, the
           *  view frustum is symmetric about the view direction. Setting an offset, this view frustum
           *  is sheared a bit. This can be used, for example, in full scene antialiasing (FSAA) or
           *  stereo rendering.\n
           *  In FSAA mode, several passes are drawn where the Camera is kept in place, but the window
           *  is slightly offset (jittered) each time. This way, a somewhat blurred image is achieved.\n
           *  In stereo mode, two passes are drawn - once with the Camera shifted half the eye distance
           *  to the left, and once to the right. To adjust the view frustum accordingly, it is sheared a
           *  bit to the right and to the left, respectively.
           *  \sa setWindowOffset */
          DP_SG_CORE_API const dp::math::Vec2f & getWindowOffset( void ) const;

          /*! \brief Get the window region
           *  \return A reference to the relative lower left und upper right positions of the window region
           *  \remarks The window region describes a relative part of the viewing window. Usually the
           *  view frustum of the Camera is symmetric about the view direction. To describe an
           *  asymmetric view frustum, you can use the window region.\n
           *  The default for the window region is (0.0,0.0) for the lower left, and (1.0,1.0) for
           *  the upper right, thus using the whole view window. To only use the upper left quarter
           *  of the complete view frustum, for example, you could specify lower left as (0.0,0.5)
           *  and upper right as (0.5,1.0).
           *  \sa getWindowSize, setWindowRegion */
          DP_SG_CORE_API const dp::math::Box2f & getWindowRegion() const;

          /*! \brief Get the window size.
           *  \return A constant reference to the size of the view window in world space
           *  \remarks The window size is the size of the world window that is viewed by this Camera.
           *  The size is the size of the view frustum at focus distance from the Camera. It is
           *  symmetric about the view direction and it thus defines the aspect ratio of the world
           *  window, as well as, for a perspective Camera, the field of view. The default window size
           *  is (1.0,1.0).
           *  \sa setWindowSize */
          DP_SG_CORE_API const dp::math::Vec2f & getWindowSize( void ) const;

          /*! \brief Set the aspect ratio of the Camera window.
           *  \param ar         Aspect ratio (width over height) of the Camera window.
           *  \param keepWidth  boolean that controls whether the width or the height of the
           *  Camera window is modified.
           *  \remarks Setting the aspect ratio of the Camera window changes either the width or the
           *  height. With \a keepWidth equal to \c false (the default), the width is changed. With \a
           *  keepWidth equal to \a true, the height is changed. Setting the aspect ratio increases the
           *  Incarnation of the Camera.
           *  \sa getAspectRatio, setWindowSize */
          DP_SG_CORE_API void setAspectRatio( float ar
                                      , bool keepWidth );

          /*! \brief Set the aspect ratio of the Camera window. Calls setAspectRatio width keepWidth = false
           *  \param ar         Aspect ratio (width over height) of the Camera window.
           *  \sa setAspectRatio
           **/
          DP_SG_CORE_API void setAspectRatio( float ar );

          /*! \brief Set the distance to the far clipping plane.
           *  \param fd The distance from the Camera to the far clipping plane
           *  \remarks The distance to the far clipping plane determines the back plane of the view
           *  frustum of this Camera. All objects in behind this plane are clipped.\n
           *  The far clipping distance has to be greater than the near clipping distance. The ratio of
           *  far-to-near gives the depth resolution of the Camera. Therefore, near should be as large
           *  as possible, while far should be as small as possible.\n
           *  \sa calcNearFarDistances, getFarDistance, setNearDistance */
          DP_SG_CORE_API void setFarDistance( float fd );

          /*! \brief Set the distance to the near clipping plane.
           *  \param nd The distance from the Camera to the near clipping plane
           *  \remarks The distance to the near clipping plane determines the front plane of the view
           *  frustum of this Camera. All objects in front of this plane are clipped.\n
           *  The near clipping distance has to be positive and less than the far clipping distance.
           *  The ratio of far-to-near gives the depth resolution of the Camera. Therefore, near
           *  should be as large as possible, while far should be as small as possible.\n
           *  \sa calcNearFarDistances, getNearDistance, setFarDistance */
          DP_SG_CORE_API void setNearDistance( float nd );

          /*! \brief Set the window offset
           *  \param offset The window offset.
           *  \remarks The window offset is a value in world space that describes how to offset the
           *  view frustum for projection purposes. By default, the offset is (0.0,0.0). That is, the
           *  view frustum is symmetric about the view direction. Setting an offset, this view frustum
           *  is sheared a bit. This can be used, for example, in full scene antialiasing (FSAA) or
           *  stereo rendering.\n
           *  In FSAA mode, several passes are drawn where the Camera is kept in place, but the window
           *  is slightly offset (jittered) each time. This way, a somewhat blurred image is achieved.\n
           *  In stereo mode, two passes are drawn - once with the Camera shifted half the eye distance
           *  to the left, and once to the right. To adjust the view frustum accordingly, it is sheared a
           *  bit to the right and to the left, respectively.
           *  \sa getWindowOffset */
          DP_SG_CORE_API void setWindowOffset( const dp::math::Vec2f &offset );

          /*! \brief Set the window region
           *  \param region A reference to the relative lower left und upper right positions of the window region
           *  \remarks The window region describes a relative part of the viewing window. Usually the
           *  view frustum of the Camera is symmetric about the view direction. To describe an
           *  asymmetric view frustum, you can use the window region.\n
           *  The default for the window region is (0.0,0.0) for the \a lowerLeft, and (1.0,1.0) for
           *  the \a upperRight, thus using the whole view window. To only use the upper left quarter
           *  of the complete view frustum, for example, you could specify \a lowerLeft as (0.0,0.5)
           *  and \a upperRight as (0.5,1.0).
           *  \sa getWindowRegion, setWindowSize */
          DP_SG_CORE_API void setWindowRegion( const dp::math::Box2f & region );

          /*! \brief Set the window size.
           *  \param size A constant reference to the size of the view in world space.
           *  \remarks The window size is the size of the world window that is viewed by this Camera.
           *  The size is the size of the view frustum at focus distance from the Camera. It is
           *  symmetric about the view direction and it thus defines the aspect ratio of the world
           *  window, as well as, for a perspective Camera, the field of view. The default window size
           *  is (1.0,1.0). Setting the window size increases the Incarnation of the Camera.
           *  \sa getWindowSize */
          DP_SG_CORE_API void setWindowSize( const dp::math::Vec2f & size );

          /*! \brief Interface for getting a pick ray in world coordinates for a given screen point.
           *  \param x          x-coordinate of the input screen point
           *  \param y          y-coordinate of the input screen point
           *  \param w          Width of the screen window in pixels
           *  \param h          Height of the screen window in pixels
           *  \param rayOrigin  Reference to an dp::math::Vec3f to get the ray origin
           *  \param rayDir     Reference to an dp::math::Vec3f to get the ray direction
           *  \remarks The function returns in \a rayOrigin the origin, and in \a rayDir the direction
           *  of the pick ray calculated from the screen point specified by its \a x and \a y
           *  coordinate.
           *  \note A screen point of (0,0) indicates the lower left of the currently considered screen
           *  window. */
          DP_SG_CORE_API virtual void getPickRay( int x
                                          , int y
                                          , int w
                                          , int h
                                          , dp::math::Vec3f & rayOrigin
                                          , dp::math::Vec3f & rayDir ) const = 0;

          /*! \brief Zoom by a positive factor.
           *  \param factor   Factor to use in zooming
           *  \remarks Zooming by a factor changes the window size of the Camera. The window size is
           *  the size of the world that is viewed at the projection plane. This plane is at a target
           *  distance along the view direction and orthogonal to the view direction.\n
           *  When the factor is less than one, the window size is reduced, and thus the viewport shows
           *  less content in the same space. That is, it is zoomed in. When the factor is greater than
           *  one, it is zoomed out.\n
           *  Zooming the Camera increases the Incarnation.
           *  \sa move, orbit, rotate, zoom */
          DP_SG_CORE_API void virtual zoom( float factor );

          /*! \brief Zoom to fit view to a sphere.
           *  \param sphere A reference to the constant sphere to zoom to.
           *  \param fovy   Optional parameter describing the field of view. Default keeps the current field of view.
           *  \param adjustClipPlanes Optional parameter to enable near and far clip pane adjustment.
           *  \remarks Zooming to a sphere moves the Camera to view the complete sphere. The direction
           *  is not changed, but the position and the window size are changed. If the automatic clip plane
           *  adjustment is on, the near and far distances are set accordingly.\n
           *  Zooming the Camera increases the Incarnation.
           *  Behavior is undefined if sphere is invalid. This can be checked using !dp::math::isValid
           *  \sa setAutoClipPlanes, setPosition, setWindowSize */
          DP_SG_CORE_API void zoom( const dp::math::Sphere3f & sphere, float fovy = -1.0f, bool adjustClipPlanes = true );

        // reflected properties
        public:
          REFLECTION_INFO_API( DP_SG_CORE_API, FrustumCamera );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( FarDistance );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( NearDistance );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( WindowOffset );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( WindowSize );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( WindowRegion );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( AspectRatio );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          /*! \brief Protected constructor to prevent explicit creation.
           *  \remarks Because FrustumCamera is an abstract class, its constructor is never used explicitly,
           *  but in the constructor of derived classes.
           *  \sa ParallelCamera, PerspectiveCamera */
          DP_SG_CORE_API FrustumCamera();

          /*! \brief Copy-constructs a FrustumCamera from another FrustumCamera.
           *  \param rhs       Reference to the constant FrustumCamera to copy from
           *  \remarks Because FrustumCamera is an abstract class, its constructor is never used explicitly,
           *  but in the constructor of derived classes.
           *  \sa ParallelCamera, PerspectiveCamera */
          DP_SG_CORE_API FrustumCamera( const FrustumCamera &rhs );

          /*! \brief Protected assignment operator 
           *  \param rhs  Reference to the constant FrustumCamera to copy from
           *  \return A reference to the assigned FrustumCamera.
           *  \remarks The assignment operator calls the assignment operator of Camera.
           *  \sa LightSource, Object */
          DP_SG_CORE_API FrustumCamera & operator=(const FrustumCamera & rhs);

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          float           m_farDist;          //!< distance from position to far clipping plane
          float           m_nearDist;         //!< distance from position to near clipping plane
          dp::math::Vec2f m_windowOffset;     //!< offset from the viewing reference point to the center of the viewing window
          dp::math::Vec2f m_windowSize;       //!< width (x) and height (y) of viewing window
          dp::math::Box2f m_windowRegion;     //!< lower left and upper right position of window region to view
      };

    } // namespace core
  } // namespace sg
} // namespace dp

