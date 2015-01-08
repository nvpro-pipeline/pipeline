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

#include <dp/sg/core/Config.h>
#include <dp/sg/core/FrustumCamera.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Class describing a perspective Camera.
       *  \par Namespace: dp::sg::core
       *  \remarks A PerspectiveCamera overwrites the projection specific functions of a Camera.
       *  \sa Camera, FrustumCamera, ParallelCamera */
      class PerspectiveCamera : public FrustumCamera
      {
        public:
          DP_SG_CORE_API static PerspectiveCameraSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~PerspectiveCamera();

        public:
          /*! \brief Get the field of view.
           *  \return The field of view in radians.
           *  \remarks The field of view is the complete view angle (in radians) in the y-direction.
           *  \sa setFieldOfView */
          DP_SG_CORE_API float getFieldOfView() const;

          /*! \brief Set the field of view.
           *  \param fov The field of view in radians.
           *  \remarks The field of view is the complete view angle (in radians) in the y-direction.
           *  \sa getFieldOfView */
          DP_SG_CORE_API void setFieldOfView( float fov );

          /*! \brief Set the distance to the projection plane.
           *  \param fd The distance from the PerspectiveCamera to the projection plane.
           *  \remarks Setting the focus distance keeps the field of view constant. Therefore, the
           *  window size is adjusted accordingly.
           *  \sa getFieldOfView, getFocusDistance */
          DP_SG_CORE_API virtual void setFocusDistance( float fd );

          /*! \brief Get the projection matrix of this PerspectiveCamera.
           *  \return The projection transformation.
           *  \sa Camera, getInverseProjection, getWorldToViewMatrix, getViewToWorldMatrix */
          DP_SG_CORE_API virtual dp::math::Mat44f getProjection() const;

          /*! \brief Get the inverse projection matrix of this PerspectiveCamera.
           *  \return The inverse projection transformation.
           *  \sa Camera, getProjection,  getWorldToViewMatrix, getViewToWorldMatrix */
          DP_SG_CORE_API virtual dp::math::Mat44f getInverseProjection() const;

          /*! \brief Get a pick ray in world coordinates for a given screen point.
           *  \param x          x-coordinate of the input screen point.
           *  \param y          y-coordinate of the input screen point.
           *  \param w          Width of the screen window in pixels.
           *  \param h          Height of the screen window in pixels.
           *  \param rayOrigin  Reference to an dp::math::Vec3f to get the ray origin.
           *  \param rayDir     Reference to an dp::math::Vec3f to get the ray direction.
           *  \remarks The function returns in \a rayOrigin the origin, and in \a rayDir the direction
           *  of the pick ray calculated from the screen point specified by its \a x and \a y
           *  coordinate. The ray origin is the camera position.
           *  \note A screen point of (0,0) indicates the lower left of the currently considered
           *  screen window. */
          DP_SG_CORE_API virtual void getPickRay( int x
                                          , int y
                                          , int w
                                          , int h
                                          , dp::math::Vec3f & rayOrigin
                                          , dp::math::Vec3f & rayDir ) const;

          /*! \brief Assignment operator
           *  \param rhs A reference to the constant PerspectiveCamera to copy from.
           *  \return A reference to the assigned PerspectiveCamera.
           *  \remarks The assignment operator calls the assignment operator of FrustumCamera. */
          DP_SG_CORE_API PerspectiveCamera & operator=(const PerspectiveCamera & rhs);

          /*! \brief Determine the CullCode of a Sphere3f relative to the view frustum.
           *  \param sphere A reference to the constant Sphere3f to determine the CullCode for.
           *  \return CC_IN, if the Sphere3f \a sphere is completely inside the view frustum; CC_OUT
           *  if it is completely out of the view frustum; otherwise CC_PART. */
          DP_SG_CORE_API virtual CullCode determineCullCode( const dp::math::Sphere3f &sphere ) const;

        protected:
          /*! \brief Default-constructs a PerspectiveCamera. 
           *  \remarks The PerspectiveCamera initially is positioned at (0.0,0.0,1.0), has the y-axis
           *  as up-vector and looks down the negative z-axis. */
          DP_SG_CORE_API PerspectiveCamera();

          /*! \brief Copy-constructs a PerspectiveCamera from another PerspectiveCamera. */
          DP_SG_CORE_API PerspectiveCamera( const PerspectiveCamera &rhs );

        // reflected properties
        public:
          REFLECTION_INFO_API( DP_SG_CORE_API, PerspectiveCamera );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( FieldOfView );
          END_DECLARE_STATIC_PROPERTIES
      };

      inline float PerspectiveCamera::getFieldOfView() const
      {
        return( 2.0f * atanf( 0.5f * getWindowSize()[1] / getFocusDistance() ) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
