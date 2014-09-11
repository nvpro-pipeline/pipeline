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

      /*! \brief Class describing a Camera by a projection matrix.
       *  \par Namespace: dp::sg::core
       *  \remarks A MatrixCamera overwrites the projection-specific functions of a Camera.
       *  \sa Camera, ParallelCamera, PerspectiveCamera */
      class MatrixCamera : public Camera
      {
        public:
          DP_SG_CORE_API static MatrixCameraSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~MatrixCamera();

        public:
          DP_SG_CORE_API void setMatrices( const dp::math::Mat44f & projection, const dp::math::Mat44f & inverse );

          /*! \brief Get the projection matrix of this MatrixCamera.
           *  \return The projection transformation.
           *  \sa Camera, getInverseProjection, getWorldToViewMatrix, getViewToWorldMatrix */
          DP_SG_CORE_API virtual dp::math::Mat44f getProjection() const;

          /*! \brief Get the inverse projection matrix of this MatrixCamera.
           *  \return The inverse projection transformation.
           *  \sa Camera, getProjection,  getWorldToViewMatrix, getViewToWorldMatrix */
          DP_SG_CORE_API virtual dp::math::Mat44f getInverseProjection()  const;

          DP_SG_CORE_API void virtual zoom( float factor );

          /*! \brief Assignment operator
           *  \param rhs A reference to the constant MatrixCamera to copy from.
           *  \return A reference to the assigned MatrixCamera.
           *  \remarks The assignment operator calls the assignment operator of Camera. */
          DP_SG_CORE_API MatrixCamera & operator=(const MatrixCamera & rhs);

          /*! \brief Determine the CullCode of a Sphere3f relative to the view volume.
           *  \param sphere A reference to the constant Sphere3f to determine the CullCode for.
           *  \return CC_IN, if the Sphere3f \a sphere is completely inside the view volume; CC_OUT
           *  if it is completely out of the view volume; otherwise CC_PART. */
          DP_SG_CORE_API virtual CullCode determineCullCode( const dp::math::Sphere3f &sphere ) const;

          REFLECTION_INFO_API( DP_SG_CORE_API, MatrixCamera );

        protected:
          /*! \brief Default-constructs a MatrixCamera. 
           *  \remarks The MatrixCamera initially is positioned at (0.0,0.0,1.0), has the y-axis
           *  as up-vector and looks down the negative z-axis. */
          DP_SG_CORE_API MatrixCamera();

          /*! \brief Copy-constructs a MatrixCamera from another MatrixCamera. */
          DP_SG_CORE_API MatrixCamera( const MatrixCamera &rhs );

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          dp::math::Mat44f  m_projection;
          dp::math::Mat44f  m_inverse;
      };

    } // namespace core
  } // namespace sg
} // namespace dp

