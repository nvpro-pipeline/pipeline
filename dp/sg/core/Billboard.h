// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/math/Trafo.h>
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/Group.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Class that adds billboard properties to a Group.
       *  \par Namespace: dp::sg::core
       *  \remarks The Billboard is a special transforming Group that performs a rotation either around
       *  and \a rotationAxis, or around the camera's \a upVector, such that the contents of the Group
       *  is either always oriented towards the viewer or screen aligned.
       *  \sa Group, Transform */
      class Billboard : public Group
      {
        public:
          DP_SG_CORE_API static BillboardSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~Billboard();

        public:
          //! Enumeration type to describe the alignment of a Billboard.
          enum class Alignment
          {
            AXIS,    //!< Axis aligned Billboard: rotate around a specified axis.
            VIEWER,  //!< Viewer aligned Billboard: align the Billboard to always face the viewer.
            SCREEN   //!< Screen aligned Billboard: keep the Billboard aligned to the screen.
          };

        public:
          /*! \brief Get the current Trafo of the Billboard.
           *  \param cam Specifies the read-only Camera to use in aligning the Billboard.
           *  \param worldToModel Specifies the current mapping from world to model.
           *  \return An dp::math::Trafo, that represents the transformation of the Billboard.
           *  \remarks Depending on the alignment of the Billboard, a different Trafo is determined.
           *  \sa getAlignment, setAlignment */
          DP_SG_CORE_API dp::math::Trafo getTrafo( CameraSharedPtr const& cam, dp::math::Mat44f const& worldToModel ) const;

          /*! \brief Get the alignment type of the Billboard.
           *  \return The enumeration value describing the alignment type.
           *  \sa Billboard::Alignment, setAlignment */
          DP_SG_CORE_API Alignment getAlignment() const;

          /*! \brief Set the Billboard alignment type.
           *  \param ba The enumeration value describing the alignment type.
           *  \sa Billboard::Alignment, getAlignment */
          DP_SG_CORE_API void setAlignment( Alignment ba );

          /*! \brief Get the rotation axis.
           *  \return A reference to the constant rotation axis.
           *  \remarks The rotation axis specifies which axis to use to perform the rotation. This
           *  axis is defined in the local coordinates of the Billboard. The default is the local
           *  y-axis, that is (0.0,1.0,0.0). The rotation axis is used only if the Billboard is not
           *  to be screen aligned.
           *  \sa isViewerAligned, setRotationAxis */
          DP_SG_CORE_API const dp::math::Vec3f & getRotationAxis() const;

          /*! \brief Set the rotation axis.
           *  \param axis A reference to the constant rotation axis.
           *  \remarks The rotation axis specifies which axis to use to perform the rotation. This
           *  axis is defined in the local coordinates of the Billboard. The default is the local
           *  y-axis, that is (0.0,1.0,0.0). The rotation axis is used only if the Billboard is not
           *  to be screen aligned.
           *  \sa getRotationAxis, setViewerAligned */
          DP_SG_CORE_API void setRotationAxis( const dp::math::Vec3f &axis );

          /*! \brief Assignment operator
           *  \param rhs Reference to the constant Billboard to copy from
           *  \return A reference to the assigned Billboard
           *  \remarks The assignment operator first calls the assignment operator of Group. Then the
           *  rotation axis and the screen alignment flag are copied.
           *  \sa Group */
          DP_SG_CORE_API Billboard & operator=( const Billboard & rhs );

          /*! \brief Test for equivalence with another Billboard.
           *  \param p            Pointer to the constant Billboard to test for equivalence with
           *  \param ignoreNames  Optional parameter to ignore the names of the objects; default is \c
           *  true
           *  \param deepCompare  Optional parameter to perform a deep comparison; default is \c false
           *  \return \c true if the Billboard \a p is equivalent to \c this, otherwise \c false.
           *  \remarks If \a p and \c this are equivalent as a group, they are equivalent if they are
           *  both screen aligned, or if they are not screen aligned but have the same rotation axis.
           *  \note The behavior is undefined if \a p is not a Billboard nor derived from one.
           *  \sa Group, getRotationAxis, isViewerAligned */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

        public:
          // reflected properties
          REFLECTION_INFO_API( DP_SG_CORE_API, Billboard );

          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Alignment );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( RotationAxis );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          /*! \brief Default-constructs a Billboard.
           *  \remarks. The Billboard initially has the y-axis as rotation axis and is not screen
           *  aligned. */
          DP_SG_CORE_API Billboard();

          /*! \brief Copy-constructs a Billboard from another Billboard.
           *  \param rhs Source Billboard. */
          DP_SG_CORE_API Billboard( const Billboard &rhs );

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          void getTrafoAxisAligned( CameraSharedPtr const& camera, dp::math::Mat44f const& worldToModel, dp::math::Trafo & trafo ) const;
          void getTrafoScreenAligned( CameraSharedPtr const& camera, dp::math::Mat44f const& worldToModel, dp::math::Trafo & trafo ) const;
          void getTrafoViewerAligned( CameraSharedPtr const& camera, dp::math::Mat44f const& worldToModel, dp::math::Trafo & trafo ) const;

        private:
          Alignment       m_alignment;
          dp::math::Vec3f m_rotationAxis;
      };

      inline const dp::math::Vec3f & Billboard::getRotationAxis() const
      {
        return( m_rotationAxis );
      }

      inline Billboard::Alignment Billboard::getAlignment() const
      {
        return( m_alignment );
      }

      inline void Billboard::setRotationAxis( const dp::math::Vec3f &rotationAxis )
      {
        if ( m_rotationAxis != rotationAxis )
        {
          m_rotationAxis = rotationAxis;
          notify( PropertyEvent( this, PID_RotationAxis ) );
        }
      }

      inline void Billboard::setAlignment( Alignment ba )
      {
        if ( m_alignment != ba )
        {
          m_alignment = ba;
          notify( PropertyEvent( this, PID_Alignment ) );
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp

