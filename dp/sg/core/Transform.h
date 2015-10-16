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

#include <dp/sg/core/Config.h>
#include <dp/math/Quatt.h>
#include <dp/math/Trafo.h>
#include <dp/math/Vecnt.h>
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/Group.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Transform node.
       *  \par Namespace: dp::sg::core
       *  \remarks A Transform is a special Group node. In addition to the children held by a
       *  Transform like a Group, it contains a transformation that is appended to the current
       *  transformation. The children then are positioned relative to this transformation.
       *
       *  Beginning with SceniX 7.2 each property P will send out only notify PID_P and PID_Matrix.
       *  PID_Invert is implied with all property changes.
       *  \sa Group */
      class Transform : public Group
      {
        friend class Skin;

        public:
          DP_SG_CORE_API static TransformSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~Transform(void);

        public:
          /*! \brief Get the transformation represented by this Transform.
           *  \return A reference to the constant dp::math::Trafo, representing the transformation.
           *  \sa dp::math::Trafo, setTrafo */
          virtual const dp::math::Trafo& getTrafo( void ) const;

          /*! \brief Set the transformation for this Transform.
           *  \param trafo A reference to the constant dp::math::Trafo to use.
           *  \remarks The transformation \a trafo is copied to the internal one, and the bounding
           *  sphere of the Transform is invalidated.
           *  \sa dp::math::Trafo, getTrafo */
          DP_SG_CORE_API virtual void setTrafo( const dp::math::Trafo & trafo );

          /*! \brief Test if a Transform is a Joint.
           *  \return \c true, if this Transform is a Joint, otherwise \c false.
           *  \remarks When a Transform is added as a Joint to a Skin, its joint counter is incremented.
           *  If the joint counter of a Transform is positive, it is considered to be a Joint. */
          bool isJoint() const;

          /*! \brief Assignment operator
           *  \param rhs A reference to the constant Transform to copy from
           *  \return A reference to the assigned Transform
           *  \remarks The assignment operator calls the assignment operator of Group, and copies the
           *  transformation.
           *  \sa Group, dp::math::Trafo */
          DP_SG_CORE_API Transform & operator=(const Transform & rhs);

          /*! \brief Test for equivalence with an other Transform.
           *  \param p A reference to the constant Transform to test for equivalence with.
           *  \param ignoreNames Optional parameter to ignore the names of the objects; default is \c true.
           *  \param deepCompare Optional parameter to perform a deep comparsion; default is \c false.
           *  \return \c true if the Transform \a p is equivalent to \c this, otherwise \c false.
           *  \remarks If \a p and \c this are equivalent as a Group, they are equivalent if they have
           *  the same transformation.
           *  \sa Group, dp::math::Trafo */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          REFLECTION_INFO_API( DP_SG_CORE_API, Transform );

        protected:
          /*! \brief Default-constructs a Transform.
           */
          DP_SG_CORE_API Transform(void);

          /*! \brief Constructs a Transform as a copy of another Transform.
          */
          DP_SG_CORE_API Transform( const Transform &rhs );

          /*! \brief Calculates the bounding box of this Transform.
           *  \return The axis-aligned bounding box of this Transform.
           *  \remarks This function is called by the framework when re-calculation
           *  of the bounding box is required for this Transform. */
          DP_SG_CORE_API virtual dp::math::Box3f calculateBoundingBox() const;

          /*! \brief Calculate the bounding sphere of this Transform.
           *  \return A dp::math::Sphere3f that contains the complete Transform.
           *  \remarks This function is called by the framework to determine a sphere that completely
           *  contains the Transform. First, the bounding sphere of the Group is determined, then
           *  this sphere is transformed according to the transformation. */
          DP_SG_CORE_API virtual dp::math::Sphere3f calculateBoundingSphere() const;

          /*! \brief Increments the number of references as a joint.
           *  \remarks A Transform can be used as a joint by a Skin. For each Skin, that uses this
           *  Transform, its joint count is incremented. Whenever a joint is removed from a Skin, the
           *  joint count of the corresponding Transform is decremented. That is, when the joint count
           *  is larger than zero, this Transform is a joint, otherwise it isn't. This function is
           *  exclusively used by the Skin class.
           *  \sa decrementJointCount */
          DP_SG_CORE_API void incrementJointCount();

          /*! \brief Decrements the number of references as a joint.
           *  \remarks A Transform can be used as a joint by a Skin. For each Skin, that uses this
           *  Transform, its joint count is incremented. Whenever a joint is removed from a Skin, the
           *  joint count of the corresponding Transform is decremented. That is, when the joint count
           *  is larger than zero, this Transform is a joint, otherwise it isn't. This function is
           *  exclusively used by the Skin class.
           *  \sa decrementJointCount */
          DP_SG_CORE_API void decrementJointCount();

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          dp::math::Trafo m_trafo;
          unsigned int    m_jointCount;

        public:
          // Property framework
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Center );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Orientation );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( ScaleOrientation );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Scaling );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Translation );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Matrix );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Inverse );
          END_DECLARE_STATIC_PROPERTIES

          const dp::math::Vec3f&  getCenter( void ) const           { return getTrafo().getCenter(); }
          const dp::math::Quatf&  getOrientation( void ) const      { return getTrafo().getOrientation(); }
          const dp::math::Quatf&  getScaleOrientation( void ) const { return getTrafo().getScaleOrientation(); }
          const dp::math::Vec3f&  getScaling( void ) const          { return getTrafo().getScaling(); }
          const dp::math::Vec3f&  getTranslation( void ) const      { return getTrafo().getTranslation(); }
                dp::math::Mat44f  getMatrix( void ) const           { return getTrafo().getMatrix(); }
                dp::math::Mat44f  getInverse( void ) const          { return getTrafo().getInverse(); }

          DP_SG_CORE_API void setCenter( const dp::math::Vec3f& value );
          DP_SG_CORE_API void setOrientation( const dp::math::Quatf& value );
          DP_SG_CORE_API void setScaleOrientation( const dp::math::Quatf& value );
          DP_SG_CORE_API void setScaling( const dp::math::Vec3f& value );
          DP_SG_CORE_API void setTranslation( const dp::math::Vec3f& value );
          DP_SG_CORE_API void setMatrix( const dp::math::Mat44f& value);
      };

      inline const dp::math::Trafo& Transform::getTrafo( void ) const
      {
        return( m_trafo );
      }

      inline bool Transform::isJoint() const
      {
        return( 0 < m_jointCount );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

