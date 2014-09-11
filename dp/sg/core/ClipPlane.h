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
#include <dp/sg/core/OwnedObject.h>
#include <dp/util/HashGenerator.h>
#include <dp/math/Vecnt.h>
#include <dp/math/Planent.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Defines a user clipping plane 
       * \remarks 
       * A ClipPlane will be defined through a normal vector, that is - a vector perpendicular to the plane,
       * and the distance, or offset, to the origin along the ClipPlane's normal. The ClipPlane is specified 
       * in object space. All points x in object space that satisfy normal * x + offset >=0 lie in the half-space 
       * defined by the ClipPlane. All points outside this half-space will be clipped away during rendering. 
       * \n\n
       * A ClipPlane in SceniX can be added to a Group using the Group::addClipPlane API. If a ClipPlane is enabled,
       * it will affect all children of the Group it was added to. Also, a ClipPlane will be transformed by the 
       * exact same hierarchy of Transforms that apply to the Group it was added to.
       * \par Namespace: dp::sg::core
       */ 
      class ClipPlane : public OwnedObject<Group>
      {
      public:
        DP_SG_CORE_API static ClipPlaneSharedPtr create();

        DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

        DP_SG_CORE_API virtual ~ClipPlane();

      public:
        /*! \brief Defines the orientation of the ClipPlane
         * \param normal 
         * Defines the normal vector of the ClipPlane which defines the orientation of the plane in object space.
         * \remarks
         * This function lets you specify the orientation of the ClipPlane in object space.
         * The ClipPlane's normal vector together with the ClipPlane's offset completely defines
         * the ClipPlane. All points x in object space that satisfy normal * x + offset >= 0 lie in the
         * half-space defined by the ClipPlane. All points outside this half-space are clipped away. 
         * \sa setOffset, getNormal */
        DP_SG_CORE_API void setNormal(const dp::math::Vec3f& normal);

        /*! \brief Retrieves the ClipPlane's normal vector.
         * \return 
         * The function returns the normal vector of the ClipPlane as last specified through setNormal.
         * \sa setNormal */
        DP_SG_CORE_API const dp::math::Vec3f& getNormal() const;

        /*! \brief Defines the ClipPlane's offset
         * \param offset 
         * Specifies the offset of the ClipPlane in object space.
         * \remarks
         * This function lets you specify the distance of the ClipPlane to the object space origin 
         * along the ClipPlane's normal vector.
         * The ClipPlane's normal vector together with the ClipPlane's offset completely defines
         * the ClipPlane. All points x in object space that satisfy normal * x + offset >= 0 lie in the
         * half-space defined by the ClipPlane. All points outside this half-space are clipped away. 
         * \sa setNormal, getOffset */
        DP_SG_CORE_API void setOffset(float offset);

        /*! \brief Retrieves the ClipPlane's offset.
         * \return 
         * The function returns the ClipPlane's offset as last specified through setOffset.
         * \sa setOffset */
        DP_SG_CORE_API float getOffset() const;

        /*! \brief Specifies the enable state of the ClipPlane
         * \param onOff
         * Indicates the new enable state. If you pass \c false, the ClipPlane will be further on ignored
         * during rendering until you re-enable it. Passing \c true enables the ClipPlane for rendering. 
         * \sa isEnabled */
        DP_SG_CORE_API void setEnabled(bool onOff);

        /*! \brief Retrieves the ClipPlane's current enable state
         * \return
         * \c true if the ClipPlane is enabled for rendering, \c false otherwise.
         * \remarks
         * If a ClipPlane is disabled, that is - if this function returns \c false, it will be
         * ignored during rendering.
         * \sa setEnabled */
        DP_SG_CORE_API bool isEnabled() const;

        /*! \brief Returns the distance of the specified point to this ClipPlane
         * \param point The point, in object space, for which to calculate the distance to this ClipPlane.
         * \return 
         * The function returns the distance of the given point to the ClipPlane along the ClipPlane's normal. 
         * \remarks 
         * The function expects the point to evaluate to be in object space.
         * A negative return value indicates that the point lies outside the half-space defined by this ClipPlane.
         * If the ClipPlane is enabled, all points outside this half-space will be clipped away during rendering. 
         * \sa isInside */
        DP_SG_CORE_API float getDistance(const dp::math::Vec3f& point) const;
  
        /*! \brief Evaluates if the indicated point lies inside the half-space defined by the ClipPlane
         * \param point
         * Indicates the point in object space to evaluate.
         * \return
         * The function returns \c true if the indicated point lies inside the half-space defined by the 
         * ClipPlane.
         * \remarks
         * Use this function to test if a given point in object space lies inside the half-space defined
         * by the ClipPlane. If the function returns \c false, the indicated point lies outside
         * this half-space and would be clipped away during rendering if the ClipPlane is enabled. 
         * Alternative to this function, you can also use getDistance and evaluate its return value. 
         * If getDistance returns a negative value, the point lies outside the half-space defined by this
         * ClipPlane. Otherwise, it lies inside the half-space. 
         * \sa getDistance */
        DP_SG_CORE_API bool isInside(const dp::math::Vec3f& point) const;

        /*! \brief Tests whether the indicated object is equivalent to this ClipPlane.
         * \param object 
         * The object to test against this ClipPlane.
         * \param ignoreNames
         * Specifies if Object names should be considered for the test or not. If \c true, object names will
         * be ignored while testing.
         * \param deepCompare 
         * This parameter does not apply for testing ClipPlanes and can be ignored.
         * \remarks
         * This function overrides Object::isEquivalent. */
        DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object , bool ignoreNames, bool deepCompare ) const;

        REFLECTION_INFO_API( DP_SG_CORE_API, ClipPlane );

      protected:
        /*! \brief Default-constructs a ClipPlane
         * \remarks
         * This constructor will be called on instantiation through ClipPlane::create().
         * All plane coefficients are set to 0 after default construction, and the ClipPlane is
         * enabled. */
        DP_SG_CORE_API ClipPlane();

        /*! \brief Copy-constructs a ClipPlane from a source ClipPlane. 
         * \param rhs
         * ClipPlane object that is taken as source.
         * \remarks
         * This copy-constructor will be called when ClipPlaneHandle::clone is invoked.
         * It completely copies the state of the right-hand-sided object. */
        DP_SG_CORE_API ClipPlane(const ClipPlane& rhs);

        /*! \brief Feed the data of this object into the provied HashGenerator.
         *  \param hg The HashGenerator to update with the data of this object.
         *  \sa getHashKey */
        DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

      private:
        dp::math::Plane3f m_plane; // internal representation 
        bool m_enabled;
      };

      inline const dp::math::Vec3f& ClipPlane::getNormal() const
      {
        return m_plane.getNormal();
      }

      inline float ClipPlane::getOffset() const
      {
        return m_plane.getOffset();
      }

      inline bool ClipPlane::isEnabled() const
      {
        return m_enabled;
      }

      inline float ClipPlane::getDistance(const dp::math::Vec3f& point) const
      {
        return m_plane(point);
      }

      inline bool ClipPlane::isInside(const dp::math::Vec3f& point) const
      {
        return m_plane(point) >= 0.0f;
      }

    } // namespace core
  } // namespace sg
} // namespace dp
