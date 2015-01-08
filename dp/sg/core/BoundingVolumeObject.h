// Copyright NVIDIA Corporation 2012-2015
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

#include <dp/sg/core/Object.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Class to manage bounding volume information.
       * This class acts as a base class for all Object classes that need an additional
       * bounding volume functionality.
       * Override the protected virtual functions calculateBoundingBox and calculateBoundingSphere
       * in the derived class to calculate the right bounding volume data for the class.
       */
      class BoundingVolumeObject : public Object
      {
      public: 
        /*! \brief Get the bounding box of the object.
         * \return
         * The function returns the current bounding box of the object.
         * \remarks This class caches the bounding box and only recalculates it
         * by calling calculateBoundingBox when the DP_BOUNDING_BOX dirty flag
         * is set.
         */
        const dp::math::Box3f & getBoundingBox() const;

        /*! \brief Get the bounding sphere of the object.
         * \return
         * The function returns the current bounding sphere of the object.
         * \remarks This class caches the bounding sphere and only recalculates it
         * by calling calculateBoundingBox when the DP_BOUNDING_SPHERE dirty flag
         * is set.
         */
        const dp::math::Sphere3f & getBoundingSphere() const;

      protected:
    
        /*! \brief Protected default constructor to prevent instantiation of a BoundingVolumeObject.
         *  \remarks A BoundingVolumeObject is not intended to be instantiated, but only classes derived from it.
         */
        BoundingVolumeObject();
    
        /*! \brief Protected copy constructor from a BoundingVolumeObject.
         *  \remarks A BoundingVolumeObject is not intended to be instantiated, but only classes derived from it.
         */
        BoundingVolumeObject( const BoundingVolumeObject &rhs );

        /*! \brief Protected destructor of a BoundingVolumeObject.
         *  \remarks A BoundingVolumeObject is not intended to be instantiated, but only classes derived from it.
         */
        ~BoundingVolumeObject();
    
        /*! \brief Assigns new content from another Object. 
         *  \param rhs Reference to an Object from which to assign the new content.
         *  \return A reference to this object.
         */
        BoundingVolumeObject& operator=( const BoundingVolumeObject& rhs );
    
        /*! \brief Calculate the bounding box of the object.
         *  \return The function calculates and returns the bounding box of the object.
         *  \remarks This base class implementation just returns a default-contructed bounding box.
         *  Classes deriving from this class need to override this function.
         */
        virtual dp::math::Box3f calculateBoundingBox() const;

        /*! \brief Calculate the bounding sphere of the object.
         *  \return The function calculates and returns the bounding sphere of the object.
         *  \remarks This base class implementation just returns a default-contructed bounding sphere.
         *  Classes deriving from this class need to override this function. 
         */
        virtual dp::math::Sphere3f calculateBoundingSphere() const;

      private:
        mutable dp::math::Box3f     m_boundingBox;    //!< The cached bounding box of the object
        mutable dp::math::Sphere3f  m_boundingSphere; //!< The cached bounding sphere of the object.
      };


      inline const dp::math::Box3f & BoundingVolumeObject::getBoundingBox() const
      {
        if( !!( this->m_dirtyState & this->DP_SG_BOUNDING_BOX ) )
        {
          this->m_dirtyState &= ~this->DP_SG_BOUNDING_BOX;
          m_boundingBox = calculateBoundingBox();
        }
        return m_boundingBox;
      }

      inline const dp::math::Sphere3f & BoundingVolumeObject::getBoundingSphere() const
      {
        if( !!( this->m_dirtyState & this->DP_SG_BOUNDING_SPHERE ) )
        {
          this->m_dirtyState &= ~this->DP_SG_BOUNDING_SPHERE;
          m_boundingSphere = calculateBoundingSphere();
        }
        return m_boundingSphere;
      }

      inline BoundingVolumeObject::BoundingVolumeObject()
      {
      }

      inline BoundingVolumeObject::BoundingVolumeObject( const BoundingVolumeObject &rhs )
        : Object( rhs )
        , m_boundingBox( rhs.m_boundingBox )
        , m_boundingSphere( rhs.m_boundingSphere )
      {
      }

      inline BoundingVolumeObject::~BoundingVolumeObject()
      {
      }

      inline BoundingVolumeObject & BoundingVolumeObject::operator=( const BoundingVolumeObject& rhs )
      {
        Object::operator=(rhs);
        if (&rhs != this)
        {
          m_boundingBox    = rhs.m_boundingBox;
          m_boundingSphere = rhs.m_boundingSphere;
        }
        return *this;
      }

      inline dp::math::Box3f BoundingVolumeObject::calculateBoundingBox() const
      {
        return dp::math::Box3f();
      }

      inline dp::math::Sphere3f BoundingVolumeObject::calculateBoundingSphere() const
      {
        return dp::math::Sphere3f();
      }

    } // namespace core
  } // namespace sg
} // namespace dp
