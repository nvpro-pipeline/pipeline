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


#include <dp/sg/core/Transform.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Transform, Center );
      DEFINE_STATIC_PROPERTY( Transform, Orientation );
      DEFINE_STATIC_PROPERTY( Transform, ScaleOrientation );
      DEFINE_STATIC_PROPERTY( Transform, Scaling );
      DEFINE_STATIC_PROPERTY( Transform, Translation );
      DEFINE_STATIC_PROPERTY( Transform, Matrix );
      DEFINE_STATIC_PROPERTY( Transform, Inverse );

      BEGIN_REFLECTION_INFO( Transform )
        DERIVE_STATIC_PROPERTIES( Transform, Group );

        INIT_STATIC_PROPERTY_RW( Transform, Center,           Vec3f,  Semantic::POSITION, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Transform, Orientation,      Quatf,  Semantic::DIRECTION, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Transform, ScaleOrientation, Quatf,  Semantic::DIRECTION, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Transform, Scaling,          Vec3f,  Semantic::SCALING, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Transform, Translation,      Vec3f,  Semantic::POSITION, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Transform, Matrix,           Mat44f, Semantic::VALUE, value,           const_reference );
        INIT_STATIC_PROPERTY_RO( Transform, Inverse,          Mat44f, Semantic::VALUE, value );
      END_REFLECTION_INFO

      TransformSharedPtr Transform::create()
      {
        return( std::shared_ptr<Transform>( new Transform() ) );
      }

      HandledObjectSharedPtr Transform::clone() const
      {
        return( std::shared_ptr<Transform>( new Transform( *this ) ) );
      }

      Transform::Transform( void )
      : m_jointCount(0)
      {
        m_objectCode = ObjectCode::TRANSFORM;
      }

      Transform::Transform( const Transform &rhs )
      : Group(rhs)
      , m_trafo(rhs.m_trafo)
      , m_jointCount(0)
      {
        m_objectCode = ObjectCode::TRANSFORM;
      }

      Transform::~Transform( void )
      {
      }


      void Transform::setTrafo( const Trafo &trafo )
      {
        if ( m_trafo != trafo )
        {
          m_trafo = trafo;
          // TODO Is it really necessary to send Object::Event here?
          // For now it's only required for the bounding box hierarchy
          // in the SceneGraph
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_Center ) );
          notify( PropertyEvent( this, PID_Orientation ) );
          notify( PropertyEvent( this, PID_ScaleOrientation ) );
          notify( PropertyEvent( this, PID_Scaling ) );
          notify( PropertyEvent( this, PID_Translation ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      void Transform::setCenter( const Vec3f& value )
      {
        if ( m_trafo.getCenter() != value )
        {
          m_trafo.setCenter( value );
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_Center ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      void Transform::setOrientation( const Quatf& value )
      {
        if ( m_trafo.getOrientation() != value )
        {
          m_trafo.setOrientation( value );
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_Orientation ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      void Transform::setScaleOrientation( const Quatf& value )
      {
        if (m_trafo.getScaleOrientation() != value)
        {
          m_trafo.setScaleOrientation( value );
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_ScaleOrientation ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      void Transform::setScaling( const Vec3f& value )
      {
        if (m_trafo.getScaling() != value)
        {
          m_trafo.setScaling( value );
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_Scaling ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      void Transform::setTranslation( const Vec3f& value )
      {
        if (m_trafo.getTranslation() != value)
        {
          m_trafo.setTranslation( value );
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_Translation ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      void Transform::setMatrix( const Mat44f& value)
      {
        if ( m_trafo.getMatrix() != value )
        {
          m_trafo.setMatrix( value );
          notify( Object::Event( this ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
      }

      Box3f Transform::calculateBoundingBox() const
      {
        Box3f bbox = Group::calculateBoundingBox();
        //
        // never transform an empty bounding box!
        // the result would be undefined!
        //
        if ( isValid(bbox) )
        {
          Mat44f mat(getTrafo().getMatrix());

          float lx = bbox.getLower()[0];
          float ly = bbox.getLower()[1];
          float lz = bbox.getLower()[2];
          float ux = bbox.getUpper()[0];
          float uy = bbox.getUpper()[1];
          float uz = bbox.getUpper()[2];

          Box3f tbbox;
          tbbox.update(Vec3f(Vec4f(lx, ly, lz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(lx, ly, uz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(lx, uy, lz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(lx, uy, uz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(ux, ly, lz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(ux, ly, uz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(ux, uy, lz, 1.0f) * mat));
          tbbox.update(Vec3f(Vec4f(ux, uy, uz, 1.0f) * mat));

          return tbbox;
        }
        return bbox;
      }

      Sphere3f Transform::calculateBoundingSphere() const
      {
        Sphere3f sphere = Group::calculateBoundingSphere();
        //
        // never transform an invalid bounding sphere!
        // the result would be undefined!
        //
        if ( isValid(sphere) )
        {
          sphere.setCenter( Vec3f( Vec4f( sphere.getCenter(), 1.0f ) * getTrafo().getMatrix() ) );
          sphere.setRadius( sphere.getRadius() * maxElement( getTrafo().getScaling() ) );
        }
        return( sphere );
      }

      Transform & Transform::operator=(const Transform & rhs)
      {
        if (&rhs != this)
        {
          Group::operator=(rhs);
          m_trafo = rhs.m_trafo;
          notify(Object::Event(this));
          notify( PropertyEvent( this, PID_Center ) );
          notify( PropertyEvent( this, PID_Orientation ) );
          notify( PropertyEvent( this, PID_ScaleOrientation ) );
          notify( PropertyEvent( this, PID_Scaling ) );
          notify( PropertyEvent( this, PID_Translation ) );
          notify( PropertyEvent( this, PID_Matrix ) );
        }
        return *this;
      }

      bool Transform::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object.get() == this )
        {
          return( true );
        }

        bool equi = std::dynamic_pointer_cast<Transform>(object) && Group::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          TransformSharedPtr const& t = std::static_pointer_cast<Transform>(object);
          equi =    ( m_trafo == t->m_trafo )
                &&  ! ( m_jointCount || t->m_jointCount );    // Joints can't be equivalent to each other or to non-joints!
        }
        return( equi );
      }

      void Transform::incrementJointCount()
      {
        if ( m_jointCount == 0 )
        {
          addHints( DP_SG_HINT_ALWAYS_VISIBLE );    // makes that a Joint is never culled !
        }
        m_jointCount++;
      }

      void Transform::decrementJointCount()
      {
        DP_ASSERT( 0 < m_jointCount );
        m_jointCount--;
        if ( m_jointCount == 0 )
        {
          removeHints( DP_SG_HINT_ALWAYS_VISIBLE );  // a non-Joint can be culled again
        }
      }

      void Transform::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Group::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_trafo), sizeof(m_trafo) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
