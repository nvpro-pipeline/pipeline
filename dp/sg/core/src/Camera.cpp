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


#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/LightSource.h>

#include <iterator>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Camera, Position )
      DEFINE_STATIC_PROPERTY( Camera, Orientation );
      DEFINE_STATIC_PROPERTY( Camera, Direction );
      DEFINE_STATIC_PROPERTY( Camera, UpVector );
      DEFINE_STATIC_PROPERTY( Camera, FocusDistance );

      BEGIN_REFLECTION_INFO( Camera )
        DERIVE_STATIC_PROPERTIES( Camera, Object );
        INIT_STATIC_PROPERTY_RW( Camera, Position,      Vec3f,    Semantic::POSITION,  const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Camera, Orientation,   Quatf,    Semantic::POSITION,  const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Camera, Direction,     Vec3f,    Semantic::DIRECTION, value,           const_reference );
        INIT_STATIC_PROPERTY_RW( Camera, UpVector,      Vec3f,    Semantic::VALUE,     value,           const_reference );
        INIT_STATIC_PROPERTY_RW( Camera, FocusDistance, float,    Semantic::VALUE,     value,           value );
      END_REFLECTION_INFO

      Camera::Camera(void)
      : m_orientation( 0.0f, 0.0f, 0.0f, 1.0f )
      , m_position( 0.0f, 0.0f, 1.0f )
      , m_focusDist( 1.0f )
      , m_viewToWorldValid(false)
      , m_worldToViewValid(false)
      {
      }

      Camera::Camera(const Camera& rhs)
      : Object(rhs)
      , m_orientation(rhs.m_orientation)
      , m_position(rhs.m_position)
      , m_focusDist(rhs.m_focusDist)
      , m_viewToWorld(rhs.m_viewToWorld)
      , m_viewToWorldValid(rhs.m_viewToWorldValid)
      , m_worldToView(rhs.m_worldToView)
      , m_worldToViewValid(rhs.m_worldToViewValid)
      {
        copyHeadLights(rhs);
      }

      Camera::~Camera(void)
      {
        removeHeadLights();
      }

      Camera::HeadLightIterator Camera::addHeadLight( const LightSourceSharedPtr & light )
      {
        m_headLights.push_back( light );
        notify( Event(this ) );
        return( HeadLightIterator( m_headLights.end() - 1 ) );
      }

      bool Camera::replaceHeadLight( const LightSourceSharedPtr & newLight, const LightSourceSharedPtr & oldLight )
      {
        if ( oldLight != newLight )
        {
          HeadLightContainer::iterator hlci = find( m_headLights.begin(), m_headLights.end(), oldLight );
          if ( hlci != m_headLights.end() )
          {
            *hlci = newLight;
            notify( Event(this ) );
            return( true );
          }
        }
        return( false );
      }

      bool Camera::replaceHeadLight( const LightSourceSharedPtr & newLight, const HeadLightIterator & oldIt )
      {
        if ( ( oldIt.m_iter != m_headLights.end() ) && ( *oldIt != newLight ) )
        {
          *oldIt.m_iter = newLight;
          notify( Event(this ) );
          return( true );
        }
        return( false );
      }

      bool Camera::removeHeadLight( const LightSourceSharedPtr & light )
      {
        HeadLightContainer::iterator hlci = find( m_headLights.begin(), m_headLights.end(), light );
        return( doRemoveHeadLight( hlci ) );
      }

      bool Camera::removeHeadLight( const HeadLightIterator & hli )
      {
        return( doRemoveHeadLight( hli.m_iter ) );
      }

      void Camera::clearHeadLights( void )
      {
        if ( getNumberOfHeadLights() )
        {
          removeHeadLights();
        }
      }

      void Camera::setViewToWorldMatrix( const Mat44f & m )
      {
        Vec3f trans, scale;
        Quatf orient, scaleOrient;
        decompose( m, trans, orient, scale, scaleOrient );
        setPosition( trans );
        setOrientation( orient );
      }

      const Mat44f& Camera::getViewToWorldMatrix( void ) const
      {
        if ( ! m_viewToWorldValid )
        {
          m_viewToWorld = Mat44f( m_orientation, m_position );
          m_viewToWorldValid = true;
        }
        return( m_viewToWorld );
      }

      void Camera::setWorldToViewMatrix( const Mat44f & m )
      {
        Mat44f invM;
        DP_VERIFY( invert( m, invM ) );
        setViewToWorldMatrix( invM );
      }

      const Mat44f& Camera::getWorldToViewMatrix( void ) const
      {
        if ( ! m_worldToViewValid )
        {
          m_worldToView =   Mat44f( Quatf( 0.0f, 0.0f, 0.0f, 1.0f ), -m_position )
                          * Mat44f( -m_orientation, Vec3f( 0.0f, 0.0f, 0.0f ) );
          m_worldToViewValid = true;
        }
        return( m_worldToView );
      }

      void Camera::move( const Vec3f & delta, bool cameraRelative /* = true */ )
      {
        if ( !isNull( delta ) )
        {
          if ( cameraRelative )
          {
            Vec3f yAxis = getUpVector();
            Vec3f zAxis = - getDirection();
            Vec3f xAxis = yAxis ^ zAxis;
            m_position += delta[0] * xAxis + delta[1] * yAxis + delta[2] * zAxis;
          }
          else
          {
            m_position += delta;
          }
          m_viewToWorldValid = false;
          m_worldToViewValid = false;
          notify( PropertyEvent( this, PID_Position ) );
        }
      }

      void Camera::orbit( const Vec3f & axis, float targetDistance, float angle, bool cameraRelative /*= true*/ )
      {
        DP_ASSERT( isNormalized( axis, 2*FLT_EPSILON ) );
        Vec3f target = m_position + targetDistance * getDirection();  //  determine the current target position
        rotate( -axis, angle, cameraRelative );                       //  rotate the camera
        setPosition( target - targetDistance * getDirection());       //  determine the new camera position
      }

      void Camera::rotate( const Vec3f & axis, float angle, bool cameraRelative /*= true*/ )
      {
        DP_ASSERT( isNormalized( axis, 2*FLT_EPSILON ) );

        if ( angle != 0.0f )
        {
          if ( cameraRelative )
          {
            setOrientation( Quatf( axis, angle ) * getOrientation() );
          }
          else
          {
            Quatf orientation = getOrientation();
            // Added the normalize step. It shouldn't be necessary
            // but it seems that rounding errors can hit us even in the case when
            // m_orientation and axis are normalized
            Vec3f newAxis( axis * ~orientation );
            newAxis.normalize();
            setOrientation( Quatf( newAxis, angle ) * orientation );
          }
          DP_ASSERT( isNormalized( m_orientation ) );
          m_viewToWorldValid = false;
          m_worldToViewValid = false;
          notify( PropertyEvent( this, PID_Orientation ) );
        }
      }

      void Camera::setDirection( const Vec3f &dir )
      {
        DP_ASSERT( isNormalized( dir ) );
        Vec3f oldDir = getDirection();
        if ( oldDir != dir )
        {
          if ( areCollinear( dir, oldDir ) )
          {
            if ( length( dir + oldDir ) < FLT_EPSILON )
            {
              // old and new dir are opposite to each other => rotate by PI around the upVector
              m_orientation *= Quatf( getUpVector(), PI );
            }
          }
          else
          {
            m_orientation *= Quatf( oldDir, dir );
          }
          DP_ASSERT( isNormalized( m_orientation ) );
          m_viewToWorldValid = false;
          m_worldToViewValid = false;
          notify( PropertyEvent( this, PID_Direction ) );
          notify( PropertyEvent( this, PID_Orientation ) );
        }
      }

      void Camera::setFocusDistance( float fd )
      {
        DP_ASSERT( FLT_EPSILON < fd );
        if ( m_focusDist != fd )
        {
          m_focusDist = fd;
          notify( PropertyEvent( this, PID_FocusDistance ) );
        }
      }

      void  Camera::setOrientation( const Quatf &quat )
      {
        DP_ASSERT( isNormalized( quat ) );
        if ( m_orientation != quat )
        {
          m_orientation = quat;
          m_viewToWorldValid = false;
          m_worldToViewValid = false;
          notify( PropertyEvent( this, PID_Direction ) );
          notify( PropertyEvent( this, PID_Orientation ) );
        }
      }

      void  Camera::setOrientation( const Vec3f &dir, const Vec3f &up )
      {
        DP_ASSERT( areOrthonormal( dir, up ) );
        Vec3f scaling;
        Quatf scaleOrientation, orientation;
        Mat33f  m = ~Mat33f( { dir ^ up, up, -dir } );
        decompose( m, orientation, scaling, scaleOrientation );
        setOrientation( ~orientation );
      }

      void  Camera::setPosition( const Vec3f& pos )
      {
        if ( m_position != pos )
        {
          m_position = pos;
          m_viewToWorldValid = false;
          m_worldToViewValid = false;
          notify( PropertyEvent( this, PID_Position ) );
        }
      }

      void Camera::setUpVector( const Vec3f &up )
      {
        DP_ASSERT( isNormalized( up ) );
        Vec3f oldUp = getUpVector();
        if ( oldUp != up )
        {
          if ( areCollinear( up, oldUp ) )
          {
            if ( length( up + oldUp ) < FLT_EPSILON )
            {
              // old and new up are opposite to each other => rotate by PI around the direction
              m_orientation *= Quatf( getDirection(), PI );
            }
          }
          else
          {
            m_orientation *= Quatf( oldUp, up );
          }
          DP_ASSERT( isNormalized( m_orientation ) );
          m_viewToWorldValid = false;
          m_worldToViewValid = false;
          notify( PropertyEvent( this, PID_UpVector ) );
        }
      }

      void Camera::copyHeadLights(const Camera& rhs)
      {
        DP_ASSERT(this!=&rhs);
        DP_ASSERT(m_headLights.empty());
        transform( rhs.m_headLights.begin()
                 , rhs.m_headLights.end()
                 , back_inserter(m_headLights)
                 , dp::util::CloneObject() );
        notify( Event(this ) );
      }

      void Camera::removeHeadLights()
      {
        m_headLights.clear();
        notify( Event(this ) );
      }

      Camera & Camera::operator=(const Camera & rhs)
      {
        if (&rhs != this)
        {
          Object::operator=(rhs);

          removeHeadLights();
          copyHeadLights(rhs);

          m_orientation       = rhs.m_orientation;
          m_position          = rhs.m_position;
          m_focusDist         = rhs.m_focusDist;

          m_viewToWorld       = rhs.m_viewToWorld;
          m_viewToWorldValid  = rhs.m_viewToWorldValid;
          m_worldToView       = rhs.m_worldToView;
          m_worldToViewValid  = rhs.m_worldToViewValid;
        }
        return *this;
      }

      void Camera::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Object::feedHashGenerator( hg );
        for ( HeadLightContainer::const_iterator hlcci = m_headLights.begin() ; hlcci != m_headLights.end() ; ++hlcci )
        {
          hg.update( *hlcci );
        }
        hg.update( reinterpret_cast<const unsigned char *>(&m_orientation), sizeof(m_orientation) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_position), sizeof(m_position) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_focusDist), sizeof(m_focusDist) );
      }

      bool Camera::doRemoveHeadLight( const HeadLightContainer::iterator & hlci )
      {
        if ( hlci != m_headLights.end() )
        {
          m_headLights.erase( hlci );
          notify( Event(this ) );
          return( true );
        }
        return( false );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

