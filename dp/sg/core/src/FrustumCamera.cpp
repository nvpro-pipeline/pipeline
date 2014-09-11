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


#include <dp/sg/core/FrustumCamera.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( FrustumCamera, FarDistance );
      DEFINE_STATIC_PROPERTY( FrustumCamera, NearDistance );
      DEFINE_STATIC_PROPERTY( FrustumCamera, WindowOffset );
      DEFINE_STATIC_PROPERTY( FrustumCamera, WindowSize );
      DEFINE_STATIC_PROPERTY( FrustumCamera, WindowRegion );
      DEFINE_STATIC_PROPERTY( FrustumCamera, AspectRatio );

      BEGIN_REFLECTION_INFO( FrustumCamera )
        DERIVE_STATIC_PROPERTIES ( FrustumCamera, Camera );
        INIT_STATIC_PROPERTY_RW( FrustumCamera, FarDistance,   float,    SEMANTIC_VALUE,     value,           value );
        INIT_STATIC_PROPERTY_RW( FrustumCamera, NearDistance,  float,    SEMANTIC_VALUE,     value,           value );
        INIT_STATIC_PROPERTY_RW( FrustumCamera, WindowOffset,  Vec2f,    SEMANTIC_VALUE,     const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( FrustumCamera, WindowSize,    Vec2f,    SEMANTIC_VALUE,     const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( FrustumCamera, WindowRegion,  Box2f,    SEMANTIC_VALUE,     const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( FrustumCamera, AspectRatio,   float,    SEMANTIC_VALUE,     value,           value );
      END_REFLECTION_INFO

      FrustumCamera::FrustumCamera(void)
      : m_farDist( 10.0f )
      , m_nearDist( 0.1f )
      , m_windowOffset( 0.0f, 0.0f )
      , m_windowSize( 1.0f, 1.0f )
      , m_windowRegion( Vec2f( 0.0f, 0.0f ), Vec2f( 1.0f, 1.0f ) )
      {
      }

      FrustumCamera::FrustumCamera( const FrustumCamera &rhs )
      : Camera(rhs)
      , m_farDist(rhs.m_farDist)
      , m_nearDist(rhs.m_nearDist)
      , m_windowOffset(rhs.m_windowOffset)
      , m_windowSize(rhs.m_windowSize)
      , m_windowRegion(rhs.m_windowRegion)
      {
      }

      FrustumCamera::~FrustumCamera(void)
      {
      }

      FrustumCamera & FrustumCamera::operator=(const FrustumCamera & rhs)
      {
        if (&rhs != this)
        {
          Camera::operator=(rhs);
          m_farDist           = rhs.m_farDist;
          m_nearDist          = rhs.m_nearDist;
          m_windowOffset      = rhs.m_windowOffset;
          m_windowSize        = rhs.m_windowSize;
          m_windowRegion      = rhs.m_windowRegion;
        }
        return *this;
      }

      /** Calculate the vector from the camera position to the point on the bounding 
       *  sphere in the camera direction and project this vector onto the direction 
       *  vector of the camera. The length is the distance from the camera to the 
       *  far / near clipping plane.
       *  far: ps = center + radius * camdir , near: ps = center - radius * camdir 
       *  v  = ps - campos
       *  far = |(camdir * v) * camdir)|
       */
      void FrustumCamera::calcNearFarDistances( const Sphere3f &sphere )
      {
        DP_ASSERT( isPositive( sphere ) );
        DP_ASSERT( isNormalized( getOrientation() ) );

        float prevNear = m_nearDist;
        float prevFar  = m_farDist;
        //  NOTE: this ratio needs to depend on the z-buffer depth (24bits or more?)
        //        with a ratio of 16384 we have 10 bits of precision on a 24 bit z-buffer, giving good results in most cases...
      const float cFarToNearRatio = 16384.0f;
        m_farDist  = cFarToNearRatio;
        m_nearDist = 1.0f;

        // calculate the far clipping plane
        Vec3f dir = getDirection();
        Vec3f vf  = sphere.getCenter() + sphere.getRadius() * dir - getPosition();
  
        // scene is behind us ?
        float farDist = dir * vf;
        if ( 0.0f < farDist )
        {
          m_farDist = farDist;

          // calculate the near clipping plane
          Vec3f vn = sphere.getCenter() - sphere.getRadius() * dir - getPosition();

          m_nearDist = dir * vn;
          if ( ( m_nearDist <= 0.0f ) || ( m_farDist <= m_nearDist ) )
          {
            // part of the scene is behind us,
            // or scene is that far away, that we can't distinguish near and far
            m_nearDist = m_farDist / cFarToNearRatio;
          }
          if ( prevNear != m_nearDist )
          {
            notify( PropertyEvent( this, PID_NearDistance ) );
          }
          if ( prevFar != m_farDist )
          {
            notify( PropertyEvent( this, PID_FarDistance ) );
          }
        }
      }

      float FrustumCamera::getAspectRatio( void ) const 
      { 
        return( m_windowSize[0] / m_windowSize[1] );  
      }

      float FrustumCamera::getFarDistance( void ) const
      { 
        DP_ASSERT( m_nearDist < m_farDist );
        return( m_farDist ); 
      }

      float FrustumCamera::getNearDistance( void ) const
      { 
        DP_ASSERT( m_nearDist < m_farDist );
        return( m_nearDist );  
      }

      const Vec2f & FrustumCamera::getWindowOffset( void ) const
      {
        return m_windowOffset;
      }

      const Box2f & FrustumCamera::getWindowRegion() const
      {
        return( m_windowRegion );
      }

      const Vec2f & FrustumCamera::getWindowSize( void ) const 
      { 
        return( m_windowSize ); 
      }

      void FrustumCamera::setAspectRatio( float ar )
      {
        setAspectRatio( ar, false );
      }

      void FrustumCamera::setAspectRatio( float ar, bool keepWidth )
      {
        DP_ASSERT( ar >= FLT_EPSILON );

        if ( m_windowSize[0] / m_windowSize[1] != ar )
        {
          if ( keepWidth )
          {
            m_windowSize[1] = m_windowSize[0] / ar;
          }
          else
          {
            m_windowSize[0] = m_windowSize[1] * ar;
          }
          notify( PropertyEvent( this, PID_AspectRatio ) );
          notify( PropertyEvent( this, PID_WindowSize ) );
        }
      }

      void FrustumCamera::setFarDistance( float fd )
      {
        DP_ASSERT( ( FLT_EPSILON < fd ) );
        if ( m_farDist != fd )
        {
          m_farDist = fd;  
          notify( PropertyEvent( this, PID_FarDistance ) );
        }
      }

      void FrustumCamera::setNearDistance( float nd )
      { 
        DP_ASSERT( ( FLT_EPSILON <= nd ) );
        if ( m_nearDist != nd )
        {
          m_nearDist = nd; 
          notify( PropertyEvent( this, PID_NearDistance ) );
        }
      }

      void FrustumCamera::setWindowOffset( const Vec2f & offset )
      {
        if ( m_windowOffset != offset )
        {
          m_windowOffset = offset;
          notify( PropertyEvent( this, PID_WindowOffset ) );
        }
      }

      void FrustumCamera::setWindowRegion( const Box2f & region )
      {
        if ( m_windowRegion != region )
        {
          m_windowRegion = region;
          notify( PropertyEvent( this, PID_WindowRegion ) );
        }
      }

      void FrustumCamera::setWindowSize( const Vec2f & size )
      {
        if ( m_windowSize != size )
        {
          m_windowSize = size;  
          notify( PropertyEvent( this, PID_WindowSize ) );
        }
      }

      void FrustumCamera::zoom( float factor )
      {
        DP_ASSERT( FLT_EPSILON < factor );
        if ( factor != 1.0f )
        {
          m_windowSize *= factor;
          notify( PropertyEvent( this, PID_WindowSize ) );
        }
      }

      void FrustumCamera::zoom( const Sphere3f &sphere, float fovy, bool adjustClipPlanes )
      {
        DP_ASSERT( isPositive(sphere) );

        //  with some simple trigonometry, we get some factors for the distance and the window size:
        //  alpha = fovy / 2
        //  dist from camera position to center = p =>
        //    sin( alpha ) = r / ( p * r ) = 1 / p
        //  => p = 1 / sin( alpha )
        //  window size is 2 * q * r =>
        //    tan( alpha ) = ( q * r ) / ( p * r ) = q / p = q * sin( alpha )
        //  => q = tan( alpha ) / sin( alpha ) = 1 / cos( alpha )

        float aspectRatio = getAspectRatio();

        if ( fovy < 0.0f )
        {
          //  with negative fovy we keep the current
 
          // consider the aspect ratio with default fovy calculation
          float windowSize = (aspectRatio <= 1.0) ? getWindowSize()[0] : getWindowSize()[1];
          fovy = (float)( 2.0 * atan( windowSize / ( 2.0 * getFocusDistance() ) ) );
        }

        float alpha = fovy / 2.0f;
        float p = (float)( 1.0f / sinf( alpha ) );
        float q = (float)( 1.0f / cosf( alpha ) );

        // from the center go p*radius steps along the direction as the new position
        setPosition( sphere.getCenter() - p * sphere.getRadius() * getDirection() );

        float radius = sphere.getRadius();
        setFocusDistance( p * radius );

        float size = q * 2.0f * radius;
        if ( aspectRatio <= 1.0f )
        {
          //  the window is higher than wide, so fit to the width and keep the aspect ratio
          setWindowSize( Vec2f( size, size / aspectRatio ) );
        }
        else
        {
          //  the window is wider than high, so fit to the height and keep the aspect ratio
          setWindowSize( Vec2f( size / aspectRatio, size ) );
        }

        // note: don't touch near/far distance if AutoClipPlanes is off
        if ( adjustClipPlanes )
        { 
          //  adjust the near and far distances to best fit
          m_nearDist = ( p - 1.0f ) * radius;
          m_farDist  = ( p + 1.0f ) * radius;

          notify( PropertyEvent( this, PID_NearDistance ) );
          notify( PropertyEvent( this, PID_FarDistance ) );
        }
      }

      void FrustumCamera::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Camera::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_farDist), sizeof(m_farDist) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_nearDist), sizeof(m_nearDist) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_windowOffset), sizeof(m_windowOffset) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_windowSize), sizeof(m_windowSize) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_windowRegion), sizeof(m_windowRegion) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
