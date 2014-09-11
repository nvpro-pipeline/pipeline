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


#include <dp/sg/core/MatrixCamera.h>
#include <cstring>

#if defined(_M_IX86) || defined(_X86_) || defined(_M_X64) || defined(__x86_64__)
#define SCENIX_MATRIXCAMERA_USE_SSE
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( MatrixCamera )
        DERIVE_STATIC_PROPERTIES( MatrixCamera, Camera );
      END_REFLECTION_INFO

      #if defined(SCENIX_MATRIXCAMERA_USE_SSE)
      namespace {
        struct Mat
        {
          union
          {
            __m128 sse[4];
            float  data[16];
          };
        };

        struct Vec
        {
          union
          {
            __m128 sse;
            float  data[4];
          };

          Vec()
          {
          }

          Vec( float value )
          {
            sse = _mm_set_ss( value );
            sse = _mm_shuffle_ps( sse, sse, _MM_SHUFFLE(0,0,0,0) );
          }

          Vec( const Vec4f &vec )
          {
            //memcpy( data, vec.getPtr(), 4 * sizeof(float));
            sse = _mm_set_ps( vec[3], vec[2], vec[1], vec[0] );
          }

          Vec( const __m128 rhs )
          {
            sse = rhs;
          }

          Vec( float x, float y, float z, float w )
          {
            sse = _mm_set_ps( w, z, y, x );
          }

          Vec &operator +=( const Vec &vec )
          {
            sse = _mm_add_ps( sse, vec.sse );
            return *this;
          }

          Vec &operator -=( const Vec &vec )
          {
            sse = _mm_sub_ps( sse, vec.sse );
            return *this;
          }

          Vec operator-( const Vec &vec )
          {
            Vec result(*this);
            result -= vec;
            return result;
          }

          Vec operator+( const Vec &vec )
          {
            Vec result(*this);
            result += vec;
            return result;
          }

          void operator=( const Vec &vec )
          {
            sse = vec.sse;
          }

          Vec &operator=( __m128 rhs )
          {
            sse = rhs;
            return *this;
          }

          float operator[](size_t idx) const { return data[idx];}
          float &operator[](size_t idx) { return data[idx];}
        };

        inline Vec multiplyHomogen( const Vec &vec, const Mat &mat )
        {
          Vec result;

          __m128 vvec = vec.sse;
    
          result.sse = mat.sse[3];
          result.sse = _mm_add_ps( result.sse, _mm_mul_ps( mat.sse[0], _mm_shuffle_ps( vvec, vvec, _MM_SHUFFLE(0,0,0,0) ) ) );
          result.sse = _mm_add_ps( result.sse, _mm_mul_ps( mat.sse[1], _mm_shuffle_ps( vvec, vvec, _MM_SHUFFLE(1,1,1,1) ) ) );
          result.sse = _mm_add_ps( result.sse, _mm_mul_ps( mat.sse[2], _mm_shuffle_ps( vvec, vvec, _MM_SHUFFLE(2,2,2,2) ) ) );

          return result;
        }
      }
      #endif

      MatrixCameraSharedPtr MatrixCamera::create()
      {
        return( std::shared_ptr<MatrixCamera>( new MatrixCamera() ) );
      }

      HandledObjectSharedPtr MatrixCamera::clone() const
      {
        return( std::shared_ptr<MatrixCamera>( new MatrixCamera( *this ) ) );
      }

      MatrixCamera::MatrixCamera(void)
      {
        m_objectCode = OC_MATRIXCAMERA;
      }

      MatrixCamera::MatrixCamera( const MatrixCamera &rhs )
      : Camera(rhs)
      , m_projection(cIdentity44f)
      , m_inverse(cIdentity44f)
      {
        m_objectCode = OC_MATRIXCAMERA;
      }

      MatrixCamera::~MatrixCamera(void)
      {
      }

      Mat44f  MatrixCamera::getProjection( void ) const
      {
        return( m_projection );
      }

      Mat44f  MatrixCamera::getInverseProjection( void ) const
      {
        return( m_inverse );
      }

      void MatrixCamera::zoom( float factor )
      {
        DP_ASSERT( FLT_EPSILON < factor );

        if ( FLT_EPSILON < abs( factor - 1.0f ) )
        {
          float invFactor = 1.0f / factor;
          for ( int i=0 ; i<3 ; i++ )
          {
            m_projection[i][i] *= invFactor;
            m_inverse[i][i] *= factor;
          }
          notify( Event(this ) );
        }
      }

      MatrixCamera & MatrixCamera::operator=(const MatrixCamera & rhs)
      {
        if (&rhs != this)
        {
          Camera::operator=(rhs);
          m_projection = rhs.m_projection;
          m_inverse    = rhs.m_inverse;
        }
        return *this;
      }

      void MatrixCamera::setMatrices( const Mat44f & projection, const Mat44f & inverse )
      {
        if ( ( m_projection != projection ) || ( m_inverse != inverse ) )
        {
          m_projection = projection;
          m_inverse    = inverse;
          notify( Event(this ) );
        }
      }

      #ifdef SCENIX_MATRIXCAMERA_USE_SSE
      namespace
      {
        inline void updateCullCode( const Vec& point, unsigned int &cfo, unsigned int &cfa )
        {
          __m128 homogen = _mm_shuffle_ps( point.sse, point.sse, _MM_SHUFFLE( 3,3,3,3 ) );
          static const __m128 zero = { 0.0f, 0.0f, 0.0f, 0.0f };
          __m128 negHomogen = _mm_sub_ps( zero, homogen );

          __m128i resultP = _mm_castps_si128(_mm_cmple_ps( homogen, point.sse ));
          __m128i resultN = _mm_castps_si128(_mm_cmple_ps( point.sse, negHomogen));

          unsigned int maskP = _mm_movemask_epi8( resultP );
          unsigned int maskN = _mm_movemask_epi8( resultN );
          unsigned int mask = maskP << 16 | maskN;

          cfo |= mask;
          cfa &= mask;
        }
      }
      #else
        inline void determineCullFlags( const Vec4f &p, unsigned int & cfo, unsigned int & cfa )
        {
          unsigned int cf = 0;

          if ( p[0] <= -p[3] )
          {
            cf |= 0x01;
          }
          else if ( p[3] <= p[0] )
          {
            cf |= 0x02;
          }
          if ( p[1] <= -p[3] )
          {
            cf |= 0x04;
          }
          else if ( p[3] <= p[1] )
          {
            cf |= 0x08;
          }
          if ( p[2] <= -p[3] )
          {
            cf |= 0x10;
          }
          else if ( p[3] <= p[2] )
          {
            cf |= 0x20;
          }
          cfo |= cf;
          cfa &= cf;
        }
      #endif

      CullCode MatrixCamera::determineCullCode( const Sphere3f &sphere ) const
      {
        Vec3f center = sphere.getCenter();
        float radius = sphere.getRadius();

      #ifdef SCENIX_MATRIXCAMERA_USE_SSE
        Vec source( center[0], center[1], center[2], 1.0f );
        Vec r( radius, radius, radius, 0.0f );
        source -= r;

        Mat mat;
        memcpy( &mat.data, m_projection.getPtr(), 16*sizeof(float) );

      #if 1
        Vec vectors[8];
        vectors[0] = multiplyHomogen( source, mat );

        Vec r2( 2 * radius );
        Vec x,y,z;

        unsigned int cfo = 0;
        unsigned int cfa = 0xfff0fff; // set only the bits which are of interest. 8 bits per component.

        x.sse = _mm_mul_ps( mat.sse[0], r2.sse );
        y.sse = _mm_mul_ps( mat.sse[1], r2.sse );
        z.sse = _mm_mul_ps( mat.sse[2], r2.sse );
        vectors[1] = vectors[0] + x;
        vectors[2] = vectors[0] + y;
        vectors[4] = vectors[0] + z;

        vectors[3] = vectors[1] + y;
        vectors[5] = vectors[1] + z;
        vectors[6] = vectors[2] + z;
        vectors[7] = vectors[3] + z;

        for ( unsigned int i = 0;i < 8;++i )
        {
          updateCullCode( vectors[i], cfo, cfa );
        }
        cfo &= 0xfff0fff; // ignore the upper 8 bits of each word.
      #else // this version is slightly slower than the version obove.
        unsigned int cfo = 0;
        unsigned int cfa = 0xfff0fff; // set only the bits which are of interest. 8 bits per component.

        Vec vector;
        vector = multiplyHomogen( source, mat );

        Vec r2( 2 * radius );
        Vec x = _mm_mul_ps( mat.sse[0], r2.sse );
        Vec y = _mm_mul_ps( mat.sse[1], r2.sse );
        Vec z = _mm_mul_ps( mat.sse[2], r2.sse );

        updateCullCode( vector, cfo, cfa ); 
        vector += x;
        updateCullCode( vector, cfo, cfa );
        vector += y;
        updateCullCode( vector, cfo, cfa );
        vector -= x;
        updateCullCode( vector, cfo, cfa );
        vector += z;
        updateCullCode( vector, cfo, cfa );
        vector += x;
        updateCullCode( vector, cfo, cfa );
        vector -= y;
        updateCullCode( vector, cfo, cfa );
        vector -= x;
        updateCullCode( vector, cfo, cfa );

        cfo &= 0xfff0fff; // ignore the upper 8 bits of each word.
      #endif

      #else
        unsigned int cfo = 0;
        unsigned int cfa = ~0;

        Vec4f vectors[8];
        vectors[0] = Vec4f( center[0] - radius, center[1] - radius, center[2] - radius, 1.0f ) * m_projection;

        float r2 = 2.0f*radius;
        const float *projection = m_projection.getPtr();
        Vec4f x( r2 * projection[0], r2 * projection[1], r2 * projection[2], r2 * projection[3] );
        Vec4f y( r2 * projection[4], r2 * projection[5], r2 * projection[6], r2 * projection[7] );
        Vec4f z( r2 * projection[8], r2 * projection[9], r2 * projection[10], r2 * projection[11] );

        vectors[1] = vectors[0] + x;
        vectors[2] = vectors[0] + y;
        vectors[3] = vectors[1] + y;
        vectors[4] = vectors[0] + z;
        vectors[5] = vectors[1] + z;
        vectors[6] = vectors[2] + z;
        vectors[7] = vectors[3] + z;

        for ( unsigned int i = 0;i < 8; ++i )
        {
          determineCullFlags( vectors[i], cfo, cfa );
        }
      #endif
  
        return( !cfo ? CC_IN : cfa ? CC_OUT : CC_PART );
      }

      void MatrixCamera::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Camera::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_projection), sizeof(m_projection) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_inverse), sizeof(m_inverse) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
