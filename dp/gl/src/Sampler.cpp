// Copyright NVIDIA Corporation 2015
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


#include <dp/gl/Sampler.h>

namespace dp
{
  namespace gl
  {
    SamplerSharedPtr Sampler::create()
    {
      return( std::shared_ptr<Sampler>( new Sampler() ) );
    }

    Sampler::Sampler()
    {
      glGenSamplers( 1, &m_id );
    }

    Sampler::~Sampler( )
    {
      if ( m_id )
      {
        if ( getShareGroup() )
        {
          DEFINE_PTR_TYPES( CleanupTask );
          class CleanupTask : public ShareGroupTask
          {
            public:
              static CleanupTaskSharedPtr create( GLuint id )
              {
                return( std::shared_ptr<CleanupTask>( new CleanupTask( id ) ) );
              }

              virtual void execute() { glDeleteSamplers( 1, &m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
          };

          // make destructor exception safe
          try
          {
            getShareGroup()->executeTask( CleanupTask::create( m_id ) );
          } catch (...) {}
        }
        else
        {
          glDeleteSamplers(1, &m_id );
        }
      }
    }

    void Sampler::bind( GLuint unit )
    {
#if !defined(NDEBUG)
      DP_VERIFY( m_boundUnits.insert( unit ).second );
#endif
      glBindSampler( unit, m_id );
    }

    void Sampler::unbind( GLuint unit )
    {
#if !defined(NDEBUG)
      DP_VERIFY( m_boundUnits.erase( unit ) );
#endif
      glBindSampler( unit, 0 );
    }

    void Sampler::setBorderColor( float color[4] )
    {
      glSamplerParameterfv( m_id, GL_TEXTURE_BORDER_COLOR, color );
    }

    void Sampler::setBorderColor( int color[4] )
    {
      glSamplerParameteriv( m_id, GL_TEXTURE_BORDER_COLOR, color );
    }

    void Sampler::setBorderColor( unsigned int color[4] )
    {
      glSamplerParameterIuiv( m_id, GL_TEXTURE_BORDER_COLOR, color );
    }

    void Sampler::setCompareParameters( GLenum mode, GLenum func )
    {
      glSamplerParameteri( m_id, GL_TEXTURE_COMPARE_MODE, mode );
      glSamplerParameteri( m_id, GL_TEXTURE_COMPARE_FUNC, func );
    }

    void Sampler::setFilterParameters( GLenum minFilter, GLenum magFilter )
    {
      glSamplerParameteri( m_id, GL_TEXTURE_MIN_FILTER, minFilter );
      glSamplerParameteri( m_id, GL_TEXTURE_MAG_FILTER, magFilter );
    }

    void Sampler::setLODParameters( float minLOD, float maxLOD )
    {
      glSamplerParameterf( m_id, GL_TEXTURE_MIN_LOD, minLOD );
      glSamplerParameterf( m_id, GL_TEXTURE_MAX_LOD, maxLOD );
    }

    void Sampler::setWrapParameters( GLenum wrapS, GLenum wrapT, GLenum wrapR )
    {
      glSamplerParameteri( m_id, GL_TEXTURE_WRAP_S, wrapS );
      glSamplerParameteri( m_id, GL_TEXTURE_WRAP_T, wrapT );
      glSamplerParameteri( m_id, GL_TEXTURE_WRAP_R, wrapR );
    }

  } // namespace gl
} // namespace dp
