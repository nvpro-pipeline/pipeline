// Copyright NVIDIA Corporation 2010
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


#include <dp/gl/Renderbuffer.h>

namespace dp
{
  namespace gl
  {
    Renderbuffer::Renderbuffer(GLenum internalFormat, int width, int height)
      : m_colorSamples( 0 )
      , m_coverageSamples( 0 )
      , m_width(0)
      , m_height(0)
      , m_internalFormat( internalFormat )
      , m_resizeFunc( &Renderbuffer::resizeNoAA )
    {
      init( width, height );
    }

    Renderbuffer::Renderbuffer(const MSAA &msaa, GLenum internalFormat, int width, int height)
      : m_colorSamples( msaa.m_colorSamples )
      , m_coverageSamples( 0 )
      , m_width(0)
      , m_height(0)
      , m_internalFormat( internalFormat )
      , m_resizeFunc( &Renderbuffer::resizeMSAA ) 
    {
      DP_ASSERT( isMSAAAvailable() );
      init( width, height );
    }

    Renderbuffer::Renderbuffer(const CSAA &csaa, GLenum internalFormat, int width, int height)
      : m_colorSamples( csaa.m_colorSamples )
      , m_coverageSamples( csaa.m_coverageSamples )
      , m_width(0)
      , m_height(0)
      , m_internalFormat( internalFormat )
      , m_resizeFunc( &Renderbuffer::resizeCSAA ) 
    {
      DP_ASSERT( isCSAAAvailable() );
      init( width, height );
    }

    dp::util::SmartPtr<Renderbuffer> Renderbuffer::create(GLenum internalFormat, int width, int height )
    {
      return new Renderbuffer( internalFormat, width, height );
    }

    dp::util::SmartPtr<Renderbuffer> Renderbuffer::create(const MSAA &msaa, GLenum internalFormat, int width, int height )
    {
      return new Renderbuffer( msaa, internalFormat, width, height );
    }

    dp::util::SmartPtr<Renderbuffer> Renderbuffer::create(const CSAA &csaa, GLenum internalFormat, int width, int height )
    {
      return new Renderbuffer( csaa, internalFormat, width, height );
    }

    void Renderbuffer::init( int width, int height )
    {
      GLuint id;
      glGenRenderbuffers( 1, &id );
      setGLId( id );

      resize( width, height );
    }

    Renderbuffer::~Renderbuffer()
    {
      class CleanupTask : public ShareGroupTask
      {
        public:
          CleanupTask( GLuint id ) : m_id( id ) {}

          virtual void execute() { glDeleteRenderbuffers( 1, &m_id ); }

        private:
          GLuint m_id;
      };

      if ( getGLId() && getShareGroup() )
      {
        // make destructor exception safe
        try
        {
          getShareGroup()->executeTask( new CleanupTask( getGLId() ) );
        } catch (...) {}
      }
    }

    void Renderbuffer::resize( int width, int height )
    {
      if ( m_width != width || m_height != height )
      {
        m_width = width;
        m_height = height;
        glBindRenderbuffer( GL_RENDERBUFFER_EXT, getGLId() );
        (this->*m_resizeFunc)();
        glBindRenderbuffer( GL_RENDERBUFFER_EXT, 0 );
      }
    }

    void Renderbuffer::resizeNoAA()
    {
      glRenderbufferStorage( GL_RENDERBUFFER_EXT, m_internalFormat, m_width, m_height );
    }

    void Renderbuffer::resizeMSAA()
    {
      glRenderbufferStorageMultisample( GL_RENDERBUFFER_EXT, m_colorSamples, m_internalFormat, m_width, m_height );
    }

    void Renderbuffer::resizeCSAA()
    {
      glRenderbufferStorageMultisampleCoverageNV( GL_RENDERBUFFER_EXT, m_coverageSamples, m_colorSamples, m_internalFormat, m_width, m_height );
    }

    bool Renderbuffer::isMSAAAvailable()
    {
      return /*!!GLEW_ARB_framebuffer_object || */!!GLEW_EXT_framebuffer_multisample;
    }

    bool Renderbuffer::isCSAAAvailable()
    {
      return !!GLEW_NV_framebuffer_multisample_coverage;
    }

    GLint Renderbuffer::getMaxMSAASamples()
    {
      if ( !isMSAAAvailable() )
      {
        return 0;
      }
      else
      {
        GLint maxSamples;
        glGetIntegerv( GL_MAX_SAMPLES_EXT, &maxSamples );
        return maxSamples;
      }
    }

    std::vector<Renderbuffer::CSAA> Renderbuffer::getAvailableCSAAModes()
    {
      std::vector<CSAA> csaaModes;

      if ( isCSAAAvailable() )
      {
        // Get the number of unique multisample coverage modes
        int numModes = 0;
        glGetIntegerv( GL_MAX_MULTISAMPLE_COVERAGE_MODES_NV, &numModes );

        if ( numModes )
        {
          // Provide room for the pairs of coverageSamples and colorSamples.
          std::vector<int> modes( 2 * numModes );

          // Get the list of modes.
          glGetIntegerv( GL_MULTISAMPLE_COVERAGE_MODES_NV, &modes[0] );
        
          for ( std::vector<int>::iterator it = modes.begin(); it != modes.end(); it += 2 )
          {
            csaaModes.push_back( CSAA(it[0], it[1]) );
          }
        }
      }
      return csaaModes;
    }
  }
} // namespace nvgl
