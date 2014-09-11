// Copyright NVIDIA Corporation 2010-2011
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

#include <dp/gl/Config.h>
#include <dp/util/Types.h>
#include <set>

#if defined(DP_OS_WINDOWS)
#include <windows.h>
#elif defined(DP_OS_LINUX)
#include <GL/glxew.h>
#endif

namespace dp
{
  namespace gl
  {

    DP_GL_API bool isExtensionExported( char const *extName );

    class RenderContextFormat
    {
    public:
      DP_GL_API RenderContextFormat();
      DP_GL_API virtual ~RenderContextFormat();

      DP_GL_API void setStereo(bool stereo);
      DP_GL_API bool isStereo() const;

      DP_GL_API void setThirtyBit(bool thirtyBit);
      DP_GL_API bool isThirtyBit() const;

      DP_GL_API void setSRGB(bool sRGB);
      DP_GL_API bool isSRGB() const;

      DP_GL_API void setMultisample( unsigned int samples ); //<! 0 -> do not request multisampling fb
      DP_GL_API unsigned int getMultisample( ) const;

      // requires NV_multisample_coverage extension
      DP_GL_API void setMultisampleCoverage( unsigned int colorSamples, unsigned int coverageSamples );
      DP_GL_API void getMultisampleCoverage( unsigned int & colorSamples, unsigned int & coverageSamples ) const;

      DP_GL_API void setStencil( bool stencil );
      DP_GL_API bool isStencil() const;

      struct FormatInfo
      {
        FormatInfo()
          : stereoSupported( false )
          , stencilSupported( false )
          , thirtyBitSupported( false )
          , sRGBSupported( false )
          , msSupported( false )
          , msCoverageSupported( false )
        {}

        bool stereoSupported;
        bool stencilSupported;
        bool thirtyBitSupported;
        bool sRGBSupported;
        bool msSupported;
        bool msCoverageSupported;
        std::set< std::pair < unsigned int, unsigned int > > multisampleModesSupported;
      };

      //
      // NOTE: both of these require a current context!
      //
      DP_GL_API bool getFormatInfo( FormatInfo & info ) const;
      DP_GL_API bool isAvailable() const;

      DP_GL_API bool operator==( const RenderContextFormat &rhs ) const;
      DP_GL_API bool operator!=( const RenderContextFormat &rhs ) const;

  #if defined(DP_OS_WINDOWS)
      DP_GL_API int getPixelFormat() const;
      DP_GL_API void syncFormat( HDC hdc );
  #elif defined(DP_OS_LINUX)
      DP_GL_API GLXFBConfig getGLXFBConfig( Display *display, int screen ) const;
      DP_GL_API static GLXFBConfig getGLXFBConfig( Display *display, GLXContext context );
      DP_GL_API static int getGLXFBConfigAttrib( Display *display, GLXFBConfig config, int attribute );
      DP_GL_API void syncFormat( Display *display, GLXContext context );
  #endif

    protected:
      bool m_sRGB;
      bool m_stereo;
      bool m_thirtyBit;
      unsigned m_msColorSamples;
      unsigned m_msCoverageSamples;
      bool m_stencil;
    };

    inline void RenderContextFormat::setStereo(bool stereo)
    {
      m_stereo = stereo;
    }

    inline bool RenderContextFormat::isStereo() const
    {
      return m_stereo;
    }

    inline void RenderContextFormat::setThirtyBit(bool thirtyBit)
    {
      m_thirtyBit = thirtyBit;
    }

    inline bool RenderContextFormat::isThirtyBit() const
    {
      return m_thirtyBit;
    }

    inline void RenderContextFormat::setSRGB(bool sRGB)
    {
      m_sRGB = sRGB;
    }

    inline bool RenderContextFormat::isSRGB() const
    {
      return m_sRGB;
    }

    inline void RenderContextFormat::setMultisample( unsigned int multisample )
    {
      m_msColorSamples    = multisample;
      m_msCoverageSamples = multisample;
    }

    inline unsigned int RenderContextFormat::getMultisample( ) const
    {
      return m_msColorSamples;
    }

    inline void RenderContextFormat::setMultisampleCoverage( unsigned int color, unsigned int coverage )
    {
      m_msColorSamples = color;

      // can't have less coverage samples than color samples
      m_msCoverageSamples = std::max( color, coverage );
    }

    inline void RenderContextFormat::getMultisampleCoverage( unsigned int & color, unsigned int & coverage ) const
    {
      color    = m_msColorSamples;
      coverage = m_msCoverageSamples;
    }

    inline void RenderContextFormat::setStencil( bool stencil )
    {
      m_stencil = stencil;
    }

    inline bool RenderContextFormat::isStencil() const
    {
      return m_stencil;
    }

  } // namespace gl
} // namespace dp
