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


#include <dp/gl/RenderContextFormat.h>

#if defined(DP_OS_WINDOWS)
#include <GL/glew.h>
#include <GL/wglew.h>
#endif

#include <vector>
#include <cstring>

namespace dp
{
  namespace gl
  {

    bool isExtensionExported( char const *extName )
    {
      /*
        ** Search for extName in the extensions string.  Use of strstr()
        ** is not sufficient because extension names can be prefixes of
        ** other extension names.  Could use strtok() but the constant
        ** string returned by glGetString can be in read-only memory.
        */
      char *p = (char *) glGetString(GL_EXTENSIONS);
      char *end;
      size_t extNameLen;

      extNameLen = strlen(extName);
      end = p + strlen(p);

      while (p < end) {
          size_t n = strcspn(p, " ");
          if ((extNameLen == n) && (strncmp(extName, p, n) == 0)) {
              return( true );
          }
          p += (n + 1);
      }
      return( false );
    }

    RenderContextFormat::RenderContextFormat()
      : m_sRGB(false)
      , m_stereo(false)
      , m_thirtyBit(false)
      , m_msColorSamples(0)
      , m_msCoverageSamples(0)
      , m_stencil(false)
    {
    }

    RenderContextFormat::~RenderContextFormat()
    {
    }

    bool RenderContextFormat::operator==( const RenderContextFormat &rhs ) const
    {
      return m_sRGB == rhs.m_sRGB
          && m_stereo == rhs.m_stereo
          && m_thirtyBit == rhs.m_thirtyBit
          && m_msColorSamples == rhs.m_msColorSamples
          && m_msCoverageSamples == rhs.m_msCoverageSamples
          && m_stencil == rhs.m_stencil;
    }

    bool RenderContextFormat::operator!=( const RenderContextFormat &rhs ) const
    {
      return !((*this) == rhs);
    }

  #if defined(WIN32)
    struct HiddenWGLWindow
    {
      HDC   hdcCurrent;
      HGLRC hglrcCurrent;
      // Dummy window state:
      HWND  hwnd;
      HDC   hdc;
      HGLRC hglrc;
      int   pfmt;
      bool  initialized;

      bool opened()
      {
        return initialized; 
      }

      HiddenWGLWindow()
      : hdcCurrent(0)
      , hglrcCurrent(0)
      // Dummy window state:
      , hwnd(0)
      , hdc(0)
      , hglrc(0)
      , pfmt(0)
      , initialized(false)
      {
        hdcCurrent   = wglGetCurrentDC();
        hglrcCurrent = wglGetCurrentContext();

        // Create temporary dummy OpenGL context. 
        // Note that we only need this to initialize the WGL extensions.
        // After proper initialization of WGL extensions, we use WGLChoosePixelFormat
        // to get the requested pixel format.

        // !!!!! do not use WS_MINIMIZE this will not work with dual view !!!!!!!!!!
        hwnd = ::CreateWindow( "STATIC"
                             , NULL
                             , WS_CLIPCHILDREN | WS_CLIPSIBLINGS | CS_OWNDC
                             , 0, 0, 0, 0, NULL, NULL, NULL, NULL );

        if (hwnd)
        {
          hdc = ::GetDC(hwnd);
          if (hdc)
          {
            static PIXELFORMATDESCRIPTOR pfd = 
            {
              sizeof(PIXELFORMATDESCRIPTOR),                                //WORD nSize
                1,                                                          //WORD nVersion
                PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER, //DWORD dwFlags
                PFD_TYPE_RGBA,    //BYTE iPixelType
                32,               //BYTE cColorBits
                0,0,0,0,0,0,      //BYTE cRedBits BYTE cRedShift BYTE cGreenBits BYTE cGreenShift BYTE cBlueBits BYTE cBlueShift
                8,                //BYTE cAlphaBits
                0,                //BYTE cAlphaShift
                64,               //BYTE cAccumBits
                16,16,16,16,      //BYTE cAccumRedBits BYTE cAccumGreenBits BYTE cAccumBlueBits BYTE cAccumAlphaBits
                24,               //BYTE cDepthBits
                0,                //BYTE cStencilBits
                0,                //BYTE cAuxBuffers
                PFD_MAIN_PLANE,   //BYTE iLayerType
                0,                //BYTE bReserved
                0,0,0,            //DWORD dwLayerMask DWORD dwVisibleMask DWORD dwDamageMask
            }; 

            pfmt = ChoosePixelFormat(hdc, &pfd);

            if (pfmt > 0)
            {
              // so we fill the descriptor and we are able to check on HW acceleration
              DescribePixelFormat(hdc, pfmt, sizeof(PIXELFORMATDESCRIPTOR), &pfd);   

              // Check on HW acceleration
              if ( (pfd.dwFlags & PFD_GENERIC_FORMAT) 
                || (pfd.dwFlags & PFD_GENERIC_ACCELERATED) )
              {
                //reportError(log, "PFD_GENERIC_FORMAT", "No HW accelerated pixel format.");
              }
              else
              {
                if (SetPixelFormat(hdc, pfmt, &pfd))
                {
                  hglrc = wglCreateContext(hdc);
                  if (hglrc)
                  {
                    if (wglMakeCurrent(hdc, hglrc))
                    {
                      glewInit();
                      initialized = true;
                    }
                  }
                }
              }
            }
          }
        }
      }

      ~HiddenWGLWindow()
      {
        // shut everything down
        if( hdc )
        {
          wglMakeCurrent(hdc, NULL);
          if( hglrc )
          {
            wglDeleteContext(hglrc);
          }
          if( hwnd )
          {
            ::ReleaseDC(hwnd, hdc);
          }
        }

        if( hwnd )
        {
          ::DestroyWindow(hwnd);
        }

        // make sure original context is re-activated
        wglMakeCurrent( hdcCurrent, hglrcCurrent );
      }

    };

    int RenderContextFormat::getPixelFormat() const
    {
      HiddenWGLWindow wglwin;

      // initialize to invalid
      int pixelFormat = 0;

      if( wglwin.opened() )
      {
        // Now try to choose the right pixel format 
        // for the render area
        std::vector<int> iAttributes;

        iAttributes.push_back( WGL_DRAW_TO_WINDOW_ARB );
        iAttributes.push_back( GL_TRUE );
        iAttributes.push_back( WGL_SUPPORT_OPENGL_ARB );
        iAttributes.push_back( GL_TRUE );
        iAttributes.push_back( WGL_ACCELERATION_ARB );
        iAttributes.push_back( WGL_FULL_ACCELERATION_ARB );
        iAttributes.push_back( WGL_DEPTH_BITS_ARB );
        iAttributes.push_back( 24 );
        iAttributes.push_back( WGL_DOUBLE_BUFFER_ARB );
        iAttributes.push_back( GL_TRUE );

        if ( isStencil() )
        {
          iAttributes.push_back( WGL_STENCIL_BITS_ARB );
          iAttributes.push_back( 8 );
        }

        if ( isSRGB() && GLEW_EXT_framebuffer_sRGB )
        {
          iAttributes.push_back( WGL_FRAMEBUFFER_SRGB_CAPABLE_EXT );
          iAttributes.push_back( GL_TRUE );
        }

        unsigned int color;
        unsigned int coverage;
        getMultisampleCoverage( color, coverage );

        if( color )
        {
          iAttributes.push_back( WGL_SAMPLE_BUFFERS_ARB );
          iAttributes.push_back( GL_TRUE );

          if( GLEW_NV_multisample_coverage )
          {
            iAttributes.push_back( WGL_COLOR_SAMPLES_NV );
            iAttributes.push_back( color );
            iAttributes.push_back( WGL_COVERAGE_SAMPLES_NV );
            iAttributes.push_back( coverage );
          }
          else
          {
            iAttributes.push_back( WGL_SAMPLES_ARB );
            iAttributes.push_back( color );
          }
        }

        if ( isStereo() )
        {
          iAttributes.push_back( WGL_STEREO_ARB );
          iAttributes.push_back( GL_TRUE );
        }

        if( isThirtyBit() )
        {
          // keep accum bits unrequested if a 10-bpc format is requested!
          // for 10-bpc formats, accum bits are currently not exported 
          // by the driver! 
          //
          iAttributes.push_back( WGL_RED_BITS_ARB );
          iAttributes.push_back( 10 );
          iAttributes.push_back( WGL_GREEN_BITS_ARB );
          iAttributes.push_back( 10 );
          iAttributes.push_back( WGL_BLUE_BITS_ARB );
          iAttributes.push_back( 10 );
        }
        else
        {
          iAttributes.push_back( WGL_COLOR_BITS_ARB );
          iAttributes.push_back( 24 );
          iAttributes.push_back( WGL_ALPHA_BITS_ARB );
          iAttributes.push_back( 8 );

          // implicitly request accum bits for 8-bpc formats
          iAttributes.push_back( WGL_ACCUM_BITS_ARB );
          iAttributes.push_back( 64 );
        }

        iAttributes.push_back( 0 );
        iAttributes.push_back( 0 );

        // Now lets choose the pixel format
        unsigned int numFormats = 0;
        float fAttributes[] = {0.0f, 0.0f};
        if ( wglChoosePixelFormatARB( 
               wglwin.hdc,
               &iAttributes[0],
               fAttributes,
               1, // return only one format, so &pixelFormat is OK
               &pixelFormat,
               &numFormats ) != GL_TRUE )
        {
          pixelFormat = 0;
        }
      }

      return pixelFormat;
    }
  #elif defined(LINUX)

    DP_GL_API GLXFBConfig RenderContextFormat::getGLXFBConfig( Display *display, int screen ) const
    {
      std::vector<int> attributes;

      attributes.push_back( GLX_RENDER_TYPE );
      attributes.push_back( GLX_RGBA_BIT );

      attributes.push_back( GLX_DOUBLEBUFFER );
      attributes.push_back( True );

      if ( isStencil() )
      {
        attributes.push_back( GLX_STENCIL_SIZE );
        attributes.push_back( 8 );
      }

      if ( isSRGB() && isExtensionExported("GL_EXT_framebuffer_sRGB") )
      {
        attributes.push_back( GLX_FRAMEBUFFER_SRGB_CAPABLE_EXT );
        attributes.push_back( True );
      }

      unsigned int color;
      unsigned int coverage;
      getMultisampleCoverage( color, coverage );

      if( color )
      {
        attributes.push_back( GLX_SAMPLE_BUFFERS_ARB );
        attributes.push_back( True );

        if( isExtensionExported("GL_NV_multisample_coverage") )
        {
          attributes.push_back( GLX_COLOR_SAMPLES_NV );
          attributes.push_back( color );
          attributes.push_back( GLX_COVERAGE_SAMPLES_NV );
          attributes.push_back( coverage );
        }
        else
        {
          attributes.push_back( GLX_SAMPLES_ARB );
          attributes.push_back( color );
        }
      }

      if ( isStereo() )
      {
        attributes.push_back( GLX_STEREO );
        attributes.push_back( true );
      }
      if( isThirtyBit() )
      {
        // keep accum bits unrequested if a 10-bpc format is requested!
        // for 10-bpc formats, accum bits are currently not exported
        // by the driver!
        //

        attributes.push_back( GLX_RED_SIZE);
        attributes.push_back( 10 );
        attributes.push_back( GLX_GREEN_SIZE);
        attributes.push_back( 10 );
        attributes.push_back( GLX_BLUE_SIZE);
        attributes.push_back( 10 );
      }
      else
      {
        attributes.push_back( GLX_RED_SIZE);
        attributes.push_back( 8 );
        attributes.push_back( GLX_GREEN_SIZE);
        attributes.push_back( 8 );
        attributes.push_back( GLX_BLUE_SIZE);
        attributes.push_back( 8 );
        attributes.push_back( GLX_ALPHA_SIZE);
        attributes.push_back( 8 );

        // implicitly request accum bits for 8-bpc formats
        attributes.push_back( GLX_ACCUM_RED_SIZE);
        attributes.push_back( 16 );
        attributes.push_back( GLX_ACCUM_GREEN_SIZE);
        attributes.push_back( 16 );
        attributes.push_back( GLX_ACCUM_BLUE_SIZE);
        attributes.push_back( 16 );
        attributes.push_back( GLX_ACCUM_ALPHA_SIZE);
        attributes.push_back( 16 );
      }

      attributes.push_back( None );

      int          numberOfConfigs;
      GLXFBConfig *configs;

      configs = glXChooseFBConfig( display, screen, &attributes[0], &numberOfConfigs );

      return configs && numberOfConfigs ? configs[0] : 0;
    }
  #endif


  #if defined(WIN32)
    void RenderContextFormat::syncFormat( HDC hdc )
    {
      std::vector<int> attributes;
      std::vector<int> values;

      // thirty bit (0-2)
      attributes.push_back( WGL_RED_BITS_ARB );
      attributes.push_back( WGL_GREEN_BITS_ARB );
      attributes.push_back( WGL_BLUE_BITS_ARB );

      // stereo (3)
      attributes.push_back( WGL_STEREO_ARB );

      // stencil (4)
      attributes.push_back( WGL_STENCIL_BITS_ARB );

      // sRGB (5?)
      bool sRGB = !!GLEW_EXT_framebuffer_sRGB;

      if( sRGB )
      {
        attributes.push_back( WGL_FRAMEBUFFER_SRGB_CAPABLE_EXT );
      }

      size_t firstMSValue = attributes.size();

      bool multisample         = !!GLEW_ARB_multisample;
      bool multisampleCoverage = !!GLEW_NV_multisample_coverage;

      if (multisample)
      {
        // firstMSValue
        attributes.push_back( WGL_SAMPLE_BUFFERS_ARB );

        if( multisampleCoverage )
        {
          // firstMSValue+1
          attributes.push_back( WGL_COLOR_SAMPLES_NV );
          // firstMSValue+2?
          attributes.push_back( WGL_COVERAGE_SAMPLES_NV );
        }
        else
        {
          // firstMSValue+1
          attributes.push_back( WGL_SAMPLES_ARB );
        }
      }

      values.resize( attributes.size() );
    
      if (wglGetPixelFormatAttribivARB( hdc, GetPixelFormat( hdc ), 0, static_cast<UINT>(attributes.size()), &attributes[0], &values[0] ))
      {
        setThirtyBit(values[0] == 10 && values[1] == 10 && values[2] == 10);
        setStencil( values[4] != 0 );
        // MMM - shouldn't these be GL_FALSE instead?
        setStereo( values[3] != 0 );
        setSRGB( sRGB && values[5] != 0 );

        if ( multisample && values[firstMSValue] )
        {
          if( multisampleCoverage )
          {
            setMultisampleCoverage( values[firstMSValue+1], values[firstMSValue+2] );
          }
          else
          {
            setMultisample( values[firstMSValue+1] );
          }
        }
        else
        {
          setMultisample( 0 );
        }
      }
    }

  #elif defined(LINUX)
    int RenderContextFormat::getGLXFBConfigAttrib( Display *display, GLXFBConfig config, int attribute )
    {
      int value = 0;
      glXGetFBConfigAttrib( display, config, attribute, &value );
      return value;
    }

    GLXFBConfig RenderContextFormat::getGLXFBConfig( Display *display, GLXContext context )
    {
      // fetch fbconfig id for current drawable
      int glxFBConfigId = 0;
      glXQueryContext( display, context, GLX_FBCONFIG_ID, &glxFBConfigId);

      int screen = 0;
      glXQueryContext( display, context, GLX_SCREEN, &screen);

      // fetch GLXFBConfig object for glxFBConfigId
      int numElements = 0;
      const int glxFBConfigAttributes[] = {GLX_FBCONFIG_ID, glxFBConfigId, None, None};
      GLXFBConfig *glxFBConfig = glXChooseFBConfig( display, screen, glxFBConfigAttributes, &numElements );
      DP_ASSERT( glxFBConfig );

      return glxFBConfig[0];
    }

    void RenderContextFormat::syncFormat( Display *display, GLXContext context )
    {
      GLXFBConfig config = getGLXFBConfig( display, context );
      setThirtyBit(  getGLXFBConfigAttrib( display, config, GLX_RED_SIZE) == 10
                  && getGLXFBConfigAttrib( display, config, GLX_GREEN_SIZE) == 10
                  && getGLXFBConfigAttrib( display, config, GLX_BLUE_SIZE) == 10 );

      setStencil( getGLXFBConfigAttrib( display, config, GLX_STENCIL_SIZE ) == 8 );

      setSRGB( isExtensionExported("GL_EXT_framebuffer_sRGB") 
               && getGLXFBConfigAttrib( display, config, GLX_FRAMEBUFFER_SRGB_CAPABLE_EXT ) == True );

      if ( getGLXFBConfigAttrib( display, config, GLX_SAMPLE_BUFFERS_ARB ) == True )
      {
        if( isExtensionExported("GL_NV_multisample_coverage") )
        {
          setMultisampleCoverage( getGLXFBConfigAttrib( display, config, GLX_COLOR_SAMPLES_NV ),
                                  getGLXFBConfigAttrib( display, config, GLX_COVERAGE_SAMPLES_NV ));
        }
        else
        {
          setMultisample( getGLXFBConfigAttrib( display, config, GLX_SAMPLES_ARB ) );
        }
      }

      setStereo( getGLXFBConfigAttrib( display, config, GLX_STEREO ) == True );
    }
  #endif

    bool RenderContextFormat::isAvailable() const
    {
  #if defined(WIN32)
      return getPixelFormat() != 0;
  #elif defined( LINUX )
      Display  * display = glXGetCurrentDisplay();
      GLXContext context = glXGetCurrentContext(); 
      int screen = 0;
      glXQueryContext( display, context, GLX_SCREEN, &screen);

      return !!getGLXFBConfig( display, screen );
  #else
      return false;
  #endif
    }

    bool RenderContextFormat::getFormatInfo( FormatInfo & info ) const
    {
      // give some preliminary info here
      info.sRGBSupported       = !!GLEW_EXT_framebuffer_sRGB;
      info.msSupported         = !!GLEW_ARB_multisample;
      info.msCoverageSupported = !!GLEW_NV_multisample_coverage;

  #if defined( WIN32 )
      HiddenWGLWindow wglwin;

      if(! wglwin.opened() )
      {
        return false;
      }

      int attrib = WGL_NUMBER_PIXEL_FORMATS_ARB;
      int numFormats = 0;

      if (!wglGetPixelFormatAttribivARB(wglwin.hdc, 1, 0, 1, &attrib, &numFormats)) 
      {
        return false;
      }

      if( !numFormats )
      {
        return false;
      }

      for(int i = 1; i <= numFormats; i++)
      {
        unsigned int color = 0;
        unsigned int coverage = 0;

        if( info.msCoverageSupported )
        {
          int values[2];
          int attribs[] = 
          {
            WGL_COLOR_SAMPLES_NV,
            WGL_COVERAGE_SAMPLES_NV
          };

          wglGetPixelFormatAttribivARB(wglwin.hdc, i, 0, 2, attribs, values);
          color    = (unsigned int) values[0];
          coverage = (unsigned int) values[1];
        }
        else
        {
          int value;
          attrib = WGL_SAMPLES_ARB;
          wglGetPixelFormatAttribivARB(wglwin.hdc, i, 0, 1, &attrib, &value);
          color = coverage = (unsigned int) value;
        }

        int attribs[] = 
        {
          WGL_STEREO_ARB,
          WGL_STENCIL_BITS_ARB,
          WGL_RED_BITS_ARB,
          WGL_GREEN_BITS_ARB,
          WGL_BLUE_BITS_ARB
        };

        int values[5];

        wglGetPixelFormatAttribivARB(wglwin.hdc, i, 0, 5, attribs, values);

        info.stereoSupported    = info.stereoSupported  || ( values[0] == GL_TRUE );
        // we could really check for STENCIL_SIZE > 1
        info.stencilSupported   = info.stencilSupported || ( values[1] == 8 );
        info.thirtyBitSupported = info.thirtyBitSupported || ( values[2] == 10 && values[3] == 10 && values[4] == 10 );

        // store all modes, so one will contain 0 samples, which is valid
        info.multisampleModesSupported.insert( std::make_pair( color, coverage ) );
      }

      return true;

  #elif defined( LINUX )
      Display  * display = glXGetCurrentDisplay();
      GLXContext context = glXGetCurrentContext(); 
      int screen = 0;
      glXQueryContext( display, context, GLX_SCREEN, &screen);

      if( !display || !context )
      {
        return false;
      }

      // get all the configs
      int count = 0;
      GLXFBConfig * config = glXGetFBConfigs( display, screen, &count );

      if( !count || !config )
      {
        return false;
      }

      for( int i = 0; i < count; i ++ )
      {
        unsigned int color = 0;
        unsigned int coverage = 0;

        if( info.msCoverageSupported )
        {
          color    = getGLXFBConfigAttrib( display, config[i], GLX_COLOR_SAMPLES_NV );
          coverage = getGLXFBConfigAttrib( display, config[i], GLX_COVERAGE_SAMPLES_NV );
        }
        else
        {
          color = coverage = getGLXFBConfigAttrib( display, config[i], GLX_SAMPLES_ARB );
        }

        info.stereoSupported    = info.stereoSupported  || ( getGLXFBConfigAttrib( display, config[i], GLX_STEREO ) == True );
        // we could really check for STENCIL_SIZE > 1
        info.stencilSupported   = info.stencilSupported || ( getGLXFBConfigAttrib( display, config[i], GLX_STENCIL_SIZE ) == 8 );
        info.thirtyBitSupported = info.thirtyBitSupported || (   getGLXFBConfigAttrib( display, config[i], GLX_RED_SIZE) == 10
                                                              && getGLXFBConfigAttrib( display, config[i], GLX_GREEN_SIZE) == 10
                                                              && getGLXFBConfigAttrib( display, config[i], GLX_BLUE_SIZE) == 10 );

        // store all modes, so one will contain 0 samples, which is valid
        info.multisampleModesSupported.insert( std::make_pair( color, coverage ) );
      }

      return true;
  #endif
    }
  } // namespace gl
} // namespace dp
