// Copyright NVIDIA Corporation 2009-2014
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


#include <dp/gl/RenderContext.h>
#include <boost/thread.hpp>

#include <GL/glew.h>

#if defined(DP_OS_WINDOWS)
#include <GL/wglew.h>
#endif

#if defined(DP_OS_LINUX)
#define INIT_CONTEXT_VARIABLES m_display(0), m_context(0), m_drawable(0)
#endif

namespace dp
{
  namespace gl
  {
    namespace {
      struct ThreadData
      {
        ~ThreadData() {}
        SharedRenderContext currentRenderContext;
      };

      boost::thread_specific_ptr<ThreadData> t_threadData;

      ThreadData *getThreadData()
      {
        if ( !t_threadData.get() )
        {
          t_threadData.reset( new ThreadData );
        }
        return t_threadData.get();
      }
    }

  #if defined(DP_OS_WINDOWS)
    RenderContext::NativeContext::NativeContext( HWND hwnd, bool destroyHWND, HDC hdc, bool destroyHDC, HGLRC hglrc, bool destroyHGLRC )
      : m_hwnd( hwnd )
      , m_hdc( hdc )
      , m_hglrc( hglrc )
      , m_destroyHWND( destroyHWND )
      , m_destroyHDC( destroyHDC )
      , m_destroyHGLRC( destroyHGLRC )
    {
      // get OpenGL version, only works on OpenGL 3 or higher
      glGetIntegerv( GL_MAJOR_VERSION, &m_majorVersion );
      glGetIntegerv( GL_MINOR_VERSION, &m_minorVersion );

      // query the gpu mask from the given HDC
      if (WGLEW_NV_gpu_affinity)
      {
        UINT gpuIndex = 0;
        HGPUNV gpu;
        while (wglEnumGpusFromAffinityDCNV(m_hdc, gpuIndex++, &gpu))
        {
          m_gpuMask.push_back(gpu);
        }
      }

    }
  #elif defined(DP_OS_LINUX)
    RenderContext::NativeContext::NativeContext( GLXContext context, bool destroyContext, GLXDrawable drawable, bool destroyDrawable, GLXPbuffer pbuffer, bool destroypbuffer, Display *display, bool destroyDisplay )
      : m_context(context)
      , m_destroyContext( destroyContext )
      , m_drawable( drawable )
      , m_destroyDrawable( destroyDrawable )
      , m_pbuffer( pbuffer )
      , m_destroypBuffer( destroypbuffer )
      , m_display( display )
      , m_destroyDisplay( destroyDisplay )
    {
    }
  #endif

    RenderContext::NativeContext::~NativeContext()
    {
      // destroy
  #if defined(DP_OS_WINDOWS)
      if (m_destroyHGLRC)
      {
        wglDeleteContext( m_hglrc );
        m_hglrc = 0;
      }

      if ( m_destroyHDC )
      {
        ::ReleaseDC( m_hwnd, m_hdc );
        m_hdc = 0;
      }

      if (m_destroyHWND)
      {
        ::DestroyWindow( m_hwnd );
        m_hwnd = 0;
      }

  #elif defined(DP_OS_LINUX)
      if ( m_destroyContext )
      {
        glXDestroyContext( m_display, m_context );
      }
      if ( m_pbuffer && m_destroypBuffer)
      {
        glXDestroyPbuffer( m_display, m_pbuffer );
      }
      if ( m_display && m_destroyDisplay )
      {
        XCloseDisplay( m_display );
      }
  #endif
    }

    void RenderContext::NativeContext::notifyDestroyed()
    {
  #if defined(WIN32)
      if (m_hglrc)
      {
        m_hglrc = 0;
        m_destroyHGLRC = false;
      }
      m_hdc = 0;
      m_hwnd = 0;
  #elif defined(DP_OS_LINUX)
  #endif
    }

    bool RenderContext::NativeContext::makeCurrent()
    {
  #if defined(DP_OS_WINDOWS)
      if ( wglGetCurrentDC() != m_hdc || wglGetCurrentContext() != m_hglrc )
      {
        BOOL result = wglMakeCurrent( m_hdc, m_hglrc );
        DP_ASSERT( result );
        return result == TRUE;
      }
      else
      {
        return true;
      }
  #elif defined(DP_OS_LINUX)
      return !!glXMakeCurrent( m_display, m_drawable, m_context);
  #endif
    }

    void RenderContext::NativeContext::makeNoncurrent()
    {
  #if defined(DP_OS_WINDOWS)
      // FIXME check for failure
      wglMakeCurrent( 0, 0);
  #elif defined(DP_OS_LINUX)
      glXMakeCurrent( m_display, 0, 0 );
  #endif
    }

    void RenderContext::NativeContext::swap()
    {
  #if defined(DP_OS_WINDOWS)
      DP_VERIFY( SwapBuffers( m_hdc ) );
  #elif defined(DP_OS_LINUX)
      glXSwapBuffers( m_display, m_drawable );
  #endif
    }


    RenderContext::RenderContext()
    {
    }

    RenderContext::~RenderContext()
    {
    }

    void RenderContext::notifyDestroy()
    {
      m_context->notifyDestroyed();
    }

    const RenderContextFormat &RenderContext::getFormat() const
    {
      return m_format;
    }

  #if defined(DP_OS_WINDOWS)
    HWND RenderContext::getHWND() const
    {
      return m_context->m_hwnd;
    }

    HDC RenderContext::getHDC() const
    {
      return m_context->m_hdc;
    }

    HGLRC RenderContext::getHGLRC() const
    {
      return m_context->m_hglrc;
    }

    std::vector<HGPUNV> RenderContext::enumGpusNV() const
    {
      std::vector<HGPUNV> gpus;

      if(WGLEW_NV_gpu_affinity)
      {
        HGPUNV gpu;
        for (UINT gpuIndex = 0;wglEnumGpusNV(gpuIndex, &gpu);++gpuIndex)
        {
          gpus.push_back(gpu);
        }
      }

      return gpus;
    }

    std::vector<GPU_DEVICE> RenderContext::enumGpuDevicesNV(HGPUNV gpu) const
    {
      std::vector<GPU_DEVICE> devices;

      if(WGLEW_NV_gpu_affinity)
      {
        GPU_DEVICE device;
        for (UINT deviceIndex = 0;wglEnumGpuDevicesNV(gpu, deviceIndex, &device);++deviceIndex)
        {
          devices.push_back(device);
        }
      }
      
      return devices;
    }

  #elif defined(DP_OS_LINUX)
    GLXContext RenderContext::getContext()  const
    {
      return m_context->m_context;
    }
    GLXDrawable RenderContext::getDrawable() const
    {
      return m_context->m_drawable;
    }

    Display* RenderContext::getDisplay()  const
    {
      return m_context->m_display;
    }
  #endif

  #if defined(DP_OS_WINDOWS)
    LRESULT CALLBACK RenderContext::RenderContextWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
      return DefWindowProc( hWnd, message, wParam, lParam );
    }

    HWND RenderContext::createWindow(int width, int height)
    {
      // FIXME RegisterClass should be called only once
      // FIXME error managment
      WNDCLASS wc;

      // register dummy window class, move creation to nvsg.cpp
      wc.style = CS_OWNDC;
      wc.lpfnWndProc = RenderContextWndProc;
      wc.cbClsExtra = 0;
      wc.cbWndExtra = 0;
      //wc.hInstance = AfxGetInstanceHandle();
      wc.hInstance = GetModuleHandle( NULL );
      wc.hIcon = LoadIcon( NULL, IDI_APPLICATION );
      wc.hCursor = LoadCursor( NULL, IDC_ARROW );
      wc.hbrBackground = (HBRUSH)GetStockObject( BLACK_BRUSH );
      wc.lpszMenuName = NULL;
      wc.lpszClassName = "ScenixGL";
      RegisterClass( &wc );

      // create a dummy OpenGL Window
      HWND hwnd = CreateWindow( 
        "ScenixGL", "ScenixGL", 
        WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_POPUP,
        0, 0, width, height,
        NULL, NULL, GetModuleHandle( NULL ), NULL );

      return hwnd;
    }

  #define WGL_CONTEXT_MAJOR_VERSION_ARB  0x2091
  #define WGL_CONTEXT_MINOR_VERSION_ARB  0x2092
  #define WGL_CONTEXT_LAYER_PLANE_ARB   0x2093
  #define WGL_CONTEXT_FLAGS_ARB         0x2094
  #define WGL_CONTEXT_PROFILE_MASK_ARB  0x9126

  #define WGL_CONTEXT_DEBUG_BIT_ARB               0x0001
  #define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB  0x0002

  #define WGL_CONTEXT_CORE_PROFILE_BIT_ARB          0x00000001
  #define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002

  #define GL_ERROR_INVALID_VERSION_ARB 0x2095
  #define GL_ERROR_INVALID_PROFILE_ARB 0x2096

    HGLRC RenderContext::createContext( HDC hdc, HGLRC shareContext )
    {
      HGLRC currentContext = wglGetCurrentContext();
      HDC currentDC = wglGetCurrentDC();

      HGLRC context = wglCreateContext( hdc );
      wglMakeCurrent( hdc, context );

      PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
      if ( context && wglGetExtensionsStringARB )
      {
        bool arbCreateContextSupported = false;
        std::istringstream is( wglGetExtensionsStringARB( hdc ) );
        while ( !is.eof() && !arbCreateContextSupported )
        {
          std::string extension;
          is >> extension;
          arbCreateContextSupported = extension == "WGL_ARB_create_context";
        }

        if ( arbCreateContextSupported )
        {

  #ifndef GLAPIENTRY
  #  if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)  /* Mimic <wglext.h> */
  #   define GLAPIENTRY __stdcall
  #  else
  #   define GLAPIENTRY
  #  endif
  #endif

          typedef HGLRC (GLAPIENTRY *PFNWGLCREATECONTEXTATTRIBSARB)(HDC hDC, HGLRC hShareContext, const int *attribList);
          PFNWGLCREATECONTEXTATTRIBSARB wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARB)wglGetProcAddress("wglCreateContextAttribsARB");
          DP_ASSERT( wglCreateContextAttribsARB );

  #if defined(NDEBUG)
          int attribs[] =
          {
            //WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
            0
          };
  #else
          int attribs[] =
          {
            //WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB | WGL_CONTEXT_DEBUG_BIT_ARB, // WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB not running on AMD
            WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
            0
          };
  #endif

          HGLRC hglrcNew = wglCreateContextAttribsARB( hdc, shareContext, attribs );
          wglMakeCurrent( NULL, NULL );
          wglDeleteContext( context );

          context = hglrcNew;
          DP_ASSERT( context );

          if ( shareContext )
          {
  #if 0
            nvgl::WGLNotifyShareLists( shareContext, context );
  #endif
          }
        }
        else
        {
          DP_ASSERT( false );
  #if 0
          nvgl::WGLAttachContext( hdc, context );
          // context construction failed
          if ( context && shareContext && !nvgl::WGLShareLists( shareContext, context) )
          {
            // Sharelists failed. Delete context and set return value to 0.
            nvgl::WGLDeleteContext( context );
            context = 0;
          }
  #endif
        }
      }
      else if (context)
      {
        DP_ASSERT( false );
  #if 0
        nvgl::WGLAttachContext( hdc, context );
        // context construction failed
        if ( context && shareContext && !nvgl::WGLShareLists( shareContext, context) )
        {
          // Sharelists failed. Delete context and set return value to 0.
          nvgl::WGLDeleteContext( context );
          context = 0;
        }
  #endif
      }

      wglMakeCurrent( currentDC, currentContext );
      // context construction successful
      return context;
    }

  // WIN32
  #endif 

  #if defined(DP_OS_LINUX)

    GLXPbuffer RenderContext::createPbuffer( Display *display, GLXFBConfig config )
    {
      // create a small dummy pbuffer for the context
      static int pbattrs[] =
      {
        GLX_PBUFFER_WIDTH, 16,
        GLX_PBUFFER_HEIGHT, 16,
        0
      };
      GLXPbuffer pbuffer = glXCreatePbuffer(display, config, pbattrs);
      DP_ASSERT(pbuffer);
      return pbuffer;
    }

    GLXContext RenderContext::createContext( Display *display, GLXFBConfig config, GLXContext shareContext )
    {
      GLXContext context = glXCreateNewContext( display, config, GLX_RGBA_TYPE, shareContext, true );
      DP_ASSERT( context );

      return context;
    }

  #endif

    bool RenderContext::makeCurrent()
    {
      bool result = m_context->makeCurrent();
      if ( result && ( getThreadData()->currentRenderContext.getWeakPtr() != this ) )
      {
        getThreadData()->currentRenderContext = shared_from_this();
      }

      return result;
    }

    void RenderContext::makeNoncurrent()
    {
      getThreadData()->currentRenderContext.reset();
      m_context->makeNoncurrent();
    }

    void RenderContext::swap()
    {
      return m_context->swap();
    }

    ShareGroupTask::~ShareGroupTask()
    {
    }

    SharedShareGroup RenderContext::createShareGroup( const RenderContext::SharedNativeContext &shareContext )
    {
  #if defined(DP_OS_WINDOWS)
      DP_ASSERT( shareContext && shareContext->m_hglrc );

      // First create a headless context for the ShareGroup
      HWND hwnd = createWindow(); // create a dummy window
      DP_ASSERT( hwnd );

      HDC hdc;
      if( shareContext->m_gpuMask.empty() )
      {
        hdc = GetDC( hwnd ); // create a default dc
      }
      else
      {
        std::vector<HGPUNV> gpus;
        std::copy(shareContext->m_gpuMask.begin(), shareContext->m_gpuMask.end(), std::back_inserter(gpus));
        gpus.push_back(nullptr);
        hdc = wglCreateAffinityDCNV( gpus.data() );
      }
      DP_ASSERT( hdc );

      PIXELFORMATDESCRIPTOR pfd;
      bool ok = !!SetPixelFormat( hdc, GetPixelFormat( shareContext->m_hdc ), &pfd );
      DP_ASSERT( ok );

      HGLRC hglrc = createContext( hdc, shareContext->m_hglrc );
      DP_ASSERT( hglrc );

      return ShareGroup::create( SharedNativeContext( new RenderContext::NativeContext( hwnd, true, hdc, true, hglrc, true ) ) );
  #elif defined(DP_OS_LINUX)
      Display *display = XOpenDisplay( DisplayString(shareContext->m_display) );
      DP_ASSERT( display );

      GLXFBConfig config = RenderContextFormat::getGLXFBConfig( display, shareContext->m_context );

      GLXPbuffer pbuffer = createPbuffer( display, config );
      DP_ASSERT( pbuffer );

      GLXDrawable drawable = pbuffer;

      GLXContext newContext = createContext( display, config, shareContext->m_context );
      DP_ASSERT( newContext );

      return ShareGroup::create( new RenderContext::NativeContext( newContext, true, drawable, false, pbuffer, true, display, true ));
  #endif
    }

    SharedRenderContext RenderContext::create( const Attach &creation )
    {
  #if defined(DP_OS_WINDOWS)
      HDC hdc = wglGetCurrentDC();
      DP_ASSERT( hdc );

      HGLRC hglrc = wglGetCurrentContext();
      DP_ASSERT( hglrc );

      glewInit();

      SharedNativeContext nativeContext =  SharedNativeContext( new RenderContext::NativeContext( 0, false, hdc, false, hglrc, false) );

      if ( creation.getContext() )
      {
  #if 0
        nvgl::WGLNotifyShareLists( creation.getContext()->getHGLRC(), hglrc );
  #endif
      }

  #elif defined(DP_OS_LINUX)
      // query the current render context, drawable and display
      GLXContext context  = glXGetCurrentContext();
      DP_ASSERT(context);

      GLXDrawable drawable = glXGetCurrentDrawable();
      DP_ASSERT( drawable );

      Display* display = glXGetCurrentDisplay();

      SharedNativeContext nativeContext = new RenderContext::NativeContext( context, false, drawable, false, 0, false, display, false );
  #endif

      SharedShareGroup shareGroup;
      if ( creation.getContext() )
      {
        shareGroup = creation.getContext()->getShareGroup();
      }
      else
      {
        shareGroup = createShareGroup( nativeContext );
      }

      return( std::shared_ptr<RenderContext>( new RenderContext( nativeContext, shareGroup ) ) );
    }

    SharedRenderContext RenderContext::create( const Clone &creation )
    {
  #if defined(DP_OS_WINDOWS)
      HDC hdc = creation.getContext()->m_context->m_hdc;
      DP_ASSERT( hdc );

      //HGLRC hglrc = nvgl::WGLCreateContext( hdc );
      HGLRC hglrc = createContext( hdc, creation.isShared() ? creation.getContext()->m_context->m_hglrc : NULL );
      DP_ASSERT( hglrc );

      SharedNativeContext nativeContext = SharedNativeContext( new RenderContext::NativeContext( 0, false, hdc, true, hglrc, true ) );
  #elif defined(DP_OS_LINUX)
      Display *display = XOpenDisplay( DisplayString(creation.getContext()->getDisplay()) );
      DP_ASSERT( display );

      GLXFBConfig config = RenderContextFormat::getGLXFBConfig( display, creation.getContext()->getContext() );
      GLXDrawable drawable = creation.getContext()->getDrawable();

      GLXContext newContext = createContext( display, config, creation.isShared() ? creation.getContext()->getContext() : 0 );
      DP_ASSERT( newContext );

      SharedNativeContext nativeContext = new NativeContext( newContext, true, drawable, false, 0, false, display, true );
  #endif
      SharedShareGroup shareGroup = creation.isShared() ? creation.getContext()->getShareGroup() : createShareGroup( nativeContext );
      return( std::shared_ptr<RenderContext>( new RenderContext( nativeContext, shareGroup ) ) );
    }

    SharedRenderContext RenderContext::create( const Headless &creation )
    {
  #if defined(DP_OS_WINDOWS)
      int pixelFormat = creation.getFormat()->getPixelFormat();
      if (!pixelFormat)
      {
        return nullptr;
      }

      HWND hwnd = createWindow(); // create a dummy window
      DP_ASSERT( hwnd );

      HDC hdc;
      if( creation.getGpus().empty() )
      {
        hdc = GetDC( hwnd ); // create a default dc
      }
      else
      {
        std::vector< HGPUNV > gpus;
        gpus = creation.getGpus();
        gpus.push_back( nullptr );

        hdc = wglCreateAffinityDCNV( &gpus[0] );
      }
      DP_ASSERT( hdc );

      PIXELFORMATDESCRIPTOR pfd;
      bool ok = !!SetPixelFormat( hdc, pixelFormat, &pfd );
      DP_ASSERT( ok );

      HGLRC hglrc = createContext( hdc, creation.getContext() ? creation.getContext()->m_context->m_hglrc : 0 );
      DP_ASSERT( hglrc );

      SharedNativeContext nativeContext( new NativeContext( hwnd, true, hdc, true, hglrc, true ) );

      if (creation.getContext())
      {
        DP_ASSERT( false );
  #if 0
        WGLShareLists( creation.getContext()->m_context->m_hglrc, hglrc );
  #endif
      }
  #elif defined(DP_OS_LINUX)
      Display *display = XOpenDisplay( creation.getContext() ? DisplayString(creation.getContext()->m_context->m_context ) : creation.getDisplay() );
      DP_ASSERT( display );

      int screen = 0;
      if ( creation.getContext() )
      {
        glXQueryContext( display, creation.getContext()->m_context->m_context, GLX_SCREEN, &screen);
      }
      else
      {
        screen = creation.getScreen();
      }

      GLXFBConfig config = creation.getContext() ?
                             RenderContextFormat::getGLXFBConfig( display, creation.getContext()->m_context->m_context )
                           : creation.getFormat()->getGLXFBConfig( display, screen );

      GLXPbuffer pbuffer = createPbuffer( display, config );
      DP_ASSERT( pbuffer );

      GLXDrawable drawable = pbuffer;

      GLXContext context = createContext( display, config, creation.getContext() ? creation.getContext()->m_context->m_context : 0 );
      DP_ASSERT( context );

      SharedNativeContext nativeContext = new NativeContext( context, true, drawable, false, pbuffer, true, display, true );
  #endif
      SharedShareGroup shareGroup = creation.getContext() ? creation.getContext()->getShareGroup() : createShareGroup( nativeContext );
      return( std::shared_ptr<RenderContext>( new RenderContext( nativeContext, shareGroup, *creation.getFormat() ) ) );
    }

    SharedRenderContext RenderContext::create( const Windowed &creation )
    {
  #if defined(DP_OS_WINDOWS)
      int pixelFormat = creation.getFormat()->getPixelFormat();
      if (!pixelFormat)
      {
        return nullptr;
      }

      HWND hwnd = createWindow( creation.getWidth(), creation.getHeight() ); // create a dummy window
      DP_ASSERT( hwnd );

      HDC hdc = GetDC( hwnd ); // create a default dc
      DP_ASSERT( hdc );


      PIXELFORMATDESCRIPTOR pfd;
      bool ok = !!SetPixelFormat( hdc, pixelFormat, &pfd );
      DP_ASSERT( ok );

      HGLRC hglrc = createContext( hdc, creation.getContext() ? creation.getContext()->m_context->m_hglrc : 0 );
      DP_ASSERT( hglrc );

      SharedNativeContext nativeContext( new NativeContext( hwnd, true, hdc, true, hglrc, true ) );

      if (creation.getContext())
      {
        DP_ASSERT( false );
  #if 0
        WGLShareLists( creation.getContext()->m_context->m_hglrc, hglrc );
  #endif
      }
  #elif defined(DP_OS_LINUX)
      assert(0 && "not yet supported"); 
      /*
      Display *display = XOpenDisplay( creation.getContext() ? DisplayString(creation.getContext()->m_context->m_context ) : creation.getDisplay() );
      DP_ASSERT( display );

      int screen = 0;
      if ( creation.getContext() )
      {
        glXQueryContext( display, creation.getContext()->m_context->m_context, GLX_SCREEN, &screen);
      }
      else
      {
        screen = creation.getScreen();
      }

      GLXFBConfig config = creation.getContext() ?
        RenderContextFormat::getGLXFBConfig( display, creation.getContext()->m_context->m_context )
        : creation.getFormat()->getGLXFBConfig( display, screen );

      GLXPbuffer pbuffer = createPbuffer( display, config );
      DP_ASSERT( pbuffer );

      GLXDrawable drawable = pbuffer;

      GLXContext context = createContext( display, config, creation.getContext() ? creation.getContext()->m_context->m_context : 0 );
      DP_ASSERT( context );

      SharedNativeContext nativeContext = new NativeContext( context, true, drawable, false, pbuffer, true, display, true );
      */
      SharedNativeContext nativeContext;
  #endif
      SharedShareGroup shareGroup = creation.getContext() ? creation.getContext()->getShareGroup() : createShareGroup( nativeContext );
      return( std::shared_ptr<RenderContext>( new RenderContext( nativeContext, shareGroup, *creation.getFormat() ) ) );

    }

  #if defined(DP_OS_WINDOWS)
    SharedRenderContext RenderContext::create( const FromHDC &creation )
    {
      HGLRC hglrc = createContext( creation.getHDC(), creation.getContext() ? creation.getContext()->m_context->m_hglrc : 0 );
      DP_ASSERT( hglrc );

      SharedNativeContext nativeContext( new NativeContext( 0, false, creation.getHDC(), false, hglrc, true ) );

      SharedShareGroup shareGroup;
      if (creation.getContext())
      {
        shareGroup = creation.getContext()->getShareGroup();
        DP_ASSERT( false );
  #if 0
        WGLShareLists( creation.getContext()->m_context->m_hglrc, hglrc );
  #endif
      }
      else
      {
        shareGroup = createShareGroup( nativeContext );
      }

      return( std::shared_ptr<RenderContext>( new RenderContext( nativeContext, shareGroup ) ) );
    }

    SharedRenderContext RenderContext::create( const FromHWND &creation )
    {
      int pixelFormat = creation.getFormat()->getPixelFormat();
      if ( !pixelFormat )
      {
        return nullptr;
      }

      HDC hdc = GetDC( creation.getHWND() );
      DP_ASSERT( hdc );

      PIXELFORMATDESCRIPTOR pfd;
      bool ok = !!SetPixelFormat( hdc, pixelFormat, &pfd );
      DP_ASSERT( ok );

      HGLRC hglrc = createContext( hdc, creation.getContext() ? creation.getContext()->m_context->m_hglrc : 0 );
      DP_ASSERT( hglrc );

      SharedNativeContext nativeContext( new NativeContext( creation.getHWND(), false, hdc, false, hglrc, true ) );

      SharedShareGroup shareGroup;
      if (creation.getContext())
      {
        shareGroup = creation.getContext()->getShareGroup();
        wglShareLists( creation.getContext()->getHGLRC(), nativeContext->m_hglrc );
      }
      else
      {
        shareGroup = createShareGroup( nativeContext );
      }

      return( std::shared_ptr<RenderContext>( new RenderContext( nativeContext, shareGroup, *creation.getFormat() ) ) );
    }
  #elif defined(DP_OS_LINUX)
    SharedRenderContext RenderContext::create( const FromDrawable &creation )
    {
      Display *display = creation.shared ? XOpenDisplay( DisplayString(creation.shared->m_context->m_context ) ): creation.display;
      DP_ASSERT( display );

      int screen = 0;
      if ( creation.shared )
      {
        glXQueryContext( display, creation.shared->m_context->m_context, GLX_SCREEN, &screen);
      }
      else
      {
        screen = creation.screen;
      }

      glXQueryDrawable = (PFNGLXQUERYDRAWABLEPROC)glXGetProcAddress((GLubyte*)"glXQueryDrawable");
      glXChooseFBConfig = (PFNGLXCHOOSEFBCONFIGPROC)glXGetProcAddress((GLubyte*)"glXChooseFBConfig");
      glXCreateNewContext = (PFNGLXCREATENEWCONTEXTPROC)glXGetProcAddress((GLubyte*)"glXCreateNewContext");
#if 0
      unsigned int glxFBConfigId = 0;
      glXQueryDrawable( display, creation.drawable, GLX_FBCONFIG_ID, &glxFBConfigId );
      DP_ASSERT( glxFBConfigId );

      // fetch GLXFBConfig object for glxFBConfigId
      int numElements = 0;
      const int glxFBConfigAttributes[] = {GLX_FBCONFIG_ID, int(glxFBConfigId), None, None};
      GLXFBConfig *glxFBConfigs = glXChooseFBConfig( display, screen, glxFBConfigAttributes, &numElements );
      DP_ASSERT( glxFBConfigs );

#else
      int numElements = 0;
      static int fb_attribs[] = {
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_DOUBLEBUFFER, True,
        GLX_RED_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        0
      };
      GLXFBConfig *glxFBConfigs = glXChooseFBConfig(display, screen, fb_attribs, &numElements);
#endif

      GLXDrawable drawable = creation.drawable;
      GLXContext context = createContext( display, glxFBConfigs[0], creation.shared ? creation.shared->m_context->m_context : 0 );
      DP_ASSERT( context );

      SharedNativeContext nativeContext = new NativeContext( context, true, drawable, false, 0, false, display, true );
      nativeContext->makeCurrent();
      glewInit();

      SharedShareGroup shareGroup = creation.shared ? creation.shared->getShareGroup() : createShareGroup( nativeContext );
      return new RenderContext( nativeContext, shareGroup );
    }
  #endif

    RenderContext::RenderContext( const RenderContext::SharedNativeContext &nativeContext, const SharedShareGroup &shareGroup )
      : m_context(nativeContext)
      , m_shareGroup(shareGroup)
    {
  #if defined(DP_OS_WINDOWS)
      m_format.syncFormat( getHDC() );
  #elif defined(DP_OS_LINUX)
      m_format.syncFormat( getDisplay(), getContext() );
  #endif
    }

    RenderContext::RenderContext( const RenderContext::SharedNativeContext &nativeContext, const SharedShareGroup &shareGroup, const RenderContextFormat &format )
      : m_context(nativeContext)
      , m_shareGroup(shareGroup)
    {
      unsigned int color, coverage;
      format.getMultisampleCoverage( color, coverage );

      m_format.setStereo( format.isStereo() );
      m_format.setThirtyBit( format.isThirtyBit() );
      m_format.setStencil( format.isStencil() );
      m_format.setMultisampleCoverage( color, coverage );
      m_format.setSRGB( format.isSRGB() );
    }

    RenderContextStack::StackEntry::StackEntry()
  #if defined(DP_OS_WINDOWS)
      : hdc(0)
      , hglrc(0)
      , hwnd(0)
  #elif defined(DP_OS_LINUX)
  #endif
    {
    }

    RenderContextStack::StackEntry::StackEntry( const StackEntry &rhs )
  #if defined(DP_OS_WINDOWS)
      : hdc(rhs.hdc)
      , hglrc( rhs.hglrc )
      , hwnd( rhs.hwnd )
  #elif defined(DP_OS_LINUX)
      : context(rhs.context)
      , drawable(rhs.drawable)
      , display(rhs.display)      
  #endif
      , renderContextGL( rhs.renderContextGL )
    {
    }

    RenderContextStack::~RenderContextStack()
    {
      DP_ASSERT( empty() );
    }

    void RenderContextStack::push( const SharedRenderContext &renderContextGL )
    {
      StackEntry entry;
      entry.renderContextGL = RenderContext::getCurrentRenderContext();
  #if defined(DP_OS_WINDOWS)
      entry.hdc = wglGetCurrentDC();
      entry.hglrc = wglGetCurrentContext();
      if (    !(entry.renderContextGL == renderContextGL)
           || entry.hdc != renderContextGL->getHDC()
           || entry.hglrc != renderContextGL->getHGLRC()
         )
      {
        renderContextGL->makeCurrent();
      }

  #elif defined(DP_OS_LINUX)
      entry.context = glXGetCurrentContext();
      if (entry.context)
      {
        entry.drawable = glXGetCurrentDrawable();
        entry.display = glXGetCurrentDisplay();
      }
      else
      {
        entry.drawable = 0;
        entry.display = renderContextGL->getDisplay();
      }
      entry.renderContextGL = RenderContext::getCurrentRenderContext();

      if (    (entry.renderContextGL != renderContextGL)
           || (entry.drawable != renderContextGL->getDrawable())
           || (entry.display != renderContextGL->getDisplay()))
      {
        renderContextGL->makeCurrent();
      }

  #endif
      m_stack.push( entry );
    }

    void RenderContextStack::pop( )
    {
      if (m_stack.empty())
      {
        return;
      }

      const StackEntry& entry = m_stack.top();
  #if defined(DP_OS_WINDOWS)
      if (entry.hglrc != wglGetCurrentContext())
      {
        wglMakeCurrent( entry.hdc, entry.hglrc );
      }
      else
      {
        DP_ASSERT( wglGetCurrentDC() == entry.hdc );
      }
      getThreadData()->currentRenderContext = entry.renderContextGL;
  #elif defined(DP_OS_LINUX)
      if (entry.context != glXGetCurrentContext() || entry.drawable != glXGetCurrentDrawable() || entry.display != glXGetCurrentDisplay() )
      {
        if ( entry.context && entry.drawable && entry.display )
        {
          glXMakeCurrent( entry.display, entry.drawable, entry.context );
        }
      }
      getThreadData()->currentRenderContext = entry.renderContextGL;
  #endif
       m_stack.pop();
    }

    bool RenderContextStack::empty() const
    {
      return m_stack.empty();
    }


    const SharedRenderContext & RenderContext::getCurrentRenderContext()
    {
      return getThreadData()->currentRenderContext;
    }

    SharedShareGroup RenderContext::getShareGroup() const
    {
      return m_shareGroup;
    }

    /********************/
    /* ShareGroupImpl */
    /********************/
    class ShareGroupImpl
    {
    public:
      ShareGroupImpl( const RenderContext::SharedNativeContext &nativeContext );
      virtual ~ShareGroupImpl();

      void executeTask( const SharedShareGroupTask &task );

      void cleanupThread( );
    protected:
      RenderContext::SharedNativeContext m_context;
      volatile bool m_exit;
      volatile bool m_exitDone;
      boost::thread m_thread;
      boost::mutex  m_mutex;
      boost::condition_variable m_taskDoWork;
      boost::condition_variable m_taskDone;
      SharedShareGroupTask m_task;
    };

    ShareGroupImpl::ShareGroupImpl( const RenderContext::SharedNativeContext &nativeContext )
      : m_context( nativeContext )
      , m_exit( false )
      , m_exitDone( false )
    {
      m_thread = boost::thread( boost::bind(&ShareGroupImpl::cleanupThread, this ) );
    }

    ShareGroupImpl::~ShareGroupImpl()
    {
      {
        boost::mutex::scoped_lock lock( m_mutex );
        m_exit = true;
      }

      m_taskDoWork.notify_one();

      {
        boost::mutex::scoped_lock lock( m_mutex );
        while ( !m_exitDone )
        {
          m_taskDone.wait( lock );
        }
      }
    }

    void ShareGroupImpl::executeTask( const SharedShareGroupTask &task )
    {
      {
        boost::mutex::scoped_lock lock( m_mutex );
        m_task = task;
      }

      m_taskDoWork.notify_one();

      {
        boost::mutex::scoped_lock lock( m_mutex );
        while ( m_task )
        {
          m_taskDone.wait( lock );
        }
      }

    }

    void ShareGroupImpl::cleanupThread()
    {
      boost::mutex::scoped_lock lock( m_mutex );

      // keep context current for the lifetime of the thread
    
      do
      {
        if ( !m_exit )
        {
          m_taskDoWork.wait( lock );
        }

        if ( m_task.get() )
        {
          m_context->makeCurrent();
          m_task->execute();
          m_task.reset();
          m_context->makeNoncurrent();
        }
        m_taskDone.notify_one();
      } while ( !m_exit );

      // make context noncurrent so that deletion in other thread will not fail

      m_exitDone = true;
      m_taskDone.notify_one();

    }

    /****************/
    /* ShareGroup */
    /****************/
    SharedShareGroup ShareGroup::create( const RenderContext::SharedNativeContext &nativeContext )
    {
      return( std::shared_ptr<ShareGroup>( new ShareGroup( nativeContext ) ) );
    }

    ShareGroup::ShareGroup( const RenderContext::SharedNativeContext &nativeContext )
      : m_nativeContext( nativeContext )
      , m_impl( new ShareGroupImpl( nativeContext ) )
    {
      /* Nothing to do */
    }

    ShareGroup::ShareGroup( const ShareGroup &)
    {
      DP_ASSERT( 0 && "should not be called" );
      /* Nothing to do */
    }

    ShareGroup::~ShareGroup()
    {
      delete m_impl;
    }

    SharedShareGroupResource ShareGroup::registerResource( size_t key, unsigned int type, const SharedObject &resource )
    {
      if ( m_resources.find( Key(key, type ) ) == m_resources.end() )
      {
        ShareGroupResourceHolder *resourceHolder = new ShareGroupResourceHolder( shared_from_this(), Key(key, type), resource );
        m_resources[ Key(key, type) ] = std::pair<int, ShareGroupResourceHolder *>(1, resourceHolder);
        return( SharedShareGroupResource( new ShareGroupResourceHolder(*resourceHolder) ) );
      }
      else
      {
        DP_ASSERT( 0 && "Tried to register a resource under an existing key" );
        return 0;
      }
    }

    SharedShareGroupResource ShareGroup::getResource( size_t key, unsigned int type )
    {
      ResourceMap::iterator it = m_resources.find( Key(key, type ) );
      if (it != m_resources.end() )
      {
        ++it->second.first;
        return( SharedShareGroupResource( new ShareGroupResourceHolder( *(it->second.second) ) ) ); // create a new sharedptr here to ensure unregister gets called
      }
      return 0;
    }

    void ShareGroup::unregisterResource( const ShareGroup::Key &key )
    {
      ResourceMap::iterator it = m_resources.find( key );
      if (it != m_resources.end())
      {
        --it->second.first;
        if (!it->second.first)
        {
          delete it->second.second;
          m_resources.erase(it);
        }
      }
    }

    void ShareGroup::executeTask( const SharedShareGroupTask &task, bool async )
    {
      m_impl->executeTask(task);
    }

    ShareGroupResourceHolder::ShareGroupResourceHolder( const SharedShareGroup &shareGroup, ShareGroup::Key key, const SharedObject &resource )
      : m_shareGroup( shareGroup )
      , m_key( key )
      , m_resource( resource )
    {
    }

    ShareGroupResourceHolder::ShareGroupResourceHolder( const ShareGroupResourceHolder &rhs )
      : m_shareGroup( rhs.m_shareGroup )
      , m_key( rhs.m_key )
      , m_resource( rhs.m_resource )
    {
    }

    ShareGroupResourceHolder::~ShareGroupResourceHolder( )
    {
      m_shareGroup->unregisterResource( m_key );
    }

    SharedObject ShareGroupResourceHolder::getResource()
    {
      return m_resource;
    }

  } // namespace gl
} // namespace dp
