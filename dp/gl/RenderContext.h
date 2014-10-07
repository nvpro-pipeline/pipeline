// Copyright NVIDIA Corporation 2009-2011
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
#include <dp/gl/RenderContextFormat.h>
#include <dp/gl/Types.h>
#include <vector>

#if defined( DP_OS_WINDOWS)
#include <GL/wglew.h>
#endif

#include <stack>
#include <map>

namespace dp
{
  namespace gl
  {
    class ShareGroupImpl;

    /** \brief RenderContext wraps an OpenGL context. 
    **/
    class RenderContext : public std::enable_shared_from_this<RenderContext>
    {
    protected:
      class NativeContext;
      typedef std::shared_ptr<NativeContext> SharedNativeContext;

    public:
      // Constructor parameters
      struct Attach
      {
        Attach()
        {
        }

        Attach( const SharedRenderContext &pSharedContext )
          : m_shared(pSharedContext)
        {
        }

        Attach( const Attach &rhs )
         : m_shared( rhs.m_shared )
        {
        }

        ~Attach()
        {
        }

        SharedRenderContext getContext() const
        {
          return( m_shared );
        }

      private:
        // prohibit assignment
        Attach &operator=( const Attach & );

        SharedRenderContext  m_shared;
      };

      struct Clone
      {
        Clone( const SharedRenderContext &pContext, bool pShare )
        : m_context( pContext )
        , m_share(pShare)
        {
        }

        Clone( bool pShare )
          : m_share(pShare)
        {
        }

        Clone( const Clone &rhs )
         : m_context( rhs.m_context )
         , m_share( rhs.m_share )
        {
        }

        ~Clone()
        {
        }

        SharedRenderContext getContext() const
        {
          return( m_context );
        }

        bool isShared() const
        {
          return( m_share );
        }

      private:
        // prohibit assignment
        Clone &operator=( const Clone & );

        SharedRenderContext  m_context;
        bool                  m_share;
      };

      struct Headless
      {
        Headless( RenderContextFormat *pFormat, const SharedRenderContext &pShared )
          : m_format(pFormat)
          , m_shared(pShared)
  #if defined(LINUX)
          , m_display(0)
          , m_screen(0)
  #endif
        {
        }

  #if defined(DP_OS_WINDOWS)
        Headless( RenderContextFormat *pFormat, const SharedRenderContext &pShared, std::vector< HGPUNV > gpus )
          : m_format(pFormat)
          , m_shared(pShared)
          , m_gpus( gpus )
        {
        }
  #endif

  #if defined(DP_OS_LINUX)
        Headless( RenderContextFormat *pFormat, const char *pDisplay = 0, int pScreen = 0 )
          : m_format(pFormat)
          , m_display(pDisplay)
          , m_screen(pScreen)
        {
        }
  #endif

        Headless( const Headless &rhs )
        : m_format( rhs.m_format)
        , m_shared( rhs.m_shared )
  #if defined(LINUX)
        , m_display( rhs.m_display )
        , m_screen( rhs.m_screen )
  #endif
        {
        }

        ~Headless()
        {
        }

        const RenderContextFormat * getFormat() const
        {
          return( m_format );
        }

        SharedRenderContext getContext() const
        {
          return( m_shared );
        }

  #if defined( DP_OS_WINDOWS)
        std::vector< HGPUNV > const& getGpus() const
        {
          return m_gpus;
        }
  #endif

  #if defined(LINUX)
        const char * getDisplay() const
        {
          return( m_display );
        }

        int getScreen() const
        {
          return( m_screen );
        }
  #endif

      private:
        // prohibit assignment
        Headless &operator=( const Headless & );

        RenderContextFormat * m_format;
        SharedRenderContext    m_shared;
  #if defined(DP_OS_WINDOWS)
        std::vector< HGPUNV > m_gpus;
  #endif
  #if defined(DP_OS_LINUX)
        const char *          m_display;
        int                   m_screen;
  #endif
      };



      struct Windowed
      {
        Windowed( RenderContextFormat *pFormat, const SharedRenderContext &pShared, unsigned int width, unsigned int height )
          : m_format(pFormat)
          , m_shared(pShared)
  #if defined(LINUX)
          , m_display(0)
          , m_screen(0)
  #endif
        {
          m_width = width;
          m_height = height;
        }

  #if defined(LINUX)
        Windowed( RenderContextFormat *pFormat, const char *pDisplay = 0, int pScreen = 0 )
          : m_format(pFormat)
          , m_display(pDisplay)
          , m_screen(pScreen)
        {
        }
  #endif

        Windowed( const Windowed &rhs )
          : m_format( rhs.m_format)
          , m_shared( rhs.m_shared )
  #if defined(LINUX)
          , m_display( rhs.m_display )
          , m_screen( rhs.m_screen )
  #endif
        {
        }

        ~Windowed()
        {
        }

        const RenderContextFormat * getFormat() const
        {
          return( m_format );
        }

        SharedRenderContext getContext() const
        {
          return( m_shared );
        }

        inline unsigned int getWidth() const
        {
          return m_width;
        }

        inline unsigned int getHeight() const
        {
          return m_height;
        }

  #if defined(LINUX)
        const char * getDisplay() const
        {
          return( m_display );
        }

        int getScreen() const
        {
          return( m_screen );
        }
  #endif

      private:
        // prohibit assignment
        Windowed &operator=( const Windowed & );

        RenderContextFormat * m_format;
        SharedRenderContext    m_shared;
        unsigned int          m_width;
        unsigned int          m_height;
  #if defined(LINUX)
        const char *          m_display;
        int                   m_screen;
  #endif
      };


  #if defined( WIN32 )
      struct FromHDC
      {
        FromHDC( HDC pHdc, const SharedRenderContext &pShared )
          : m_hdc(pHdc)
          , m_shared(pShared)
        {
        }

        FromHDC( const FromHDC &rhs )
         : m_hdc( rhs.m_hdc )
         , m_shared( rhs.m_shared )
        {
        }

        ~FromHDC()
        {
        }

        HDC getHDC() const
        {
          return( m_hdc );
        }

        SharedRenderContext getContext() const
        {
          return( m_shared );
        }

      private:
        // prohibit assignment
        FromHDC &operator=( const FromHDC & );

        HDC                   m_hdc;
        SharedRenderContext  m_shared;
      };

      struct FromHWND
      {
        FromHWND( HWND pHwnd, RenderContextFormat const * pFormat, SharedRenderContext const & pShared )
          : m_hwnd(pHwnd)
          , m_format(pFormat)
          , m_shared(pShared)
        {
        }

        FromHWND( FromHWND const & rhs )
          : m_hwnd( rhs.m_hwnd )
          , m_format( rhs.m_format )
          , m_shared( rhs.m_shared )
        {
        }

        ~FromHWND()
        {
        }

        HWND getHWND() const
        {
          return( m_hwnd );
        }

        RenderContextFormat const * getFormat() const
        {
          return( m_format );
        }

        SharedRenderContext getContext() const
        {
          return( m_shared );
        }

      private:
        // prohibit assignment
        FromHWND &operator=( FromHWND const & );

        HWND                        m_hwnd;
        RenderContextFormat const * m_format;
        SharedRenderContext          m_shared;
      };
  #endif

  #if defined(LINUX)
      struct FromDrawable
      {
        FromDrawable( Display *pDisplay, int pScreen, GLXDrawable pDrawable, const SharedRenderContext &pShared )
          : display(pDisplay)
          , screen( pScreen )
          , drawable(pDrawable)
          , shared(pShared)
        {
        }

        FromDrawable( const FromDrawable &rhs)
        : display( rhs.display )
        , screen( rhs.screen )
        , drawable( rhs.drawable )
        , shared( rhs.shared )
        {
        }

        Display *display;
        int screen;
        GLXDrawable drawable;
        SharedRenderContext shared;
      private:
        // prohibit assignment
        FromDrawable &operator=( const FromDrawable & );
      };
  #endif

    public:
      DP_GL_API virtual ~RenderContext();

    protected:
      // abstract base class
      DP_GL_API RenderContext();

      DP_GL_API RenderContext( const SharedNativeContext &nativeContext, const SharedShareGroup &shareGroup );
      DP_GL_API RenderContext( const SharedNativeContext &nativeContext, const SharedShareGroup &shareGroup, const RenderContextFormat &format );
      DP_GL_API RenderContext( const Attach &creation );
      DP_GL_API RenderContext( const Headless &headless);
      DP_GL_API RenderContext( const Clone &creation );
  #if defined(LINUX)
      DP_GL_API RenderContext( const FromDrawable &creation );
  #endif
      // no copy
      DP_GL_API RenderContext(const RenderContext &);
      DP_GL_API RenderContext &operator=(const RenderContext &);

    public:
      DP_GL_API static SharedRenderContext create( const Attach &creation );
      DP_GL_API static SharedRenderContext create( const Clone &creation );
      DP_GL_API static SharedRenderContext create( const Headless &creation );
      DP_GL_API static SharedRenderContext create( const Windowed &creation );
  #if defined(WIN32)
      DP_GL_API static SharedRenderContext create( const FromHDC &creation );
      DP_GL_API static SharedRenderContext create( const FromHWND &creation );
  #elif defined(LINUX)
      DP_GL_API static SharedRenderContext create( const FromDrawable &creation );
  #endif

      // may introduce locking to ensure that context is been used only by one thread at the same time
      DP_GL_API bool makeCurrent();

      /* Not sure if we want to have the uncurrent calls. They could be useful for multithreaded programs.
         Unfortunately it seems they'll introduce a lot of new complexity which cannot be solved easily.
      **/
      // make context uncurrent
      DP_GL_API void makeNoncurrent();

      DP_GL_API static const SharedRenderContext & getCurrentRenderContext();
      DP_GL_API SharedShareGroup getShareGroup() const;

      DP_GL_API void swap();

      // call this before context is being destroyed
      DP_GL_API virtual void notifyDestroy();

      DP_GL_API const RenderContextFormat &getFormat() const;

      GLint getMinorVersion() const
      {
        return m_context->m_minorVersion;
      }

      GLint getMajorVersion() const
      {
        return m_context->m_majorVersion;
      }

  #if defined(_WIN32)
      DP_GL_API HWND  getHWND()  const;
      DP_GL_API HDC   getHDC()   const;
      DP_GL_API HGLRC getHGLRC() const;

  #elif defined(LINUX)
      GLXContext  getContext()  const;
      GLXDrawable getDrawable() const;
      Display*    getDisplay()  const;
  #endif

    protected:
      // clone a context
  #if defined(LINUX)
      DP_GL_API void clone( GLXContext context, bool share );
  #elif defined(_WIN32)
      DP_GL_API void clone( HDC hdc, HGLRC shareContext );
  #endif

    protected:
      // platform specific variables and functions
    #if defined (_WIN32)
      DP_GL_API static LRESULT CALLBACK RenderContextWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
      // create a native invisible dummy window
      DP_GL_API static HWND createWindow(int width=1, int height=1);
      DP_GL_API static HGLRC createContext( HDC hdc, HGLRC shareContext );
    #elif defined(LINUX)
      DP_GL_API static GLXContext createContext( Display *display, GLXFBConfig config, GLXContext shareContext );
      DP_GL_API static GLXPbuffer createPbuffer( Display *display, GLXFBConfig config );
    #endif
      DP_GL_API static SharedShareGroup createShareGroup( const RenderContext::SharedNativeContext &context );

      /** Hold a native OGL context **/
      class NativeContext
      {
      public:
  #if defined(WIN32)
        DP_GL_API NativeContext( HWND hwnd, bool destroyHWND, HDC hdc, bool destroyHDC, HGLRC hglrc, bool destroyHGLRC );
  #elif defined(LINUX)
        DP_GL_API NativeContext( GLXContext context, bool destroyContext, GLXDrawable drawable, bool destroyDrawable, GLXPbuffer pbuffer, bool destroyPbuffer, Display *display, bool destroyDisplay );
  #endif
        DP_GL_API virtual ~NativeContext();

        DP_GL_API bool makeCurrent();
        DP_GL_API void makeNoncurrent();
        DP_GL_API void swap();

        DP_GL_API void notifyDestroyed();

        GLint                   m_majorVersion;
        GLint                   m_minorVersion;

  #if defined(_WIN32)
        HDC   m_hdc;
        HWND  m_hwnd;
        HGLRC m_hglrc;

        bool m_destroyHDC;
        bool m_destroyHWND;
        bool m_destroyHGLRC;

        std::vector< HGPUNV > m_gpuMask;
  #elif defined(LINUX)
        GLXContext  m_context;
        GLXDrawable m_drawable;
        GLXPbuffer  m_pbuffer;
        Display*    m_display;

        bool        m_destroyContext;
        bool        m_destroyDrawable;
        bool        m_destroypBuffer;
        bool        m_destroyDisplay;
  #endif
      };

      RenderContextFormat m_format;
      SharedShareGroup     m_shareGroup;
      SharedNativeContext  m_context;

      friend class ShareGroup;
      friend class ShareGroupImpl;
    };

    class RenderContextStack {
    public:
      DP_GL_API ~RenderContextStack();

      /* \brief Push the current RenderContext on the stack and make a the given RenderContext object context active.
       * \param renderContextGL The RenderContext which should be active
       */
      DP_GL_API void push( const SharedRenderContext &renderContextGL );

      /* \brief Make the RenderContextGl object at the top of the stack active and remove it from the stack.
       */
      DP_GL_API void pop();

      /* \brief Check if there are elements on the context stack
         \return true if there is an element on the context stack, false otherwise
         **/
      DP_GL_API bool empty() const;

    private:
      struct StackEntry {
        DP_GL_API StackEntry();
        DP_GL_API StackEntry( const StackEntry &rhs );
  #if defined(_WIN32)
        HDC   hdc;
        HWND  hwnd;
        HGLRC hglrc;
  #elif defined(LINUX)
        GLXContext  context;
        GLXDrawable drawable;
        Display*    display;
  #endif
        SharedRenderContext renderContextGL;
      };
      std::stack<StackEntry> m_stack;
    };

    /**********************/
    /* ShareGroup classes */
    /**********************/

    class ShareGroupResourceHolder;
    typedef std::shared_ptr<ShareGroupResourceHolder> ShareGroupResource;
  
    /** \brief A ShareGroupTask is a small task which can be executed in the GL thread of a ShareGroup.
    **/
    // FIXME does it make sense to introduce a generic task function?
    class ShareGroupTask
    {
    public:
      DP_GL_API virtual ~ShareGroupTask();

      /** \brief This function is being called by the ShareGroup.
          \remarks At the time this function is called a GL context of the ShareGroup is active.
      **/
      DP_GL_API virtual void execute() = 0;
    };

    typedef std::shared_ptr<ShareGroupTask> SharedShareGroupTask;

    class ShareGroupResourceHolder;
    typedef std::shared_ptr<ShareGroupResourceHolder> SharedShareGroupResource;

    class ShareGroup : public std::enable_shared_from_this<ShareGroup>
    {
    public:
      DP_GL_API static SharedShareGroup create( const RenderContext::SharedNativeContext &nativeContext );
      DP_GL_API virtual ~ShareGroup();

      DP_GL_API SharedShareGroupResource registerResource( size_t key, unsigned int type, const SharedObject &resource );
      DP_GL_API SharedShareGroupResource getResource( size_t key, unsigned int type );

      DP_GL_API void executeTask( const SharedShareGroupTask &task, bool async = true );

    protected:
      friend class ShareGroupResourceHolder;

      DP_GL_API ShareGroup( const RenderContext::SharedNativeContext &nativeContext );
      DP_GL_API ShareGroup( const ShareGroup & );

      /** \brief Key used for ResourceMap **/
      struct Key
      {
        Key( size_t key, unsigned int type )
          : m_key(key)
          , m_type(type)
        {
        }

        bool operator<(const Key &rhs) const
        {
          return m_key < rhs.m_key || (m_key == rhs.m_key && m_type < rhs.m_type);
        }

        size_t       m_key;
        unsigned int m_type;
      };

      void unregisterResource( const ShareGroup::Key &key );

      typedef std::map<Key, std::pair<int, ShareGroupResourceHolder*> > ResourceMap;
      ResourceMap m_resources;

      RenderContext::SharedNativeContext m_nativeContext;  // headless context for resource cleanup

      ShareGroupImpl *m_impl;
    };

    /** \brief A ShareGroupResourceHolder holds a reference to a Resource in a ShareGroup.
               It keeps the use count of the resource always up to date.
    **/
    class ShareGroupResourceHolder
    {
    public:
      DP_GL_API ShareGroupResourceHolder( const SharedShareGroup &shareGroup, ShareGroup::Key key, const SharedObject &resource );
      DP_GL_API ShareGroupResourceHolder( const ShareGroupResourceHolder &rhs );
      DP_GL_API virtual ~ShareGroupResourceHolder( );

      DP_GL_API SharedObject getResource();

    protected:
      ShareGroupResourceHolder &operator=( const ShareGroupResourceHolder &rhs );

      SharedShareGroup  m_shareGroup;
      ShareGroup::Key   m_key;
      SharedObject      m_resource;
    };

  } // namespace gl
} // namespace dp
