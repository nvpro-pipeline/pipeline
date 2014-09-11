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


#include <dp/gl/RenderTarget.h>
#include <dp/gl/RenderContext.h>
#include <dp/util/DPAssert.h>

namespace dp
{
  namespace gl
  {

    RenderTarget::RenderTarget( const SmartRenderContext &glContext )
     : m_renderContext(glContext)
     , m_width(0)
     , m_height(0)
     , m_x(0)
     , m_y(0)
     , m_clearDepth(1.0)
     , m_clearStencil(0)
     , m_current(false)
    {
    }

    RenderTarget::~RenderTarget()
    {
    }

    bool RenderTarget::beginRendering()
    {
      makeCurrent();
      return true;
    }

    void RenderTarget::endRendering()
    {
      //m_renderContext->swap();
      makeNoncurrent();
    }

    TargetBufferMask RenderTarget::getClearMask()
    {
      return m_clearMask;
    }

    void RenderTarget::setClearMask( TargetBufferMask clearMask )
    {
      m_clearMask = clearMask;
    }

    void RenderTarget::setClearDepth( GLclampd depth )
    {
      m_clearDepth = depth;
    }

    void RenderTarget::setClearStencil( GLint stencil )
    {
      m_clearStencil = stencil;
    }

    // RenderTarget interface
    void RenderTarget::setSize( unsigned int width, unsigned int height )
    {
      m_width = width;
      m_height = height;
    }

    void RenderTarget::getSize( unsigned int &width, unsigned int &height ) const
    {
      width = m_width;
      height = m_height;
    }

    void RenderTarget::setPosition( int x, int y)
    {
      m_x = x;
      m_y = y;
    }

    void RenderTarget::getPosition( int &x, int &y ) const
    {
      x = m_x;
      y = m_y;
    }

#if 0
    bool RenderTarget::copyToBuffer( GLenum mode, dp::sg::core::Image::PixelFormat pixelFormat, dp::sg::core::Image::PixelDataType pixelDataType, const dp::sg::core::BufferSharedPtr & buffer )
    {
      // FIXME use C++ object for current/noncurrent for exception safety
      makeCurrent();

      size_t components = 0;
      size_t bytesPerComponent = 0;

      // set up alignments
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
      glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
      glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glPixelStorei(GL_PACK_ROW_LENGTH, 0);
      glPixelStorei(GL_PACK_SKIP_ROWS, 0);
      glPixelStorei(GL_PACK_SKIP_PIXELS, 0);

      // determine OpenGL format
      GLenum format = ~0;
      switch (pixelFormat)
      {
      case dp::util::PixelFormat::PF_RGB:
        format = GL_RGB;
        components = 3;
      break;
      case dp::util::PixelFormat::PF_RGBA:
        format = GL_RGBA;
        components = 4;
      break;
      case dp::util::PixelFormat::PF_BGR:
        format = GL_BGR;
        components = 3;
      break;
      case dp::util::PixelFormat::PF_BGRA:
        format = GL_BGRA;
        components = 4;
      break;
      case dp::util::PixelFormat::PF_LUMINANCE:
        format = GL_LUMINANCE;
        components = 1;
      break;
      case dp::util::PixelFormat::PF_ALPHA:
        format = GL_ALPHA;
        components = 1;
      break;
      case dp::util::PixelFormat::PF_LUMINANCE_ALPHA:
        format = GL_LUMINANCE_ALPHA;
        components = 2;
      break;
      case dp::util::PixelFormat::PF_DEPTH_COMPONENT:
        format = GL_DEPTH_COMPONENT;
        components = 1;
      break;
      case dp::util::PixelFormat::PF_DEPTH_STENCIL:
        format = GL_DEPTH24_STENCIL8;
        components = 1;
      break;
      default:
        DP_ASSERT(0 && "unsupported PixelFormat");
      };

      GLenum dataType = ~0;
      switch (pixelDataType)
      {
      case dp::util::PixelFormat::PF_BYTE:
        dataType = GL_BYTE;
        bytesPerComponent = 1;
      break;
      case dp::util::PixelFormat::PF_UNSIGNED_BYTE:
        dataType = GL_UNSIGNED_BYTE;
        bytesPerComponent = 1;
      break;
      case dp::util::PixelFormat::PF_SHORT:
        dataType = GL_SHORT;
        bytesPerComponent = 2;
      break;
      case dp::util::PixelFormat::PF_UNSIGNED_SHORT:
        dataType = GL_UNSIGNED_SHORT;
        bytesPerComponent = 2;
      break;
      case dp::util::PixelFormat::PF_INT:
        dataType = GL_INT;
        bytesPerComponent = 4;
      break;
      case dp::util::PixelFormat::PF_UNSIGNED_INT:
        dataType = GL_UNSIGNED_INT;
        bytesPerComponent = 4;
      break;
      case dp::util::PixelFormat::PF_FLOAT32:
        dataType = GL_FLOAT;
        bytesPerComponent = 4;
      break;
      case dp::util::PixelFormat::PF_FLOAT16:
        dataType = GL_HALF_FLOAT;
        bytesPerComponent = 2;
      break;
      default:
        DP_ASSERT(0 && "unsupported PixelDataType");
      }

      BufferLock(buffer)->setSize(m_width * m_height * components * bytesPerComponent);

      // read the pixels
      glWindowPos2i(0,0);
      glReadBuffer( mode );

      bool isBufferGL = buffer.isPtrTo<BufferGL>();
      if ( isBufferGL )
      {
        GLint oldPBO;
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &oldPBO);
        // FIXME it's necessary to check wheter the buffer object shared data with the current context...
        BufferGLLock bufferGL( sharedPtr_cast<BufferGL>( buffer ) );
        bufferGL->bind( GL_PIXEL_PACK_BUFFER );
        glReadPixels(0, 0, m_width, m_height, format, dataType, 0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, (GLuint)oldPBO);
      }
      else
      {
        Buffer::DataWriteLock bufferLock(buffer, Buffer::MAP_WRITE);
        glReadPixels(0, 0, m_width, m_height, format, dataType, bufferLock.getPtr());
      }

      makeNoncurrent();

      return true;
    }
#endif

    std::vector<dp::util::Uint8> RenderTarget::getImagePixels( GLenum mode, dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType )
    {
      // FIXME use C++ object for current/noncurrent for exception safety
      makeCurrent();

      size_t components = 0;
      size_t bytesPerComponent = 0;

      // set up alignments
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
      glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
      glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glPixelStorei(GL_PACK_ROW_LENGTH, 0);
      glPixelStorei(GL_PACK_SKIP_ROWS, 0);
      glPixelStorei(GL_PACK_SKIP_PIXELS, 0);

      // determine OpenGL format
      GLenum format = ~0;
      switch (pixelFormat)
      {
      case dp::util::PF_RGB:
        format = GL_RGB;
        components = 3;
        break;
      case dp::util::PF_RGBA:
        format = GL_RGBA;
        components = 4;
        break;
      case dp::util::PF_BGR:
        format = GL_BGR;
        components = 3;
        break;
      case dp::util::PF_BGRA:
        format = GL_BGRA;
        components = 4;
        break;
      case dp::util::PF_LUMINANCE:
        format = GL_LUMINANCE;
        components = 1;
        break;
      case dp::util::PF_ALPHA:
        format = GL_ALPHA;
        components = 1;
        break;
      case dp::util::PF_LUMINANCE_ALPHA:
        format = GL_LUMINANCE_ALPHA;
        components = 2;
        break;
      case dp::util::PF_DEPTH_COMPONENT:
        format = GL_DEPTH_COMPONENT;
        components = 1;
        break;
      case dp::util::PF_DEPTH_STENCIL:
        format = GL_DEPTH_STENCIL;
        components = 1;
        break;
      case dp::util::PF_STENCIL_INDEX:
        format = GL_STENCIL_INDEX;
        components = 1;
        break;
      default:
        DP_ASSERT(0 && "unsupported PixelFormat");
      };

      GLenum dataType = ~0;
      switch (pixelDataType)
      {
      case dp::util::DT_INT_8:
        dataType = GL_BYTE;
        bytesPerComponent = 1;
        break;
      case dp::util::DT_UNSIGNED_INT_8:
        dataType = GL_UNSIGNED_BYTE;
        bytesPerComponent = 1;
        break;
      case dp::util::DT_INT_16:
        dataType = GL_SHORT;
        bytesPerComponent = 2;
        break;
      case dp::util::DT_UNSIGNED_INT_16:
        dataType = GL_UNSIGNED_SHORT;
        bytesPerComponent = 2;
        break;
      case dp::util::DT_INT_32:
        dataType = GL_INT;
        bytesPerComponent = 4;
        break;
      case dp::util::DT_UNSIGNED_INT_32:
        dataType = GL_UNSIGNED_INT;
        bytesPerComponent = 4;
        break;
      case dp::util::DT_FLOAT_32:
        dataType = GL_FLOAT;
        bytesPerComponent = 4;
        break;
      case dp::util::DT_FLOAT_16:
        dataType = GL_HALF_FLOAT;
        bytesPerComponent = 2;
        break;
      default:
        DP_ASSERT(0 && "unsupported PixelDataType");
      }

      size_t imageSizeInBytes = m_width * m_height * components * bytesPerComponent;
      std::vector<dp::util::Uint8> output(imageSizeInBytes);

      if ( imageSizeInBytes )
      {
        // read the pixels
        glWindowPos2i(0,0);
        glReadBuffer( mode );

        glReadPixels(0, 0, m_width, m_height, format, dataType, &output[0]);
      }

      makeNoncurrent();

      return output;
    }

    dp::util::SmartImage
    RenderTarget::getImage( dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType, unsigned int index )
    {
      return getTargetAsImage( GL_FRONT, pixelFormat, pixelDataType );
    }

    dp::util::SmartImage
    RenderTarget::getTargetAsImage( GLenum mode, dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType )
    {
      // FIXME use C++ object for current/noncurrent for exception safety
      makeCurrent();

      dp::util::SmartImage image = dp::util::Image::create();
      std::vector<dp::util::Uint8> pixels = getImagePixels( mode, pixelFormat, pixelDataType );
      if ( !pixels.empty() )
      {
        image->setSingleLayerData( m_width, m_height, pixelFormat, pixelDataType, &pixels[0] );
      }

      makeNoncurrent();
      return image;
    }

    bool RenderTarget::isValid()
    {
      // Under which condition is an OpenGL RenderTarget invalid?
      return true;
    }

    void RenderTarget::makeCurrent()
    {
      DP_ASSERT( m_renderContext );
      if ( m_renderContext )
      {
        m_current = true;
        m_contextStack.push( m_renderContext );
      }
    }

    void RenderTarget::makeNoncurrent()
    {
      DP_ASSERT( m_renderContext );
      DP_ASSERT( !m_contextStack.empty() );
      if ( m_renderContext && !m_contextStack.empty() )
      {
        m_contextStack.pop();
        m_current = !m_contextStack.empty();
      }
    }

    bool RenderTarget::isCurrent()
    {
      return m_current;
    }

  }
}  // namespace nvgl
