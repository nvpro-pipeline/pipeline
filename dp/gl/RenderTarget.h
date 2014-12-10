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
#include <dp/gl/RenderContext.h>
#include <dp/ui/RenderTarget.h>
#include <dp/util/BitMask.h>

#include <GL/glew.h>

namespace dp
{
  namespace gl
  {
    typedef dp::util::Uint32 TargetBufferMask;

    static const TargetBufferMask TBM_COLOR_BUFFER   = BIT0;
    static const TargetBufferMask TBM_DEPTH_BUFFER   = BIT30;
    static const TargetBufferMask TBM_STENCIL_BUFFER = BIT31;

    /** \brief RenderTarget is the base class for OpenGL based RenderTargets like dp::gl::RenderTargetFB
               and dp::gl::RenderTargetFBO. It contains basic OpenGL functionality like OpenGL context 
               managment, framebuffer grabbing and clearing the background.
    **/
    class RenderTarget : public dp::ui::RenderTarget
    {
      // RenderTarget interface
    public:
      DP_GL_API virtual ~RenderTarget();

      DP_GL_API virtual bool beginRendering();
      /** \brief Make the current OpenGL context noncurrent.
      **/
      DP_GL_API virtual void endRendering();

#if 0
      /** \brief Copy the content of the framebuffer attached to this RenderTarget into an dp::sg::core::Buffer.
          \param mode OpenGL color buffer mode (i.e. GL_FRONT_LEFT or GL_COLOR_ATTACHMENT_0 ).
          \param pixelFormat SceniX Format to use for pixels in the resulting buffer.
          \param pixelDataType SceniX datatype to use for pixels in the resulting buffer.
          \param buffer A SceniX buffer which is being used for the resulting pixels.
          \return true If operation succeeded, false otherwise.
      **/
      DP_GL_API virtual bool copyToBuffer( GLenum  mode, dp::sg::core::Image::PixelFormat pixelFormat, dp::sg::core::Image::PixelDataType pixelDataType, const dp::sg::core::BufferSharedPtr &buffer );
#endif
      std::vector<unsigned char> getImagePixels( GLenum mode, dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType );

      // clear interface
      /** \brief Sets the TargetBufferMask bits used to derive the OpenGL clear mask from.
          \param clearMask Bit mask which indicates which buffers to clear.
          \remarks This is not matching the GL_*_BUFFER_BIT enums! The initial value for the clear mask is 0.
      **/
      DP_GL_API virtual void setClearMask( TargetBufferMask clearMask );

      /** \brief Set the background color for glClear calls
          \param r red value
          \param g green value
          \param b blue value
          \param a alpha value
          \param index color buffer index (if supported by implementation)
          \remarks The initial values for all components are 0.0.
      */
      DP_GL_API virtual void setClearColor( GLclampf r, GLclampf g, GLclampf b, GLclampf a, unsigned int index = 0 ) = 0;

      /** \brief Gets the current TargetBufferMask bits which can be used to derive the OpenGL clear mask from.
          \return Current TargetBufferMask clear mask (not the OpenGL GL_*_BUFFER_BITs). Default is 0, which means no buffer is cleared.
      **/
      DP_GL_API virtual TargetBufferMask getClearMask();

      /** \brief Set clear depth for glClear calls
          \param depth Depth value used to fill the depth buffer when clearing the target
          \remarks The initial value for the depth is 1.0.
      **/
      DP_GL_API void setClearDepth( GLclampd depth );

      /** \brief Set clear stencil value for glClear calls.
          \param stencil stencil value used to fill the stencil buffer when clearing the target.
          \remarks The initial value for the stencil is 0.
      **/
      DP_GL_API void setClearStencil( GLint stencil );

      // RenderTarget interface
      DP_GL_API virtual void setSize( unsigned int  width, unsigned int  height );
      DP_GL_API virtual void getSize( unsigned int &width, unsigned int &height ) const;

      DP_GL_API virtual void setPosition( int  x, int  y ); //!< Set the lower left point of the viewport
      DP_GL_API virtual void getPosition( int &x, int &y ) const; //!< Get the lower left point of the viewport

      // "screenshot"
      /** \brief Grab a screenshot of the current color buffer.
          \param pixelFormat PixelFormat to use when grabbing the pixels.
          \param pixelDataType DataType to use for each pixel component.
          \return A TextureHostSharedPtr containing a texture with the content of the surface.
      */
      DP_GL_API virtual dp::util::ImageSharedPtr getImage( dp::util::PixelFormat pixelFormat = dp::util::PF_BGRA
                                                         , dp::util::DataType pixelDataType = dp::util::DT_UNSIGNED_INT_8
                                                         , unsigned int index = 0 );

      /** \brief Grab a screenshot of the specified color buffer.
          \param mode GL color buffer target to grab (i.e. GL_FRONT_LEFT or GL_COLOR_ATTACHMENT0).
          \param pixelFormat PixelFormat to use when grabbing the pixels.
          \param pixelDataType DataType to use for each pixel component.
          \return A TextureHostSharedPtr containing a texture with the content of the surface.
      **/
      DP_GL_API virtual dp::util::ImageSharedPtr getTargetAsImage(GLenum mode
                                                                 , dp::util::PixelFormat pixelFormat = dp::util::PF_BGRA
                                                                 , dp::util::DataType pixelDataType = dp::util::DT_UNSIGNED_INT_8 );

      DP_GL_API virtual bool isValid();

      /** \brief Get the OpenGL context used by this RenderTarget
          \return RenderContext used by this RenderTarget
      **/
      DP_GL_API RenderContextSharedPtr const& getRenderContext();

    protected:
      DP_GL_API RenderTarget( const RenderContextSharedPtr &glContext );

      /** \brief Check if this RenderTarget is current
          \return true if this RenderTarget is current, false otherwise.
      **/
      DP_GL_API virtual bool isCurrent();

      // make context temporary current for exception safety
      class TmpCurrent
      {
      public:
        TmpCurrent(RenderTarget *target)
        {
          m_target = target;
          target->makeCurrent();
        }

        ~TmpCurrent()
        {
          m_target->makeNoncurrent();
        }
      private:
        RenderTarget *m_target;
      };

      friend class TmpCurrent;

      DP_GL_API virtual void makeCurrent();
      DP_GL_API virtual void makeNoncurrent();

      // clear interface
      TargetBufferMask  m_clearMask;
      GLclampd          m_clearDepth;
      GLuint            m_clearStencil;

    private:
      bool                    m_current;
      RenderContextSharedPtr  m_renderContext;

    protected:
      int                 m_x;
      int                 m_y;
      unsigned int        m_width;
      unsigned int        m_height;

      RenderContextStack  m_contextStack;
    };

    inline RenderContextSharedPtr const& RenderTarget::getRenderContext()
    {
      return m_renderContext;
    }
  } // namespace gl
}  // namespace dp
