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

#include <dp/gl/RenderTarget.h>
#include <dp/gl/RenderContext.h>

namespace dp
{
  namespace gl
  {
    /** \brief RenderTarget for the OpenGL framebuffer. If the attached OpenGL context has a stereo 
               pixelformat stereo will be enabled automatically.
    **/
    class RenderTargetFB : public RenderTarget
    {
    public:
      // RenderTarget interface
      DP_GL_API static RenderTargetFBSharedPtr create( const RenderContextSharedPtr &glContext );

      DP_GL_API virtual ~RenderTargetFB();

      DP_GL_API virtual bool isValid();

      /** \brief Grab a screenshot of the current framebuffer. If stereo is enabled a side-by-side stereo image
                 will be created.
          \return A TextureHost with the content of the color buffer.
      **/
      DP_GL_API virtual dp::util::ImageSharedPtr getImage( dp::PixelFormat pixelFormat = dp::PixelFormat::BGRA
        , dp::DataType pixelDataType = dp::DataType::UNSIGNED_INT_8, unsigned int index = 0 );

      /** \brief Set the background color for glClear calls
          \param r red value
          \param g green value
          \param b blue value
          \param a alpha value
          \param index color buffer index (ignored for this subclass)
          \remarks The initial values for all components are 0.0.
      */
      DP_GL_API virtual void setClearColor( GLclampf r, GLclampf g, GLclampf b, GLclampf a, unsigned int index = 0 );

      // stereo api
      DP_GL_API virtual bool isStereoEnabled() const;
      DP_GL_API virtual bool setStereoTarget( StereoTarget target );
      DP_GL_API virtual StereoTarget getStereoTarget() const;

      DP_GL_API virtual bool beginRendering();
      DP_GL_API virtual void endRendering();

      DP_GL_API virtual void setSwapBuffersEnabled(bool enabled);

    protected:
      // make context current
      DP_GL_API virtual void makeCurrent();
      // make context uncurrent
      DP_GL_API virtual void makeNoncurrent();

      GLenum getStereoTargetBuffer( StereoTarget stereoTarget, bool backbuffer );
      void setDrawBuffer( StereoTarget stereoTarget );

      DP_GL_API RenderTargetFB( const RenderContextSharedPtr &glContext );

      GLbitfield   m_clearMaskGL;
      GLclampf     m_clearColorR, m_clearColorG, m_clearColorB, m_clearColorA;
      StereoTarget m_stereoTarget;    //!< The currently active stereo target
      bool         m_swapBuffersEnabled;
      GLint        m_oldDrawBuffer;   //!< Keeps the drawbuffer which had been current before making this target current
      bool         m_stereoEnabled; //!< True is stereo is supported on the OpenGL context
    };

  } // namespace gl
}  // namespace dp
