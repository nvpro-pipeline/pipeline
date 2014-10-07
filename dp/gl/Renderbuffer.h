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
#include <dp/gl/Object.h>
#include <vector>

namespace dp
{
  namespace gl
  {
    /** \brief Renderbuffer is a wrapper around OpenGL renderbuffers. It can be used as attachment
               by dp::g:l:RenderTargetFBO or as standalone object. If the buffer is used as attachment
               the size is managed by the RenderTarget.
    **/
    class Renderbuffer : public Object
    {
    public:
      struct MSAA {
        MSAA( GLsizei colorSamples ) 
          : m_colorSamples(colorSamples)
        {
        }

        unsigned int m_colorSamples;
      };

      struct CSAA {
        CSAA( GLsizei coverageSamples, GLsizei colorSamples ) 
          : m_colorSamples(colorSamples)
          , m_coverageSamples(coverageSamples)
        {
        }

        unsigned int m_colorSamples;
        unsigned int m_coverageSamples;
      };

    public:
      /** \brief Create a new Renderbuffer.
          \param internalFormat OpenGL format of the newly created renderbuffer.
          \param width Width of the newly created Renderbuffer.
          \param height Height of the newly created Renderbuffer.
          \return RefCounted Renderbuffer.
      **/
      DP_GL_API static SharedRenderbuffer create(GLenum internalFormat, int width = 1, int height = 1);

      /** \brief Create a new Renderbuffer with MSAA enabled.
          \param msaa MSAA settings of the newly created Renderbuffer.
          \param internalFormat OpenGL format of the newly created renderbuffer.
          \param width Width of the newly created Renderbuffer.
          \param height Height of the newly created Renderbuffer.
          \return RefCounted Renderbuffer.
      **/
      DP_GL_API static SharedRenderbuffer create(const MSAA &msaa, GLenum internalFormat, int width = 1, int height = 1);

      /** \brief Create a new Renderbuffer with CSAA enabled.
          \param csaa CSAA settings of the newly created Renderbuffer.
          \param internalFormat OpenGL format of the newly created renderbuffer.
          \param width Width of the newly created Renderbuffer.
          \param height Height of the newly created Renderbuffer.
          \return RefCounted Renderbuffer.
      **/
      DP_GL_API static SharedRenderbuffer create(const CSAA &csaa, GLenum internalFormat, int width = 1, int height = 1);

      DP_GL_API virtual ~Renderbuffer();

      /** \brief Change the size of the renderbuffer.
          \param width New width of the renderbuffer.
          \param height New height of the renderbuffer.
          \note It's not necessary to call this function if the renderbuffer is used as dp::gl::RenderTargetFBO::Attachment.
                The size of attachments is managed by dp::gl::RenderTargetFBO.
      **/
      DP_GL_API void resize( int width, int height );

      /** \brief Test if MSAA is available for the current active context.
          \return true if MSAA is supported, false otherwise.
      **/
      DP_GL_API static bool isMSAAAvailable();

      /** \brief Test if CSAA is available for the current active context.
          \return true if CSAA is supported, false otherwise.
      **/
      DP_GL_API static bool isCSAAAvailable();

      /** \brief Determine the maximum number of MSAA samples supported by the current active context.
          \return Maximum number of supported MSAA samples.
      **/
      DP_GL_API static GLint getMaxMSAASamples();

      /** \brief Get a list of all supported CSAA modes supported by the current active context.
          \return vector of all supported CSAA modes
      **/
      DP_GL_API static std::vector<CSAA> getAvailableCSAAModes();
    protected:
      void init( int width, int height );

    private:
      DP_GL_API Renderbuffer(GLenum internalFormat, int width = 1, int height = 1);
      DP_GL_API Renderbuffer(const MSAA &msaa, GLenum internalFormat, int width = 1, int height = 1);
      DP_GL_API Renderbuffer(const CSAA &csaa, GLenum internalFormat, int width = 1, int height = 1);

    private:
      void resizeNoAA( );
      void resizeMSAA( );
      void resizeCSAA( );

      GLsizei m_width;
      GLsizei m_height;
      GLenum  m_internalFormat;

      GLsizei m_colorSamples;     // will be 0 when msaa and csaa are not being used
      GLsizei m_coverageSamples;  // will be 0 for msaa and no multisampling
      void (Renderbuffer::*m_resizeFunc)( );
    };
  } // namespace gl
} // namespace dp
