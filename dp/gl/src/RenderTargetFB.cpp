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


#include <dp/gl/Config.h>
#include <dp/gl/RenderContext.h>
#include <dp/gl/RenderTargetFB.h>
#include <dp/util/DPAssert.h>

namespace dp
{
  namespace gl
  {
    RenderTargetFB::RenderTargetFB( const SmartRenderContext &glContext )
     : RenderTarget( glContext)
     , m_swapBuffersEnabled(false)
     , m_stereoEnabled( glContext->getFormat().isStereo() )
     , m_stereoTarget( glContext->getFormat().isStereo() ? LEFT_AND_RIGHT : LEFT )
     , m_clearColorR(0.0f)
     , m_clearColorG(0.0f)
     , m_clearColorB(0.0f)
     , m_clearColorA(0.0f)
    {
      TmpCurrent tmpCurrent(this);
      
      //sync the viewport size
      GLint viewport[4];
      glGetIntegerv( GL_VIEWPORT, viewport );
      this->setPosition(viewport[0], viewport[1]);
      this->setSize(viewport[2]-viewport[0], viewport[3]-viewport[1]);
    }

    dp::util::SmartPtr<RenderTargetFB> RenderTargetFB::create( const SmartRenderContext &glContext )
    {
      return new RenderTargetFB( glContext );
    }

    RenderTargetFB::~RenderTargetFB()
    {
    }


    void RenderTargetFB::makeCurrent()
    {
      RenderTarget::makeCurrent();

      // save current draw buffer and set the new one
      glGetIntegerv( GL_DRAW_BUFFER, &m_oldDrawBuffer );
      setDrawBuffer( m_stereoTarget );
    }

    void RenderTargetFB::makeNoncurrent()
    {
      // restore the old draw buffer
      glDrawBuffer( m_oldDrawBuffer );

      RenderTarget::makeNoncurrent();
    }

    dp::util::SmartImage RenderTargetFB::getImage( dp::util::PixelFormat pixelFormat
                                                   , dp::util::DataType pixelDataType
                                                   , unsigned int index )
    {
      if (! m_stereoEnabled )
      {
        return getTargetAsImage( isCurrent() ? GL_BACK : GL_FRONT );
      }
      else
      {
        StereoTarget target = getStereoTarget();

        // Grab left and right image
        dp::util::SmartImage texLeft = getTargetAsImage( getStereoTargetBuffer( LEFT, isCurrent() ), pixelFormat, pixelDataType );
        dp::util::SmartImage texRight = getTargetAsImage( getStereoTargetBuffer( RIGHT, isCurrent() ), pixelFormat, pixelDataType );
#if 0
        return createStereoTextureHost( texLeft, texRight );
#else
        DP_ASSERT( !"There's no equivalent for createStereoTextureHost for dp::util::Image yet" );
        return nullptr;
#endif
      }
    }

    bool RenderTargetFB::isValid()
    {
      // Under which condition is an OpenGL RenderTarget invalid?
      return true;
    }

    void RenderTargetFB::setClearColor( GLclampf r, GLclampf g, GLclampf b, GLclampf a, unsigned int index )
    {
      m_clearColorR = r;
      m_clearColorG = g;
      m_clearColorB = b;
      m_clearColorA = a;
    }

    bool RenderTargetFB::isStereoEnabled() const
    {
      return m_stereoEnabled;
    }

    bool RenderTargetFB::setStereoTarget( StereoTarget target )
    {
      // Only left/mono available in non stereo mode
      if ( !m_stereoEnabled && target != LEFT )
      {
        return false;
      }

      if ( m_stereoTarget != target )
      {
        m_stereoTarget = target;
        if ( isCurrent() )
        {
          setDrawBuffer( m_stereoTarget );
        }
      }

      return true;
    }

    RenderTargetFB::StereoTarget RenderTargetFB::getStereoTarget() const
    {
      return m_stereoTarget;
    }

    GLenum RenderTargetFB::getStereoTargetBuffer( StereoTarget stereoTarget, bool backbuffer )
    {
      // set new draw buffer
      if ( m_stereoEnabled )
      {
        switch ( stereoTarget )
        {
          case LEFT:
            return backbuffer ? GL_BACK_LEFT : GL_FRONT_LEFT;
          break;
          case RIGHT:
            return backbuffer ? GL_BACK_RIGHT : GL_FRONT_RIGHT;
          break;
          case LEFT_AND_RIGHT:
            return backbuffer ? GL_BACK : GL_FRONT;
          break;
          default:
            DP_ASSERT( 0 && "invalid stereoTarget" );
            return GL_BACK;
        }
      }
      else
      {
        // mono is always GL_BACK
       return backbuffer ? GL_BACK : GL_FRONT;
      }
    }

    void RenderTargetFB::setDrawBuffer( StereoTarget stereoTarget )
    {
      glDrawBuffer( getStereoTargetBuffer( stereoTarget, isCurrent() ) );
    }

    bool RenderTargetFB::beginRendering()
    {
      RenderTarget::beginRendering();

      glViewport( m_x, m_y, m_width, m_height );

      if ( m_clearMask )
      {
        GLbitfield clearMaskGL = 0;
        if ( (m_clearMask & TBM_COLOR_BUFFER) )
        {
          clearMaskGL |= GL_COLOR_BUFFER_BIT;
          glClearColor( m_clearColorR, m_clearColorG, m_clearColorB, m_clearColorA );
        }
        if ( (m_clearMask & TBM_DEPTH_BUFFER) )
        {
          clearMaskGL |= GL_DEPTH_BUFFER_BIT;
          glClearDepth( m_clearDepth );
        }
        if ( (m_clearMask & TBM_STENCIL_BUFFER) )
        {
          clearMaskGL |= GL_STENCIL_BUFFER_BIT;
          glClearStencil( m_clearStencil );
        }
        glClear( clearMaskGL );
      }

      return true;
    }

    void RenderTargetFB::endRendering()
    {
      if( m_swapBuffersEnabled )
      {
        getRenderContext()->swap();
      }

      RenderTarget::endRendering();
    }

    void RenderTargetFB::setSwapBuffersEnabled( bool enabled )
    {
      m_swapBuffersEnabled = enabled;
    }

  } // namespace gl
}  // namespace dp
