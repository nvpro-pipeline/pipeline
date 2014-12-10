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


#include <dp/sg/renderer/rix/gl/FSQRenderer.h>
#include <dp/sg/renderer/rix/gl/inc/FSQRendererImpl.h>
#include <dp/util/SharedPtr.h>

namespace dp
{
  namespace sg
  {
    namespace renderer
    {
      namespace rix
      {
        namespace gl
        {

          BEGIN_REFLECTION_INFO( FSQRenderer )
            DERIVE_STATIC_PROPERTIES( FSQRenderer, dp::sg::ui::Renderer );
          END_REFLECTION_INFO

            // this turns a zero-based dp::sg::core::TextureTarget into its corresponding GL target
            static const GLenum cTexTarget[] = { GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D,
            GL_TEXTURE_CUBE_MAP, GL_TEXTURE_1D_ARRAY_EXT,
            GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_RECTANGLE_ARB }; 

          FSQRenderer::FSQRenderer( const dp::gl::RenderTargetSharedPtr &target )
            : Renderer( dp::util::shared_cast<dp::ui::RenderTarget>(target) )
          {  
          }

          FSQRendererSharedPtr FSQRenderer::create( const dp::gl::RenderTargetSharedPtr &renderTarget )
          {
            return( RendererFSQImpl::create( renderTarget ) );
          }

          FSQRenderer::~FSQRenderer(void)
          {
          }

          static void drawQuad( GLint umin, GLint umax, GLint vmin, GLint vmax )
          {
            // disable various state here
            glDisable( GL_DEPTH_TEST );

            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glLoadIdentity();

            glMatrixMode( GL_PROJECTION );
            glPushMatrix();
            glLoadIdentity();

            // draws a viewport-filling quad on z=0
            glBegin( GL_QUADS );
            glTexCoord2i( umin, vmin );
            glVertex2f( -1.f, -1.f );

            glTexCoord2i( umax, vmin );
            glVertex2f( 1.f, -1.f );

            glTexCoord2i( umax, vmax );
            glVertex2f( 1.f, 1.f );

            glTexCoord2i( umin, vmax );
            glVertex2f( -1.f, 1.f);
            glEnd();

            glPopMatrix();
            glMatrixMode( GL_MODELVIEW );
            glPopMatrix();

            glEnable( GL_DEPTH_TEST );
          }

          //
          // NOTE: leaves with GL_TEXTURE_2D disabled GL_DEPTH_TEST enabled, and ActiveTexture = 0
          //
          void FSQRenderer::presentTexture2D( const dp::gl::Texture2DSharedPtr &tex2d, const dp::gl::RenderTargetSharedPtr &target, bool callRTBeginEnd )
          {
            DP_ASSERT( tex2d );

            if( target && callRTBeginEnd )
            {
              target->beginRendering();
            }

            glActiveTexture( GL_TEXTURE0 );

            // mode is texunit state not texture state
            GLint envMode;
            glGetTexEnviv( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &envMode );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

            GLint binding2d;
            glGetIntegerv( GL_TEXTURE_BINDING_2D, &binding2d );

            tex2d->bind();
            glEnable( GL_TEXTURE_2D );
            glDisable(GL_TEXTURE_3D);
            glDisable(GL_TEXTURE_CUBE_MAP);
            glDisable( GL_TEXTURE_RECTANGLE);

            drawQuad( 0, 1, 0, 1 );

            glDisable( GL_TEXTURE_2D );
            glBindTexture( GL_TEXTURE_2D, binding2d );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, envMode );

            if( target && callRTBeginEnd )
            {
              target->endRendering();
            }
          }

          //
          // NOTE: leaves with GL_TEXTURE_RECTANGLE_ARB disabled, GL_DEPTH_TEST enabled, and ActiveTexture = 0
          //
          void FSQRenderer::presentTextureRectangle( const dp::gl::TextureRectangleSharedPtr &tex, const dp::gl::RenderTargetSharedPtr &target, bool callRTBeginEnd )
          {
            DP_ASSERT( tex );

            if( target && callRTBeginEnd )
            {
              target->beginRendering();
            }

            glActiveTexture( GL_TEXTURE0 );

            // mode is texunit state not texture state
            GLint envMode;
            glGetTexEnviv( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &envMode );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

            GLint bindingRect;
            glGetIntegerv( GL_TEXTURE_BINDING_RECTANGLE_ARB, &bindingRect );

            tex->bind();
            glEnable( GL_TEXTURE_RECTANGLE_ARB );
            glDisable(GL_TEXTURE_3D);
            glDisable(GL_TEXTURE_CUBE_MAP);

            drawQuad( 0, tex->getWidth(), 0, tex->getHeight() );

            glDisable( GL_TEXTURE_RECTANGLE_ARB );
            glBindTexture( GL_TEXTURE_RECTANGLE_ARB, bindingRect );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, envMode );

            if( target && callRTBeginEnd )
            {
              target->endRendering();
            }
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

