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

#include <dp/gl/Renderbuffer.h>
#include <dp/gl/RenderTarget.h>
#include <dp/gl/RenderTargetFB.h>
#include <dp/gl/Texture.h>
#include <dp/math/Vecnt.h>

#include <map>
#include <vector>

namespace dp
{
  namespace gl
  {

    static const TargetBufferMask TBM_COLOR_BUFFER0  = TBM_COLOR_BUFFER;
    static const TargetBufferMask TBM_COLOR_BUFFER1  = BIT1;
    static const TargetBufferMask TBM_COLOR_BUFFER2  = BIT2;
    static const TargetBufferMask TBM_COLOR_BUFFER3  = BIT3;
    static const TargetBufferMask TBM_COLOR_BUFFER4  = BIT4;
    static const TargetBufferMask TBM_COLOR_BUFFER5  = BIT5;
    static const TargetBufferMask TBM_COLOR_BUFFER6  = BIT6;
    static const TargetBufferMask TBM_COLOR_BUFFER7  = BIT7;
    static const TargetBufferMask TBM_COLOR_BUFFER8  = BIT8;
    static const TargetBufferMask TBM_COLOR_BUFFER9  = BIT9;
    static const TargetBufferMask TBM_COLOR_BUFFER10 = BIT10;
    static const TargetBufferMask TBM_COLOR_BUFFER11 = BIT11;
    static const TargetBufferMask TBM_COLOR_BUFFER12 = BIT12;
    static const TargetBufferMask TBM_COLOR_BUFFER13 = BIT13;
    static const TargetBufferMask TBM_COLOR_BUFFER14 = BIT14;
    static const TargetBufferMask TBM_COLOR_BUFFER15 = BIT15;
    static const TargetBufferMask TBM_COLOR_BUFFER_MASK = 0x0000FFFF;
    static const TargetBufferMask TBM_MAX_NUM_COLOR_BUFFERS = 16;

    /** \brief This RenderTarget supports OpenGL offscreen rendering with FBOs. It is possible to render to
               dp::gl::Texture objects for streaming textures and dp::gl::Renderbuffer objects. Stereo is also
               supported by attaching separate objects to the left and right eye targets.

               RenderTarget renders to Attachments. Currently there are two types of Attachments available,
               nvgl::RenderTargetFBO::AttachmentTexture for rendering to OpenGL textures and
               nvgl::RenderTargetFBO::AttachmentRenderbuffer for rendering on OpenGL renderbuffers.

               Attachments can be attached with the setAttachment methods. It is possible to share attachments between
               multiple RenderTargetFBO classes or the left and right eye.

               It is not necessary to resize attachments to the size of the RenderTarget. Attachments are resized
               automatically during RenderTargetFBO::beginRendering.
     **/
    class RenderTargetFBO : public RenderTarget
    {
    public:
      /***************/
      /* Attachment */
      /***************/
      /** \brief Base class for all attachments of nvgl::RenderTargetFBO.
          \sa    nvgl::RenderTargetFBO::AttachmentTexture and nvgl::RenderTargetFBO::AttachmentRenderbuffer **/
      class Attachment
      {
      protected:
        friend class RenderTargetFBO;

        /** \brief Interface function to resize the attachment.
            \param width New width for the attachment.
            \param height New height for the attachment.
        **/
        DP_GL_API virtual void resize(int width, int height) = 0;

        /** \brief Bind the attachment to the current framebuffer.
            \param target Target to bind the attachment to.
        **/
        DP_GL_API virtual void bind( GLenum target ) = 0;

        /** \brief Remove the binding for the given target.
            \param param target The binding for the given target will be removed.
        **/
        DP_GL_API virtual void unbind( GLenum target ) = 0;
      };

      typedef std::shared_ptr<Attachment> SharedAttachment;

      /*********************/
      /* AttachmentTexture */
      /*********************/
      /** \brief Class to attach a dp::gl::Texture object to a RenderTargetFBO object.
          \sa nvgl::RenderTargetFBO::setAttachment */
      class AttachmentTexture : public Attachment
      {
      public:
        /** \brief Constructor for 1D textures **/
        DP_GL_API AttachmentTexture( const SharedTexture1D &texture, int level = 0 );

        /** \brief Constructor for 2D textures **/
        DP_GL_API AttachmentTexture( const SharedTexture2D &texture, int level = 0 );

        /** \brief Constructor for 3D textures **/
        DP_GL_API AttachmentTexture( const SharedTexture3D &texture, int zoffset, int level = 0 );

        /** \brief Constructor for 1D array textures **/
        DP_GL_API AttachmentTexture( const SharedTexture1DArray &texture, int layer, int level = 0 );

        /** \brief Constructor for 2D array textures **/
        DP_GL_API AttachmentTexture( const SharedTexture2DArray &texture, int layer, int level = 0);

        /** \brief Constructor for cubemap textures **/
        DP_GL_API AttachmentTexture( const SharedTextureCubemap &texture, int face, int level = 0 );

        /** \brief Constructor for rectangle textures **/
        DP_GL_API AttachmentTexture( const SharedTextureRectangle &texture );

        /** \brief Get the attached textureGL object.
            \return The attached textureGL object.
        **/
        DP_GL_API SharedTexture getTexture() const;

      protected:
        /** \brief Resize the texture to the given size.
            \param width New width for the texture.
            \param height New height for the texture.
        **/
        DP_GL_API virtual void resize(int width, int height);

        /** \brief Bind the texture to the given target of the framebuffer.
            \param param target. The texture will be bound to the given target of the current framebuffer.
        **/
        DP_GL_API virtual void bind( GLenum target );

        /** \brief Remove the texture binding for given a target of the current framebuffer.
            \param param target The binding for the given target will be removed.
        ´**/
        DP_GL_API virtual void unbind( GLenum target );

        DP_GL_API void init( const SharedTexture &texture, GLenum target, GLenum level, GLenum zoffset );

        /** \brief Resize function for 1D textures. **/
        DP_GL_API void resizeTexture1D( int width, int height );

        /** \brief Resize function for 2D textures. **/
        DP_GL_API void resizeTexture2D( int width, int height );

        /** \brief Resize function for 3D textures. **/
        DP_GL_API void resizeTexture3D( int width, int height );

        /** \brief Resize function for 1D Array textures. **/
        DP_GL_API void resizeTexture1DArray( int width, int height );

        /** \brief Resize function for 2D array textures. **/
        DP_GL_API void resizeTexture2DArray( int width, int height );

        /** \brief Resize function for cubemap textures. **/
        DP_GL_API void resizeTextureCubemap( int width, int height );

        /** \brief Bind a 1D texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name which to bind.
        **/
        DP_GL_API void bind1D( GLenum attachment, GLuint textureId );

        /** \brief Bind a 2D texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name which to bind.
        **/
        DP_GL_API void bind2D( GLenum attachment, GLuint textureId );

        /** \brief Bind a 3D texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name which to bind.
        **/
        DP_GL_API void bind3D( GLenum attachment, GLuint textureId );

        /** \brief Bind a layered texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name to bind.
        **/
        DP_GL_API void bindLayer( GLenum attachment, GLuint textureId );

      private:
        /** \brief Function pointer to resize the attached texture type **/
        void (AttachmentTexture::*m_resizeFunc)(int width, int height);

        /** \brief Function pointer to bind the attached texture type to the framebuffer. **/
        void (AttachmentTexture::*m_bindFunc)( GLenum target, GLuint textureId );

        GLenum        m_textureTarget;
        GLuint        m_level;
        GLuint        m_zoffset;
        SharedTexture m_texture;
      };

      typedef std::shared_ptr<AttachmentTexture> SharedAttachmentTexture;

      /**************************/
      /* AttachmentRenderbuffer */
      /**************************/
      /** \brief Class to attach an OpenGL renderbuffer to an nvgl::RenderTargetFBO.
          \sa nvgl::RenderTargetFBO::setAttachment */
      class AttachmentRenderbuffer : public Attachment
      {
      public:
        /** \brief Constructor for an Attachment with a renderbuffer.
            \param renderbuffer Renderbuffer to use for this attachment.
        **/
        DP_GL_API AttachmentRenderbuffer( SharedRenderbuffer renderbuffer );
        DP_GL_API virtual ~AttachmentRenderbuffer();

      protected:
        /** \brief Resize the renderbuffer.
            \param width New width for the renderbuffer.
            \param height New height for the renderbuffer.
        **/
        DP_GL_API virtual void resize(int width, int height);

        /** \brief Bind the renderbuffer to the current framebuffer.
            \param param target. The renderbuffer will be bound to the given target of the current framebuffer.
        ´**/
        DP_GL_API virtual void bind( GLenum target );

        /** \brief Remove the renderbuffer binding for given a target of the current framebuffer.
            \param param target The binding for the given target will be removed.
        ´**/
        DP_GL_API virtual void unbind( GLenum target );

        /** \brief Get the RenderBufferGL object of this attachment.
            \return SharedRenderbuffer object used by this attachment.
        **/
        DP_GL_API SharedRenderbuffer getRenderbuffer() const;

      private:
        SharedRenderbuffer m_renderbuffer;
      };

      typedef std::shared_ptr<AttachmentRenderbuffer> SharedAttachmentRenderbuffer;

      // RenderTarget interface
      enum {
         COLOR_ATTACHMENT0           = GL_COLOR_ATTACHMENT0_EXT
        ,COLOR_ATTACHMENT1           = GL_COLOR_ATTACHMENT1_EXT
        ,COLOR_ATTACHMENT2           = GL_COLOR_ATTACHMENT2_EXT
        ,COLOR_ATTACHMENT3           = GL_COLOR_ATTACHMENT3_EXT
        ,COLOR_ATTACHMENT4           = GL_COLOR_ATTACHMENT4_EXT
        ,COLOR_ATTACHMENT5           = GL_COLOR_ATTACHMENT5_EXT
        ,COLOR_ATTACHMENT6           = GL_COLOR_ATTACHMENT6_EXT
        ,COLOR_ATTACHMENT7           = GL_COLOR_ATTACHMENT7_EXT
        ,COLOR_ATTACHMENT8           = GL_COLOR_ATTACHMENT8_EXT
        ,COLOR_ATTACHMENT9           = GL_COLOR_ATTACHMENT9_EXT
        ,COLOR_ATTACHMENT10          = GL_COLOR_ATTACHMENT10_EXT
        ,COLOR_ATTACHMENT11          = GL_COLOR_ATTACHMENT11_EXT
        ,COLOR_ATTACHMENT12          = GL_COLOR_ATTACHMENT12_EXT
        ,COLOR_ATTACHMENT13          = GL_COLOR_ATTACHMENT13_EXT
        ,COLOR_ATTACHMENT14          = GL_COLOR_ATTACHMENT14_EXT
        ,COLOR_ATTACHMENT15          = GL_COLOR_ATTACHMENT15_EXT
        ,DEPTH_ATTACHMENT            = GL_DEPTH_ATTACHMENT_EXT
        ,STENCIL_ATTACHMENT          = GL_STENCIL_ATTACHMENT_EXT
        ,DEPTH_STENCIL_ATTACHMENT    = GL_DEPTH_STENCIL_ATTACHMENT
      };

      enum {
         COLOR_BUFFER_BIT            = GL_COLOR_BUFFER_BIT
        ,DEPTH_BUFFER_BIT            = GL_DEPTH_BUFFER_BIT
        ,STENCIL_BUFFER_BIT          = GL_STENCIL_BUFFER_BIT
      };

      enum {
         NEAREST                     = GL_NEAREST
        ,LINEAR                      = GL_LINEAR
      };

      typedef unsigned int AttachmentTarget;
      typedef unsigned int BlitMask;
      typedef unsigned int BlitFilter;

    protected:
      DP_GL_API RenderTargetFBO( const SharedRenderContext &glContext );

    public:
      /** \brief Create a new RenderTargetFBO object using the given context */
      DP_GL_API static SharedRenderTargetFBO create( const SharedRenderContext &glContext );

      DP_GL_API virtual ~RenderTargetFBO();
 
      DP_GL_API virtual dp::util::SmartImage getImage( 
        dp::util::PixelFormat pixelFormat = dp::util::PF_BGRA, 
        dp::util::DataType pixelDataType = dp::util::DT_UNSIGNED_INT_8,
        unsigned int index = 0 );

      DP_GL_API virtual bool isValid();

      DP_GL_API virtual bool beginRendering();

      // Stereo API
      /** \brief Change stereo state on this RenderTargetFBO.
          \param stereoEnabled Stereo will be enabled if stereoEnabled is true. Otherwise it'll be disabled.
      **/
      DP_GL_API virtual void setStereoEnabled( bool stereoEnabled);
      DP_GL_API virtual bool isStereoEnabled() const;

      DP_GL_API virtual bool setStereoTarget( StereoTarget target );
      DP_GL_API virtual RenderTarget::StereoTarget getStereoTarget() const;

      /** \brief Remove all attachments for the given stereo target.
          \param stereoTarget All attachments of the given stereoTarget will be removed.
      **/
      DP_GL_API void clearAttachments( StereoTarget stereoTarget = RenderTarget::LEFT_AND_RIGHT );

      /** \brief Sets the attachment for a given target.
          \param target The attachment will be attached to the given target.
          \param attachment Attachment to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedAttachment &attachment, StereoTarget stereoTarget = RenderTarget::LEFT );

      // convenience functions to set an attachment
      /** \brief Attach a 1d texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param level Mipmap level to use for the attachment
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTexture1D &texture, StereoTarget stereoTarget = RenderTarget::LEFT, int level = 0 );

      /** \brief Attach a 2d texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param level Mipmap level to use for the attachment
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTexture2D &texture, StereoTarget stereoTarget = RenderTarget::LEFT, int level = 0 );

      /** \brief Attach a 3d texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param zoffset zoffset of the 3d texture to use for the attachment.
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTexture3D &texture, StereoTarget stereoTarget = RenderTarget::LEFT, int zoffset = 0, int level = 0 );

      /** \brief Attach a 1d texture array.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param layer Layer of the array to use for the attachment
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTexture1DArray &texture, StereoTarget stereoTarget = RenderTarget::LEFT, int layer = 0, int level = 0 );

      /** \brief Attach a 2d texture array.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param layer Layer of the array to use for the attachment
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTexture2DArray &texture, StereoTarget stereoTarget = RenderTarget::LEFT, int layer = 0, int level = 0 );

      /** \brief Attach a cubemap.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param face Face of the cubemap to use for the attachment
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTextureCubemap &texture, StereoTarget stereoTarget = RenderTarget::LEFT, int face = 0, int level = 0 );

      /** \brief Attach a 2d rectangular texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedTextureRectangle &texture, StereoTarget stereoTarget = RenderTarget::LEFT );

      /** \brief Attach a renderbuffer.
          \param target The attachment will be attached to the given target.
          \param buffer Renderbuffer to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedRenderbuffer &buffer, StereoTarget stereoTarget = RenderTarget::LEFT );

      /** \brief Get the attachment for a given target
          \param target Target of the FramebufferObject for the query.
          \param stereoTarget LEFT or RIGHT for the eye.
          \return Attachment for the given parameters.
      **/
      DP_GL_API SharedAttachment getAttachment( AttachmentTarget target, StereoTarget stereoTarget = RenderTarget::LEFT );

      /** \brief Set which targets of the framebuffer object should be active
          \param drawBuffers Vector of GLenums with attachment names
      **/
      DP_GL_API void setDrawBuffers( const std::vector<GLenum> &drawBuffers );

      /** \brief Get the targets of the framebuffer being active
      **/
      DP_GL_API std::vector<GLenum> const& getDrawBuffers() const;

      /** \brief Select the attachment which should be used for read operations on the framebuffer object
          \param readBuffer attachment name of the buffer to read.
      **/
      DP_GL_API void setReadBuffer( GLenum readBuffer );

      /** \brief Set the background color for glClear calls
          \param r red value
          \param g green value
          \param b blue value
          \param a alpha value
          \param index color buffer index (if supported by implementation)
          \remarks The initial values for all components are 0.0.
      */
      DP_GL_API virtual void setClearColor( GLclampf r, GLclampf g, GLclampf b, GLclampf a, unsigned int index = 0 );

      struct BlitRegion
      {
        BlitRegion( int x, int y, int width, int height ) :
          x(x),y(y),width(width),height(height)
        {
        }
        BlitRegion() :
          x(0), y(0), width(0), height(0)
        {
        }
      
        int x;
        int y;
        int width;
        int height;
      };

   
      DP_GL_API void blit( const SharedRenderTargetFBO & destination, const BlitMask & mask = COLOR_BUFFER_BIT, 
                          const BlitFilter & filter = NEAREST );
      DP_GL_API void blit( const SharedRenderTargetFB & destination, const BlitMask & mask = COLOR_BUFFER_BIT, 
                          const BlitFilter & filter = NEAREST );
      DP_GL_API void blit( const SharedRenderTargetFBO & destination, const BlitMask & mask, 
                          const BlitFilter & filter, const BlitRegion & destRegion, 
                          const BlitRegion & srcRegion );
      DP_GL_API void blit( const SharedRenderTargetFB & destination, const BlitMask & mask, 
                          const BlitFilter & filter, const BlitRegion & destRegion, 
                          const BlitRegion & srcRegion );

      /** \brief Get the OpenGL framebuffer name of this object.
          \return OpenGL name of this object.
      **/
      GLuint getFramebufferId( void ) const { return m_framebuffer; }

      /** \brief Test if framebuffer objects are supported.
          \return true if framebuffer objects are supported, false otherwise.
      **/
      DP_GL_API static bool isSupported();
    
      /** \brief Test if multiple rendertargets are supported.
          \return true if multiple rendertargets are supported, false otherwise.
      **/
      DP_GL_API static bool isMultiTargetSupported();

      /** \brief Test if blitting between two framebuffers is supported.
          \return true if blitting is supported, false otherwise.
      **/
      DP_GL_API static bool isBlitSupported();

    protected:
      DP_GL_API int getStereoTargetId( StereoTarget stereoTarget ) const; // 0 -> left eye/mono, 1 -> right eye, assert on LEFT_AND_RIGHT

      DP_GL_API void blit( const int & framebufferId, const BlitMask & mask, const BlitFilter & filter,
                          const BlitRegion & destRegion, const BlitRegion & srcRegion );

      DP_GL_API virtual void makeCurrent();
      DP_GL_API virtual void makeNoncurrent();

      DP_GL_API void bindAttachments( StereoTarget stereoTarget );
      DP_GL_API void resizeAttachments( StereoTarget stereoTarget );
      DP_GL_API bool isFramebufferComplete();

      GLuint m_framebuffer;

      std::vector<GLenum> m_drawBuffers; //!< List of drawbuffers to activate for rendering
      GLenum              m_readBuffer;  //!< read buffer to activate for rendering
      std::vector<GLint> m_bindingStack;    //!< Bind stack for FBO

      typedef std::map< GLenum,SharedAttachment > AttachmentMap;
      typedef std::map< unsigned int, dp::math::Vec4f > ClearColorMap;

      AttachmentMap m_attachments[2]; //<! left/right eye attachments
      AttachmentMap m_attachmentChanges[2];

      dp::math::Vec4f m_attachmentsClearColor[2][TBM_MAX_NUM_COLOR_BUFFERS];

      // Stereo API
      bool         m_stereoEnabled;
      StereoTarget m_stereoTarget;
      int          m_currentlyBoundAttachments;
    };
  } // namespace gl
} // namespace dp
