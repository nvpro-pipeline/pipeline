// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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
      // RenderTarget interface
      enum class AttachmentTarget
      {
         COLOR0           = GL_COLOR_ATTACHMENT0
        ,COLOR1           = GL_COLOR_ATTACHMENT1
        ,COLOR2           = GL_COLOR_ATTACHMENT2
        ,COLOR3           = GL_COLOR_ATTACHMENT3
        ,COLOR4           = GL_COLOR_ATTACHMENT4
        ,COLOR5           = GL_COLOR_ATTACHMENT5
        ,COLOR6           = GL_COLOR_ATTACHMENT6
        ,COLOR7           = GL_COLOR_ATTACHMENT7
        ,COLOR8           = GL_COLOR_ATTACHMENT8
        ,COLOR9           = GL_COLOR_ATTACHMENT9
        ,COLOR10          = GL_COLOR_ATTACHMENT10
        ,COLOR11          = GL_COLOR_ATTACHMENT11
        ,COLOR12          = GL_COLOR_ATTACHMENT12
        ,COLOR13          = GL_COLOR_ATTACHMENT13
        ,COLOR14          = GL_COLOR_ATTACHMENT14
        ,COLOR15          = GL_COLOR_ATTACHMENT15
        ,DEPTH            = GL_DEPTH_ATTACHMENT
        ,STENCIL          = GL_STENCIL_ATTACHMENT
        ,DEPTH_STENCIL    = GL_DEPTH_STENCIL_ATTACHMENT
      };

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
        DP_GL_API virtual void bind( AttachmentTarget target ) = 0;

        /** \brief Remove the binding for the given target.
            \param param target The binding for the given target will be removed.
        **/
        DP_GL_API virtual void unbind( AttachmentTarget target ) = 0;
      };

      typedef std::shared_ptr<Attachment> SharedAttachment;

      /*********************/
      /* AttachmentTexture */
      /*********************/
      class AttachmentTexture;
      typedef std::shared_ptr<AttachmentTexture> SharedAttachmentTexture;

      /** \brief Class to attach a dp::gl::Texture object to a RenderTargetFBO object.
          \sa nvgl::RenderTargetFBO::setAttachment */
      class AttachmentTexture : public Attachment
      {
      public:
        /** \brief Constructor for 1D textures **/
        DP_GL_API static SharedAttachmentTexture create( const Texture1DSharedPtr &texture, int level = 0 );

        /** \brief Constructor for 2D textures **/
        DP_GL_API static SharedAttachmentTexture create( const Texture2DSharedPtr &texture, int level = 0 );

        /** \brief Constructor for 3D textures **/
        DP_GL_API static SharedAttachmentTexture create( const Texture3DSharedPtr &texture, int zoffset, int level = 0 );

        /** \brief Constructor for 1D array textures **/
        DP_GL_API static SharedAttachmentTexture create( const Texture1DArraySharedPtr &texture, int layer, int level = 0 );

        /** \brief Constructor for 2D array textures **/
        DP_GL_API static SharedAttachmentTexture create( const Texture2DArraySharedPtr &texture, int layer, int level = 0);

        /** \brief Constructor for cubemap textures **/
        DP_GL_API static SharedAttachmentTexture create( const TextureCubemapSharedPtr &texture, int face, int level = 0 );

        /** \brief Constructor for rectangle textures **/
        DP_GL_API static SharedAttachmentTexture create( const TextureRectangleSharedPtr &texture );

        /** \brief Get the attached textureGL object.
            \return The attached textureGL object.
        **/
        DP_GL_API TextureSharedPtr getTexture() const;

      protected:
        /** \brief Constructor for 1D textures **/
        DP_GL_API AttachmentTexture( const Texture1DSharedPtr &texture, int level = 0 );

        /** \brief Constructor for 2D textures **/
        DP_GL_API AttachmentTexture( const Texture2DSharedPtr &texture, int level = 0 );

        /** \brief Constructor for 3D textures **/
        DP_GL_API AttachmentTexture( const Texture3DSharedPtr &texture, int zoffset, int level = 0 );

        /** \brief Constructor for 1D array textures **/
        DP_GL_API AttachmentTexture( const Texture1DArraySharedPtr &texture, int layer, int level = 0 );

        /** \brief Constructor for 2D array textures **/
        DP_GL_API AttachmentTexture( const Texture2DArraySharedPtr &texture, int layer, int level = 0);

        /** \brief Constructor for cubemap textures **/
        DP_GL_API AttachmentTexture( const TextureCubemapSharedPtr &texture, int face, int level = 0 );

        /** \brief Constructor for rectangle textures **/
        DP_GL_API AttachmentTexture( const TextureRectangleSharedPtr &texture );
        /** \brief Resize the texture to the given size.
            \param width New width for the texture.
            \param height New height for the texture.
        **/
        DP_GL_API virtual void resize(int width, int height);

        /** \brief Bind the texture to the given target of the framebuffer.
            \param param target. The texture will be bound to the given target of the current framebuffer.
        **/
        DP_GL_API virtual void bind( AttachmentTarget target );

        /** \brief Remove the texture binding for given a target of the current framebuffer.
            \param param target The binding for the given target will be removed.
        ´**/
        DP_GL_API virtual void unbind( AttachmentTarget target );

        DP_GL_API void init( const TextureSharedPtr &texture, GLenum target, GLenum level, GLenum zoffset );

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
        DP_GL_API void bind1D( AttachmentTarget attachment, GLuint textureId );

        /** \brief Bind a 2D texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name which to bind.
        **/
        DP_GL_API void bind2D( AttachmentTarget attachment, GLuint textureId );

        /** \brief Bind a 3D texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name which to bind.
        **/
        DP_GL_API void bind3D( AttachmentTarget attachment, GLuint textureId );

        /** \brief Bind a layered texture to an attachment.
            \param attachment Attachment to bind the texture to.
            \param textureId OpenGL texture name to bind.
        **/
        DP_GL_API void bindLayer( AttachmentTarget attachment, GLuint textureId );

      private:
        /** \brief Function pointer to resize the attached texture type **/
        void (AttachmentTexture::*m_resizeFunc)(int width, int height);

        /** \brief Function pointer to bind the attached texture type to the framebuffer. **/
        void (AttachmentTexture::*m_bindFunc)( AttachmentTarget target, GLuint textureId );

        GLenum            m_textureTarget;
        GLuint            m_level;
        GLuint            m_zoffset;
        TextureSharedPtr  m_texture;
      };

      /**************************/
      /* AttachmentRenderbuffer */
      /**************************/
      class AttachmentRenderbuffer;
      typedef std::shared_ptr<AttachmentRenderbuffer> SharedAttachmentRenderbuffer;

      /** \brief Class to attach an OpenGL renderbuffer to an nvgl::RenderTargetFBO.
          \sa nvgl::RenderTargetFBO::setAttachment */
      class AttachmentRenderbuffer : public Attachment
      {
      public:
        DP_GL_API static SharedAttachmentRenderbuffer create( RenderbufferSharedPtr const& renderbuffer );
        DP_GL_API virtual ~AttachmentRenderbuffer();

      protected:
        /** \brief Constructor for an Attachment with a renderbuffer.
            \param renderbuffer Renderbuffer to use for this attachment.
        **/
        DP_GL_API AttachmentRenderbuffer( RenderbufferSharedPtr const& renderbuffer );

        /** \brief Resize the renderbuffer.
            \param width New width for the renderbuffer.
            \param height New height for the renderbuffer.
        **/
        DP_GL_API virtual void resize(int width, int height);

        /** \brief Bind the renderbuffer to the current framebuffer.
            \param param target. The renderbuffer will be bound to the given target of the current framebuffer.
        ´**/
        DP_GL_API virtual void bind( AttachmentTarget target );

        /** \brief Remove the renderbuffer binding for given a target of the current framebuffer.
            \param param target The binding for the given target will be removed.
        ´**/
        DP_GL_API virtual void unbind( AttachmentTarget target );

        /** \brief Get the RenderBufferGL object of this attachment.
            \return RenderbufferSharedPtr object used by this attachment.
        **/
        DP_GL_API RenderbufferSharedPtr getRenderbuffer() const;

      private:
        RenderbufferSharedPtr m_renderbuffer;
      };

      enum class BlitMask
      {
         COLOR_BUFFER_BIT            = GL_COLOR_BUFFER_BIT
        ,DEPTH_BUFFER_BIT            = GL_DEPTH_BUFFER_BIT
        ,STENCIL_BUFFER_BIT          = GL_STENCIL_BUFFER_BIT
      };

      enum class BlitFilter
      {
         NEAREST                     = GL_NEAREST
        ,LINEAR                      = GL_LINEAR
      };

    protected:
      DP_GL_API RenderTargetFBO( const RenderContextSharedPtr &glContext );

    public:
      /** \brief Create a new RenderTargetFBO object using the given context */
      DP_GL_API static RenderTargetFBOSharedPtr create( const RenderContextSharedPtr &glContext );

      DP_GL_API virtual ~RenderTargetFBO();

      DP_GL_API virtual dp::util::ImageSharedPtr getImage(
        dp::PixelFormat pixelFormat = dp::PixelFormat::BGRA,
        dp::DataType pixelDataType = dp::DataType::UNSIGNED_INT_8,
        unsigned int index = 0 );

      DP_GL_API virtual bool isValid();

      DP_GL_API virtual bool beginRendering();
      DP_GL_API virtual void endRendering();

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
      DP_GL_API void clearAttachments( StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT_AND_RIGHT );

      /** \brief Sets the attachment for a given target.
          \param target The attachment will be attached to the given target.
          \param attachment Attachment to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const SharedAttachment &attachment, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT );

      // convenience functions to set an attachment
      /** \brief Attach a 1d texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param level Mipmap level to use for the attachment
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const Texture1DSharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT, int level = 0 );

      /** \brief Attach a 2d texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param level Mipmap level to use for the attachment
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const Texture2DSharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT, int level = 0 );

      /** \brief Attach a 3d texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param zoffset zoffset of the 3d texture to use for the attachment.
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const Texture3DSharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT, int zoffset = 0, int level = 0 );

      /** \brief Attach a 1d texture array.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param layer Layer of the array to use for the attachment
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const Texture1DArraySharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT, int layer = 0, int level = 0 );

      /** \brief Attach a 2d texture array.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param layer Layer of the array to use for the attachment
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const Texture2DArraySharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT, int layer = 0, int level = 0 );

      /** \brief Attach a cubemap.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \param face Face of the cubemap to use for the attachment
          \param level Mipmap level to use for the attachment.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const TextureCubemapSharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT, int face = 0, int level = 0 );

      /** \brief Attach a 2d rectangular texture.
          \param target The attachment will be attached to the given target.
          \param texture Texture to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const TextureRectangleSharedPtr &texture, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT );

      /** \brief Attach a renderbuffer.
          \param target The attachment will be attached to the given target.
          \param buffer Renderbuffer to attach.
          \param stereoTarget For stereo rendering it's possible to assign the attachment to the LEFT, RIGHT or LEFT_AND_RIGHT eye.
          \return true if the operation was successful, false otherwise.
      **/
      DP_GL_API bool setAttachment( AttachmentTarget target, const RenderbufferSharedPtr &buffer, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT );

      /** \brief Get the attachment for a given target
          \param target Target of the FramebufferObject for the query.
          \param stereoTarget LEFT or RIGHT for the eye.
          \return Attachment for the given parameters.
      **/
      DP_GL_API SharedAttachment getAttachment( AttachmentTarget target, StereoTarget stereoTarget = RenderTarget::StereoTarget::LEFT );

      /** \brief Set which targets of the framebuffer object should be active
          \param drawBuffers Vector of GLenums with attachment names
      **/
      DP_GL_API void setDrawBuffers( const std::vector<AttachmentTarget> &drawBuffers );

      /** \brief Get the targets of the framebuffer being active
      **/
      DP_GL_API std::vector<AttachmentTarget> const& getDrawBuffers() const;

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


      DP_GL_API void blit( const RenderTargetFBOSharedPtr & destination, const BlitMask & mask = BlitMask::COLOR_BUFFER_BIT,
                          const BlitFilter & filter = BlitFilter::NEAREST );
      DP_GL_API void blit( const RenderTargetFBSharedPtr & destination, const BlitMask & mask = BlitMask::COLOR_BUFFER_BIT,
                          const BlitFilter & filter = BlitFilter::NEAREST );
      DP_GL_API void blit( const RenderTargetFBOSharedPtr & destination, const BlitMask & mask,
                          const BlitFilter & filter, const BlitRegion & destRegion,
                          const BlitRegion & srcRegion );
      DP_GL_API void blit( const RenderTargetFBSharedPtr & destination, const BlitMask & mask,
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

      /** \brief Test it isMulticastSupported
          \return true if GL_NVX_linked_gpu_multicast is supported
      **/
      DP_GL_API bool isMulticastSupported();

      /** \brief Set multicast state for this FBO
          \param enabled If enabled is true multicast will be enabled, otherwise it'll be disabled
          \remarks If multicast is not supported this function will throw a std::runtime_error
      **/
      DP_GL_API void setMulticastEnabled(bool enabled);

      /** \brief Check multicast state for this FBO
          \return true if multicast extension is enabled for this FBO
      **/
      DP_GL_API virtual bool isMulticastEnabled() const;

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

      std::vector<AttachmentTarget> m_drawBuffers;    //!< List of drawbuffers to activate for rendering
      GLenum                        m_readBuffer;     //!< read buffer to activate for rendering
      std::vector<GLint>            m_bindingStack;   //!< Bind stack for FBO

      typedef std::map< AttachmentTarget,SharedAttachment > AttachmentMap;
      typedef std::map< unsigned int, dp::math::Vec4f > ClearColorMap;

      AttachmentMap m_attachments[2]; //<! left/right eye attachments
      AttachmentMap m_attachmentChanges[2];

      dp::math::Vec4f m_attachmentsClearColor[2][TBM_MAX_NUM_COLOR_BUFFERS];

      // Stereo API
      bool         m_stereoEnabled;
      bool         m_multicastEnabled;
      StereoTarget m_stereoTarget;
      int          m_currentlyBoundAttachments;

    };
  } // namespace gl
} // namespace dp
