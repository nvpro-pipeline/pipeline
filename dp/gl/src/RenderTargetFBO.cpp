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


#include <dp/gl/RenderTargetFBO.h>

#include <dp/util/BitMask.h>
#include <dp/util/SharedPtr.h>

namespace dp
{
  namespace gl
  {
    RenderTargetFBO::RenderTargetFBO( const SharedRenderContext &glContext)
      : RenderTarget( glContext )
      , m_framebuffer( 0 )
      , m_stereoTarget( LEFT )
      , m_stereoEnabled( false )
      , m_currentlyBoundAttachments( 0 )
    {
      DP_ASSERT( glContext );

      // cannot use this->makeCurrent() here because it sets the drawbuffer to GL_NONE
      RenderTarget::makeCurrent();

      // requires the following extension
      DP_ASSERT( isSupported() );

      glGenFramebuffers( 1, &m_framebuffer );

      // get default drawbuffer
      GLint binding;
      glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &binding );
      glBindFramebuffer( GL_FRAMEBUFFER_EXT, m_framebuffer );

      // get default buffer bindings
      GLint buffer;
      glGetIntegerv( GL_DRAW_BUFFER, &buffer );
      m_drawBuffers.push_back( buffer );

      glGetIntegerv( GL_READ_BUFFER, &buffer );
      m_readBuffer = buffer;

      glBindFramebuffer( GL_FRAMEBUFFER_EXT, binding );
      RenderTarget::makeNoncurrent();
    }

    SharedRenderTargetFBO RenderTargetFBO::create( const SharedRenderContext &glContext )
    {
      return( std::shared_ptr<RenderTargetFBO>( new RenderTargetFBO( glContext ) ) );
    }

    RenderTargetFBO::~RenderTargetFBO()
    {
      DP_ASSERT( m_bindingStack.empty() );
      DP_ASSERT( !isCurrent() );
    
      RenderTarget::makeCurrent();
      glDeleteFramebuffers( 1, &m_framebuffer );
      clearAttachments();
      RenderTarget::makeNoncurrent();
    }

    void RenderTargetFBO::makeCurrent()
    {
      RenderTarget::makeCurrent();
    
      GLint binding;
      glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &binding );
      m_bindingStack.push_back( binding );

      glBindFramebuffer( GL_FRAMEBUFFER_EXT, m_framebuffer );
      resizeAttachments( m_stereoTarget );
      bindAttachments( m_stereoTarget );

      // FIXME should it be possible to disable all draw buffers?

      // Choose Read/DrawBuffers. Note that Read/DrawBuffer state is bound to the FBO and thus does not
      // need to be reset in makeNoncurrent.
    
      glReadBuffer( m_readBuffer );

      if ( m_drawBuffers.empty() )
      {
        glDrawBuffer( GL_NONE );
      }
      else if ( m_drawBuffers.size() == 1 )
      {
        glDrawBuffer( m_drawBuffers[0] );
      }
      else
      {
        // extension is being checked in setDrawBuffers 
        glDrawBuffers( dp::util::checked_cast<GLsizei>(m_drawBuffers.size()), &m_drawBuffers[0] ); 
      }
    }

    void RenderTargetFBO::makeNoncurrent()
    {
      DP_ASSERT( !m_bindingStack.empty() );

      glBindFramebuffer( GL_FRAMEBUFFER_EXT, m_bindingStack.back() );
      m_bindingStack.pop_back();

      RenderTarget::makeNoncurrent();
    }

    dp::util::SmartImage RenderTargetFBO::getImage( dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType, unsigned int index )
    {
      if (! m_stereoEnabled )
      {
        return getTargetAsImage( GL_COLOR_ATTACHMENT0_EXT + index, pixelFormat, pixelDataType );
      }
      else
      {
        StereoTarget target = getStereoTarget();

        // Grab left and right image
        setStereoTarget( LEFT );
        dp::util::SmartImage texLeft = getTargetAsImage( GL_COLOR_ATTACHMENT0_EXT + index, pixelFormat, pixelDataType );

        setStereoTarget( RIGHT );
        dp::util::SmartImage texRight = getTargetAsImage( GL_COLOR_ATTACHMENT0_EXT + index, pixelFormat, pixelDataType );

        setStereoTarget( target );
#if 0
        return createStereoTextureHost( texLeft, texRight );
#else
        DP_ASSERT( !"There's no equivalent for createStereoTextureHost for dp::util::Image yet" );
        return nullptr;
#endif
      }
    }

    bool RenderTargetFBO::isValid()
    {
      bool valid = RenderTarget::isValid() && isFramebufferComplete();
      return valid;
    }

    void RenderTargetFBO::clearAttachments( StereoTarget stereoTarget )
    {
      switch (stereoTarget)
      {
      case LEFT:
        m_attachments[0].clear();
      break;
      case RIGHT:
        m_attachments[1].clear();
      break;
      case LEFT_AND_RIGHT:
        m_attachments[0].clear();
        m_attachments[1].clear();
      break;
      default:
        DP_ASSERT( 0 && "invalid stereoTarget" );
      }
    }

    bool RenderTargetFBO::setAttachment( GLenum target, const SharedAttachment &attachment, StereoTarget stereoTarget )
    {
      int stereoId = getStereoTargetId( stereoTarget );

      m_attachments[stereoId][target] = attachment;

      if ( isCurrent() )
      {
        attachment->bind( target );
      }
      else
      {
        // keep track of changed attachments. This reduces the number of bind calls in bindAttachments()
        // which is being called on beginRendering().
        m_attachmentChanges[stereoId][target] = attachment;
      }
      return true;
    }

    int RenderTargetFBO::getStereoTargetId( StereoTarget stereoTarget ) const
    {
      switch (stereoTarget)
      {
      case LEFT:
        return 0;
      case RIGHT:
        return 1;
      default:
        DP_ASSERT( 0 && "invalid stereoTarget" );
        return 0;
      }
    }

    RenderTargetFBO::SharedAttachment RenderTargetFBO::getAttachment( GLenum target, StereoTarget stereoTarget )
    {
      int stereoId = getStereoTargetId( stereoTarget );

      AttachmentMap::iterator it = m_attachments[stereoId].find( target );
      return it == m_attachments[stereoId].end() ? SharedAttachment() : it->second;
    }

    void RenderTargetFBO::bindAttachments( StereoTarget stereoTarget )
    {
      int stereoId = getStereoTargetId( stereoTarget );

      if ( m_currentlyBoundAttachments != stereoId )
      {
        // rebind all attachments
        AttachmentMap &oldAttachments = m_attachments[m_currentlyBoundAttachments];
        AttachmentMap &newAttachments = m_attachments[stereoId];
        for (AttachmentMap::iterator it = newAttachments.begin(); it != newAttachments.end(); ++it)
        {
          // bind only if attachments are different
          AttachmentMap::iterator oldIt = oldAttachments.find( it->first );
          if ( oldIt == oldAttachments.end() || oldIt->second != it->second )
          {
            it->second->bind( it->first );
          }
        }
        m_currentlyBoundAttachments = stereoId;
      }
      else // Check if there have been changes to the active stereoId attachments.
      {
        AttachmentMap &attachmentChanges = m_attachmentChanges[stereoId];
        for (AttachmentMap::iterator it = attachmentChanges.begin(); it != attachmentChanges.end(); ++it)
        {
          // Attachment has changed
          if ( it->second )
          {
            it->second->bind( it->first );
          }
          // Attachment has been removed
          else
          {
            it->second->unbind( it->first );
          }
        }
      }
      // Attachment changes of this stereoId have been handled in either of the above cases.
      m_attachmentChanges[stereoId].clear();
    }

    void RenderTargetFBO::resizeAttachments( StereoTarget stereoTarget )
    {
      int stereoId = getStereoTargetId( stereoTarget );

      for (AttachmentMap::iterator it = m_attachments[stereoId].begin(); it != m_attachments[stereoId].end(); ++it)
      {
        it->second->resize( getWidth(), getHeight() );
      }
    }

    bool RenderTargetFBO::isFramebufferComplete()
    {
      GLenum status;
      status = (GLenum) glCheckFramebufferStatus(GL_FRAMEBUFFER_EXT);
      bool complete = false;

      // Every other value of status other than GL_FRAMEBUFFER_COMPLETE_EXT
      // indicates the reason why the framebuffer is not complete
      // At this point then, the user would typically
      // specify new parameters to make it work with different creation
      // parameters. Refer to  
      // http://www.nvidia.com/dev_content/nvopenglspecs/GL_EXT_framebuffer_object.txt
      // on strategies to implement this.
      // The official OpenGL extension repository is http://www.opengl.org/registry/

      switch (status) 
      {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
          complete = true;
          break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
          // Unsupported framebuffer format
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT: 
          // Framebuffer incomplete attachment
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
          // Framebuffer incomplete, missing attachment
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
          // Framebuffer incomplete, attached images must have same dimensions
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
          // Framebuffer incomplete, attached images must have same format
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
          // Framebuffer incomplete, missing draw buffer
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
          // Framebuffer incomplete, missing read buffer
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT:
          // all attachments must have the same number of samples
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT:
          // If any framebuffer attachment is layered, all populated attachments
          // must be layered.  Additionally, all populated color attachments must
          // be from textures of the same target (i.e., three-dimensional, cube
          // map, or one- or two-dimensional array textures).
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_EXT:
          // If any framebuffer attachment is layered, all attachments must have
          // the same layer count.  For three-dimensional textures, the layer count
          // is the depth of the attached volume.  For cube map textures, the layer
          // count is always six.  For one- and two-dimensional array textures, the
          // layer count is simply the number of layers in the array texture.
          break;
        default:
          // Unknown error
          DP_ASSERT(0);
          break;
      }

      return complete;
    }

    void RenderTargetFBO::setClearColor( GLclampf r, GLclampf g, GLclampf b, GLclampf a, unsigned int index /*= 0 */ )
    {
      m_attachmentsClearColor[ getStereoTargetId(m_stereoTarget) ][index] = dp::math::Vec4f(r, g, b, a);
    }


    /*********************/
    /* AttachmentTexture */
    /*********************/
    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTexture1D const& texture, int level )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture, level ) ) );
    }

    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTexture2D const& texture, int level )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture, level ) ) );
    }

    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTexture3D const& texture, int zoffset, int level )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture, zoffset, level ) ) );
    }

    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTexture1DArray const& texture, int layer, int level )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture, layer, level ) ) );
    }

    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTexture2DArray const& texture, int layer, int level )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture, layer, level ) ) );
    }

    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTextureCubemap const& texture, int face, int level )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture, face, level ) ) );
    }

    RenderTargetFBO::SharedAttachmentTexture RenderTargetFBO::AttachmentTexture::create( SharedTextureRectangle const& texture )
    {
      return( std::shared_ptr<AttachmentTexture>( new AttachmentTexture( texture ) ) );
    }


    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTexture1D &texture, int level )
      : m_bindFunc( &AttachmentTexture::bind1D )
      , m_resizeFunc( &AttachmentTexture::resizeTexture1D )
    {
      init( texture, texture->getTarget(), level, 0 );
    }

    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTexture2D &texture, int level )
      : m_bindFunc( &AttachmentTexture::bind2D )
      , m_resizeFunc( &AttachmentTexture::resizeTexture2D )
    {
      init( texture, texture->getTarget(), level, 0 );
    }

    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTexture3D &texture, int zoffset, int level )
      : m_bindFunc( &AttachmentTexture::bind3D )
      , m_resizeFunc( &AttachmentTexture::resizeTexture3D )
    {
      init( texture, texture->getTarget(), level, zoffset );
    }

    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTexture1DArray &texture, int layer, int level )
      : m_bindFunc( &AttachmentTexture::bindLayer )
      , m_resizeFunc( &AttachmentTexture::resizeTexture1DArray )
    {
      init( texture, texture->getTarget(), level, layer );
    }

    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTexture2DArray &texture, int layer, int level )
      : m_bindFunc( &AttachmentTexture::bindLayer )
      , m_resizeFunc( &AttachmentTexture::resizeTexture2DArray )
    {
      init( texture, texture->getTarget(), level, layer );
    }

    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTextureCubemap &texture, int face, int level )
      : m_bindFunc( &AttachmentTexture::bind2D )
      , m_resizeFunc( &AttachmentTexture::resizeTextureCubemap )
    {
      init( texture, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face , level, 0 );
    }

    RenderTargetFBO::AttachmentTexture::AttachmentTexture( const SharedTextureRectangle &texture )
      : m_bindFunc( &AttachmentTexture::bind2D )
      , m_resizeFunc( &AttachmentTexture::resizeTexture2D )
    {
      init( texture, texture->getTarget(), 0, 0 );
    }

    SharedTexture RenderTargetFBO::AttachmentTexture::getTexture() const
    {
      return m_texture;
    }

    void RenderTargetFBO::AttachmentTexture::resize(int width, int height)
    {
      DP_ASSERT( m_resizeFunc );

      (this->*m_resizeFunc)( width, height );
    }

    void RenderTargetFBO::AttachmentTexture::bind( GLenum attachment )
    {
      DP_ASSERT( m_bindFunc );
      (this->*m_bindFunc)( attachment, m_texture->getGLId() );
    
    }

    void RenderTargetFBO::AttachmentTexture::unbind( GLenum attachment )
    {
      (this->*m_bindFunc)( attachment, 0 );
    }

    void RenderTargetFBO::AttachmentTexture::init( const SharedTexture &texture, GLenum texTarget, GLuint level, GLuint zoffset )
    {
      m_texture = texture;
      m_textureTarget = texTarget;
      m_level = level;
      m_zoffset = zoffset;
    }

    /********************/
    /* Resize functions */
    /********************/
    void RenderTargetFBO::AttachmentTexture::resizeTexture1D( int width, int height )
    {
      DP_ASSERT( height == 1 );
      dp::util::shared_cast<Texture1D>(m_texture)->resize( width );
    }

    void RenderTargetFBO::AttachmentTexture::resizeTexture2D( int width, int height )
    {
      dp::util::shared_cast<Texture2D>(m_texture)->resize( width, height );
    }

    void RenderTargetFBO::AttachmentTexture::resizeTexture3D( int width, int height )
    {
      dp::util::shared_cast<Texture3D>(m_texture)->resize( width, height, dp::util::shared_cast<Texture3D>(m_texture)->getDepth() );
    }

    void RenderTargetFBO::AttachmentTexture::resizeTexture1DArray( int width, int height )
    {
      DP_ASSERT( height == 1 );
      dp::util::shared_cast<Texture1DArray>(m_texture)->resize( width, dp::util::shared_cast<Texture1DArray>(m_texture)->getLayers() );
    }

    void RenderTargetFBO::AttachmentTexture::resizeTexture2DArray( int width, int height )
    {
      dp::util::shared_cast<Texture2DArray>(m_texture)->resize( width, height, dp::util::shared_cast<Texture2DArray>(m_texture)->getLayers() );
    }

    void RenderTargetFBO::AttachmentTexture::resizeTextureCubemap( int width, int height )
    {
      dp::util::shared_cast<TextureCubemap>(m_texture)->resize( width, height );
    }

    /******************/
    /* Bind functions */
    /******************/
    void RenderTargetFBO::AttachmentTexture::bind1D( GLenum attachment, GLuint textureId )
    {
      glFramebufferTexture1D( GL_FRAMEBUFFER_EXT, attachment, m_textureTarget, textureId, m_level );
    }

    void RenderTargetFBO::AttachmentTexture::bind2D( GLenum attachment, GLuint textureId )
    {
      glFramebufferTexture2D( GL_FRAMEBUFFER_EXT, attachment, m_textureTarget, textureId, m_level );
    }

    void RenderTargetFBO::AttachmentTexture::bind3D( GLenum attachment, GLuint textureId )
    {
      // INFO this could use bindLayer too, but will fail if the GL_EXT_texture_array extension is not available.
      glFramebufferTexture3D( GL_FRAMEBUFFER_EXT, attachment, m_textureTarget, textureId, m_level, m_zoffset );
    }

    void RenderTargetFBO::AttachmentTexture::bindLayer( GLenum attachment, GLuint textureId )
    {
      glFramebufferTextureLayer( GL_FRAMEBUFFER_EXT, attachment, textureId, m_level, m_zoffset );
    }

    /**************************/
    /* AttachmentRenderbuffer */
    /**************************/
    RenderTargetFBO::SharedAttachmentRenderbuffer RenderTargetFBO::AttachmentRenderbuffer::create( SharedRenderbuffer const& renderbuffer )
    {
      return( std::shared_ptr<AttachmentRenderbuffer>( new AttachmentRenderbuffer( renderbuffer ) ) );
    }

    RenderTargetFBO::AttachmentRenderbuffer::AttachmentRenderbuffer( SharedRenderbuffer const& renderbuffer )
      : m_renderbuffer( renderbuffer )
    {
    }

    RenderTargetFBO::AttachmentRenderbuffer::~AttachmentRenderbuffer()
    {
    }

    SharedRenderbuffer RenderTargetFBO::AttachmentRenderbuffer::getRenderbuffer() const
    {
      return m_renderbuffer;
    }

    void RenderTargetFBO::AttachmentRenderbuffer::resize(int width, int height)
    {
      m_renderbuffer->resize( width, height );
    }

    void RenderTargetFBO::AttachmentRenderbuffer::bind( AttachmentTarget attachment )
    {
      glFramebufferRenderbuffer( GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, m_renderbuffer->getGLId() );
    }

    void RenderTargetFBO::AttachmentRenderbuffer::unbind( AttachmentTarget attachment )
    {
      glFramebufferRenderbuffer( GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, 0 );
    }

    void RenderTargetFBO::setDrawBuffers( const std::vector<GLenum> &drawBuffers )
    {
      DP_ASSERT( m_drawBuffers.size() <= 1 || isMultiTargetSupported() );
      m_drawBuffers = drawBuffers;
    }

    std::vector<GLenum> const& RenderTargetFBO::getDrawBuffers() const
    {
      return( m_drawBuffers );
    }

    void RenderTargetFBO::setReadBuffer( GLenum readBuffer )
    {
      m_readBuffer = readBuffer;
    }

    void RenderTargetFBO::blit( const SharedRenderTargetFBO & destination, const BlitMask & mask, const BlitFilter & filter )
    {
      BlitRegion destRegion(0,0, destination->getWidth(), destination->getHeight());
      BlitRegion srcRegion(0,0, this->getWidth(), this->getHeight());
      blit(destination->getFramebufferId(), mask, filter, destRegion, srcRegion);
    } 

    void RenderTargetFBO::blit( const SharedRenderTargetFB & destination, const BlitMask & mask, const BlitFilter & filter )
    {
      GLint binding;

      TmpCurrent current = TmpCurrent(this);

      glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &binding );
      m_bindingStack.push_back( binding );
      glBindFramebuffer( GL_FRAMEBUFFER_EXT, m_framebuffer );

      resizeAttachments( m_stereoTarget );
      bindAttachments( m_stereoTarget );

      // restore old binding
      glBindFramebuffer( GL_FRAMEBUFFER_EXT, binding );
      m_bindingStack.pop_back();

      BlitRegion destRegion(0,0, destination->getWidth(), destination->getHeight());
      BlitRegion srcRegion(0,0, this->getWidth(), this->getHeight());
      blit(0, mask, filter, destRegion, srcRegion);
    }

    void RenderTargetFBO::blit( const SharedRenderTargetFBO & destination, const BlitMask & mask, const BlitFilter & filter,
                                const BlitRegion & destRegion, const BlitRegion & srcRegion )
    {
      GLint binding;

      TmpCurrent current = TmpCurrent(this);

      glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &binding );
      m_bindingStack.push_back( binding );

      glBindFramebuffer( GL_FRAMEBUFFER_EXT, m_framebuffer );
      resizeAttachments( m_stereoTarget );
      bindAttachments( m_stereoTarget );

      glBindFramebuffer( GL_FRAMEBUFFER_EXT, destination->m_framebuffer );
      destination->resizeAttachments( m_stereoTarget );
      destination->bindAttachments( destination->m_stereoTarget );

      // restore old binding
      glBindFramebuffer( GL_FRAMEBUFFER_EXT, binding );
      m_bindingStack.pop_back();

      blit(destination->getFramebufferId(), mask, filter, destRegion, srcRegion);
    } 

    void RenderTargetFBO::blit( const SharedRenderTargetFB & destination, const BlitMask & mask, const BlitFilter & filter,
                                const BlitRegion & destRegion, const BlitRegion & srcRegion )
    {
      GLint binding;

      TmpCurrent current = TmpCurrent(this);

      glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &binding );
      m_bindingStack.push_back( binding );
      glBindFramebuffer( GL_FRAMEBUFFER_EXT, m_framebuffer );

      resizeAttachments( m_stereoTarget );
      bindAttachments( m_stereoTarget );

      // restore old binding
      glBindFramebuffer( GL_FRAMEBUFFER_EXT, binding );
      m_bindingStack.pop_back();

      blit(0, mask, filter, destRegion, srcRegion);
    }

    void RenderTargetFBO::blit( const int & framebufferID, const BlitMask & mask, const BlitFilter & filter, 
                                  const BlitRegion & destRegion, const BlitRegion & srcRegion )
    {
      glBindFramebuffer( GL_READ_FRAMEBUFFER, m_framebuffer );
      glBindFramebuffer( GL_DRAW_FRAMEBUFFER, framebufferID );
      glBlitFramebuffer( srcRegion.x, srcRegion.y, srcRegion.x + srcRegion.width, srcRegion.y + srcRegion.height,  
                         destRegion.x, destRegion.y, destRegion.x + destRegion.width, destRegion.y + destRegion.height,
                         mask, filter );                       
    }

    bool RenderTargetFBO::isSupported()
    {
      return /*!!GLEW_ARB_framebuffer_object ||*/ !!GLEW_EXT_framebuffer_object;
    }

    bool RenderTargetFBO::isMultiTargetSupported()
    {
      return GLEW_VERSION_2_0 || !!GLEW_ARB_draw_buffers;
    }
  
    bool RenderTargetFBO::isBlitSupported()
    {
      return /*!!GLEW_ARB_framebuffer_object ||*/ !!GLEW_EXT_framebuffer_blit;
    }

    // Stereo API
    void RenderTargetFBO::setStereoEnabled( bool stereoEnabled )
    {
      if ( stereoEnabled != m_stereoEnabled )
      {
        m_stereoEnabled = stereoEnabled;

        // ensure that mono target is being used in non stereo mode
        if ( !m_stereoEnabled )
        {
          setStereoTarget( LEFT );
        }
      }
    }

    bool RenderTargetFBO::isStereoEnabled() const
    {
      return m_stereoEnabled;
    }

    bool RenderTargetFBO::setStereoTarget( StereoTarget stereoTarget )
    {
      if ( stereoTarget != m_stereoTarget )
      {
        if (  (!m_stereoEnabled && stereoTarget != LEFT)          // only mono target supported for non stereo mode
            ||( m_stereoEnabled && stereoTarget == LEFT_AND_RIGHT ) ) // not supported to render on left/right target at the same time
        {
          return false;
        }

        m_stereoTarget = stereoTarget;
        if ( isCurrent() )
        {
          resizeAttachments( stereoTarget );
          bindAttachments( stereoTarget );
        }
      }
      return true;
    }

    RenderTarget::StereoTarget RenderTargetFBO::getStereoTarget() const
    {
      return m_stereoTarget;
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTexture1D &texture, StereoTarget stereoTarget, int level )
    {
      return setAttachment( target, AttachmentTexture::create( texture, level ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTexture2D &texture, StereoTarget stereoTarget, int level )
    {
      return setAttachment( target, AttachmentTexture::create( texture, level ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTexture3D &texture, StereoTarget stereoTarget, int zoffset, int level )
    {
      return setAttachment( target, AttachmentTexture::create( texture, zoffset, level ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTexture1DArray &texture, StereoTarget stereoTarget, int layer, int level )
    {
      return setAttachment( target, AttachmentTexture::create( texture, layer, level ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTexture2DArray &texture, StereoTarget stereoTarget, int layer, int level )
    {
      return setAttachment( target, AttachmentTexture::create( texture, layer, level ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTextureCubemap &texture, StereoTarget stereoTarget, int face, int level )
    {
      return setAttachment( target, AttachmentTexture::create( texture, face, level ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedTextureRectangle &texture, StereoTarget stereoTarget )
    {
      return setAttachment( target, AttachmentTexture::create( texture ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::setAttachment( AttachmentTarget target, const SharedRenderbuffer &buffer, StereoTarget stereoTarget )
    {
      return setAttachment( target, AttachmentRenderbuffer::create( buffer ).inplaceCast<Attachment>(), stereoTarget );
    }

    bool RenderTargetFBO::beginRendering()
    {
      makeCurrent();

      glViewport( m_x, m_y, m_width, m_height );
      
      unsigned int colorBufferMask = m_clearMask & TBM_COLOR_BUFFER_MASK;
      unsigned int i = 0;
      while ( colorBufferMask )
      {
        if ( colorBufferMask & 1 )
        {
          glClearBufferfv(GL_COLOR, i, &m_attachmentsClearColor[ getStereoTargetId( m_stereoTarget ) ][i][0] );
        }
        colorBufferMask >>= 1;
        ++i; 
      }

      if ( m_clearMask & TBM_DEPTH_BUFFER )
      {
        float clearDepth = float(m_clearDepth);
        glClearBufferfv( GL_DEPTH, 0, &clearDepth );
      }

      if ( m_clearMask & TBM_STENCIL_BUFFER )
      {
        glClearBufferuiv( GL_STENCIL, 0, &m_clearStencil );
      }

      return true;
    }

  } // namespace gl
} // namespace dp
