// Copyright NVIDIA Corporation 2011-2014
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


#include <dp/DP.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITClosestList.h>
#include <dp/sg/renderer/rix/gl/inc/DrawableManagerDefault.h>
#include <dp/util/File.h>

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

          TransparencyManagerOITClosestListSharedPtr TransparencyManagerOITClosestList::create( dp::math::Vec2ui const & size, unsigned int layersCount, float fragmentsCountFactor )
          {
            return( std::shared_ptr<TransparencyManagerOITClosestList>( new TransparencyManagerOITClosestList( size, layersCount, fragmentsCountFactor ) ) );
          }

          TransparencyManagerOITClosestList::TransparencyManagerOITClosestList( dp::math::Vec2ui const & size, unsigned int layersCount, float fragmentsCountFactor )
            : TransparencyManager( TM_ORDER_INDEPENDENT_CLOSEST_LIST )
            , m_fragmentsCount(0)
            , m_fragmentsCountFactor(fragmentsCountFactor)
            , m_initializedBuffers( false )
            , m_initializedHandles( false )
            , m_samplesPassedQuery(-1)
          {
            setLayersCount( layersCount );
            setViewportSize( size );
          }

          TransparencyManagerOITClosestList::~TransparencyManagerOITClosestList()
          {
            if ( m_initializedHandles )
            {
              glDeleteQueries( 1, &m_samplesPassedQuery );

              m_initializedHandles = false;
            }
          }

          void TransparencyManagerOITClosestList::addFragmentCodeSnippets( bool transparent, bool depth, std::vector<std::string> & snippets )
          {
            std::string file = depth ? "emitColorDepth.glsl" : ( transparent ? "emitColorOITClosestList.glsl" : "emitColor.glsl" );
            snippets.push_back( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/" + file ) );
          }

          void TransparencyManagerOITClosestList::viewportSizeChanged()
          {
            m_initializedBuffers = false;
          }

          void TransparencyManagerOITClosestList::beginTransparentPass( dp::rix::core::Renderer * renderer )
          {
            DP_ASSERT( renderer );

            TransparencyManager::beginTransparentPass( renderer );

            if ( !m_initializedHandles )
            {
              // create the VBO for the full screen quad
              GLfloat fullScreenQuadVertices[8] = { -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
              m_fullScreenQuad = dp::gl::Buffer::create( GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), fullScreenQuadVertices, GL_STATIC_DRAW );

              glGenQueries( 1, &m_samplesPassedQuery );

              dp::gl::VertexShaderSharedPtr vertexShader = dp::gl::VertexShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/passThroughPosition_vs.glsl" ) );
              dp::gl::FragmentShaderSharedPtr fragmentShader = dp::gl::FragmentShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitClosestListClear_fs.glsl" ) );
              m_clearProgram = dp::gl::Program::create( vertexShader, fragmentShader );

              // create fragment shader source
              std::string resolveFragmentCode = dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitClosestListResolve_fs.glsl" );
              std::ostringstream oss;
              oss << getLayersCount();
              std::unique_ptr<char[]> resolveFragmentSource( new char[resolveFragmentCode.length() + oss.str().length()] );
              sprintf( resolveFragmentSource.get(), resolveFragmentCode.c_str(), oss.str().c_str() );

              // vertexShader is the same as for m_clearProgram !
              fragmentShader = dp::gl::FragmentShader::create( resolveFragmentSource.get() );
              m_resolveProgram = dp::gl::Program::create( vertexShader, fragmentShader );

              m_initializedHandles = true;
            }

            if ( !m_initializedBuffers )
            {
              dp::math::Vec2ui viewportSize = getViewportSize();
              m_perFragmentOffsetsTextureGL = dp::gl::Texture2D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, viewportSize[0], viewportSize[1] );

              m_counterTextureGL = dp::gl::Texture1D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1 );

              unsigned int fragmentsCount = (unsigned int)( m_fragmentsCountFactor * getViewportSize()[0] * getViewportSize()[1] );
              m_fragmentsTextureGL = dp::gl::TextureBuffer::create( GL_RGBA32UI, fragmentsCount * 4 + sizeof(GLuint), nullptr, GL_DYNAMIC_COPY );

              renderer->textureSetData( m_counterTexture, dp::rix::gl::TextureDataGLTexture( m_counterTextureGL ) );
              renderer->textureSetData( m_fragmentsTexture, dp::rix::gl::TextureDataGLTexture( m_fragmentsTextureGL ) );
              renderer->textureSetData( m_offsetsTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentOffsetsTextureGL ) );

              m_clearProgram->setImageTexture( "counterAccu", m_counterTextureGL, GL_WRITE_ONLY );
              m_clearProgram->setImageTexture( "perFragmentOffset", m_perFragmentOffsetsTextureGL, GL_WRITE_ONLY );

              m_resolveProgram->setImageTexture( "samplesBuffer", m_fragmentsTextureGL, GL_READ_ONLY );
              m_resolveProgram->setImageTexture( "perFragmentOffset", m_perFragmentOffsetsTextureGL, GL_READ_ONLY );

              m_initializedBuffers = true;
            }

            {
              // clear the oit buffers
              dp::gl::ProgramUseGuard pug( m_clearProgram );

              glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
              glDepthMask( GL_FALSE );
              drawQuad();
              glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
              glDepthMask( GL_TRUE );
              glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
            }

            // settings for oit path
            glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
            glDepthMask( GL_FALSE );                              // don't use depth

            glBeginQuery( GL_SAMPLES_PASSED, m_samplesPassedQuery );
          }

          bool TransparencyManagerOITClosestList::endTransparentPass()
          {
            glEndQuery( GL_SAMPLES_PASSED );

            // post-transparent pass handling
            // reset oit path settings
            glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
            glDepthMask( GL_TRUE );

            {
              // resolve the oit buffers
              dp::gl::ProgramUseGuard pug( m_resolveProgram );

              glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
              glEnable( GL_BLEND );
              glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
              glDisable( GL_DEPTH_TEST );
              drawQuad();
              glDisable( GL_BLEND );
              glEnable( GL_DEPTH_TEST );
            }

            // check if the transparent render step needed more fragments than were available in the fragmentsBuffer
            // -> do it after resolving to prevent stalling the render pipeline just to get the result
            unsigned int counter = 0;
            glGetQueryObjectuiv( m_samplesPassedQuery, GL_QUERY_RESULT, &counter );

            unsigned int fragmentsBufferSize = counter * 4 * sizeof(GLuint);
            if ( m_fragmentsTextureGL->getBuffer()->getSize() < fragmentsBufferSize )
            {
              m_fragmentsCountFactor = (float)counter / ( getViewportSize()[0] * getViewportSize()[1] );
              m_fragmentsTextureGL->getBuffer()->setData( GL_TEXTURE_BUFFER, fragmentsBufferSize, nullptr, GL_DYNAMIC_COPY );
              return( false );
            }

            return( TransparencyManager::endTransparentPass() );
          }

          void TransparencyManagerOITClosestList::initializeParameterContainer( dp::rix::core::Renderer * renderer, dp::math::Vec2ui const & viewportSize )
          {
            m_counterTexture   = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TT_2D, dp::rix::core::ITF_R32UI, dp::PF_R, dp::DT_UNSIGNED_INT_32 ) );
            m_fragmentsTexture = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TT_BUFFER, dp::rix::core::ITF_RGBA32UI, dp::PF_RGBA, dp::DT_UNSIGNED_INT_32 ) );
            m_offsetsTexture   = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TT_2D, dp::rix::core::ITF_R32UI, dp::PF_R, dp::DT_UNSIGNED_INT_32 ) );

            std::vector<dp::rix::core::ProgramParameter> parameters;
            parameters.push_back( dp::rix::core::ProgramParameter( "counterAccu", dp::rix::core::CPT_IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "samplesBuffer", dp::rix::core::CPT_IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentOffset", dp::rix::core::CPT_IMAGE, 0 ) );

            m_parameterContainerDescriptor = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( parameters.data(), parameters.size() ) );
            m_parameterContainer = renderer->containerCreate( m_parameterContainerDescriptor );

            renderer->containerSetData( m_parameterContainer, renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, "counterAccu" )
                                      , dp::rix::core::ContainerDataImage( m_counterTexture.get(), 0, false, 0, dp::rix::core::AT_READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer, renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, "samplesBuffer" )
                                      , dp::rix::core::ContainerDataImage( m_fragmentsTexture.get(), 0, false, 0, dp::rix::core::AT_READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer, renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, "perFragmentOffset" )
                                      , dp::rix::core::ContainerDataImage( m_offsetsTexture.get(), 0, false, 0, dp::rix::core::AT_READ_WRITE ) );
          }

          void TransparencyManagerOITClosestList::useParameterContainer( dp::rix::core::Renderer * renderer, dp::rix::core::RenderGroupSharedHandle const & transparentRenderGroup )
          {
            DP_ASSERT( renderer && !!transparentRenderGroup );
            renderer->renderGroupUseContainer( transparentRenderGroup, m_parameterContainer );
          }

          void TransparencyManagerOITClosestList::drawQuad()
          {
            glEnableVertexAttribArray( 0 );
            glVertexAttribFormat( 0, 2, GL_FLOAT, GL_FALSE, 0 );
            glVertexAttribBinding( 0, 0 );
            glBindVertexBuffer( 0, m_fullScreenQuad->getGLId(), 0, 8 );
            glDrawArrays( GL_QUADS, 0, 4 );
          }

          bool TransparencyManagerOITClosestList::needsSortedRendering() const
          {
            return( false );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
