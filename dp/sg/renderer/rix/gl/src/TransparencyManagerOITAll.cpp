// Copyright NVIDIA Corporation 2014
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
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITAll.h>
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

          TransparencyManagerOITAllSharedPtr TransparencyManagerOITAll::create( dp::math::Vec2ui const & size )
          {
            return( std::shared_ptr<TransparencyManagerOITAll>( new TransparencyManagerOITAll( size ) ) );
          }

          TransparencyManagerOITAll::TransparencyManagerOITAll( dp::math::Vec2ui const & size )
            : TransparencyManager( TM_ORDER_INDEPENDENT_ALL )
            , m_initializedBuffers( false )
            , m_initializedHandles( false )
            , m_samplesPassedQuery(-1)
          {
            setViewportSize( size );
          }

          TransparencyManagerOITAll::~TransparencyManagerOITAll()
          {
            if ( m_initializedHandles )
            {
              glDeleteQueries( 1, &m_samplesPassedQuery );

              m_initializedHandles = false;
            }
          }

          void TransparencyManagerOITAll::addFragmentCodeSnippets( bool transparent, bool depth, std::vector<std::string> & snippets )
          {
            std::string file = depth ? ( transparent ? "emitColorOITAllCounter.glsl" : "emitColorDepth.glsl" )
                                     : ( transparent ? "emitColorOITAllSample.glsl" : "emitColor.glsl" );
            snippets.push_back( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/" + file ) );
          }

          void TransparencyManagerOITAll::viewportSizeChanged()
          {
            m_initializedBuffers = false;
          }

          void TransparencyManagerOITAll::beginTransparentPass( dp::rix::core::Renderer * renderer )
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
              dp::gl::FragmentShaderSharedPtr fragmentShader = dp::gl::FragmentShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitAllClear_fs.glsl" ) );
              m_clearProgram = dp::gl::Program::create( vertexShader, fragmentShader );

              // vertexShader is the same as for m_clearProgram !
              fragmentShader = dp::gl::FragmentShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitAllResolveCounters_fs.glsl" ) );
              m_resolveCountersProgram = dp::gl::Program::create( vertexShader, fragmentShader );

              // vertexShader is the same as for m_clearProgram !
              fragmentShader = dp::gl::FragmentShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitAllResolveSamples_fs.glsl" ) );
              m_resolveSamplesProgram = dp::gl::Program::create( vertexShader, fragmentShader );

              m_initializedHandles = true;
            }

            if ( !m_initializedBuffers )
            {
              m_counterAccuTexture = dp::gl::Texture1D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1 );

              dp::math::Vec2ui viewportSize = getViewportSize();
              m_perFragmentCountTextureGL   = dp::gl::Texture2D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, viewportSize[0], viewportSize[1] );
              m_perFragmentOffsetTextureGL  = dp::gl::Texture2D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, viewportSize[0], viewportSize[1] );

              m_samplesTextureGL = dp::gl::TextureBuffer::create( GL_RG32UI, 0, nullptr, GL_DYNAMIC_COPY );

              m_clearProgram->setImageTexture( "counterAccu", m_counterAccuTexture, GL_WRITE_ONLY );
              m_clearProgram->setImageTexture( "perFragmentCount", m_perFragmentCountTextureGL, GL_WRITE_ONLY );

              m_resolveCountersProgram->setImageTexture( "counterAccu", m_counterAccuTexture, GL_READ_WRITE );
              m_resolveCountersProgram->setImageTexture( "perFragmentCount", m_perFragmentCountTextureGL, GL_WRITE_ONLY );
              m_resolveCountersProgram->setImageTexture( "perFragmentOffset", m_perFragmentOffsetTextureGL, GL_WRITE_ONLY );

              m_resolveSamplesProgram->setImageTexture( "perFragmentCount", m_perFragmentCountTextureGL, GL_READ_ONLY );
              m_resolveSamplesProgram->setImageTexture( "perFragmentOffset", m_perFragmentOffsetTextureGL, GL_READ_ONLY );
              m_resolveSamplesProgram->setImageTexture( "samplesBuffer", m_samplesTextureGL, GL_READ_WRITE );

              renderer->textureSetData( m_perFragmentCountTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentCountTextureGL ) );
              renderer->textureSetData( m_perFragmentOffsetTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentOffsetTextureGL ) );
              renderer->textureSetData( m_samplesTexture, dp::rix::gl::TextureDataGLTexture( m_samplesTextureGL ) );

              m_initializedBuffers = true;
            }

            {
              // clear the oit buffers
              dp::gl::ProgramUseGuard pug( m_clearProgram );

              glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
              glDepthMask( GL_FALSE );
              drawQuad();
              glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
            }

            glBeginQuery( GL_SAMPLES_PASSED, m_samplesPassedQuery );
          }

          void TransparencyManagerOITAll::resolveDepthPass()
          {
            glEndQuery( GL_SAMPLES_PASSED );

            {
              // resolve the counters buffer
              dp::gl::ProgramUseGuard pug( m_resolveCountersProgram );

              glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
              glDisable( GL_DEPTH_TEST );
              drawQuad();
              glEnable( GL_DEPTH_TEST );
            }

            unsigned int samplesCount = -1;
#if 0
            m_counterAccuTexture->getData( &samplesCount );
#else
            glGetQueryObjectuiv( m_samplesPassedQuery, GL_QUERY_RESULT, &samplesCount );
#endif

            unsigned int sampleBufferSize = samplesCount * 2 * sizeof(GLuint);
            if ( m_samplesTextureGL->getBuffer()->getSize() < sampleBufferSize )
            {
              m_samplesTextureGL->getBuffer()->setData( GL_TEXTURE_BUFFER, sampleBufferSize, nullptr, GL_DYNAMIC_COPY );
            }
          }

          bool TransparencyManagerOITAll::endTransparentPass()
          {
            // post-transparent pass handling
            // reset oit path settings
            glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
            glDepthMask( GL_TRUE );

            {
              // resolve the oit buffers
              dp::gl::ProgramUseGuard pug( m_resolveSamplesProgram );

              glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
              glEnable( GL_BLEND );
              glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
              glDisable( GL_DEPTH_TEST );
              drawQuad();
              glDisable( GL_BLEND );
              glEnable( GL_DEPTH_TEST );
            }

            return( TransparencyManager::endTransparentPass() );
          }

          void TransparencyManagerOITAll::initializeParameterContainer( dp::rix::core::Renderer * renderer, dp::math::Vec2ui const & viewportSize )
          {
            m_perFragmentCountTexture   = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TT_2D, dp::rix::core::ITF_R32UI, dp::util::PF_R, dp::util::DT_UNSIGNED_INT_32 ) );
            m_perFragmentOffsetTexture  = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TT_2D, dp::rix::core::ITF_R32UI, dp::util::PF_R, dp::util::DT_UNSIGNED_INT_32 ) );
            m_samplesTexture  = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TT_BUFFER, dp::rix::core::ITF_RG32UI, dp::util::PF_RG, dp::util::DT_UNSIGNED_INT_32 ) );

            std::vector<dp::rix::core::ProgramParameter> parameters;
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentCount", dp::rix::core::CPT_IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentOffset", dp::rix::core::CPT_IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "samplesBuffer", dp::rix::core::CPT_IMAGE, 0 ) );

            m_parameterContainerDescriptor = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( parameters.data(), parameters.size() ) );
            m_parameterContainer = renderer->containerCreate( m_parameterContainerDescriptor );

            renderer->containerSetData( m_parameterContainer, renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, "perFragmentCount" )
                                      , dp::rix::core::ContainerDataImage( m_perFragmentCountTexture.get(), 0, false, 0, dp::rix::core::AT_READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer, renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, "perFragmentOffset" )
                                      , dp::rix::core::ContainerDataImage( m_perFragmentOffsetTexture.get(), 0, false, 0, dp::rix::core::AT_READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer, renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, "samplesBuffer" )
                                      , dp::rix::core::ContainerDataImage( m_samplesTexture.get(), 0, false, 0, dp::rix::core::AT_READ_WRITE ) );
          }

          void TransparencyManagerOITAll::useParameterContainer( dp::rix::core::Renderer * renderer, dp::rix::core::RenderGroupSharedHandle const & transparentRenderGroup )
          {
            DP_ASSERT( renderer && !!transparentRenderGroup );
            renderer->renderGroupUseContainer( transparentRenderGroup, m_parameterContainer );
          }

          void TransparencyManagerOITAll::drawQuad()
          {
            glEnableVertexAttribArray( 0 );
            glVertexAttribFormat( 0, 2, GL_FLOAT, GL_FALSE, 0 );
            glVertexAttribBinding( 0, 0 );
            glBindVertexBuffer( 0, m_fullScreenQuad->getGLId(), 0, 8 );
            glDrawArrays( GL_QUADS, 0, 4 );
          }

          bool TransparencyManagerOITAll::needsSortedRendering() const
          {
            return( false );
          }

          bool TransparencyManagerOITAll::supportsDepthPass() const
          {
            return( true );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
