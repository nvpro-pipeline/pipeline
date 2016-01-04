// Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/gl/Program.h>
#include <dp/gl/ProgramInstance.h>
#include <dp/gl/Shader.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITClosestArray.h>
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

          TransparencyManagerOITClosestArraySharedPtr TransparencyManagerOITClosestArray::create( dp::math::Vec2ui const & size, unsigned int depth )
          {
            return( std::shared_ptr<TransparencyManagerOITClosestArray>( new TransparencyManagerOITClosestArray( size, depth ) ) );
          }

          TransparencyManagerOITClosestArray::TransparencyManagerOITClosestArray( dp::math::Vec2ui const & size, unsigned int depth )
            : TransparencyManager( TransparencyMode::ORDER_INDEPENDENT_CLOSEST_ARRAY )
            , m_initializedBuffers( false )
            , m_initializedHandles( false )
          {
            setLayersCount( depth );
            setViewportSize( size );
          }

          TransparencyManagerOITClosestArray::~TransparencyManagerOITClosestArray()
          {
          }

          void TransparencyManagerOITClosestArray::addFragmentCodeSnippets( bool transparent, bool depth, std::vector<std::string> & snippets )
          {
            std::string file = depth ? "emitColorDepth.glsl" : ( transparent ? "emitColorOITClosestArray.glsl" : "emitColor.glsl" );
            snippets.push_back( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/" + file ) );
          }

          void TransparencyManagerOITClosestArray::addFragmentParameters( std::vector<dp::rix::core::ProgramParameter> & parameters )
          {
            TransparencyManager::addFragmentParameters( parameters );
            parameters.push_back( dp::rix::core::ProgramParameter( "sys_OITDepth", dp::rix::core::ContainerParameterType::UINT_32 ) );
          }

          void TransparencyManagerOITClosestArray::addFragmentParameterSpecs( std::vector<dp::fx::ParameterSpec> & specs )
          {
            TransparencyManager::addFragmentParameterSpecs( specs );
            specs.push_back( dp::fx::ParameterSpec( "sys_OITDepth", dp::fx::PT_UINT32, dp::util::Semantic::VALUE ) );
          }

          void TransparencyManagerOITClosestArray::updateFragmentParameters()
          {
            TransparencyManager::updateFragmentParameters();

            DP_ASSERT( getShaderManager() );
            unsigned int lc = getLayersCount();
            getShaderManager()->updateFragmentParameter( std::string( "sys_OITDepth" ), dp::rix::core::ContainerDataRaw( 0, &lc, sizeof(unsigned int) ) );
          }

          void TransparencyManagerOITClosestArray::layersCountChanged()
          {
            if ( getShaderManager() )
            {
              unsigned int lc = getLayersCount();
              getShaderManager()->updateFragmentParameter( std::string( "sys_OITDepth" ), dp::rix::core::ContainerDataRaw( 0, &lc, sizeof(unsigned int) ) );
            }
          }

          void TransparencyManagerOITClosestArray::viewportSizeChanged()
          {
            m_initializedBuffers = false;
          }

          void TransparencyManagerOITClosestArray::beginTransparentPass( dp::rix::core::Renderer * renderer )
          {
            DP_ASSERT( renderer );

            TransparencyManager::beginTransparentPass( renderer );

            if ( !m_initializedHandles )
            {
              // create the VBO for the full screen quad
              GLfloat fullScreenQuadVertices[8] = { -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
              m_fullScreenQuad = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STATIC_DRAW, GL_ARRAY_BUFFER);
              m_fullScreenQuad->setSize(8 * sizeof(GLfloat));
              m_fullScreenQuad->update(fullScreenQuadVertices);

              dp::gl::VertexShaderSharedPtr vertexShader = dp::gl::VertexShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/passThroughPosition_vs.glsl" ) );
              dp::gl::FragmentShaderSharedPtr fragmentShader = dp::gl::FragmentShader::create( dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitClosestArrayClear_fs.glsl" ) );
              m_clearProgram = dp::gl::ProgramInstance::create( dp::gl::Program::create( vertexShader, fragmentShader ) );

              // create fragment shader source
              std::string resolveFragmentCode = dp::util::loadStringFromFile( dp::home() + "/media/dpfx/oitClosestArrayResolve_fs.glsl" );
              std::ostringstream oss;
              oss << getLayersCount();
              std::unique_ptr<char[]> resolveFragmentSource( new char[resolveFragmentCode.length() + oss.str().length()] );
              sprintf( resolveFragmentSource.get(), resolveFragmentCode.c_str(), oss.str().c_str() );

              // vertexShader is the same as for m_clearProgram !
              fragmentShader = dp::gl::FragmentShader::create( resolveFragmentSource.get() );
              m_resolveProgram = dp::gl::ProgramInstance::create( dp::gl::Program::create( vertexShader, fragmentShader ) );

              m_initializedHandles = true;
            }
            if ( !m_initializedBuffers )
            {
              dp::math::Vec2ui viewportSize = getViewportSize();
              m_perFragmentCountTextureGL       = dp::gl::Texture2D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, viewportSize[0], viewportSize[1] );
              m_perFragmentIndexTextureGL       = dp::gl::Texture2D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, viewportSize[0], viewportSize[1] );
              m_perFragmentSpinLockTextureGL    = dp::gl::Texture2D::create( GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, viewportSize[0], viewportSize[1] );
              m_perFragmentSamplesAccuTextureGL = dp::gl::Texture2D::create( GL_RGBA32F, GL_RGBA, GL_FLOAT, viewportSize[0], viewportSize[1] );
              m_perFragmentSamplesAccuTextureGL->setFilterParameters( GL_NEAREST, GL_NEAREST );

              m_samplesTextureGL = dp::gl::TextureBuffer::create( GL_RG32UI, getLayersCount() * viewportSize[0] * viewportSize[1] * 2 * sizeof(GLuint), nullptr, GL_DYNAMIC_COPY );

              m_clearProgram->setImageUniform( "perFragmentCount", m_perFragmentCountTextureGL, GL_WRITE_ONLY );
              m_clearProgram->setImageUniform( "perFragmentIndex", m_perFragmentIndexTextureGL, GL_WRITE_ONLY );
              m_clearProgram->setImageUniform( "perFragmentSpinLock", m_perFragmentSpinLockTextureGL, GL_WRITE_ONLY );
              m_clearProgram->setImageUniform( "perFragmentSamplesAccu", m_perFragmentSamplesAccuTextureGL, GL_WRITE_ONLY );

              m_resolveProgram->setImageUniform( "perFragmentCount", m_perFragmentCountTextureGL, GL_READ_ONLY );
              m_resolveProgram->setImageUniform( "perFragmentSamplesAccu", m_perFragmentSamplesAccuTextureGL, GL_READ_ONLY );
              m_resolveProgram->setImageUniform( "samplesBuffer", m_samplesTextureGL, GL_READ_ONLY );

              m_initializedBuffers = true;
            }

            // clear the oit buffers
            glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
            glDepthMask( GL_FALSE );
            m_clearProgram->apply();
            drawQuad();
            glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
            glDepthMask( GL_TRUE );
            glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

            // settings for oit path
            glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
            glDepthMask( GL_FALSE );                              // don't use depth

            renderer->textureSetData( m_perFragmentCountTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentCountTextureGL ) );
            renderer->textureSetData( m_perFragmentIndexTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentIndexTextureGL ) );
            renderer->textureSetData( m_perFragmentSpinLockTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentSpinLockTextureGL ) );
            renderer->textureSetData( m_samplesTexture, dp::rix::gl::TextureDataGLTexture( m_samplesTextureGL ) );
            renderer->textureSetData( m_perFragmentSamplesAccuTexture, dp::rix::gl::TextureDataGLTexture( m_perFragmentSamplesAccuTextureGL ) );
          }

          bool TransparencyManagerOITClosestArray::endTransparentPass()
          {
            // post-transparent pass handling
            // reset oit path settings
            glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
            glDepthMask( GL_TRUE );

            // resolve the oit buffers
            glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
            glEnable( GL_BLEND );
            glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
            glDisable( GL_DEPTH_TEST );
            m_resolveProgram->apply();
            drawQuad();
            glDisable( GL_BLEND );
            glEnable( GL_DEPTH_TEST );

            return( TransparencyManager::endTransparentPass() );
          }

          void TransparencyManagerOITClosestArray::initializeParameterContainer( dp::rix::core::Renderer * renderer, dp::math::Vec2ui const & viewportSize )
          {
            dp::rix::core::TextureDescription tti(dp::rix::core::TextureType::_2D, dp::rix::core::InternalTextureFormat::R32UI, dp::PixelFormat::R, dp::DataType::UNSIGNED_INT_32);
            m_perFragmentCountTexture       = renderer->textureCreate( tti );
            m_perFragmentIndexTexture       = renderer->textureCreate( tti );
            m_perFragmentSpinLockTexture    = renderer->textureCreate( tti );
            m_samplesTexture                = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TextureType::BUFFER, dp::rix::core::InternalTextureFormat::RG32UI, dp::PixelFormat::RG, dp::DataType::UNSIGNED_INT_32 ) );
            m_perFragmentSamplesAccuTexture = renderer->textureCreate( dp::rix::core::TextureDescription( dp::rix::core::TextureType::_2D, dp::rix::core::InternalTextureFormat::RGBA32F, dp::PixelFormat::RGBA, dp::DataType::FLOAT_32 ) );

            std::vector<dp::rix::core::ProgramParameter> parameters;
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentCount", dp::rix::core::ContainerParameterType::IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentIndex", dp::rix::core::ContainerParameterType::IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentSpinLock", dp::rix::core::ContainerParameterType::IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "samplesBuffer", dp::rix::core::ContainerParameterType::IMAGE, 0 ) );
            parameters.push_back( dp::rix::core::ProgramParameter( "perFragmentSamplesAccu", dp::rix::core::ContainerParameterType::IMAGE, 0 ) );

            m_parameterContainerDescriptor = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( parameters.data(), parameters.size() ) );
            m_parameterContainer = renderer->containerCreate( m_parameterContainerDescriptor );

            renderer->containerSetData( m_parameterContainer
                                      , renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, parameters[0].m_name )
                                      , dp::rix::core::ContainerDataImage( m_perFragmentCountTexture.get(), 0, false, 0, dp::rix::core::AccessType::READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer
                                      , renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, parameters[1].m_name )
                                      , dp::rix::core::ContainerDataImage( m_perFragmentIndexTexture.get(), 0, false, 0, dp::rix::core::AccessType::READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer
                                      , renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, parameters[2].m_name )
                                      , dp::rix::core::ContainerDataImage( m_perFragmentSpinLockTexture.get(), 0, false, 0, dp::rix::core::AccessType::READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer
                                      , renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, parameters[3].m_name )
                                      , dp::rix::core::ContainerDataImage( m_samplesTexture.get(), 0, false, 0, dp::rix::core::AccessType::READ_WRITE ) );
            renderer->containerSetData( m_parameterContainer
                                      , renderer->containerDescriptorGetEntry( m_parameterContainerDescriptor, parameters[4].m_name )
                                      , dp::rix::core::ContainerDataImage( m_perFragmentSamplesAccuTexture.get(), 0, false, 0, dp::rix::core::AccessType::READ_WRITE ) );
          }

          void TransparencyManagerOITClosestArray::useParameterContainer( dp::rix::core::Renderer * renderer, dp::rix::core::RenderGroupSharedHandle const & transparentRenderGroup )
          {
            DP_ASSERT( renderer && !!transparentRenderGroup );
            renderer->renderGroupUseContainer( transparentRenderGroup, m_parameterContainer );
          }

          void TransparencyManagerOITClosestArray::drawQuad()
          {
            glEnableVertexAttribArray( 0 );
            glVertexAttribFormat( 0, 2, GL_FLOAT, GL_FALSE, 0 );
            glVertexAttribBinding( 0, 0 );
            glBindVertexBuffer( 0, m_fullScreenQuad->getGLId(), 0, 8 );
            glDrawArrays( GL_QUADS, 0, 4 );
          }

          bool TransparencyManagerOITClosestArray::needsSortedRendering() const
          {
            return( false );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
