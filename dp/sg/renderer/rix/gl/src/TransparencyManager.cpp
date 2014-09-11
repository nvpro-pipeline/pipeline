// Copyright NVIDIA Corporation 2011-2012
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


#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>
#include <dp/sg/renderer/rix/gl/TransparencyManager.h>

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

          TransparencyManager::TransparencyManager( TransparencyMode transparencyMode )
            : m_layersCount( 0 )
            , m_shaderManager( nullptr )
            , m_transparencyMode( transparencyMode )
            , m_transparentPass( false )
            , m_viewportSize( 0, 0 )
          {
          }

          bool TransparencyManager::supportsDepthPass() const
          {
            return( false );
          }

          void TransparencyManager::resolveDepthPass()
          {
          }

          unsigned int TransparencyManager::getLayersCount() const
          {
            return( m_layersCount );
          }

          void TransparencyManager::setLayersCount( unsigned int count )
          {
            if ( m_layersCount != count )
            {
              m_layersCount = count;
              layersCountChanged();
            }
          }

          ShaderManager * TransparencyManager::getShaderManager() const
          {
            return( m_shaderManager );
          }

          void TransparencyManager::setShaderManager( ShaderManager * shaderManager )
          {
            m_shaderManager = shaderManager;
          }

          TransparencyMode TransparencyManager::getTransparencyMode() const
          {
            return( m_transparencyMode );
          }

          dp::math::Vec2ui const& TransparencyManager::getViewportSize() const
          {
            return( m_viewportSize );
          }

          void TransparencyManager::setViewportSize( dp::math::Vec2ui const & size )
          {
            if ( m_viewportSize != size )
            {
              m_viewportSize = size;
              viewportSizeChanged();
            }
          }

          void TransparencyManager::layersCountChanged()
          {
          }

          void TransparencyManager::viewportSizeChanged()
          {
          }

          void TransparencyManager::addFragmentParameters( std::vector<dp::rix::core::ProgramParameter> & parameters )
          {
            parameters.push_back( dp::rix::core::ProgramParameter( "sys_TransparentPass", dp::rix::core::CPT_BOOL ) );
          }

          void TransparencyManager::addFragmentParameterSpecs( std::vector<dp::fx::ParameterSpec> & specs )
          {
            specs.push_back( dp::fx::ParameterSpec( "sys_TransparentPass", dp::fx::PT_BOOL, dp::util::SEMANTIC_VALUE ) );
          }

          void TransparencyManager::updateFragmentParameters()
          {
            m_shaderManager->updateFragmentParameter( std::string( "sys_TransparentPass" ), dp::rix::core::ContainerDataRaw( 0, &m_transparentPass, sizeof(bool) ) );
          }

          void TransparencyManager::beginTransparentPass( dp::rix::core::Renderer * renderer )
          {
            DP_ASSERT( renderer && m_shaderManager );
            DP_ASSERT( !m_transparentPass );

            m_transparentPass = true;
            m_shaderManager->updateFragmentParameter( std::string( "sys_TransparentPass" ), dp::rix::core::ContainerDataRaw( 0, &m_transparentPass, sizeof(bool) ) );
          }

          bool TransparencyManager::endTransparentPass()
          {
            DP_ASSERT( m_transparentPass );
            m_transparentPass = false;
            m_shaderManager->updateFragmentParameter( std::string( "sys_TransparentPass" ), dp::rix::core::ContainerDataRaw( 0, &m_transparentPass, sizeof(bool) ) );
            return( true );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
