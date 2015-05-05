// Copyright NVIDIA Corporation 2010-2015
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


#include <dp/sg/renderer/rix/gl/inc/FSQRendererImpl.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManagerRiXFx.h>
#include <dp/sg/gl/TextureGL.h>
#include <dp/gl/RenderTarget.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/util/SharedPtr.h>

using namespace dp::fx;
using namespace dp::math;
using namespace dp::rix::fx;
using namespace dp::rix::gl;
using namespace dp::util;
using namespace dp::sg::core;

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

          BEGIN_REFLECTION_INFO( RendererFSQImpl )
            DERIVE_STATIC_PROPERTIES( RendererFSQImpl, FSQRenderer );
          END_REFLECTION_INFO

            // this turns a zero-based dp::sg::core::TextureTarget into its corresponding GL target
            static const GLenum cTexTarget[] = { GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D,
            GL_TEXTURE_CUBE_MAP, GL_TEXTURE_1D_ARRAY_EXT,
            GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_RECTANGLE_ARB }; 

          void RendererFSQImpl::setTexCoord1( const dp::gl::RenderTargetSharedPtr & target )
          {
            int x, y;
            unsigned int w, h;
            target->getPosition( x, y);
            target->getSize( w, h );

            if ( ( x != m_targetX ) || ( y != m_targetY ) || ( w != m_targetW ) || ( h != m_targetH ) )
            {
              m_targetX = x;
              m_targetY = y;
              m_targetW = w;
              m_targetH = h;

              Vec2f texCoord1[4];
              texCoord1[0] = Vec2f( (float)m_targetX, (float)m_targetY );
              texCoord1[1] = Vec2f( (float)m_targetX + m_targetW, (float)m_targetY );
              texCoord1[2] = Vec2f( (float)m_targetX + m_targetW, (float)m_targetY + m_targetH );
              texCoord1[3] = Vec2f( (float)m_targetX, (float)m_targetY + m_targetH );

              m_vertexAttributeSet->setVertexData( VertexAttributeSet::DP_SG_TEXCOORD1, 2, dp::DT_FLOAT_32, &texCoord1[0], 0, 4 );
            }
          }

          RendererFSQImpl::RendererFSQImpl( const dp::gl::RenderTargetSharedPtr &target )
            : FSQRenderer( target )
            , m_effectsValid(false)
            , m_rendererGLLib(nullptr)
            , m_renderer(nullptr)
          {  
            m_vertexAttributeSet = VertexAttributeSet::create();
            {
              static Vec3f vertices[4] = { Vec3f( -1.0f, -1.0f, 0.0f ), Vec3f( 1.0f, -1.0f, 0.0f ), Vec3f( 1.0f, 1.0f, 0.0f ), Vec3f( -1.0f, 1.0f, 0.0f ) };
              static Vec2f texCoord0[4] = { Vec2f( 0.0f, 0.0f ), Vec2f( 1.0f, 0.0f ), Vec2f( 1.0f, 1.0f ), Vec2f( 0.0f, 1.0f ) };

              m_vertexAttributeSet->setVertexData( VertexAttributeSet::DP_SG_POSITION, 3, dp::DT_FLOAT_32, &vertices[0], 0, 4 );
              m_vertexAttributeSet->setVertexData( VertexAttributeSet::DP_SG_TEXCOORD0, 2, dp::DT_FLOAT_32, &texCoord0[0], 0, 4 );
            }
            setTexCoord1( target );

            m_primitive = Primitive::create( PRIMITIVE_QUADS );
            m_primitive->setVertexAttributeSet( m_vertexAttributeSet );

#if defined(DP_OS_WINDOWS)
             m_rendererGLLib = DynamicLibrary::createFromFile( "RiXGL.rdr" );
#else
            m_rendererGLLib = DynamicLibrary::createFromFile( "libRiXGL.rdr" );
#endif            
            DP_ASSERT( m_rendererGLLib );

            dp::rix::core::PFNCREATERENDERER createRenderer = (dp::rix::core::PFNCREATERENDERER)m_rendererGLLib->getSymbol( "createRenderer" );
            DP_ASSERT( createRenderer );

            m_renderer = (*createRenderer)( "vertex=VBO" );
            DP_ASSERT( m_renderer );

            m_resourceManager = ResourceManager::create( m_renderer, dp::fx::MANAGER_UNIFORM );
            DP_ASSERT( m_resourceManager );
          }

          RendererFSQImplSharedPtr RendererFSQImpl::create( const dp::gl::RenderTargetSharedPtr &renderTarget )
          {
            return( std::shared_ptr<RendererFSQImpl>( new RendererFSQImpl( renderTarget ) ) );;
          }

          RendererFSQImpl::~RendererFSQImpl(void)
          {
            m_instance.reset();
            m_resourcePrimitive.reset();
            m_resourceEffectData.reset();
            m_rixFxManager.reset();
            m_renderGroup.reset();
            m_geometryInstance.reset();

            DP_ASSERT( m_renderer );
            m_renderer->deleteThis();
          }

          //
          //  Draws a viewport filling quad on Z = 0.
          //
          //       Vertex Coords                 Texture Coords (units 0 and 1)
          //
          //   (-1,1) 3-------2 (1,1)          0:(0,1)   3-------2 0:(1,1)
          //          |       |                1:(x,y+h) |       | 1:(x+w,y+h)
          //          |       |                          |       |
          //  (-1,-1) 0-------1 (1,-1)         0:(0,0)   0-------1 0:(1,0)
          //                                   1:(x,y)             1:(x+w,y)
          //
          //                                    Where: X,Y,W,H are either Viewport Parameters or
          //                                           X,Y=0 and W,H are output RenderTarget dimensions
          //
          void RendererFSQImpl::doRender( const dp::ui::RenderTargetSharedPtr &renderTarget )
          {
            dp::gl::RenderTargetSharedPtr rtgl = dp::util::shared_cast<dp::gl::RenderTarget>( renderTarget );

            if( !rtgl )
            {
              return;
            }

            setTexCoord1( rtgl );

            rtgl->beginRendering();
            if ( !m_instance )
            {
              static_cast<RiXGL *>(m_renderer)->registerContext();

              m_resourcePrimitive = ResourcePrimitive::get( m_primitive, m_resourceManager );
              DP_ASSERT( m_resourcePrimitive );

              m_geometryInstance = m_renderer->geometryInstanceCreate();
              DP_ASSERT( m_geometryInstance );
              m_renderer->geometryInstanceSetGeometry( m_geometryInstance, m_resourcePrimitive->m_geometryHandle );

              m_rixFxManager = dp::rix::fx::Manager::create( MANAGER_UNIFORM, m_renderer );
              DP_ASSERT( m_rixFxManager );

              m_resourceEffectData = ResourceEffectDataRiXFx::get( m_effect, m_rixFxManager, m_resourceManager );
              DP_ASSERT( m_resourceEffectData );

              m_instance = m_rixFxManager->instanceCreate( m_geometryInstance.get() );
              DP_ASSERT( m_instance );
            }

            if ( !m_effectsValid )
            {
              dp::fx::EffectSpecSharedPtr es = dp::fx::EffectSpec::create( "sys_matrices", dp::fx::EffectSpec::EST_SYSTEM, EffectSpec::ParameterGroupSpecsContainer() );
              dp::fx::EffectSpecSharedPtr ec = dp::fx::EffectSpec::create( "sys_camera", dp::fx::EffectSpec::EST_SYSTEM, EffectSpec::ParameterGroupSpecsContainer() );
              dp::rix::fx::Manager::SystemSpecs s;
              s["sys_matrices"] = dp::rix::fx::Manager::EffectSpecInfo( es, false );
              s["sys_camera"] = dp::rix::fx::Manager::EffectSpecInfo( ec, true );
              s["sys_Fragment"] = dp::rix::fx::Manager::EffectSpecInfo( dp::fx::EffectSpec::create( "sys_Fragment", dp::fx::EffectSpec::EST_SYSTEM, EffectSpec::ParameterGroupSpecsContainer() ), true );

              dp::rix::fx::ProgramSharedHandle program = m_rixFxManager->programCreate( m_effect->getEffectSpec()
                                                                                     , s
                                                                                     //, dp::rix::fx::Manager::SystemSpecs()
                                                                                     , "forward", nullptr, 0 ); 
              m_rixFxManager->instanceSetProgram( m_instance.get(), program.get() );

              std::vector<dp::rix::fx::GroupDataSharedHandle> groupDatas;

              ResourceEffectDataRiXFx::GroupDatas gec = m_resourceEffectData->getGroupDatas();
              std::copy( gec.begin(), gec.end(), std::back_inserter(groupDatas) ) ;

              for (size_t i = 0; i < groupDatas.size(); i++)
              {
                m_rixFxManager->instanceUseGroupData( m_instance.get(), groupDatas[i].get() );
              }
              m_effectsValid = true;
            }

            if ( !m_renderGroup )
            {
              // renderGroupAddGeometryInstance needs an instance with specified program pipeline
              // that program pipeline is set above with m_rixFxManager->instanceSetProgram()
              m_renderGroup = m_renderer->renderGroupCreate();
              DP_ASSERT( m_renderGroup );
              m_renderer->renderGroupAddGeometryInstance( m_renderGroup, m_geometryInstance );
            }

            m_resourceManager->updateResources();
            glDisable( GL_DEPTH_TEST );
            m_renderer->render( m_renderGroup );
            glEnable( GL_DEPTH_TEST );

            rtgl->endRendering();
          }

          //
          // Sets texcoord attributes on the given texture unit to be submitted when the FSQ is rendered.
          // unit may range from 2 to 7 currently, since texture coordinates 0 and 1 are currenly filled with
          // some default values.
          //
          // Quad is drawn as shown in render()
          //
          // coords is expected to be a vector of Vec4f with at least four or zero elements, 
          // latter indicating to not send attributes from this texcoord unit.
          //
          void RendererFSQImpl::setTexCoords( unsigned int unit, const std::vector<Vec4f> &coords )
          {
            DP_ASSERT( 2 <= unit && unit <= 7 );

            if ( coords.empty() )
            {
              m_vertexAttributeSet->setEnabled( VertexAttributeSet::DP_SG_TEXCOORD0 + unit, false );
            }
            else
            {
              m_vertexAttributeSet->setVertexData( VertexAttributeSet::DP_SG_TEXCOORD0 + unit, 4, dp::DT_FLOAT_32, &coords[0], 0, 4 );
            }
          }

          bool RendererFSQImpl::setSamplerByName( const std::string & samplerName, const SamplerSharedPtr & sampler )
          {
            DP_ASSERT( m_effect );
            const EffectSpecSharedPtr & es = m_effect->getEffectSpec();
            for ( EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
            {
              ParameterGroupSpec::iterator pit = (*it)->findParameterSpec( samplerName );
              if ( pit != (*it)->endParameterSpecs() )
              {
                if ( ! m_effect->getParameterGroupData( it ) )
                {
                  m_effect->setParameterGroupData( it, dp::sg::core::ParameterGroupData::create( *it ) );
                }
                DP_ASSERT( m_effect->getParameterGroupData( it ) );
                m_effect->getParameterGroupData( it )->setParameter<SamplerSharedPtr>( samplerName, sampler );
                return( true );
              }
            }
            return( false );
          }

          void RendererFSQImpl::setEffect( const dp::sg::core::EffectDataSharedPtr & effect )
          {
            if ( m_effect != effect )
            {
              m_effect = effect;
              m_effectsValid = false;
            }
          }

          const dp::sg::core::EffectDataSharedPtr & RendererFSQImpl::getEffect() const
          {
            return( m_effect );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

