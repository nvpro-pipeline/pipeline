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

#include <test/sgrdr/framework/SgRdrBackend.h>
#include <dp/rix/gl/RiXGL.h>

#include <dp/util/File.h>

#include <windows.h>
#include <GL/freeglut.h>

#include <dp/Assert.h>
#include <iostream>
#include <tchar.h>

#include <dp/sg/renderer/rix/gl/SceneRenderer.h>

#include <boost/program_options.hpp>

namespace options = boost::program_options;

namespace dp
{
  namespace sgrdr
  {
    namespace test
    {
      namespace framework
      {

        // RenderData

        RenderDataSgRdr::RenderDataSgRdr()
          : m_viewState(nullptr)
        {
        }

        RenderDataSgRdr::RenderDataSgRdr( dp::sg::ui::ViewStateSharedPtr const& viewState )
        {
          m_viewState = viewState;
        }

        RenderDataSgRdr::~RenderDataSgRdr()
        {
        }

        void RenderDataSgRdr::setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState )
        {
          m_viewState = viewState;
        }

        dp::sg::ui::ViewStateSharedPtr const& RenderDataSgRdr::getViewState() const
        {
          return m_viewState;
        }

        dp::fx::Manager getShaderManager( std::string const& name )
        {
          std::map<std::string, dp::fx::Manager> shaderManager;
          shaderManager["rixfx:uniform"] = dp::fx::Manager::UNIFORM;
          shaderManager["rixfx:shaderbufferload"] = dp::fx::Manager::SHADERBUFFER;
          shaderManager["rixfx:ubo140"] = dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX_FX;
          shaderManager["rixfx:ssbo140"] = dp::fx::Manager::SHADER_STORAGE_BUFFER_OBJECT;
          if ( shaderManager.find(name) != shaderManager.end() )
          {
            return shaderManager[name];
          }
          else
          {
            return dp::fx::Manager::UNIFORM;
          }
        }

        // Backend

        SgRdrBackend::SgRdrBackend( const std::string& rendererName
                                  , const std::vector<std::string>& options )
        {
          // Initialize freeglut
          char* dummyChar[] = {"nothing"};
          int dummyInt = 0;
          glutInit(&dummyInt, nullptr);

          // Parse options
          options::options_description od("Usage: DPTApp");
          od.add_options()
            ( "renderengine", options::value<std::string>(), "The renderengine to use" )
            ( "shadermanager", options::value<std::string>(), "The shader manager to use" )
            ( "cullingengine", options::value<std::string>(), "The culling engine to use" )
            ;

          options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>( options ).options( od ).allow_unregistered().run();

          options::variables_map optsMap;
          options::store( parsedOpts, optsMap );

          std::string renderEngine = optsMap["renderengine"].as<std::string>();
          std::string cullingEngine = optsMap["cullingengine"].as<std::string>();
          std::string shaderManager = optsMap["shadermanager"].as<std::string>();


          bool disableCulling = false;

          // Create the renderer as specified
          dp::culling::Mode cullingMode = dp::culling::Mode::AUTO;

          if ( cullingEngine == "cpu" )
          {
            cullingMode = dp::culling::Mode::CPU;
          }
          else if ( cullingEngine == "gl_compute" )
          {
            cullingMode = dp::culling::Mode::OPENGL_COMPUTE;
          }
          else if ( cullingEngine == "cuda" )
          {
            cullingMode = dp::culling::Mode::CUDA;
          }
          else if ( cullingEngine == "none" )
          {
            disableCulling = true;
          }
          else if ( cullingEngine != "auto" )
          {
            std::cerr << "unknown culling engine, turning culling off" << std::endl;
            disableCulling = true;
          }


          dp::fx::Manager smt = getShaderManager( shaderManager );

          //The chosen renderer takes precedence over the chosen shader manager.
          if( rendererName == "RiXGL.rdr" )
          {
            m_renderer = dp::sg::renderer::rix::gl::SceneRenderer::create
            (   renderEngine.c_str()
              , smt
              , cullingMode
            );
          }
          else
          {
            DP_ASSERT(!"Unknown Renderer");
          }

          m_renderer->setCullingEnabled( !disableCulling );
        }

        SgRdrBackend::~SgRdrBackend()
        {
          //The Sg Renderer must be destroyed prior to destroying the window
          m_renderer = dp::sg::renderer::rix::gl::SceneRendererSharedPtr();
          if ( m_windowId )
          {
            glutDestroyWindow( m_windowId );
            glutLeaveMainLoop();
            glutMainLoopEvent();
#if defined(DP_OS_WINDOWS)
            // As long as DPTRiXGL.bkd gets loaded and unloaded several times during test
            // execution this workaround is necessary to prevent freeglut from crashing.
            UnregisterClass( _T("FREEGLUT"), NULL);
#endif
          }
        }

        dp::sg::ui::SceneRendererSharedPtr SgRdrBackend::getRenderer() const
        {
          return m_renderer;
        }

        void SgRdrBackend::render( dp::testfw::core::RenderData* renderData, dp::ui::RenderTargetSharedPtr renderTarget )
        {
          DP_ASSERT(m_renderer);
          DP_ASSERT(!!renderTarget);

          m_renderer->render( static_cast<RenderDataSgRdr*>(renderData)->getViewState(), renderTarget );

          glutSwapBuffers();
        }

        void SgRdrBackend::finish()
        {
          glFinish();
        }

        dp::ui::RenderTargetSharedPtr SgRdrBackend::createDisplay( int width, int height, bool visible )
        {
          glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA | GLUT_BORDERLESS );
          glutInitWindowSize( width, height );
          glutInitWindowPosition(0, 0);

          m_windowId = glutCreateWindow( "DPT" );

          glewInit();

          dp::ui::RenderTargetSharedPtr displayTarget = dp::gl::RenderTargetFB::create( dp::gl::RenderContext::create( dp::gl::RenderContext::Attach() ) );
          displayTarget->setSize(width, height);
          std::static_pointer_cast<dp::gl::RenderTargetFB>(displayTarget)->setClearMask(dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER);

          return displayTarget;
        }

      } // namespace framework
    } // namespace test
  } // namespace sgrdr
} // namespace dp
