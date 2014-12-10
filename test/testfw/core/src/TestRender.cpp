// Copyright NVIDIA Corporation 2011
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


#include <test/testfw/core/TestRender.h>

#include <dp/util/File.h>

#include <iostream>

#include <boost/program_options.hpp>

namespace options = boost::program_options;

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      std::string TestRender::m_backendName;

      TestRender::TestRender()
        : m_width(0)
        , m_height(0)
        , m_rendererSpecified(false)
        , m_backendLib(nullptr)
      {
      }

      TestRender::~TestRender()
      {
      }

      util::ImageSharedPtr TestRender::getScreenshot() const
      {
        util::ImageSharedPtr image = m_displayTarget->getImage();
        return image;
      }

      void TestRender::render( RenderData* renderData, dp::ui::RenderTargetSharedPtr renderTarget )
      {
        m_backend->render(renderData, renderTarget);
      }

      bool TestRender::run( MeasurementFunctor & mf, const std::string& name )
      {
        if(!m_backend)
        {
          if(!m_rendererSpecified)
          {
            std::cerr << "Error: A renderer wasn't specified. Aborting program.\n";
            std::cerr << "Use --renderer flag to specify a renderer.\n";
          }
          return false;
        }

        DP_ASSERT(m_width && m_height);
        m_displayTarget = m_backend->createDisplay(m_width, m_height, true);

        return Test::run(mf, name);
      }

      bool TestRender::option( const std::vector<std::string>& optionString )
      {
        options::options_description od("Usage: DPTApp");
        od.add_options()
          ( "width", options::value<unsigned int>()->default_value(1024), "Width of the render window" )
          ( "height", options::value<unsigned int>()->default_value(768), "Height of the render window" )
          ( "renderer", options::value<std::string>(), "The renderer that will be used" )
          ( "backend", options::value<std::string>(), "The backend to use" )
          ;

        options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>( optionString ).options( od ).allow_unregistered().run();

        options::variables_map optsMap;
        options::store( parsedOpts, optsMap );

        // Width
        m_width = optsMap["width"].as<unsigned int>();

        // Height
        m_height = optsMap["height"].as<unsigned int>();

        // All additional options

        std::vector<std::string> possibleRendererOptions = options::collect_unrecognized( parsedOpts.options, options::include_positional );

        // Backend

        if( !optsMap["backend"].empty() )
        {
          m_backendName = optsMap["backend"].as<std::string>();
        }

        // Renderer

        if( optsMap["renderer"].empty() )
        {
          std::cerr << "Error: A renderer must be specified\n";
          return false;
        }
        else
        {
          m_backend = createBackend( optsMap["renderer"].as<std::string>(), possibleRendererOptions );
          m_rendererSpecified = true;
        }

        return !!m_backend;
      }

      BackendSharedPtr TestRender::createBackend( const std::string& rendererName, const std::vector<std::string>& options )
      {
        typedef core::Backend * (*ContextCreator)(const char*, const std::vector<std::string>*);

        if ( m_backendName.empty() )
        {
          std::vector<std::string> backendNames;
          dp::util::findFiles( ".bkd", dp::util::getModulePath(), backendNames );

          typedef bool (*IsRendererSupported)(const char*);

          for( std::vector<std::string>::iterator it = backendNames.begin(); it != backendNames.end(); ++it )
          {
            dp::util::DynamicLibrarySharedPtr backendLib = dp::util::DynamicLibrary::createFromFile( *it );
            IsRendererSupported isRendererSupported = (IsRendererSupported)backendLib->getSymbol( "isRendererSupported" );
            DP_ASSERT( isRendererSupported );
            if ( isRendererSupported( rendererName.c_str() ) )
            {
              m_backendLib = backendLib;
              m_backendName = *it;
              break;
            }
          }

          if(!m_backendLib)
          {
            std::cerr << "No supported back end was found for renderer library """ + rendererName + """\n";
            return nullptr;
          }
        }
        else
        {
          m_backendLib = util::DynamicLibrary::createFromFile( m_backendName );
          if( !m_backendLib )
          {
            std::cerr << "Error: Backend named '" << m_backendName << "' doesn't exist.\n";
            return nullptr;
          }
        }

        ContextCreator creator = ( ContextCreator ) m_backendLib->getSymbol("create");
        if(!creator)
        {
          std::cerr << "Error: Invalid backend '" << m_backendName << "'\n";
          return nullptr;
        }
        return creator( rendererName.c_str(), &options )->shared_from_this();
      }

      BackendSharedPtr TestRender::getBackend() const
      {
        return( m_backend );
      }

    } // namespace core
  } // namespace testfw
} // namespace dp
