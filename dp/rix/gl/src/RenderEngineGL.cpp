// Copyright NVIDIA Corporation 2011-2015
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


#include <RenderEngineGL.h>

#include <GL/glew.h>
#include <GeometryGL.h>
#include <GeometryInstanceGL.h>
#include <GeometryDescriptionGL.h>
#include <IndicesGL.h>
#include <ProgramGL.h>
#include <VertexAttributesGL.h>
#include <boost/algorithm/string.hpp>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      RenderEngineMap &getRenderEngineMap()
      {
        static RenderEngineMap renderEngineMap;
        return renderEngineMap;
      }

      bool registerRenderEngine( const char *renderEngine, RenderEngineCreator creator )
      {
        RenderEngineMap& renderEngineMap = getRenderEngineMap();
        RenderEngineMap::iterator it = renderEngineMap.find( renderEngine );
        if ( it == renderEngineMap.end() )
        {
          renderEngineMap[renderEngine] = creator;
          return true;
        }
        DP_ASSERT( !"renderEngine already registered!" );
        return false;
      }

      RenderEngineGL* getRenderEngine( const char *renderEngineOptions )
      {
        std::map<std::string, std::string> options;
        std::vector<std::string> tokens;
        std::string renderEngineOptionsString(renderEngineOptions);
        boost::split(tokens, renderEngineOptionsString, boost::is_any_of(";"));
        for (auto it = tokens.begin(); it != tokens.end(); ++it)
        {
          std::vector<std::string> values;
          if (it->empty())
          {
            throw std::runtime_error("Empty tokens are not allowed in the RiX config string.");
          }
          boost::split(values, *it, boost::is_any_of("="));
          if (values.empty() || values.size() > 2)
          {
            throw std::runtime_error(std::string("invalid token") + *it + "\n");
          }
          options[values[0]] = (values.size() == 2) ? values[1] : "";
        }

        RenderEngineMap& renderEngineMap = getRenderEngineMap();

        auto itRenderEngine = options.find("vertex");
        RenderEngineMap::iterator it = renderEngineMap.find(itRenderEngine != options.end() ? itRenderEngine->second : "VAB");
        if ( it != renderEngineMap.end() )
        {
          return it->second(options);
        }
        DP_ASSERT( !"renderEngine not found!" );
        return nullptr;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
 
