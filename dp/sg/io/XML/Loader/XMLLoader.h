// Copyright NVIDIA Corporation 2002-2005
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


//
// XMLLoader.h
//

#pragma once

#if ! defined( DOXYGEN_IGNORE )

#include <string>
#include <vector>
#include <map>

#include "tinyxml.h"

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/PlugInterface.h>

#ifdef _WIN32
// microsoft specific storage-class defines
# ifdef XMLLOADER_EXPORTS
#  define XMLLOADER_API __declspec(dllexport)
# else
#  define XMLLOADER_API __declspec(dllimport)
# endif
#else
# define XMLLOADER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
XMLLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPlugIn & pi);
XMLLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

#define INVOKE_CALLBACK(cb) if( callback() ) callback()->cb

SMART_TYPES( XMLLoader );

class XMLLoader : public dp::sg::io::SceneLoader
{
public:
  static SmartXMLLoader create();
  virtual ~XMLLoader(void);

  dp::sg::core::SceneSharedPtr load( std::string const& filename
                                   , std::vector<std::string> const& searchPaths
                                   , dp::sg::ui::ViewStateSharedPtr & viewState );

protected:
  XMLLoader();

private:
  dp::sg::core::SceneSharedPtr lookupFile( const std::string & name );
  void buildScene( dp::sg::core::GroupSharedPtr const& topLevel, TiXmlDocument & doc, 
                    TiXmlNode * node );

  std::map< std::string, dp::sg::core::SceneSharedPtr > m_fileCache;
  std::vector< std::string > m_searchPath;
  dp::sg::ui::ViewStateWeakPtr m_viewState;
};

#endif // ! defined( DOXYGEN_IGNORE )
