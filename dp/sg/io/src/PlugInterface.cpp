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


#include <dp/sg/io/PlugInterface.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/util/File.h>

using namespace dp::sg::core;

using std::string;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace io
    {
      SceneLoader::SceneLoader()
      {
      }

      SceneLoader::~SceneLoader()
      {
      }

      SceneSaver::SceneSaver()
      {
      }

      SceneSaver::~SceneSaver()
      {
      }

      TextureLoader::TextureLoader()
      {
      }

      TextureLoader::~TextureLoader()
      {
      }

      dp::sg::core::TextureHostSharedPtr TextureLoader::load( const string & filename
                                              , const vector<string> & searchPaths
                                              , const unsigned int& creationFlags )
      {
        // serialize invocations from concurrent threads
        bool success = true;
        dp::sg::core::TextureHostSharedPtr texImgHdl( dp::sg::core::TextureHost::create( filename ) );
        {
          texImgHdl->setCreationFlags(creationFlags);

          success = onLoad( texImgHdl, searchPaths );
        }

        if ( !success )
        {
          texImgHdl.reset();
        }

        return( texImgHdl );
      }

      bool TextureLoader::reload( const dp::sg::core::TextureHostSharedPtr & tih, const vector<string> & searchPaths )
      {
        // serialize invocations from concurrent threads
        DP_ASSERT( dp::util::fileExists( tih->getFileName() ) );
        DP_ASSERT( tih->getNumberOfImages() == 0 );
        return( onLoad( tih, searchPaths ) );
      }

      TextureSaver::TextureSaver()
      {
      }

      TextureSaver::~TextureSaver()
      {
      }

      ShaderLoader::~ShaderLoader()
      {
      }

    } // namespace io
  } // namespace sg
} // namespace dp
