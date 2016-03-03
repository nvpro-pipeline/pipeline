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


#include <dp/sg/io/IO.h>
#include <dp/sg/io/PlugInterface.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/util/File.h>
#include <dp/util/FileFinder.h>

#if defined(DP_OS_WINDOWS)
#include <windows.h>
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#endif

using namespace dp::sg::core;

using namespace std;

namespace dp
{
  namespace sg
  {
    namespace io
    {

      dp::sg::ui::ViewStateSharedPtr loadScene( const std::string & filename, dp::util::FileFinder const& fileFinder, dp::util::PlugInCallbackSharedPtr const& callback )
      {
        dp::sg::ui::ViewStateSharedPtr viewState;

        // appropriate search paths for the
        // loader dll and the sample file.
        string dir;
        dp::util::FileFinder localFF( fileFinder );
        localFF.addSearchPath( dp::util::getCurrentPath() );
        localFF.addSearchPath( dp::util::getModulePath() );

        // receive the unique scene loader plug interface ID from the file extension
        std::string ext = dp::util::getFileExtension( filename );
        dp::util::UPIID piid = dp::util::UPIID( ext.c_str(), dp::util::UPITID( UPITID_SCENE_LOADER, UPITID_VERSION ) );

        {
          dp::util::PlugInSharedPtr plug;
          if ( !dp::util::getInterface(localFF, piid, plug) )
          {
            throw std::runtime_error( std::string( "Scene Plugin not found: " + ext ) );
          }
          SceneLoaderSharedPtr loader = std::static_pointer_cast<SceneLoader>(plug);
          loader->setCallback( callback );

          // add the scene path, if it's not the current path (-> dir.empty()) and not yet added
          localFF.addSearchPath( dp::util::getFilePath( filename ) );

          // for supplied models add additional resource paths to make sure
          // that required resources will be found by subsequent loaders
          size_t pos = dir.rfind("media");
          if ( pos != string::npos )
          {
            string sdk(dir.substr(0, pos));
            localFF.addSearchPath(sdk + "media/effects");
            localFF.addSearchPath(sdk + "media/textures");
          }

          dp::sg::core::SceneSharedPtr scene;
          try
          {
            scene = loader->load( filename, localFF, viewState );
          }
          catch (...)
          {
            // TODO another non RAII pattern, callback should be passed to load
            loader->setCallback( dp::util::PlugInCallbackSharedPtr() );
            throw;
          }
          if ( !scene )
          {
            throw std::runtime_error( std::string("Failed to load scene: " + filename ) );
          }

          // create a new viewstate if necessary
          if ( !viewState )
          {
            viewState = dp::sg::ui::ViewState::create();
          }
          if ( !viewState->getSceneTree() )
          {
            viewState->setSceneTree( dp::sg::xbar::SceneTree::create( scene ) );
          }
        }

        // FIXME interface needs to be released since the cleanup order (first dp::sg::core, then dp::util) causes problems upon destruction.
        dp::util::releaseInterface(piid);

        return viewState;
      }

      bool saveScene( std::string const& filename, dp::sg::ui::ViewStateSharedPtr const& viewState, dp::util::PlugInCallback *callback )
      {
        bool result = false;
        // define a unique plug-interface ID for SceneLoader
        const dp::util::UPITID PITID_SCENE_SAVER(UPITID_SCENE_SAVER, UPITID_VERSION);

        dp::util::FileFinder fileFinder( dp::util::getCurrentPath() );
        fileFinder.addSearchPath( dp::util::getModulePath() );
#if defined(DP_OS_WINDOWS)
        fileFinder.addSearchPath( dp::util::getModulePath( reinterpret_cast<HMODULE>(&__ImageBase) ) );
#endif

        dp::util::UPIID piid = dp::util::UPIID(dp::util::getFileExtension( filename ).c_str(), PITID_SCENE_SAVER);

        {
          dp::util::PlugInSharedPtr plug;
          if ( getInterface( fileFinder, piid, plug ) )
          {
            SceneSaverSharedPtr ss = std::static_pointer_cast<SceneSaver>(plug);
            try
            {
              dp::sg::core::SceneSharedPtr scene( viewState->getScene() ); // DAR HACK Change SceneSaver interface later.
              result = ss->save( scene, viewState, filename );
            }
            catch(...) // catch all others
            {
            }
          }
        }

        // FIXME interface needs to be released since the cleanup order (first dp::sg::core, then dp::util) causes problems upon destruction.
        dp::util::releaseInterface(piid);

        return result;
      }

      bool saveTextureHost( const std::string & filename, const dp::sg::core::TextureHostSharedPtr & tih )
      {
        // load the saver plug-in  - this should be configured
        dp::util::UPIID piid = dp::util::UPIID(dp::util::getFileExtension( filename ).c_str(), dp::util::UPITID(UPITID_TEXTURE_SAVER, UPITID_VERSION) );

        dp::util::FileFinder fileFinder( dp::util::getCurrentPath() );
        fileFinder.addSearchPath( dp::util::getModulePath() );

        bool retval = false;
        //
        // MMM - TODO - Update me for stereo images
        //
        {
          dp::util::PlugInSharedPtr plug;
          if ( getInterface( fileFinder, piid, plug ) )
          {
            TextureSaverSharedPtr ts = std::static_pointer_cast<TextureSaver>(plug);
            retval = ts->save( tih, filename );
          }
        }

        // FIXME interface needs to be released since the cleanup order (first dp::sg::core, then dp::util) causes problems upon destruction.
        dp::util::releaseInterface(piid);

        return retval;
      }

      dp::sg::core::TextureHostSharedPtr loadTextureHost( const std::string & filename, dp::util::FileFinder const& fileFinder )
      {
        dp::sg::core::TextureHostSharedPtr tih;

        dp::util::FileFinder localFF( fileFinder );
        localFF.addSearchPath( dp::util::getCurrentPath() );
        localFF.addSearchPath( dp::util::getModulePath() );

        std::string foundFile = localFF.find( filename );
        if (!foundFile.empty())
        {
          std::string ext = dp::util::getFileExtension( filename );

          dp::util::UPIID piid = dp::util::UPIID( ext.c_str(), dp::util::UPITID(UPITID_TEXTURE_LOADER, UPITID_VERSION) );

          // TODO - Update me for stereo images
          {
            dp::util::PlugInSharedPtr plug;
            if ( getInterface( fileFinder, piid, plug ) )
            {
              TextureLoaderSharedPtr tl = std::static_pointer_cast<TextureLoader>(plug);
              tih = tl->load( foundFile );
            }
          }

          // FIXME interface needs to be released since the cleanup order (first dp::sg::core, then dp::util) causes problems upon destruction.
          dp::util::releaseInterface(piid);
        }

        return tih;
      }

    } // namespace io
  } // namespace sg
} // namespace dp
