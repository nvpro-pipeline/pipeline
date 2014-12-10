// Copyright NVIDIA Corporation 2012
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

      dp::sg::ui::ViewStateSharedPtr loadScene( const std::string & filename, const std::vector<std::string> &searchPaths, dp::util::PlugInCallbackSharedPtr const& callback )
      {
        dp::sg::ui::ViewStateSharedPtr viewState;

        // appropriate search paths for the
        // loader dll and the sample file.
        string dir;
        vector<string> binSearchPaths = searchPaths;

        string curDir = dp::util::getCurrentPath();
        if ( find( binSearchPaths.begin(), binSearchPaths.end(), curDir ) == binSearchPaths.end() )
        {
          binSearchPaths.push_back(curDir);
        }

        std::string modulePath = dp::util::getModulePath();
        if ( find( binSearchPaths.begin(), binSearchPaths.end(), modulePath ) == binSearchPaths.end() )
        {
          binSearchPaths.push_back(modulePath);
        }
        
        // receive the unique scene loader plug interface ID from the file extension 
        std::string ext = dp::util::getFileExtension( filename );
        dp::util::UPIID piid = dp::util::UPIID( ext.c_str(), dp::util::UPITID( UPITID_SCENE_LOADER, UPITID_VERSION ) );

        dp::util::PlugInSharedPtr plug;
        if ( !dp::util::getInterface(binSearchPaths, piid, plug) )
        {
          throw std::runtime_error( std::string( "Scene Plugin not found: " + ext ) );
        }
        SceneLoaderSharedPtr loader = plug.staticCast<SceneLoader>();
        loader->setCallback( callback );

        vector<string> sceneSearchPaths = binSearchPaths;

        // add the scene path, if it's not the current path (-> dir.empty()) and not yet added
        dir = dp::util::getFilePath( filename );
        if ( !dir.empty() && find( sceneSearchPaths.begin(), sceneSearchPaths.end(), dir ) == sceneSearchPaths.end() )
        {
          sceneSearchPaths.push_back(dir);
        }

        // for supplied models add additional resource paths to make sure
        // that required resources will be found by subsequent loaders
        size_t pos = dir.rfind("media");
        if ( pos != string::npos )
        {
          string nvsgsdk(dir.substr(0, pos));
          sceneSearchPaths.push_back(nvsgsdk + "media/effects");
          sceneSearchPaths.push_back(nvsgsdk + "media/textures");
        }

        dp::sg::core::SceneSharedPtr scene;
        try
        {
          scene = loader->load( filename, sceneSearchPaths, viewState );
        }
        catch (...)
        {
          // TODO another non RAII pattern, callback should be passed to load
          loader->setCallback( dp::util::PlugInCallbackSharedPtr::null );
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
        if ( !viewState->getScene() )
        {
          viewState->setScene( scene );
        }

        return viewState;
      }

      bool saveScene( std::string const& filename, dp::sg::ui::ViewStateSharedPtr const& viewState, dp::util::PlugInCallback *callback )
      {
        bool result = false;
        // define a unique plug-interface ID for SceneLoader
        const dp::util::UPITID PITID_SCENE_SAVER(UPITID_SCENE_SAVER, UPITID_VERSION);

        vector<string> searchPaths;
        searchPaths.push_back( dp::util::getCurrentPath() );

        std::string modulePath = dp::util::getModulePath();
        if ( std::find( searchPaths.begin(), searchPaths.end(), modulePath ) == searchPaths.end() )
        {
          searchPaths.push_back( modulePath );
        }

#if defined(DP_OS_WINDOWS)
        modulePath = dp::util::getModulePath( reinterpret_cast<HMODULE>(&__ImageBase) );
        if ( std::find( searchPaths.begin(), searchPaths.end(), modulePath ) == searchPaths.end() )
        {
          searchPaths.push_back( modulePath );
        }
#endif

        dp::util::UPIID piid = dp::util::UPIID(dp::util::getFileExtension( filename ).c_str(), PITID_SCENE_SAVER);

        dp::util::PlugInSharedPtr plug;
        if ( getInterface( searchPaths, piid, plug ) )
        {
          SceneSaverSharedPtr ss = plug.staticCast<SceneSaver>();
          try
          {
            dp::sg::core::SceneSharedPtr scene( viewState->getScene() ); // DAR HACK Change SceneSaver interface later.
            result = ss->save( scene, viewState, filename );
          }
          catch(...) // catch all others
          {
          }
        }

        return result;
      }

      bool saveTextureHost( const std::string & filename, const dp::sg::core::TextureHostSharedPtr & tih )
      {
        // load the saver plug-in  - this should be configured
        dp::util::UPIID piid = dp::util::UPIID(dp::util::getFileExtension( filename ).c_str(), dp::util::UPITID(UPITID_TEXTURE_SAVER, UPITID_VERSION) );

        std::vector<std::string> searchPaths;
        searchPaths.push_back( dp::util::getCurrentPath() );
        searchPaths.push_back( dp::util::getModulePath() );

        bool retval = false;
        //
        // MMM - TODO - Update me for stereo images
        //
        dp::util::PlugInSharedPtr plug;
        if ( getInterface( searchPaths, piid, plug ) )
        {
          TextureSaverSharedPtr ts = plug.staticCast<TextureSaver>();
          retval = ts->save( tih, filename );
        }

        return retval;
      }

      dp::sg::core::TextureHostSharedPtr loadTextureHost( const std::string & filename, const std::vector<std::string> &searchPaths )
      {
        dp::sg::core::TextureHostSharedPtr tih;
        // appropriate search paths for the loader dll and the sample file.
        vector<string> binSearchPaths = searchPaths;

        std::string curDir = dp::util::getCurrentPath();
        if ( find( binSearchPaths.begin(), binSearchPaths.end(), curDir ) == binSearchPaths.end() )
        {
          binSearchPaths.push_back(curDir);
        }

        std::string modulePath = dp::util::getModulePath();
        if ( find( binSearchPaths.begin(), binSearchPaths.end(), modulePath ) == binSearchPaths.end() )
        {
          binSearchPaths.push_back(modulePath);
        }

        std::string foundFile = dp::util::findFile( filename, binSearchPaths );
        if (!foundFile.empty())
        {
          std::string ext = dp::util::getFileExtension( filename );

          dp::util::UPIID piid = dp::util::UPIID( ext.c_str(), dp::util::UPITID(UPITID_TEXTURE_LOADER, UPITID_VERSION) );

          // TODO - Update me for stereo images
          dp::util::PlugInSharedPtr plug;
          if ( getInterface( binSearchPaths, piid, plug ) )
          {
            TextureLoaderSharedPtr tl = plug.staticCast<TextureLoader>();
            tih = tl->load( foundFile );
          }
        }
        return tih;
      }

    } // namespace io
  } // namespace sg
} // namespace dp
