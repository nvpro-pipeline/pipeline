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


#pragma once

#include <dp/sg/io/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/ui/ViewState.h>
#include <string>
#include <dp/util/PlugInCallback.h>

namespace dp
{
  namespace sg
  {
    namespace io
    {

      /*! \brief Load a scene, internally doing all the SceneLoader handling.
       *  \param filename The name of the file to load.
       *  \param searchPaths Optional array of search paths to find the file to load.
       *  \param callback Optional pointer to a callback used by the loader for the file to load.
       *  \return A ViewState containing loaded scene on success. Throws otherwise.
       *  \sa saveScene */
      DP_SG_IO_API dp::sg::ui::ViewStateSharedPtr loadScene( std::string const& filename
                                                           , std::vector<std::string> const& searchPaths = std::vector<std::string>()
                                                           , dp::util::PlugInCallbackSharedPtr const& callback = dp::util::PlugInCallbackSharedPtr::null );

      /*! \brief Save a scene, internally doing all the SceneSaver handling.
       *  \param filename The name of the file to save to.
       *  \param viewState The view state (holding the scene) to save.
       *  \param callback Optional pointer to a callback used by the saver for the file to save.
       *  \return true if a scene could be saved, false otherwise
       *  \sa loadScene */
      DP_SG_IO_API bool saveScene( std::string const& filename
                                 , dp::sg::ui::ViewStateSharedPtr const& viewState
                                 , dp::util::PlugInCallback *callback = 0 );

      /*! \brief Load a texture image from disk
       * \param filename disk file to load image from
       * \param tih texture image handle to load to
       * \param searchPaths additional search paths
       * \return true if save was successful
       */
     DP_SG_IO_API dp::sg::core::TextureHostSharedPtr loadTextureHost( const std::string & filename, const std::vector<std::string> &searchPaths = std::vector<std::string>() );

     /*! \brief Save a texture image to disk
       * \param filename disk file to save image to
       * \param tih texture image to save
       * \return true if save was successful
       */
      DP_SG_IO_API bool saveTextureHost( const std::string & filename, const dp::sg::core::TextureHostSharedPtr & tih );

    } // namespace io
  } // namespace sg
} // namespace dp
