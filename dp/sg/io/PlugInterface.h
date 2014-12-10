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


#pragma once
/** @file */

#include <dp/sg/io/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/util/PlugIn.h> // general PlugIn class definition
#include <string>

namespace dp
{
  namespace sg
  {
    namespace io
    {

      //! Pure virtual base class for SceniX scene loader plug-ins
      /** User defined SceniX scene loader plug-ins must provide the \c SceneLoader
        * interface.
        * \par
        * Client code that needs to utilize a certain \c SceneLoader interface should
        * first query for a unique plug interface type ID through a call to \c dp::util::queryInterfaceType.
        * The unique plug interface type ID for a SceneLoader will be constructed from the \c UPITID_SCENE_SAVER
        * define and the actual SceniX version, which is coded in the UPITID_VERSION define, as shown in the code 
        * snippet below.
        * A call to \c dp::util::queryInterfaceType returns a list of all unique interface IDs (UPIIDSs)
        * found at given search paths.\n
        * After that, the client code can take the different UPIIDs to grab a scene loader interface 
        * through a call to \c dp::util::getInterface.\n
        * After usage, the client code should release the interface through a call to \c dp::util::releaseInterface:
        *
        * \code
        *   // Example:
        *   // --------
        *   // Get a scene loader interface capable to load 'nbf' files ("NVSG Binary File")
        *   
        *   // 
        *   vector<string> searchPaths;
        *   // Add appropriate search paths here. Not relevant for the example here.
        *   // ...
        *
        *   // define the unique plug interface type ID for SceneLoaders
        *   const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);
        *
        *   dp::sg::core::SmartScene theScene;
        *   dp::sg::core::SmartViewState viewState;
        *   dp::util::UPIID nbfLoaderInterfaceID;
        *   bool foundAppropriate = false;
        *   vector<dp::util::UPIID> piids;
        *   if ( dp::util::queryInterfaceType(searchPaths, PITID_SCENE_LOADER, piids) )
        *   {
        *     vector<dp::util::UPIID>::iterator it = piids.begin();
        *     for ( ; it != piids.end(); ++it )
        *     {
        *       if ( !stricmp((*it).getPlugSpecificIDString(), ".nbf") )
        *       {
        *         // found, copy the ID
        *         nbfLoaderInterfaceID = *it;
        *         foundAppropriate = true;
        *         break; // look no further
        *       }
        *     }
        *     if ( foundAppropriate )
        *     {
        *       dp::util::PlugIn * plug; 
        *       if ( dp::util::getInterface(searchPaths, nbfLoaderInterfaceID, plug) )
        *       {
        *         dp::sg::core::SceneLoader * loader = reinterpret_cast<SceneLoader*>(plug);
        *         theScene = loader->load("c:\\myscenes\\sample.nbf", searchPaths, viewState);
        *         dp::util::releaseInterface(nbfLoaderInterfaceID);
        *       }
        *     }
        *   }
        * \endcode
        */
      DEFINE_PTR_TYPES( SceneLoader );

      // SceneLoader interface
      class SceneLoader : public dp::util::PlugIn
      {
        public:
          DP_SG_IO_API virtual ~SceneLoader();

          //! Loading a scene
          /** Loads a SceniX scene from a file specified by \a filename. 
            * The function tries to look up this file as follows:
            * -# Looks at the specified location.
            * -# Looks at the current directory.
            * -# Uses the search paths stored in \a searchPaths to look up the file.
            * \returns A a SceneSharedPtr specifying the loaded scene if successful, a null pointer otherwise.
            */
          DP_SG_IO_API virtual dp::sg::core::SceneSharedPtr load( std::string const& filename                 //!< A string that holds the name of the scene file to be loaded.
                                                                , std::vector<std::string> const& searchPaths //!< A collection of search paths used to look up the file.
                                                                , dp::sg::ui::ViewStateSharedPtr & viewState  /*!< If the function succeeded, this points to the optional
                                                                                                                   ViewState stored with the scene. */
                                                                ) = 0;

        protected:
          DP_SG_IO_API SceneLoader();
      };


      DEFINE_PTR_TYPES( SceneSaver );

      //! Pure virtual base class for SceniX scene saver plug-ins
      /** User defined SceniX scene saver plug-ins must provide the \c SceneSaver
        * interface. */
      class SceneSaver : public dp::util::PlugIn
      {
        public:
          DP_SG_IO_API virtual ~SceneSaver();

          //! Saving a scene
          /** Saves a SceniX \a scene and a \a viewState to a file specified by \a filename. 
            * \returns \c true if successful, \c false otherwise.
            */
          DP_SG_IO_API virtual bool save( dp::sg::core::SceneSharedPtr const& scene
                                        , dp::sg::ui::ViewStateSharedPtr const& viewState
                                        , std::string const& filename ) = 0;

        protected:
          DP_SG_IO_API SceneSaver();
      };


      DEFINE_PTR_TYPES( TextureLoader );

      //! Pure virtual base class for texture loader plugins
      /** \note TextureLoader are not yet implemented as plugins. */
      class TextureLoader : public dp::util::PlugIn
      {
        public:
          DP_SG_IO_API virtual ~TextureLoader();

          /*! \brief Function to load a TextureHost.
           *  \param filename The name of the texture file to load.
           *  \param searchPaths Not applicable for texture files, can be left empty.
           *  This exists due to the SceniX plug-in interface and is normally used 
           *  to search for files referenced within scenes or shaders.
           *  \param creationFlags An optional set of creation flags
           *  \return The created TextureHost.
           *  \remarks Creates a TextureHost and calls \c onLoad() with \a filename, the
           *  created TextureHost, and \a searchPaths.
           *  \note The behavior is undefined if the file \a filename does not exist or is not of
           *  the appropriate type.
           *  \sa reload, onLoad */
          DP_SG_IO_API dp::sg::core::TextureHostSharedPtr load( const std::string& filename
                                                      , const std::vector<std::string>& searchPaths = std::vector<std::string>()
                                                      , const unsigned int& creationFlags = 0 );

          /*! \brief Function to reload a TextureHost.
           *  \param tih A pointer to the TextureHost to reload.
           *  \param searchPaths An optional set of search paths to use for files referenced within
           *  the texture file specified by \a filename.
           *  \return \c true if reloading the TextureHost was successful, otherwise \c false.
           *  \remarks Any previously loaded pixel data is released, and the file specified by
           *  \a tih is reloaded. It is assumed that the TextureHost is cleared, that is, it has no
           *  images.
           *  \sa load, onLoad */
          DP_SG_IO_API bool reload( const dp::sg::core::TextureHostSharedPtr & tih
                                  , const std::vector<std::string>& searchPaths
                                  = std::vector<std::string>() );

        protected:
          DP_SG_IO_API TextureLoader();

          /*! \brief Pure virtual interface called on loading a TextureHost file.
           *  \param texImg A pointer to the TextureHost to load the data into.
           *  \param searchPaths A set of search paths used for files referenced within the texture
           *  file.
           *  \return \c true, if the TextureHost was successfully loaded, otherwise \c false.
           *  \remarks Loads a TextureHost from the file specified by \a texImg. The PlugIn must
           *  create the image(s) in the OpenGL compatible orientation, that has its image origin in
           *  the lower left. TextureHost offers mirrorX() and mirrorY() routines that a Loader
           *  PlugIn can use to achieve this orientation.
           *  \sa load, reload */
          DP_SG_IO_API virtual bool onLoad( dp::sg::core::TextureHostSharedPtr const& texImg
                                          , const std::vector<std::string> & searchPaths ) = 0;
      };

      DEFINE_PTR_TYPES( TextureSaver );

      //! Pure virtual base class for texture saver plugins
      /** \note TextureSaver are not yet implemented as plugins. */
      class TextureSaver : public dp::util::PlugIn
      {
        public:
          DP_SG_IO_API virtual ~TextureSaver();

          //! Pure virtual interface function to save a TextureHost.
          /** Saves the texture image \a image to the file given by \a filename.
            * \returns  \c true, if the TextureHost was saved successfully, otherwise \c false.
            */
          DP_SG_IO_API virtual bool save( const dp::sg::core::TextureHostSharedPtr & image
                                    , const std::string & fileName ) = 0;

        protected:
          DP_SG_IO_API TextureSaver();
      };

      //! Pure virtual base class for shader loader plugins
      class ShaderLoader : public dp::util::PlugIn
      {
        public:
          typedef std::vector<std::string> BufferList;

        public:
          //! Pure virtual interface function to load a shader.
          DP_SG_IO_API virtual bool load( const std::string &fileName                 //!< A string that holds the name of the shader file to be loaded.
                                    , const std::vector<std::string> &searchPaths //!< A collection of search paths used to look up the file.
                                    , BufferList &bl                              //!< A collection of buffers that contain the generated shader files.
                                    ) = 0;

          //! Pure virtual interface function to get occured errors.
          DP_SG_IO_API virtual std::string getErrors() = 0;
        protected:
          //! Protected virtual destructor
          DP_SG_IO_API virtual ~ShaderLoader();
      };

    } // namespace io
  } // namespace dp
} //  namespace sg
