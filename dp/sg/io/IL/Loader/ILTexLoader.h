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

#include <dp/sg/io/PlugInterface.h>

#include <vector>
#include <string>

// storage-class defines 
#if defined(_WIN32)
# ifdef ILTEXLOADER_EXPORTS
#  define ILTEXLOADER_API __declspec(dllexport)
# else
#  define ILTEXLOADER_API __declspec(dllimport)
# endif
#else
#  define ILTEXLOADER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
  ILTEXLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPlugIn & pi);
  ILTEXLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

SMART_TYPES( ILTexLoader );

//! A Texture Loader that encapsulates DevIL so it can be used with the SceniX PlugIn mechanism.
class ILTexLoader : public dp::sg::io::TextureLoader
{
  public:
    static SmartILTexLoader create();
    virtual ~ILTexLoader();  

    friend bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPlugIn & pi);

  protected:
    ILTexLoader();
    
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
    bool onLoad( dp::sg::core::TextureHostSharedPtr const& texImg
               , const std::vector<std::string> & searchPaths );

  protected:
    /*! \brief Single instance of this class.
     *  \remarks To reduce the overhead due to the large number of supported extensions, we only
     *  instantiate this class once. */
    static SmartILTexLoader m_instance;
};


