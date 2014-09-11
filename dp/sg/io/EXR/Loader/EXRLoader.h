// Copyright NVIDIA Corporation 2002-2004
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
/** \file */

#include <dp/sg/io/PlugInterface.h>
#include <dp/sg/io/PlugInterfaceID.h>

// storage-class defines 
#if defined(_WIN32)
# ifdef EXRLOADER_EXPORTS
#  define EXRLOADER_API __declspec(dllexport)
# else
#  define EXRLOADER_API __declspec(dllimport)
# endif
#else
#  define EXRLOADER_API
#endif

#if !defined(DOXYGEN_IGNORE)

// exports required for a scene loader plug-in
#if defined(LINUX)
extern "C"
{
#endif
  EXRLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi);
  EXRLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
#if defined(LINUX)
}
#endif

#endif //DOXYGE_IGNORE

//! A Texture Loader PlugIn that can read OpenEXR images using the OpenEXR library
class EXRLoader : public dp::sg::io::TextureLoader
{
  public:
    EXRLoader();

    friend bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi);

    //! Realization of the pure virtual interface function of a PlugIn.
    /** \note Never call \c delete on a PlugIn, always use the member function. */
    void deleteThis( void );

  protected:
    //! Protected virtual destructor
    /** Prohibits ordinary client code from  
    * - creating a \c TextureLoader derived object on stack and
    * - calling \c delete on a pointer to \c TextureLoader.
    */
    virtual ~EXRLoader();  

    bool onLoad( dp::sg::core::TextureHostSharedPtr const& texImg
      , const std::vector<std::string> & searchPaths );
};

inline void EXRLoader::deleteThis()
{  
  delete this;
}
