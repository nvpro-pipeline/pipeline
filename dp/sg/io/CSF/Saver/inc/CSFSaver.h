// Copyright NVIDIA Corporation 2002-2012
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

#include  <set>
#include  <dp/sg/core/nvsgapi.h>
#include  <dp/sg/io/PlugInterfaceID.h>
#include  <dp/sg/io/PlugInterface.h>
#include  <dp/sg/ui/ViewState.h>


//  Don't need to document the API specifier
#if ! defined( DOXYGEN_IGNORE )
#if defined(_WIN32)
# ifdef CSFSAVER_EXPORTS
#  define CSFSAVER_API __declspec(dllexport)
# else
#  define CSFSAVER_API __declspec(dllimport)
# endif
#else
# define CSFSAVER_API
#endif
#endif  //  DOXYGEN_IGNORE

// exports required for a scene loader plug-in
extern "C"
{
//! Get the PlugIn interface for this scene saver.
/** Every PlugIn has to resolve this function. It is used to get a pointer to a PlugIn class, in this case a 
CSFSAVER.
  * If the PlugIn ID \a piid equals \c PIID_NVSG_SCENE_SAVER, a CSFSAVER is created and returned in \a pi.
  * \returns  true, if the requested PlugIn could be created, otherwise false
  */
CSFSAVER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPtr<dp::util::PlugIn> & pi);

//! Query the supported types of PlugIn Interfaces.
CSFSAVER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

//! A Scene Saver for csf files.
class CSFSaver : public dp::sg::io::SceneSaver
{
  public :
    //! Realization of the pure virtual interface function of a PlugIn.
    /** \note Never call \c delete on a PlugIn, always use the member function. */
    void  deleteThis( void ); //!< PlugIn interface

    //! Realization of the pure virtual interface function of a SceneSaver.
    /** Saves the \a scene and the \a viewState to \a filename. 
      * The \a viewState may be NULL. */
    bool  save( dp::sg::core::SceneSharedPtr   const& scene      //!<  scene to save
              , dp::sg::ui::ViewStateSharedPtr const& viewState  //!<  view state to save
              , std::string                    const& filename   //!<  file name to save to
              );
};

inline void CSFSaver::deleteThis( void )
{
  delete this;
}
