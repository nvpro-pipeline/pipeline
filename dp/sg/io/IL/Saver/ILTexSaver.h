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


namespace dp
{
  namespace util
  {
    class UPITID;
    class UPIID;
  }
}

#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/io/PlugInterface.h>
#include <vector>
#include <string>

// storage-class defines 
#if defined(_WIN32)
# ifdef ILTEXSAVER_EXPORTS
#  define ILTEXSAVER_API __declspec(dllexport)
# else
#  define ILTEXSAVER_API __declspec(dllimport)
# endif
#else
#  define ILTEXSAVER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
  ILTEXSAVER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPtr<dp::util::PlugIn> & pi);
  ILTEXSAVER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}


//! A Texture Saver that encapsulates DevIL so it can be used with the NVSG PlugIn mechanism.
class ILTexSaver : public dp::sg::io::TextureSaver
{
public:
  ILTexSaver();

  friend bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPtr<dp::util::PlugIn> & pi);

  void deleteThis( void );
  bool save( const dp::sg::core::TextureHostSharedPtr & image, const std::string& fileName );

protected:   
  virtual ~ILTexSaver();

private:
  static dp::util::SmartPtr<ILTexSaver> m_instance;
};


