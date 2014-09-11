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


#include <dp/sg/core/NVSGVersion.h>
#include <dp/sg/core/nvsg.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

#if 0
      // TODO understand c++ 11 literals. The following code gives this error for gcc.
      /home/matavenrath/perforce/vbox/sw/devtech/platform/pipeline/trunk/dp/sg/core/src/NVSGVersion.cpp: In function ‘void dp::sg::core::getCopyrightString(std::string&)’:
      /home/matavenrath/perforce/vbox/sw/devtech/platform/pipeline/trunk/dp/sg/core/src/NVSGVersion.cpp:24:63: error: inconsistent user-defined literal suffixes ‘SDK_NAME’ and ‘VERSION_STR’ in string literal
      /home/matavenrath/perforce/vbox/sw/devtech/platform/pipeline/trunk/dp/sg/core/src/NVSGVersion.cpp:24:63: error: unable to find string literal operator ‘operator"" SDK_NAME’

      void getVersionString(std::string & string)   {string = VERSION_STR;}
      void getSDKName(std::string & string)         {string = SDK_NAME;}
      void getCopyrightString(std::string & string) {string = NVSG_COPYRIGHT;}
      void getVendorName(std::string & string)      {string = SDK_VENDOR;}
#endif

    } // namespace core
  } // namespace sg
} // namespace dp
