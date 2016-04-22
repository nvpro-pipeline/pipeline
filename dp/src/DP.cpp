// Copyright NVIDIA Corporation 2013-2014
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


#include <dp/DP.h>
#include <dp/util/File.h>
#include <boost/filesystem.hpp>
#include <iostream>

#if defined(DP_OS_LINUX)
#include <linux/limits.h>
#endif

#include <cstdlib>

namespace dp
{
  // helper function to check if the given path includes the global config file:
  bool dpCfgExistsInPath( const std::string &path )
  {
    boost::filesystem::path file = boost::filesystem::path( path ) / boost::filesystem::path( "dp.cfg" );
    return boost::filesystem::exists( file );
  }

  std::string home()
  {
    // Find the home, the path where the assets, shaders etc. are located.
    // 1. Use $DPHOME if set.
    // 2. Otherwise check if the dp.cfg is located where this dll is located, then use that path.
    // 3. Last resort is to look for the dp.cfg at a CMAKE defined location. 
    static boost::filesystem::path home;

    // as long as dp.cfg has not been found, no valid home was found
    static bool dpCfgFound = false;

    // only search for the path once
    if( dpCfgFound )
    {
      return home.string();
    }

    // 1:
    if( getenv( "DPHOME" ) )
    {
      std::string dpHomeEnvironment = std::string( getenv( "DPHOME" ) );
      if( dpCfgExistsInPath( dpHomeEnvironment ) )
      {
        home = dpHomeEnvironment;
        dpCfgFound = true;
      }
      else
      {
        std::cerr << "The path pointed at by $DPHOME does not contain the file <dp.cfg>." << std::endl
                  << "Path: <" << dpHomeEnvironment << ">" << std::endl;
      }
    }

    // 2:
    if (!dpCfgFound)
    {
#if defined(DP_OS_WINDOWS)
      TCHAR binaryPath[MAX_PATH];
      if( ( 0 != GetModuleFileName(0, binaryPath, MAX_PATH) ) && ( dpCfgExistsInPath( binaryPath ) ) )
#elif defined(DP_OS_LINUX)
      char binaryPath[PATH_MAX];
      if( ( -1 != readlink ("/proc/self/exe", binaryPath, PATH_MAX) ) && ( dpCfgExistsInPath( binaryPath ) ) )
#else
  #error("unsupported os");
#endif
      {
        home = binaryPath;
        home.remove_filename();
        dpCfgFound = true;
      }
    }
  
    // 3:
    if(( !dpCfgFound ) && ( dpCfgExistsInPath( DP_HOME_FALLBACK ) ))
    {
      home = DP_HOME_FALLBACK;
      dpCfgFound = true;
    }
    
    if( !dpCfgFound )
    {
      throw std::runtime_error( "Cannot determine path of home. dp.cfg was searched at $DPHOME,"
                                " the binary location of the dp library and the source directory at compile time." );
    }

    std::cout << "found home at location: " << home.string() << std::endl;
    return home.string();
  }

} // namespace dp
