// Copyright NVIDIA Corporation 2015
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

#include <dp/util/Config.h>
#include <boost/filesystem.hpp>
#include <set>
#include <string>
#include <vector>

namespace dp
{
  namespace util
  {

    class FileFinder
    {
      public:
        DP_UTIL_API FileFinder();
        DP_UTIL_API FileFinder( std::string const& path );
        DP_UTIL_API FileFinder( std::vector<std::string> const& paths );

        DP_UTIL_API bool addSearchPath( std::string const& path );
        DP_UTIL_API void addSearchPaths( std::vector<std::string> const& paths );
        DP_UTIL_API void clear();
        DP_UTIL_API std::string find( std::string const& file ) const;
        DP_UTIL_API std::string findRecursive( std::string const& file ) const;
        DP_UTIL_API std::vector<std::string> getSearchPaths() const;
        DP_UTIL_API bool removeSearchPath( std::string const& path );

      private:
        mutable boost::filesystem::path   m_latestHit;
        std::set<boost::filesystem::path> m_searchPaths;
    };

  } // namespace util
} // namespace dp


