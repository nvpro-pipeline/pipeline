// Copyright NVIDIA Corporation 2013
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

#include <dp/util/Config.h>
#include <dp/util/Timer.h>

#include <string>
#include <vector>

namespace dp
{
  namespace util
  {

    class FrameProfiler
    {
    public:
      DP_UTIL_API FrameProfiler();
      DP_UTIL_API void beginFrame();

      /** \brief Enable/disable the frame profiler. If the profiler is disabled it'll ignore.
                 all addEntry calls. If the profiler is enabled it'll display the entries
                 of the last frame.
                 TODO Display the average values of the last frame.
      **/
      DP_UTIL_API void setEnabled( bool enabled );
      DP_UTIL_API bool isEnabled() const;

      /** \brief Time is in seconds **/
      DP_UTIL_API void addEntry( std::string const& key, double time );
      DP_UTIL_API void endFrame();

      DP_UTIL_API static FrameProfiler& instance();

    private:
      bool m_enabled;

      struct Entry
      {
        std::string key;
        double      time;
      };
      std::vector< Entry > m_entries;

      dp::util::Timer m_lastDisplay;
    };

    class ProfileEntry
    {
    public:
      DP_UTIL_API ProfileEntry( std::string const& key );
      DP_UTIL_API ~ProfileEntry();

    private:
      std::string     m_key;
      dp::util::Timer m_timer;
    };

  } // namespace util
} // namespace dp
