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
 
#include <dp/util/Config.h>

#if defined(DP_OS_WINDOWS)
  #include <Windows.h>
#elif defined(DP_OS_LINUX)
  #include <sys/time.h>
#else
  #error Unknown OS
#endif

namespace dp
{
  namespace util
  {
    /*! \brief A simple timer class.
     * This timer class can be used on Windows and Linux systems to
     * measure time intervals in seconds. 
     * The timer can be started and stopped several times and accumulates
     * time elapsed between the start() and stop() calls. */
    class Timer
    {
    public:
      //! Default constructor. Constructs a Timer, but does not start it yet. 
      DP_UTIL_API Timer();

      //! Default destructor.
      DP_UTIL_API ~Timer();

      //! Starts the timer.
      DP_UTIL_API void start();

      //! Stops the timer.
      DP_UTIL_API void stop();

      //! Resets the timer.
      DP_UTIL_API void reset();

      //! Resets the timer and starts it.
      DP_UTIL_API void restart();

      //! Returns the current time in seconds.
      DP_UTIL_API double getTime() const;

      //! Return whether the timer is still running.
      bool isRunning() const { return m_running; }

    private:
#if defined(DP_OS_WINDOWS)
      typedef LARGE_INTEGER Time;
#elif defined(DP_OS_LINUX)
      typedef timeval Time;
#else
  #error Unknown OS
#endif

    private:
      double calcDuration(Time begin, Time end) const;

    private:
#if defined(DP_OS_WINDOWS)
      LARGE_INTEGER m_freq;
#endif
      Time m_begin;
      bool m_running;
      double m_seconds;
    };

  } // namespace util
} // namespace dp

