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
/** \file */

#include <dp/sg/core/Config.h>

/**
Use this Profiling class to measure execution times in milliseconds.

The PROFILE(desc) macro can be used to start a timer. PROFILE_GETTIME() can be
used to measure an inbetween-time.  The timer will run until PROFILE_STOP() is
called  or it goes out of scope.  Thus,  to just  measure the  time a function 
takes, you only need to call PROFILE(...); at the start of the function.

The PROFILE_GETTIME() and PROFILE_STOP() macros return a double containing the
measured time in milliseconds.

The PROFILEN* macros can be used if several  timers have to be declared in one
scope.

The Profiling can be adjusted via the following defines:

ENABLE_PROFILING controls the profiling itself.  If defined to 0, no profiling 
occurs. If set to a nonzero value, the profiling takes place.

If PROFILING_OUTPUT_FUNCTION is set to one of (fileOutput, consoleOutput,
debugOutput), starting and stopping a timer will output desc to mark the begin
and end of the profiling,  together with the  measured time on stop.  If it is 
defined to an empty value, nothing is output.

The output filename (if PROFILING_OUTPUT_FUNCTION is set to fileOutput) can be
set via PROFILING_FILENAME.
*/

#include <string>

#include <dp/util/Config.h>
#include <dp/util/Timer.h>


// Define to en/disable profiling
#ifndef ENABLE_PROFILING
#define ENABLE_PROFILING 0
#endif

// Uncomment one of the following lines to set output method:
#ifndef PROFILING_OUTPUT_FUNCTION
//#define PROFILING_OUTPUT_FUNCTION fileOutput      // output to a file 
#define PROFILING_OUTPUT_FUNCTION consoleOutput     // output to console
//#define PROFILING_OUTPUT_FUNCTION debugOutput     // output to attached debugger
//#define PROFILING_OUTPUT_FUNCTION                 // don't output automatically
#endif

// Set to 1 if you want a flush after every output. Slower but useful for debugging
#ifndef PROFILING_OUTPUT_FLUSH
#define PROFILING_OUTPUT_FLUSH 0
#endif

#ifndef PROFILING_FILENAME
#define PROFILING_FILENAME "profile.txt"
#endif

#if ENABLE_PROFILING
//! Profiling macros.
#define PROFILE(desc)       dp::util::Profile profileObject(desc) //! Start a profiler
#define PROFILE_GETTIME()   profileObject.getTime()               //! Get intermediate time
#define PROFILE_TEXT(msg)   profileObject.text(msg)               //! Output a text
#define PROFILE_STOP()      profileObject.stop()                  //! Stop the profiler

#define PROFILEN(n, desc)     dp::util::Profile profileObject ## n(desc)
#define PROFILEN_GETTIME(n)   profileObject ## n.getTime()
#define PROFILEN_TEXT(n, msg) profileObject ## n.text(msg)
#define PROFILEN_STOP(n)      profileObject ## n.stop()
#else
#define PROFILE(desc)       
#define PROFILE_GETTIME()   
#define PROFILE_TEXT(msg)   
#define PROFILE_STOP()      

#define PROFILEN(n, desc)     
#define PROFILEN_GETTIME(n)   
#define PROFILEN_TEXT(n, msg) 
#define PROFILEN_STOP(n)
#endif


namespace dp
{
  namespace util
  {

    //! Profiling class.
    class Profile
    {
    public:
      //! Start a profiler
      DP_UTIL_API Profile( const std::string &description );
      DP_UTIL_API ~Profile();

      //! Output and return the current measured time in seconds
      DP_UTIL_API double getTime();

      //! Output a text
      DP_UTIL_API void text( const std::string &text );

      //! Stop the profiler, output and return measured time in seconds
      DP_UTIL_API double stop();

    private:
      void output( const std::string &s );

      void fileOutput( const std::string &s );
      void consoleOutput( const std::string &s );
      void debugOutput( const std::string &s );

    private:
      dp::util::Timer m_timer;
      std::string     m_description;
      unsigned int    m_level;
    };

  } // namespace util
} // namespace dp
