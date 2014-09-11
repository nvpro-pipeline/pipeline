// Copyright NVIDIA Corporation 2012
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
#pragma warning( push )
#pragma warning( disable: 4996 )
#endif

#include <iostream>

#if defined(DP_OS_WINDOWS)
#include <conio.h>
#include <stdio.h>
#elif defined(DP_OS_LINUX)
#include <stdio.h>
#include <stdarg.h>
#endif

//! Define for the trace log file name.
#define TRACE_LOG_FILE       "dputil.log" /* log file for debug printouts */

namespace dp
{
  namespace util
  {
    //! output to attached debugger
    struct traceDebugOutput
    { 
      //! Function call operator to output a string
      void operator()(const char * str)
      {
#if defined(DP_OS_WINDOWS)
#ifndef UNICODE
        OutputDebugStringA(str);
#endif
#elif defined(DP_OS_LINUX)
        printf("%s", str);
#endif
      }
    };

    //! output to console
    struct traceConsoleOutput
    { 
      //! Function call operator to output a string
      void operator()(const char * str)
      {
        std::cout << str;
      }
    };

    //! output to console _AND_ attached debugger
    struct traceConsoleDebugOutput
    { 
      //! Function call operator to output a string
      void operator()(const char * str)
      {
        traceDebugOutput()(str); 
        traceConsoleOutput()(str); 
      }
    };

    //! output to file
    struct traceFileOutput
    {
      //! Function call operator to output a string
      void operator()(const char * str)
      {
        FILE * fp = fopen(TRACE_LOG_FILE, "a");
        if ( fp ) 
        {
          fputs(str, fp);
          fclose(fp);
        }
      }
    };

    //! output to log file _AND_ attached debugger
    struct traceFileDebugOutput
    {
      //! Function call operator to output a string
      void operator()(const char * str)
      {
        traceFileOutput()(str);
        traceDebugOutput()(str);
      }
    };

    //! output to log file _AND_ console
    struct traceFileConsoleOutput
    {
      //! Function call operator to output a string
      void operator()(const char * str)
      {
        traceFileOutput()(str);
        traceConsoleOutput()(str);
      }
    };
  } // namespace util
} // namespace dp

#if defined(DP_OS_WINDOWS)
#pragma warning( pop )
#endif
