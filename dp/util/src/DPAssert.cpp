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


#include <iostream>
#include <sstream>
#include <cstdio>

#include <assert.h>

#include <dp/util/DPAssert.h>

namespace dp
{
  namespace util
  {

    int DefaultAssertCallback( const char *message )
    {
      std::cerr << message << std::endl;
      DP_DBGBREAK();
      return 0; // Continue execution.
    }

    static UserAssertCallback g_userAssertCallback = DefaultAssertCallback;

    void AssertionFailed(const char * expr, const char * file, unsigned int line)
    {
      if ( g_userAssertCallback )
      {
        std::stringstream text;
        text << "Assertion Failed: " << expr << "\nFile: " << file << ", Line: " << line;

        // If the user callback returns zero, simply continue.
        if ( g_userAssertCallback( text.str().c_str() ) == 0 )
        {
          return;
        }
      }

#if !defined(NDEBUG)
      #if defined(_WIN32)
      # if defined(_MSC_VER) && (_MSC_VER >= 1400)
        // _wassert expects wide character strings
        #define ccount(s) (strlen(s) + 1)
        wchar_t * wfile = new wchar_t[ccount(file) << 1]; // make the wide string twice as big
        wchar_t * wexpr = new wchar_t[ccount(expr) << 1]; // dito
        mbstowcs(wfile, file, ccount(file));
        mbstowcs(wexpr, expr, ccount(expr));
        #undef ccount
        _wassert(wexpr, wfile, line);
        delete[] wfile;
        delete[] wexpr;
      # else
        _assert(expr, file, line);
      # endif
       #else
        printf( "assert( %s ) failed!\nFile: %s, Line: %d\n", expr, file, line );
        fflush(stdout);
      # if defined(ABORT_ON_ASSERTION_FAILED)
        // c-runtime spec conform but annoying somehow
        abort();
      # elif !defined(RESUME_ON_ASSERTION_FAILED)
        // break into the attached debuger
        DP_DBGBREAK() // note: no semicolon here!
      # else
        // resume
      # endif
#endif // !defined(NDEBUG)

      #endif
    } // AssertionFailed


    UserAssertCallback SetUserAssertCallback( UserAssertCallback callback )
    {
      UserAssertCallback currentCallback = g_userAssertCallback;
      g_userAssertCallback = callback;
      return currentCallback;
    }

  }
}
