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


#include <dp/util/Profile.h>

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#ifndef LINUX
#include <conio.h>
#endif

namespace dp
{
  namespace util
  {

    namespace {
      unsigned int level(0);
      bool triedAllocConsole(false);
      bool triedOpenOutFile(false);
      bool isOpenOutFile(false);
      std::ofstream ofsProfile;
    }

    Profile::Profile( const std::string &description )
      : m_description(description), m_level(level)
    {
      ++level;

      std::stringstream s;
      s << std::setw(m_level*2) << ""; // indent 
      s << "> " << m_description << "\n";
      output( s.str() );

      m_timer.start();
    }

    Profile::~Profile()
    {
      if( m_timer.isRunning() )
      {
        stop();
      }
    }

    double Profile::getTime()
    { 
      double seconds = m_timer.getTime();
      
      std::stringstream s;
      s << std::setw(m_level*2) << ""; // indent 
      s << "- " << m_description;  
      s << " " << std::setw(20) << seconds << "s" << "\n";
      output( s.str() );

      return seconds;
    }

    void Profile::text( const std::string &text )
    {
      std::stringstream s;
      s << std::setw(m_level*2) << ""; // indent 
      s << "- " << m_description;  
      s << " " << text << "\n";
      output( s.str() );
    }

    double Profile::stop()
    {
      --level;

      m_timer.stop();
      double seconds = m_timer.getTime();

      std::stringstream s;
      s << std::setw(m_level*2) << ""; // indent 
      s << "< " << m_description;  
      s << " " << std::setw(20) << seconds << "s" << "\n";
      output( s.str() );

      return seconds;
    }

    void Profile::output( const std::string &s )
    {
      PROFILING_OUTPUT_FUNCTION( s );
    }

    void Profile::fileOutput( const std::string &s )
    {
      if( !triedOpenOutFile )
      {
        triedOpenOutFile = true;
        ofsProfile.open( PROFILING_FILENAME );
        isOpenOutFile = ofsProfile.is_open();
      }
      if( isOpenOutFile )
      {
        ofsProfile << s;
    #if PROFILING_OUTPUT_FLUSH
        ofsProfile << std::flush;
    #endif
      }
    }

    void Profile::consoleOutput( const std::string &s )
    {

    #ifndef LINUX
      if( !triedAllocConsole )
      {
        triedAllocConsole = true;
        AllocConsole();
      }
      _cprintf("%s", s.c_str());
    #else
      std::cout << s;
    #if PROFILING_OUTPUT_FLUSH
      std::cout << std::flush;
    #endif
    #endif
    }

    void Profile::debugOutput( const std::string &s )
    {
    #ifdef _WIN32
    #ifndef UNICODE
      OutputDebugStringA(s.c_str());
    #endif
    #elif defined(LINUX)
      std::cout << s;
    #if PROFILING_OUTPUT_FLUSH
      std::cout << std::flush;
    #endif
    #endif
    }
  } // namespace util
} // namespace dp

