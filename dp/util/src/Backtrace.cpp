// Copyright NVIDIA Corporation 2010-2012
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


#include <dp/util/DPAssert.h>
#include <dp/util/Backtrace.h>
#include <map>
#include <sstream>

#if defined(DP_POSIX)
#include <execinfo.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#elif defined(DP_OS_WINDOWS)
#include <windows.h>
#include <Dbghelp.h>
#endif

#define MAX_SYMBOL_LENGTH 8192

namespace dp
{
  namespace util
  {
    /* \brief This class provides functionality to create a backtraces for debugging purposes.
    **/
    class Backtrace
    {
    public:
      /* \brief Default Constructor. Initializes debug symbols */
      Backtrace();
      ~Backtrace();

      /* \brief Create a backtrace of the calling thread. Each string in the returned vector contains information about one line
       * \param firstFrame Skip the given amount of frames in the backtrace.
       * \param maxFrames Walk up the stack up to of maxFrames after skipping skipFrames.
       * \param exclusionList list of names to exclude in the backtrace
       * \return A std::vector containing strings where each string contains a textual representation for one stackframe.
      **/
      std::vector<std::string> backtrace( unsigned int skipFrames, unsigned int maxFrames, const std::vector<std::string> & exclusionList );
    private:
#if defined(__MINGW32__) // TODO enable backtrace on mingw. this requires dbghelp libraries.
#elif defined(DP_OS_WINDOWS)
      CRITICAL_SECTION m_lock;      // DbgHelp is not debugging safe. Global lock.
      HANDLE           m_hProcess;  // Handle of the running process.
      PSYMBOL_INFO     m_symbolInfo;
      std::map<PVOID,std::string> m_resolvedAddresses;
#endif
    };

    Backtrace::Backtrace()
    {
#if defined(__MINGW32__)
#elif defined(DP_OS_WINDOWS)
      // Allocate memory for symbols
      m_symbolInfo = (PSYMBOL_INFO)malloc( sizeof(SYMBOL_INFO) + MAX_SYMBOL_LENGTH );
      m_symbolInfo->SizeOfStruct = sizeof(SYMBOL_INFO);
      m_symbolInfo->MaxNameLen = MAX_SYMBOL_LENGTH;

      m_hProcess = GetCurrentProcess();
      InitializeCriticalSection( &m_lock );
      DP_VERIFY( SymInitialize( m_hProcess, 0, TRUE ) );
#endif
    }

    Backtrace::~Backtrace()
    {
#if defined(__MINGW32__)
#elif defined(DP_OS_WINDOWS)
      // free up memory allocated for symbol infos
      free(m_symbolInfo);
#endif
    }

    bool exclude( const std::string & name, const std::vector<std::string> & exclusionList )
    {
      for ( size_t i=0 ; i<exclusionList.size() ; i++ )
      {
        if ( strncmp( name.c_str(), exclusionList[i].c_str(), exclusionList[i].size() ) == 0 )
        {
          return( true );
        }
      }
      return( false );
    }

    std::vector<std::string> Backtrace::backtrace( unsigned int firstFrame, unsigned int maxFrames, const std::vector<std::string> & exclusionList )
    {
      std::vector<std::string> stackWalk;

#if defined(__MINGW32__)
#elif defined(DP_OS_WINDOWS)
      CONTEXT currentContext;
    #ifdef _M_IX86
      // on x86 assembly is required to initialize CONTEXT
      memset( &currentContext, 0, sizeof( CONTEXT ) );
      currentContext.ContextFlags = CONTEXT_CONTROL;

      // fetch registers required for StackWalk64
      __asm
      {
    Label:
        mov [currentContext.Ebp], ebp;
        mov [currentContext.Esp], esp;
        mov eax, [Label];
        mov [currentContext.Eip], eax;
      }
    #else
      // on all other platforms there's an helper function
      RtlCaptureContext( &currentContext );
    #endif

      //
      // Set up stack frame.
      //
      DWORD machineType;
      STACKFRAME64 stackFrame;
      memset( &stackFrame, 0, sizeof( STACKFRAME64 ) );

    #ifdef _M_IX86
      machineType                 = IMAGE_FILE_MACHINE_I386;
      stackFrame.AddrPC.Offset    = currentContext.Eip;
      stackFrame.AddrPC.Mode      = AddrModeFlat;
      stackFrame.AddrFrame.Offset = currentContext.Ebp;
      stackFrame.AddrFrame.Mode   = AddrModeFlat;
      stackFrame.AddrStack.Offset = currentContext.Esp;
      stackFrame.AddrStack.Mode   = AddrModeFlat;
    #elif _M_X64
      machineType                 = IMAGE_FILE_MACHINE_AMD64;
      stackFrame.AddrPC.Offset    = currentContext.Rip;
      stackFrame.AddrPC.Mode      = AddrModeFlat;
      stackFrame.AddrFrame.Offset = currentContext.Rsp;
      stackFrame.AddrFrame.Mode   = AddrModeFlat;
      stackFrame.AddrStack.Offset = currentContext.Rsp;
      stackFrame.AddrStack.Mode   = AddrModeFlat;
    #elif _M_IA64
      machineType                 = IMAGE_FILE_MACHINE_IA64;
      stackFrame.AddrPC.Offset    = currentContext.StIIP;
      stackFrame.AddrPC.Mode      = AddrModeFlat;
      stackFrame.AddrFrame.Offset = currentContext.IntSp;
      stackFrame.AddrFrame.Mode   = AddrModeFlat;
      stackFrame.AddrBStore.Offset= currentContext.RsBSP;
      stackFrame.AddrBStore.Mode  = AddrModeFlat;
      stackFrame.AddrStack.Offset = currentContext.IntSp;
      stackFrame.AddrStack.Mode   = AddrModeFlat;
    #else
      #error "Unsupported platform"
    #endif

      //
      // Dbghelp is is singlethreaded, so acquire a lock.
      //
      // Note that the code assumes that
      // SymInitialize( GetCurrentProcess(), NULL, TRUE ) has
      // already been called.
      //
      EnterCriticalSection( &m_lock );

      // Structure for line information
      IMAGEHLP_LINE64 line;
      line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

      for ( unsigned int frameNumber = firstFrame ; frameNumber < firstFrame + maxFrames ; frameNumber++ )
      {
        PVOID backTrace;
        USHORT capturedFrames = RtlCaptureStackBackTrace( frameNumber, 1, &backTrace, NULL );
        if ( !capturedFrames || !backTrace )
        {
          break;
        }
        // first check, if that address has already been resolved
        std::map<PVOID,std::string>::const_iterator it = m_resolvedAddresses.find( backTrace );
        if ( it != m_resolvedAddresses.end() )
        {
          if ( it->second.empty() )
          {
            break;
          }
          if ( exclude( it->second, exclusionList ) )
          {
            maxFrames++;
          }
          else
          {
            stackWalk.push_back( it->second );
          }
        }
        else
        {
          // fetch function name
          DWORD64 displacementSymbol = 0;
          if (!SymFromAddr( m_hProcess, reinterpret_cast<DWORD64>(backTrace), &displacementSymbol, m_symbolInfo ))
          {
            // SymFromAddr failed
            DWORD error = GetLastError();
            if ( error == ERROR_MOD_NOT_FOUND )
            {
              if (    ! SymRefreshModuleList( m_hProcess )
                  ||  ! SymFromAddr( m_hProcess, reinterpret_cast<DWORD64>(backTrace), &displacementSymbol, m_symbolInfo ) )
              {
                error = GetLastError();
                m_resolvedAddresses[backTrace] = "";
                break;
              }
            }
            else
            {
              m_resolvedAddresses[backTrace] = "";
              break;
            }
          }
          if ( exclude( m_symbolInfo->Name, exclusionList ) )
          {
            maxFrames++;
          }
          else
          {
            // fetch line
            DWORD displacement2 = 0;
            if ( SymGetLineFromAddr64( m_hProcess, reinterpret_cast<DWORD64>(backTrace), &displacement2, &line ) == TRUE )
            {
              std::ostringstream lineString;
              lineString << m_symbolInfo->Name << ", " << line.FileName << ":" << line.LineNumber;
              stackWalk.push_back( lineString.str() );
            }
            else
            {
              DWORD error = GetLastError();
              stackWalk.push_back( m_symbolInfo->Name );
            }
          }
          m_resolvedAddresses[backTrace] = m_symbolInfo->Name;
        }
      }

      LeaveCriticalSection( &m_lock );

  // defined(WIN32)
  #elif defined(DP_OS_LINUX)
      std::vector<void *> trace( firstFrame + maxFrames );
      int nptrs = ::backtrace( (void **)&trace[0], firstFrame + maxFrames );
      char ** symbols = backtrace_symbols( (void **)&trace[0], nptrs );

      for( unsigned int i = firstFrame; i < firstFrame+maxFrames; i++ )
      {
        if( symbols == nullptr )
        {
          stackWalk.push_back( "error retrieving backtrace" );
        }
        else
        {
          stackWalk.push_back( symbols[i] );
        }
      }

      if( symbols )
      {
        // must free symbols, but not individual pointers
        free( symbols );
      }

      //
      // MMM - note that this only gives us some rudamentary symbol names and
      //       not file and line numbers.  That requires about 100 more lines
      //       of code or so.
      //
  #endif

      return stackWalk;
    }

    std::vector<std::string> backtrace( int skipFrames, unsigned int maxFrames, const std::vector<std::string> & exclusionList )
    {
      static Backtrace trace;
      return trace.backtrace( skipFrames, maxFrames, exclusionList );
    }
  } // namespace util
} // namespace dp
