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


#include <dp/util/PlugIn.h>
#include <dp/util/Tokenizer.h>
#include <dp/util/File.h>

#if !defined( NDEBUG )
#include <stdio.h>
#if defined( MSVC )
#include <strsafe.h>
#endif
#endif

#include <cstring>

using std::make_pair;
using std::unique;
using std::vector;
using std::string;

namespace dp
{
  namespace util
  {
    UPITID::UPITID(unsigned short pit, const char verStr[16])
    : data0(pit)
    {
      strncpy(data1, verStr, 15);
      data1[15] = '\0'; // terminate
    }

    UPITID::UPITID(const UPITID& rhs) 
    {
      memcpy(this, &rhs, sizeof(UPITID));
    }

    UPIID::UPIID(const char idstr[16], UPITID pitid) 
    : data1(pitid)
    {
      strncpy(data0, idstr, 15);
      data0[15] = '\0'; // terminate
    }

    UPIID& UPIID::operator=(const UPIID& rhs)
    {
      if (&rhs != this)
      {
        memcpy(this, &rhs, sizeof(UPIID));
      }
      return *this;
    }

    PlugInServer::PlugInServer()
    :m_filter(LibExtStr)
    {
    }

    PlugInServer::~PlugInServer()
    {
      // release all PlugIns
      m_plugIns.clear();
    }

    void PlugInServer::gatherPlugIns( std::vector<std::string> const& searchPaths )
    {
      std::vector<std::string> plugInFiles;
      findPlugIns( searchPaths, plugInFiles );

      for ( std::vector<std::string>::const_iterator it = plugInFiles.begin() ; it != plugInFiles.end() ; ++it )
      {
        HLIB lib = MapLibrary( it->c_str() );

        if ( lib ) 
        {
          PFNQUERYPLUGINTERFACEPIIDS pfnQueryPlugInterfacePIIDs = (PFNQUERYPLUGINTERFACEPIIDS)GetFuncAddress(lib, "queryPlugInterfacePIIDs");
          if ( pfnQueryPlugInterfacePIIDs ) 
          {
            std::vector<UPIID> upiids;
            (*pfnQueryPlugInterfacePIIDs)(upiids);

            for ( std::vector<UPIID>::const_iterator upiidit = upiids.begin() ; upiidit != upiids.end() ; ++upiidit )
            {
              m_plugIns[*upiidit].fileName = *it;
            }
          }
          UnMapLibrary(lib);
        }
#if !defined( NDEBUG )
        else
        {
          std::stringstream errorStream;
#if defined( _WIN32 )
          DWORD dw = GetLastError(); 

          // If dwFlags includes FORMAT_MESSAGE_ALLOCATE_BUFFER, the function 
          // allocates a buffer using the LocalAlloc function, and places the
          // pointer to the buffer at the address specified in lpBuffer.
          LPVOID lpMsgBuf = NULL;
          FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &lpMsgBuf, 0, NULL );

          errorStream << "MapLibrary( " << *it << " ) failed with error " << dw << ": " << (LPCTSTR)lpMsgBuf;
          LocalFree( lpMsgBuf );
#elif defined(LINUX)
          errorStream << "MapLibrary failed with error: " << dlerror();
#endif // PLATFORM

          fprintf(stderr, "%s\n", errorStream.str().c_str() );
          DP_ASSERT( false && "MapLibrary failed." );  
        }
#endif // NDEBUG
      }
    }

    bool PlugInServer::getInterfaceImpl(const vector<string>& searchPath, const UPIID& piid, dp::util::SmartPtr<PlugIn> & plugIn)
    {
      plugIn.reset();
      if ( m_unsupportedUPIIDs.find( piid ) == m_unsupportedUPIIDs.end() )
      {
        PlugInMap::iterator plugInMapIt = m_plugIns.find( piid );
        if ( plugInMapIt == m_plugIns.end() )
        {
          gatherPlugIns( searchPath );
          plugInMapIt = m_plugIns.find( piid );
        }
        if ( plugInMapIt != m_plugIns.end() )
        {
          if ( plugInMapIt->second.plugIn )
          {
            plugIn = plugInMapIt->second.plugIn;
          }
          else
          {
            HLIB lib = MapLibrary( plugInMapIt->second.fileName.c_str() );
            if ( lib ) 
            {
              PFNGETPLUGINTERFACE pfnGetPlugInterface = (PFNGETPLUGINTERFACE)GetFuncAddress( lib, "getPlugInterface" );
              if ( pfnGetPlugInterface ) 
              {
                if ( (*pfnGetPlugInterface)( piid, plugIn ) )
                {
                  // keep the plug-in for later access
                  plugInMapIt->second.plugIn = plugIn;
                }
              }
              else
              {
                // free unused
                UnMapLibrary( lib );
              }
            }
          }

          if ( ! plugIn )
          {
            m_unsupportedUPIIDs.insert( piid );
          }
        }
      }
      return( !!plugIn );
    }

    bool PlugInServer::queryInterfaceTypeImpl(const vector<string>& searchPath, const UPITID& pitid, vector<UPIID>& piids)
    {
      piids.clear();

      // TODO actually calling function needs to call gather plugins each time a new search path has been added...
      if ( m_plugIns.empty() )
      {
        gatherPlugIns( searchPath );
      }

      for ( PlugInMap::iterator it = m_plugIns.begin(); it != m_plugIns.end(); ++it )
      {
        if ( it->first.getPlugInterfaceType() == pitid )
        {
          piids.push_back(it->first);
        }
      }
      return( !piids.empty() );
    }

    void PlugInServer::releaseInterfaceImpl(const UPIID& piid)
    {
      PlugInMap::iterator it = m_plugIns.find( piid );
      if ( it != m_plugIns.end() )
      {
        it->second.plugIn.reset();
      }
    }

    /* set file filter specified by filter */
    void PlugInServer::setFileFilterImpl(const std::string& filter)
    {
      if ( !filter.empty() ) 
      {
        m_filter = filter;
      }
    }

    bool PlugInServer::findPlugIns(const vector<string>& searchPath, vector<string>& plugIns)
    {
      plugIns.clear();
      vector<string> _searchPath(searchPath);                 //temporary search path variable

      for ( std::vector<std::string>::const_iterator it = getPlugInSearchPath().begin() ; it != getPlugInSearchPath().end() ; ++it )
      {
        if ( std::find( _searchPath.begin(), _searchPath.end(), *it ) == _searchPath.end() )
        {
          _searchPath.push_back( *it );
        }
      }

      StrTokenizer tok(";");
      tok.setInput( m_filter );
        
      while ( tok.hasMoreTokens() )
      {
        findFiles( tok.getNextToken().substr( 1 ), _searchPath, plugIns );
      }

      return !plugIns.empty();
    }

    bool getInterface(const std::vector<std::string>& searchPath, const UPIID& piid, dp::util::SmartPtr<PlugIn> & plugIn)
    {// call the singleton's implementation
      return PIS::instance()->getInterfaceImpl(searchPath, piid, plugIn);
    }

    bool queryInterfaceType(const std::vector<std::string>& searchPath, const UPITID& pitid, std::vector<UPIID>& piids)
    {// call the singleton's implementation
      return PIS::instance()->queryInterfaceTypeImpl(searchPath, pitid, piids);
    }

    /* release the interface identified by piid */
    void releaseInterface(const UPIID& piid)
    {// call the singleton's implementation
      PIS::instance()->releaseInterfaceImpl(piid);
    }

    /* set file filter specified by filter */
    void setPlugInFileFilter(const std::string& filter)
    {// call the singleton's implementation
      PIS::instance()->setFileFilterImpl(filter);
    }

    void addPlugInSearchPath(const std::string& path)
    {
      PIS::instance()->m_searchPath.push_back(path); 
      DP_ASSERT(PIS::instance()->m_searchPath.size()>0);
    }

    const std::vector<string>& getPlugInSearchPath()
    {  
      DP_ASSERT(PIS::instance()->m_searchPath.size()>0);
      return PIS::instance()->m_searchPath;  
    }
  } // namespace util
} // namespace dp
