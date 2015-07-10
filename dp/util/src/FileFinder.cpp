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


#include <dp/util/FileFinder.h>
#include <dp/Assert.h>
#include <dp/util/File.h>

namespace dp
{
  namespace util
  {
    void chopOffFirstElement( boost::filesystem::path & fileName )
    {
      boost::filesystem::path reducedPath;
      boost::filesystem::path::iterator pit = fileName.begin();
      for ( ++pit ; pit != fileName.end() ; ++pit )
      {
        reducedPath /= *pit;
      }
      fileName = reducedPath;
    }

    FileFinder::FileFinder()
    {
    }

    FileFinder::FileFinder( std::string const& path )
    {
      DP_VERIFY( addSearchPath( path ) );
    }

    FileFinder::FileFinder( std::vector<std::string> const& paths )
    {
      addSearchPaths( paths );
    }

    bool FileFinder::addSearchPath( std::string const& path )
    {
      bool result = false;
      boost::filesystem::path boostPath(path);
      if (boost::filesystem::exists(boostPath))
      {
        result = m_searchPaths.insert(boostPath).second;
      }
      return result;
    }

    void FileFinder::addSearchPaths( std::vector<std::string> const& paths )
    {
      for ( std::vector<std::string>::const_iterator it = paths.begin() ; it != paths.end() ; ++it )
      {
        addSearchPath( *it );
      }
    }

    void FileFinder::clear()
    {
      m_latestHit.clear();
      m_searchPaths.clear();
    }

    std::string FileFinder::find( std::string const& file ) const
    {
      boost::filesystem::path filePath( file );
      if ( boost::filesystem::is_regular_file( filePath ) )
      {
        return( file );
      }
      while ( !filePath.empty() )
      {
        for ( std::set<boost::filesystem::path>::const_iterator pit = m_searchPaths.begin() ; pit != m_searchPaths.end() ; ++pit )
        {
          boost::filesystem::path dirPath( *pit );
          dirPath /= filePath;
          if ( boost::filesystem::is_regular_file( dirPath ) )
          {
            return( dirPath.string() );
          }
        }
        chopOffFirstElement( filePath );
      }
      return( "" );
    }

    std::string FileFinder::findRecursive( std::string const& file ) const
    {
      boost::filesystem::path filePath( file );
      if ( boost::filesystem::is_regular_file( filePath ) )
      {
        return( filePath.string() );
      }
      while ( !filePath.empty() )
      {
        if ( !m_latestHit.empty() )
        {
          boost::filesystem::path dirPath( m_latestHit );
          dirPath /= filePath.filename();
          if ( boost::filesystem::is_regular_file( dirPath ) )
          {
            return( dirPath.string() );
          }
        }

        for ( std::set<boost::filesystem::path>::const_iterator pit = m_searchPaths.begin() ; pit != m_searchPaths.end() ; ++pit )
        {
          DP_ASSERT( boost::filesystem::exists( *pit ) );
          boost::filesystem::path dirPath( *pit );
          dirPath /= filePath;
          if ( boost::filesystem::is_regular_file( dirPath ) )
          {
            m_latestHit = *pit;
            return( dirPath.string() );
          }
          for ( boost::filesystem::recursive_directory_iterator it( *pit ) ; it != boost::filesystem::recursive_directory_iterator() ; ++it )
          {
            dirPath = it->path();
            // checking is_regular_file only if is_directory(*pit) is substantially slower !!
            dirPath /= filePath;
            if ( boost::filesystem::is_regular_file( dirPath ) )
            {
              m_latestHit = it->path();
              return( dirPath.string() );
            }
          }
        }
        chopOffFirstElement( filePath );
      }
      return( "" );
    }

    std::vector<std::string> FileFinder::getSearchPaths() const
    {
      std::vector<std::string> paths;
      paths.reserve( m_searchPaths.size() );
      for ( std::set<boost::filesystem::path>::const_iterator it = m_searchPaths.begin() ; it != m_searchPaths.end() ; ++it )
      {
        paths.push_back( it->string() );
      }
      return( std::move( paths ) );
    }

    bool FileFinder::removeSearchPath( std::string const& path )
    {
      std::set<boost::filesystem::path>::iterator it = m_searchPaths.find( boost::filesystem::path( path ) );
      bool found = ( it != m_searchPaths.end() );
      if ( found && ( m_latestHit == *it ) )
      {
        m_latestHit.clear();
      }
      return( found );
    }

  } // namespace util
} // namespace dp

