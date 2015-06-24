// Copyright NVIDIA Corporation 201
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


#include <cstring>
#include <boost/filesystem.hpp>
#include <dp/Assert.h>
#include <dp/util/Config.h>
#include <dp/util/File.h>

namespace dp
{
  namespace util
  {

    // convert between slashes and backslashes in paths
    // depending on the operating system
    void convertPath( std::string & path)
    {
#if defined(DP_OS_WINDOWS)
      std::string::size_type pos=path.find("/",0);
      while ( pos != std::string::npos )
      {
        path[pos]='\\';
        pos=path.find("/",pos);  
      }  
#elif defined(DP_OS_LINUX)
      std::string::size_type pos=path.find("\\",0);
      while(pos!=std::string::npos)
      {
        path[pos]='/';
        pos=path.find("\\",pos);  
      }
#endif
    }

    void convertPath(char* path)
    {  
#if defined(DP_OS_WINDOWS)
      for(size_t i=0; i<strlen(path); i++)
      {
        if(path[i]=='/')
        {
          path[i]='\\';
        }
      }
#elif defined(DP_OS_LINUX)
      for(size_t i=0; i<strlen(path); i++)
      {
        if(path[i]=='\\')
        {
          path[i]='/';
        }
      }
#endif
    }

    bool createDirectory( const std::string& dirpath )
    {
      return( boost::filesystem::create_directory( dirpath ) );
    }

    bool directoryExists( std::string const& dirpath )
    {
      boost::filesystem::path p = dirpath;
      boost::filesystem::file_status fs = boost::filesystem::status( p );
      bool isLink = ( fs.type() == boost::filesystem::reparse_file ) || ( fs.type() == boost::filesystem::symlink_file );
      return( boost::filesystem::is_directory( isLink ? boost::filesystem::read_symlink( p ) : p ) );
    }

    bool fileDelete( std::string const& filepath )
    {
      return( boost::filesystem::remove( filepath ) );
    }

    bool fileExists( std::string const& filepath )
    {
      return( boost::filesystem::exists( filepath ) && boost::filesystem::is_regular_file( filepath ) );
    }

    size_t fileSize( std::string const& filepath )
    {
      return( boost::filesystem::file_size( filepath ) );
    }

    bool findFiles( std::string const& extension, std::string const& path, std::vector<std::string> & results )
    {
      bool found = false;
      DP_ASSERT( boost::filesystem::exists(path) );
      for ( boost::filesystem::directory_iterator it( path ) ; it != boost::filesystem::directory_iterator() ; ++it )
      {
        if ( _stricmp( it->path().extension().string().c_str(), extension.c_str() ) == 0 )
        {
          results.push_back( it->path().string() );
          found = true;
        }
      }
      return( found );
    }

    bool findFiles( std::string const& extension, std::vector<std::string> const& paths, std::vector<std::string> & results )
    {
      for ( std::vector<std::string>::const_iterator it = paths.begin() ; it != paths.end() ; ++it )
      {
        findFiles( extension, *it, results );
      }
      return( results.empty() );
    }

    bool findFilesRecursive( std::string const& extension, std::string const& path, std::vector<std::string> & results )
    {
      bool found = false;
      boost::filesystem::path searchPath( path );
      DP_ASSERT( directoryExists( searchPath.string() ) );
      for ( boost::filesystem::recursive_directory_iterator dirIt( searchPath ) ; dirIt != boost::filesystem::recursive_directory_iterator() ; ++dirIt )
      {
        if ( _stricmp( dirIt->path().extension().string().c_str(), extension.c_str() ) == 0 )
        {
          results.push_back( dirIt->path().string() );
          found = true;
        }
      }
      return( found );
    }

    bool findFilesRecursive( std::string const& extension, std::vector<std::string> const& paths, std::vector<std::string> & results )
    {
      for ( std::vector<std::string>::const_iterator it = paths.begin() ; it != paths.end() ; ++it )
      {
        findFilesRecursive( extension, *it, results );
      }
      return( results.empty() );
    }

    std::string getCurrentPath()
    {
      return( boost::filesystem::current_path().string() );
    }

    std::string getFileExtension( std::string const& filePath )
    {
      return( boost::filesystem::path( filePath ).extension().string() );
    }

    std::string getFileName( std::string const& filePath )
    {
      return( boost::filesystem::path( filePath ).filename().string() );
    }

    std::string getFilePath( std::string const& filePath )
    {
      return( boost::filesystem::path( filePath ).parent_path().string() );
    }

    std::string getFileStem( std::string const& filePath )
    {
      return( boost::filesystem::path( filePath ).stem().string() );
    }

#if defined(DP_OS_WINDOWS)
    std::string getModulePath( std::string const& module )
    {
      return( dp::util::getModulePath( GetModuleHandle( module.c_str() ) ) );
    }

    std::string getModulePath()
    {
      return( dp::util::getModulePath( GetModuleHandle( NULL ) ) );
    }

    std::string getModulePath( HMODULE const hModule )
    {
      char _path[_MAX_PATH];
      if ( GetModuleFileName( hModule, _path, sizeof(_path) ) )
      {
        return( dp::util::getFilePath( _path ) );
      } 
      return( "" );
    }
#else 
#ifndef _MAX_PATH
#define _MAX_PATH 2048
#endif
    std::string getModulePath()
    {
      char linkname[64];
      int ret;
      char buf[_MAX_PATH];

      if( snprintf(linkname, sizeof(linkname), "/proc/%i/exe", getpid()) < 0)
      {
        return( "" );
      }

      ret = readlink(linkname, buf, _MAX_PATH);
      if ( ret != -1 )
      {
        buf[ret]=0;
        return( dp::util::getFilePath( buf ) );
      }
      return( "" );
    }
#endif

    bool isAbsolutePath( std::string const& path )
    {
      return( boost::filesystem::path( path ).is_absolute() );
    }

    std::string makePathRelative( std::string const & filePath, std::vector<std::string> const& basePaths )
    {
      // for each path in basePaths, check if the it equals the start of filePath
      // if so, cut that part off of filePath and return the resulting relative path
      boost::filesystem::path fp(filePath);
      for ( size_t i=0 ; i<basePaths.size() ; i++ )
      {
        // iterate over both filePath and basePath in parallel, until they differ
        boost::filesystem::path bp(basePaths[i]);
        boost::filesystem::path::iterator fpit, bpit;
        for ( fpit = fp.begin(), bpit = bp.begin() ; fpit != fp.end() && bpit != bp.end() && *fpit == *bpit ; ++fpit, ++bpit )
          ;

        // the basePaths equals the first parts of filePath iff the current element of basePath is "."
        DP_ASSERT( ( fpit != fp.end() ) && ( bpit != fp.end() ) );
        if ( bpit->string() == "." )
        {
          DP_ASSERT( ++bpit == bp.end() );

          // gather all the remaining parts of filePath into the relative path
          boost::filesystem::path rp;
          for ( ; fpit != fp.end() ; ++fpit )
          {
            rp /= *fpit;
          }
          return( rp.string() );
        }
      }
      return( filePath );
    }

    std::string replaceFileExtension( std::string const& filePath, std::string const& extension )
    {
      return( boost::filesystem::path( filePath ).replace_extension( extension ).string() );
    }

    std::string loadStringFromFile( const std::string& filename )
    {
      std::string result;
      FILE *file = fopen( filename.c_str(), "rb" );
      if ( file )
      {
        fseek( file, 0, SEEK_END );
        long int size = ftell( file );
        DP_ASSERT( size >= 0 );
        if( size >= 0 )
        {
          rewind( file );
          result.resize( size );
          fread( (char *) &result[0], 1, size, file );
        }
        fclose( file );
      }
      else
      {
        DP_ASSERT( !"failed to open file" );
      }
      return result;
    }

    bool saveStringToFile( const std::string& filename, const std::string& data )
    {
      FILE *file = fopen( filename.c_str(), "wb" );
      if ( file )
      {
        if ( data.size() )
        {
          fwrite( data.c_str(), data.size(), 1, file );
        }
        fclose( file );
        return true;
      }
      return false;
    }

  } // namespace util
} // namespace dp

