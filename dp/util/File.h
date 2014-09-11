// Copyright NVIDIA Corporation 2011
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

#include <dp/util/Config.h>
#if defined (DP_OS_WINDOWS)
# include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#include <vector>
#include <string>
#include <sstream>

#include <stdio.h>
#include <iomanip>

namespace dp
{
  namespace util
  {

    template <class T>
    std::string to_string (const T& t, const unsigned int w = 1)
    {
      std::stringstream ss;
      ss << std::setw( w ) << std::setfill( '0' ) << t;
      return ss.str();
    }

    //! Create a directory
    DP_UTIL_API bool createDirectory( const std::string& dirpath );

    //! Check if a given directory exists
    DP_UTIL_API bool directoryExists( const std::string& dirpath );

    //! Delete a file
    DP_UTIL_API bool fileDelete( const std::string& filepath );

    //! Check if a given file exists
    DP_UTIL_API bool fileExists( const std::string& filepath );

    //! Return filesize in bytes of an existing file. Supports filesizes up to 2GB only.
    DP_UTIL_API size_t fileSize( const std::string& filepath );

    DP_UTIL_API std::string findFile( std::string const& name, std::vector<std::string> const& paths );

    DP_UTIL_API std::string findFileRecursive( std::string const& name, std::vector<std::string> const& paths );

    DP_UTIL_API bool findFiles( std::string const& extension, std::string const& path, std::vector<std::string> & results );

    DP_UTIL_API bool findFiles( std::string const& extension, std::vector<std::string> const& paths, std::vector<std::string> & results );

    DP_UTIL_API bool findFilesRecursive( std::string const& extension, std::string const& path, std::vector<std::string> & results );

    DP_UTIL_API bool findFilesRecursive( std::string const& extension, std::vector<std::string> const& paths, std::vector<std::string> & results );

    //! Get the current path for the current process
    DP_UTIL_API std::string getCurrentPath();

    //! Extract file extension from a path
    DP_UTIL_API std::string getFileExtension( const std::string& filePath );

    //! Extract filename from a path
    DP_UTIL_API std::string getFileName( const std::string& filePath );

    //! Extract file path
    DP_UTIL_API std::string getFilePath( const std::string& filePath );

    DP_UTIL_API std::string getFileStem( const std::string& filePath );

#if defined(DP_OS_WINDOWS)
    //! Determine the module path **Windows only**
    /** Determines the full path to the module, given the module's name in \a module and returns it in \a path. */
    DP_UTIL_API std::string getModulePath( std::string const& module );

    //! Determine the module path **Windows only**
    /** Determines the full path of the module, given the module handle in \a hModule and returns it in \a path. */
    DP_UTIL_API std::string getModulePath( HMODULE const hModule );
#endif 

    //! Determine the executable module path **Cross Platform**
    /** Determines the full path to the current executable and returns it in \a path. */
    DP_UTIL_API std::string getModulePath();

    //! \brief Check if a path is absolute
    DP_UTIL_API bool isAbsolutePath( std::string const& path );

    /*! \brief Make an absolute path a relative one
     *  \param filePath The absolute path to make relative
     *  \param basePaths A vector of paths to make \a filePath relative to
     *  \remarks \a filePath is checked against the \a basePaths. The first base path that is the
     *  beginning of \a filePath is subtracted from \a filePath, making it relative to that path. */
    DP_UTIL_API std::string makePathRelative( std::string const& filePath, std::vector<std::string> const& basePaths );

    DP_UTIL_API std::string replaceFileExtension( std::string const& filePath, std::string const& extension );

    /*! \brief Load a file from disk.
        \param filename filename to load from disk
        \return std::string with content of file if it was found.
    **/
    DP_UTIL_API std::string loadStringFromFile( const std::string& filename );

    /*! \brief Write a string to a file on disk.
        \param filename Filename to write the data to. An already existing file will be overwritten.
        \param data The string which contents are written to disk. An empty string will result in an empty file.
        \returns \c true if the file could be created, \c false if not.
    **/
    DP_UTIL_API bool saveStringToFile( const std::string& filename, const std::string& data );

    /*! \brief convert between slashes and backslashes in paths depending on the operating system
        \param path The path to convert
    **/
    DP_UTIL_API void convertPath( std::string& path );

  } // namespace util
} // namespace dp


