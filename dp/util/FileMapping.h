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

#include <dp/util/Config.h>
#include <map>
#include <list>
#include <string>
#include <cstddef>

#if defined(LINUX)
# include <errno.h>
#endif

#if defined(DP_OS_WINDOWS)
#include <windows.h>
#endif

namespace dp
{
  namespace util
  {
    /*! \brief Helper class to ease efficient reading and writing of large files.
     *  \remarks This is the base class for class ReadMapping and class WriteMapping, that can be used
     *  to hide platform dependencies on reading or writing. This is done by selectively mapping parts
     *  of a file in and out.
     *  \note This class is not intended to be directly instantiated.
     *  \sa ReadMapping, WriteMapping */
    class FileMapping
    {
      public :
        /*! \brief Get the last error.
         *  \return An platform dependent unsigned int that describes the last error on using a mapped
         *  file.
         *  \par Example:
         *  const void * ptr = rm->mapIn( offset, count*sizeof(int) );
         *  if ( ! ptr )
         *  {
         *    unsigned int errCode = rm->getLastError();
         *    // ...
         *  }
         *  \code
         *  \endcode */
        DP_UTIL_API unsigned int getLastError() const;

        /*! \brief Test if a FileMapping is valid.
         *  \return \c true, if the FileMapping is currently valid, otherwise \c false.
         *  \remarks Before using a FileMapping, you should test for its validity.
         *  \par Example:
         *  \code
         *    ReadMapping rm = new ReadMapping( fileName );
         *    if ( rm->isValid() )
         *    {...}
         *  \endcode */
        DP_UTIL_API bool isValid() const;

        /*! \brief Maps out a previously mapped in part of a file.
         *  \param offsetPtr The constant pointer to void that was previously returned by a call to
         *  mapIn.
         *  \par Example:
         *  \code
         *  const void * ptr = rm->mapIn( offset, count*sizeof(int) );
         *  if ( ptr )
         *  {
         *    // ...
         *    rm->mapOut( ptr );
         *  }
         *  \endcode
         *  \sa mapIn */
        DP_UTIL_API void mapOut( const void * offsetPtr );

      protected:
        /*! \brief Protected default constructor.
         *  \remarks The constructor of FileMapping is protected to prevent explicit instantiation.
         *  \sa ReadMapping, WriteMapping */
        DP_UTIL_API FileMapping();

        /*! \brief Protected destructor.
         *  \remarks The destructor of FileMapping is protected to prevent explicit instantiation.
         *  \sa ReadMapping, WriteMapping */
        DP_UTIL_API ~FileMapping();

        /*! \brief Protected base function to map in part of the file.
         *  \param offset The offset that has to be part of the mapping.
         *  \param numBytes The number of bytes that, starting from \a offset, are to be part of
         *  the mapping.
         *  \return A pointer to the mapped memory location, or NULL if the mapping failed.
         *  \sa mapOut */
        DP_UTIL_API void * mapIn( size_t offset, size_t numBytes );

      private:
        struct ViewHeader
        {
          ViewHeader( size_t start, size_t size, void * base )
            : basePtr(base)
            , startOffset(start)
            , endOffset(start+size)
            , refCnt(0)
          {}

          void  * basePtr;      // points to the beginning of the view
          size_t  startOffset;  // the view's start offset inside the mapped file 
          size_t  endOffset;    // the view's 'past the end' offset inside the mapped file
          int     refCnt;       // reflects if the view is in use or not
        };

      protected:
        unsigned int                m_accessType;       //!< read-only or read/write
  #if defined(_WIN32)
        HANDLE                      m_file;             //!< file handle returned by CreateFile
        HANDLE                      m_fileMapping;      //!< handle of file mapping object returned by CreateFileMapping
  #elif defined(LINUX)
        int                         m_file;             //!< file descriptor returned by open()
  #endif
        size_t                      m_mappingSize;      //!< actual size of the file mapping
        bool                        m_isValid;          //!< file mapping is valid

      private :
        typedef std::map<const void *,std::pair<unsigned int,ViewHeader*> > OffsetPtrToCountViewHeaderMap;
        typedef std::list<ViewHeader*>                                      ViewHeaderList;

        ViewHeaderList                m_mappedViews;                    // collection of currently mapped views
        ViewHeaderList                m_unmappedViews;                  // collection of mapped views currently not used
        OffsetPtrToCountViewHeaderMap m_offsetPtrToCountViewHeaderMap;  // mapping offset pointers to number of mappings of that offset and their associated mapped view
    };

    /*! \brief Helper class to ease efficient reading of large files.
     *  \sa FileMapping */
    class ReadMapping : public FileMapping
    {
      public :
        /*! \brief Constructor using the name of the file to read.
         *  \param fileName The name of the file to read.
         *  \note If the file \a fileName does not exist, the resulting ReadMapping is marked as
         *  invalid. Only valid ReadMappings can be used for reading.
         *  \par Example:
         *  \code
         *    ReadMapping rm = new ReadMapping( fileName );
         *    if ( rm->isValid() )
         *    {...}
         *  \endcode */
        DP_UTIL_API ReadMapping( const std::string & fileName );

        /*! \brief Destructor of a read-only mapping.
         *  \remarks If the ReadMapping is valid, the opened file is closed. */
        DP_UTIL_API ~ReadMapping();

        /*! \brief Map a view of the file.
         *  \param offset The offset that has to be part of the mapping.
         *  \param numBytes The number of bytes that, starting from \a offset, are to be part of
         *  the mapping.
         *  \return A pointer to the constant mapped memory location, or NULL if the mapping failed.
         *  \sa mapOut */
        DP_UTIL_API const void * mapIn( size_t offset, size_t numBytes );
    };

    /*! \brief Helper class to ease efficient writing of large files.
     *  \sa FileMapping */
    class WriteMapping : public FileMapping
    {
      public:
        /*! \brief Constructor using the name of the file to write.
         *  \param fileName The name of the file to write.
         *  \param fileSize The size of the file to write.
         *  \remarks If there is enough free space on the disk specified by \a fileName, a file of
         *  that with the name \a fileName is created.
         *  \par Example:
         *  \code
         *    WriteMapping wm = new WriteMapping( fileName, preCalculatedFileSize );
         *    if ( wm->isValid() )
         *    {...}
         *  \endcode */
        DP_UTIL_API WriteMapping( const std::string & fileName, size_t fileSize );

        /*! \brief Destructor of a writable mapping.
        *  \remarks If the WriteMapping is valid, the written file is closed. */
        DP_UTIL_API ~WriteMapping();

        /*! \brief Map a view of the file.
         *  \param offset The offset that has to be part of the mapping.
         *  \param numBytes The number of bytes that, starting from \a offset, are to be part of
         *  the mapping.
         *  \return A pointer to the mapped memory location, or NULL if the mapping failed.
         *  \sa mapOut */
        DP_UTIL_API void * mapIn( size_t offset, size_t numBytes );

      private :
        size_t  m_endOffset;    // last accessed offset in file => eof
    };


    inline unsigned int FileMapping::getLastError() const
    {
  #if defined(_WIN32)
      return( GetLastError() );
  #elif defined(LINUX)
      return( errno );
  #else
      DP_STATIC_ASSERT( false );
  #endif
    }

    inline bool FileMapping::isValid() const
    {
      return( m_isValid );
    }


    inline const void * ReadMapping::mapIn( size_t offset, size_t numBytes )
    {
      return( FileMapping::mapIn( offset, numBytes ) );
    }
  } // namespace util
} // namespace dp
