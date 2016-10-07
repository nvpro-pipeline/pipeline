// Copyright NVIDIA Corporation 2002-2015
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


#include <dp/Assert.h>
#include <dp/util/FileMapping.h>
#include <algorithm>

#if defined(LINUX)
# include <sys/types.h>
# include <sys/stat.h>
# include <unistd.h>
# include <fcntl.h>
# include <sys/mman.h>
# include <sys/resource.h>
#endif

using std::find;
using std::list;
using std::make_pair;
using std::map;
using std::string;

namespace dp
{
  namespace util
  {
    const size_t MAX_VIEW_SIZE = 0x01000000;
    static size_t gPageSize = 0;
    const size_t MAX_UNMAPPED_VIEWS = 16;

  #if defined(_WIN32)
    // note: the __int64 cast is necessary for 32-bit platforms as the result of the right shift
    // is undefined if the right operand is greater than or equal to the number of bits in the 
    // (promoted) left operand.
    inline DWORD HIDWORD(size_t x) { return (DWORD)((__int64)x>>32); }
    inline DWORD LODWORD(size_t x) { return (DWORD)x; }
  #endif

    FileMapping::FileMapping()
      : m_isValid(false)
      , m_mappingSize(0)
  #if defined(_WIN32)
      , m_accessType(FILE_MAP_READ)
      , m_file(INVALID_HANDLE_VALUE)
      , m_fileMapping(NULL)
  #elif defined(LINUX)
      , m_accessType(PROT_READ)
      , m_file(-1)
  #endif
    {
      if ( gPageSize == 0 )
      {
  #if defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        gPageSize = (size_t)si.dwAllocationGranularity;
  #elif defined(LINUX)
        gPageSize = (size_t)getpagesize();
  #else
        DP_STATIC_ASSERT( false );
  #endif
      }
    }

    FileMapping::~FileMapping()
    {
      DP_ASSERT( m_offsetPtrToCountViewHeaderMap.empty() ); // there should be no offsets left mapped...
      DP_ASSERT( m_mappedViews.empty() );                   // ... and hence, all mapped views should have been unmapped
      for ( ViewHeaderList::const_iterator it = m_unmappedViews.begin() ; it != m_unmappedViews.end() ; ++it )
      {
        DP_ASSERT( (*it)->refCnt == 0 );
  #if defined(_WIN32)
        UnmapViewOfFile( (*it)->basePtr );
  #elif defined(LINUX)
        munmap( (*it)->basePtr, (*it)->endOffset - (*it)->startOffset );
  #else
        DP_STATIC_ASSERT( false );
  #endif
        delete *it;
      }
    }

    void * FileMapping::mapIn( size_t offset, size_t numBytes )
    {
      DP_ASSERT( m_isValid );
      DP_ASSERT( offset + numBytes <= m_mappingSize );

      ViewHeader * vh = NULL;
      void * offsetPtr = NULL;

      //  First, search for a ViewHeader that fits around offset and numBytes
      for ( ViewHeaderList::iterator vhit=m_mappedViews.begin() ; vhit!=m_mappedViews.end() ; ++vhit )
      {
        if (    ( (*vhit)->startOffset <= offset )
            &&  ( ( offset + numBytes ) <= (*vhit)->endOffset ) )
        {
          vh = *vhit;
          break;
        }
      }
      // Not found in the mapped views -> look in the 'unmapped'
      if ( !vh )
      {
        for ( ViewHeaderList::iterator vhit = m_unmappedViews.begin() ; vhit != m_unmappedViews.end() ; ++vhit )
        {
          if (    ( (*vhit)->startOffset <= offset )
              &&  ( ( offset + numBytes ) <= (*vhit)->endOffset ) )
          {
            vh = *vhit;
            // found it in an unmapped view -> move that to the mapped views
            m_mappedViews.push_front( vh );
            m_unmappedViews.erase( vhit );
            break;
          }
        }
      }

      if ( vh )
      {
        //  the ViewHeader around offset and numBytes was found => calculate offsetPtr
        offsetPtr = (char *) vh->basePtr + ( offset - vh->startOffset );
        OffsetPtrToCountViewHeaderMap::iterator it = m_offsetPtrToCountViewHeaderMap.find( offsetPtr );
        if ( it == m_offsetPtrToCountViewHeaderMap.end() )
        {
          //  only if this offsetPtr wasn't used before, vh has to be added to the map
          m_offsetPtrToCountViewHeaderMap[offsetPtr] = make_pair( 1, vh );
        }
        else
        {
          //  otherwise assert that offsetPtr is mapped to the ViewHeader we have found
          DP_ASSERT( it->second.second == vh );
          it->second.first++;
        }
        vh->refCnt++;
      }
      else
      {
        //  Create a new ViewHeader for this offset
        //  NOTE: start offset must be a multiple of the systems allocation granularity!
        size_t startOffset = ( offset / gPageSize ) * gPageSize;
        size_t viewSize = std::min( std::max( MAX_VIEW_SIZE, ( offset - startOffset ) + numBytes )
                                  , m_mappingSize - startOffset );
  #if defined(_WIN32)
        void * basePtr = MapViewOfFile( m_fileMapping, m_accessType, HIDWORD(startOffset), LODWORD(startOffset), (SIZE_T)viewSize );
  #elif defined(LINUX)
        void * basePtr = mmap( 0, viewSize, m_accessType, MAP_SHARED, m_file, startOffset );
  #else
        DP_STATIC_ASSERT( false );
  #endif
        DP_ASSERT( basePtr );

        vh = new ViewHeader( startOffset, viewSize, basePtr );
        m_mappedViews.push_front( vh );

        offsetPtr = (char *) vh->basePtr + ( offset - vh->startOffset );

        //  We want to have only one ViewHeader per offsetPtr, so handle the case when there's
        //  already one registered.
        OffsetPtrToCountViewHeaderMap::iterator it = m_offsetPtrToCountViewHeaderMap.find( offsetPtr );
        if ( it == m_offsetPtrToCountViewHeaderMap.end() )
        {
          //  only if this offsetPtr wasn't used before, vh has to be added to the map
          m_offsetPtrToCountViewHeaderMap[offsetPtr] = make_pair( 1, vh );
        }
        else
        {
          //  There is already a view registered for this offsetPtr 

          // the offset gets assigned this new view. 
          // the new view then is referenced 'm_offsetPtrToCountViewHeaderMap[offsetPtr].first' times
          vh->refCnt = it->second.first;
          it->second.second->refCnt -= it->second.first;

          if ( it->second.second->basePtr == vh->basePtr )
          {
            // same base address!
            // an unmap would destroy both, the old and the new view!
            if ( it->second.second->refCnt == 0 )
            {
              // no unmap! just delete the view header
              delete it->second.second;
            }
          }
          else
          {
            // different base addresses!
            // unmap the old view is safe!
            it->second.second->refCnt++; // avoid negative ref count! will be decremented again in mapOut below!
            mapOut( offsetPtr );
          }
          it->second.second = vh;
          it->second.first++;
        }
        vh->refCnt++; // the view is referenced one more times
      }

      return( offsetPtr );
    }

    void FileMapping::mapOut( const void * offsetPtr )
    {
      DP_ASSERT( m_isValid );

      OffsetPtrToCountViewHeaderMap::iterator it = m_offsetPtrToCountViewHeaderMap.find( (void*)offsetPtr );
      DP_ASSERT( it != m_offsetPtrToCountViewHeaderMap.end() );

      ViewHeader * vh = it->second.second;
      DP_ASSERT( 0 < vh->refCnt );
      vh->refCnt--;
      if ( vh->refCnt == 0 )
      {
        DP_ASSERT( it->second.first == 1 );
        ViewHeaderList::iterator vhit = find( m_mappedViews.begin(), m_mappedViews.end(), vh );
        DP_ASSERT( vhit != m_mappedViews.end() );

        if ( MAX_UNMAPPED_VIEWS <= m_unmappedViews.size() )
        {
          ViewHeader * unmappedVH = m_unmappedViews.front();
  #if defined(_WIN32)
          UnmapViewOfFile( unmappedVH->basePtr );
  #elif defined(LINUX)
          munmap( unmappedVH->basePtr, unmappedVH->endOffset - unmappedVH->startOffset );
  #else
          DP_STATIC_ASSERT( false );
  #endif
          delete unmappedVH;
          m_unmappedViews.pop_front();
        }
        m_unmappedViews.push_back( vh );
        m_mappedViews.erase( vhit );
      }

      it->second.first--;
      if ( it->second.first == 0 )
      {
        // this offset does not reference the view anymore
        m_offsetPtrToCountViewHeaderMap.erase( it );
      }
    }

    ReadMapping::ReadMapping( const string & fileName )
    {
      DP_ASSERT( !fileName.empty() );

  #if defined(_WIN32)
      m_file = CreateFile( fileName.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING
                         , FILE_ATTRIBUTE_READONLY, NULL );
      m_isValid = ( m_file != INVALID_HANDLE_VALUE );
      if ( m_isValid )
      {
        m_fileMapping = CreateFileMapping( m_file, NULL, PAGE_READONLY, 0, 0, NULL );
        m_isValid = ( m_fileMapping != NULL );
        if ( m_isValid )
        {
          DWORD sizeHi = 0;
          DWORD sizeLo = GetFileSize( m_file, &sizeHi );
          m_mappingSize = (static_cast<size_t>(sizeHi) << 32) | sizeLo;
        }
        else
        {
          CloseHandle( m_file );
        }
      }
  #elif defined(LINUX)
      m_file = open( fileName.c_str(), O_RDONLY );
      m_isValid = ( m_file != -1 );
      if ( m_isValid )
      {
        struct stat s;
        fstat( m_file, &s );

        //  NOTE: On Linux the mapping works on the file object directly (there is no mapping and view).
        m_mappingSize = s.st_size;
      }
  #else
      DP_STATIC_ASSERT( false );
  #endif
    }

    ReadMapping::~ReadMapping()
    {
      if ( m_isValid )
      {
  #if defined(_WIN32)
        DP_ASSERT( ( m_file != INVALID_HANDLE_VALUE ) && ( m_fileMapping != NULL ) );
        CloseHandle( m_fileMapping );
        CloseHandle( m_file );
  #elif defined(LINUX)
        DP_ASSERT( m_file != -1 );
        close( m_file );
  #else
        DP_STATIC_ASSERT( false );
  #endif
      }
    }

    WriteMapping::WriteMapping( const string &fileName, size_t fileSize )
      : m_endOffset(0)
    {
      DP_ASSERT( !fileName.empty() && ( 0 < fileSize ) );

      //  fileSize has to be a multiple of the page size
      m_mappingSize = ( 1 + fileSize / gPageSize ) * gPageSize;

      // check if the current process is allowed to save a file of that size
  #if defined(_WIN32)
      TCHAR dir[MAX_PATH+1];
      BOOL success = FALSE;
      ULARGE_INTEGER numFreeBytes;

      DWORD length = GetVolumePathName(fileName.c_str(), dir , MAX_PATH+1);

      if ( length > 0 )
      {
        success = GetDiskFreeSpaceEx( dir, NULL, NULL, &numFreeBytes );
      }

      m_isValid = (!!success) && ( m_mappingSize <= numFreeBytes.QuadPart );

      m_accessType = FILE_MAP_ALL_ACCESS;
  #elif defined(LINUX)
      struct rlimit rlim;
      getrlimit( RLIMIT_FSIZE, &rlim );
      m_isValid = ( m_mappingSize <= rlim.rlim_cur );

      m_accessType = PROT_READ | PROT_WRITE;
  #endif

      if ( m_isValid )
      {
  #if defined(_WIN32)
        m_file = CreateFile( fileName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL
                           , CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );
        m_isValid = ( m_file != INVALID_HANDLE_VALUE );
        if ( m_isValid )
        {
          m_fileMapping = CreateFileMapping( m_file, NULL, PAGE_READWRITE, HIDWORD(m_mappingSize)
                                           , LODWORD(m_mappingSize), NULL );
          m_isValid = ( m_fileMapping != NULL );
          if ( ! m_isValid )
          {
            CloseHandle( m_file );
          }
        }
  #elif defined(LINUX)
        m_file = open( fileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666 );
        m_isValid = ( m_file != -1 );
        if ( m_isValid )
        {
          // make file large enough to hold the complete scene 
          lseek(m_file, m_mappingSize-1, SEEK_SET);
          write(m_file, "", 1);
          lseek(m_file, 0, SEEK_SET);
        }
  #else
        DP_STATIC_ASSERT( false );
  #endif
      }
    }

    WriteMapping::~WriteMapping()
    {
      if ( m_isValid )
      {
  #if defined(_WIN32)
        DP_ASSERT( ( m_file != INVALID_HANDLE_VALUE ) && ( m_fileMapping != NULL ) );
        CloseHandle( m_fileMapping );

        // truncate file to minimum size
        // To work with 64-bit file pointers, you can declare a LONG, treat it as the upper half 
        // of the 64-bit file pointer, and pass its address in lpDistanceToMoveHigh. This means 
        // you have to treat two different variables as a logical unit, which is error-prone. 
        // The problems can be ameliorated by using the LARGE_INTEGER structure to create a 64-bit 
        // value and passing the two 32-bit values by means of the appropriate elements of the union.
        // (see msdn documentation on SetFilePointer)
        LARGE_INTEGER li;
        li.QuadPart = (__int64)m_endOffset;
        SetFilePointer( m_file, li.LowPart, &li.HighPart, FILE_BEGIN );

        SetEndOfFile( m_file );
        CloseHandle( m_file );
  #elif defined(LINUX)
        DP_ASSERT( m_file != -1 );
        close( m_file );
  #else
        DP_STATIC_ASSERT( false );
  #endif
      }
    }

    void * WriteMapping::mapIn( size_t offset, size_t numBytes )
    {
      void * p = FileMapping::mapIn( offset, numBytes );
      if ( p && ( m_endOffset < offset + numBytes ) )
      {
        m_endOffset = offset + numBytes;
      }
      return( p );
    }
  } // namespace util
} // namespace dp
