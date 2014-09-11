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


#include <dp/util/Allocator.h>
#ifndef NDEBUG
#include <algorithm>
#endif
#include "dp/util/Backtrace.h"

// introduce some STL stuff
#if !defined(NDEBUG)
using std::for_each;
#endif
using std::make_pair;
using std::pair;

//#define MEMORY_PROFILER
#if defined( MEMORY_PROFILER )
static bool newLoggingEnabled = true;

std::map<std::string,std::map<size_t,size_t> > & getNewHistogram()
{
  static std::map<std::string,std::map<size_t,size_t> > newHistogram;
  return( newHistogram );
}

const std::vector<std::string> & getExclusionList()
{
  static std::vector<std::string> exclusionList;
  if ( exclusionList.empty() )
  {
    exclusionList.push_back( "std::" );
    exclusionList.push_back( "dp::util::Chunk::Chunk" );
    exclusionList.push_back( "dp::util::FixedAllocator::alloc" );
  }
  return( exclusionList );
}
#endif


namespace dp
{
  namespace util
  {

    Chunk::Chunk(size_t blockSize) 
    : m_blockSize(blockSize)
    , m_chunkSize(blockSize*numBlocks)
    { 
      init(); 
    }

    Chunk::~Chunk()
    {
      freeMemory();
    }

    FixedAllocator::FixedAllocator() 
    : m_blockSize(0)
    , m_lastAlloc(nullptr)
    , m_lastDealloc(nullptr)
    {
      // nothing left to do
    }

    FixedAllocator::~FixedAllocator()
    {
      // memory leak?
      DP_ASSERT(m_chunks.size()<=1 && "Use nvsgSetDebugFlags(NVSG_DBG_LEAK_DETECTION) to track memory leaks.");
      if (m_lastDealloc)
      {
        DP_ASSERT( ( m_chunks.size() == 1 ) && ( m_chunks.begin()->second == m_lastDealloc ) );
        // memory leak?
        DP_ASSERT(m_lastDealloc->isUnused() && "Use nvsgSetDebugFlags(NVSG_DBG_LEAK_DETECTION) to track memory leaks.");
        delete m_lastDealloc;
        m_chunks.clear();
      }
    }
 
    void * FixedAllocator::alloc()
    { // allocate one memory block of size blockSize
      if (  !m_lastAlloc
         || !m_lastAlloc->blocksAvailable() 
         )
      {
        if ( m_availableChunks.empty() )
        {
          // there is currently no suitable chunk available, so add a new one
          Chunk * chunk = new Chunk( m_blockSize );
          pair<Chunks::iterator,bool> pcib = m_chunks.insert( make_pair( chunk->upperBound(), chunk ) );
          DP_ASSERT( pcib.second );
          m_lastAlloc = pcib.first->second;
          m_availableChunks.insert( m_lastAlloc );
        }
        else
        {
          m_lastAlloc = *m_availableChunks.begin();
          DP_ASSERT( m_lastAlloc->blocksAvailable() );
        }
      }

      DP_ASSERT( m_chunks.find( m_lastAlloc->upperBound() ) != m_chunks.end() );
      if ( m_lastAlloc->blocksAvailable() == 1 )
      {
        m_availableChunks.erase( m_lastAlloc );
      }

      // assume m_lastAlloc is a valid iterator into the list of chunks 
      DP_ASSERT( m_chunks.end() != m_chunks.find( m_lastAlloc->upperBound() ) );
      DP_ASSERT( m_chunks[m_lastAlloc->upperBound()] == m_lastAlloc );
      return m_lastAlloc->alloc();
    }

    void FixedAllocator::dealloc(void * p)
    { // free the single memory block pointed to by p

      if ( !m_lastDealloc || !m_lastDealloc->isInsideBounds(p) )
      {
        // pointer to deallocate does not belong to the current chunk
        Chunks::iterator I = m_chunks.lower_bound( p );
        DP_ASSERT( I != m_chunks.end() );
        DP_ASSERT( I->second->isInsideBounds( p ) );
        m_lastDealloc = I->second;
      }

      if ( ! m_lastDealloc->blocksAvailable() )
      {
        DP_ASSERT( m_availableChunks.find( m_lastDealloc ) == m_availableChunks.end() );
        m_availableChunks.insert( m_lastDealloc );
      }

      // perform deallocation
  
      // undefined behaviour if this fires 
      DP_ASSERT(m_lastDealloc->isInsideBounds(p));
      m_lastDealloc->dealloc(p);

      // check for garbage collection but keep at least one chunk
  
      if (  m_lastDealloc->isUnused() // chunk taken for last deallocation is entirely unused now
         && m_chunks.size() > 1 // there are at least two chunks in our array
         )
      {
        DP_ASSERT( m_availableChunks.find( m_lastDealloc ) != m_availableChunks.end() );
        m_availableChunks.erase( m_lastDealloc );

        m_chunks.erase( m_lastDealloc->upperBound() );
        delete m_lastDealloc;
        if ( m_lastAlloc == m_lastDealloc )
        {
          m_lastAlloc = nullptr;
        }
        m_lastDealloc = nullptr;
      }
    }

    Allocator::Allocator()
      : m_enableLeakDetection( false )
    {
      // init all the fixed size allocators
      for( int i=1; i<=maxBlockSize; ++i )
      {
         // 0. allocator manages size==1 requests
         // 1. allocator manages size==2 requests
         // ...
         // maxBlockSize-1. allocator manages size==maxBlockSize requests
        m_allocTbl[i-1].init(i);
      }
    }

    Allocator* Allocator::instance()
    {
      return Singleton<Allocator>::instance();
    }

    #if !defined(NDEBUG)
    void
    Allocator::dumpAllocations()
    {
      std::stringstream ss;
      ss << "**** DUMPING " << m_dbgAllocInfos.size() << " ALLOCATIONS ****\n";
      traceDebugOutput()(ss.str().c_str());

      for_each(m_dbgAllocInfos.begin(), m_dbgAllocInfos.end(), UsageInfo());  

      traceDebugOutput()("*************************************************\n");
    }

    void
    Allocator::dumpDiffAllocations()
    {
      // must be at least one element
      DP_ASSERT( m_dbgAllocInfoStack.size() );

      //
      // This code will display the difference between a past set of allocations and a new set.
      // All allocations in the current set that are not in the saved set are printed.
      //
      std::vector<DbgAllocInfo> diffs( m_dbgAllocInfos.size() );
      std::vector<DbgAllocInfo> sortedOld = m_dbgAllocInfoStack.back();
      std::vector<DbgAllocInfo> sortedNew = m_dbgAllocInfos;

      std::sort( sortedOld.begin(), sortedOld.end(), DbgLessThan() );
      std::sort( sortedNew.begin(), sortedNew.end(), DbgLessThan() );
      std::vector<DbgAllocInfo>::iterator sEnd = std::set_difference( sortedNew.begin(), sortedNew.end(),
                                                                      sortedOld.begin(), sortedOld.end(),
                                                                      diffs.begin(), DbgLessThan() );
      // clear unused
      diffs.erase( sEnd, diffs.end() ); 

      std::stringstream ss;
      ss << "**** DUMPING " << diffs.size() << " DIFFERENT ALLOCATIONS ****\n";
      traceDebugOutput()(ss.str().c_str());

      for_each(diffs.begin(), diffs.end(), UsageInfo());  

      traceDebugOutput()("*************************************************\n");
    }
    #endif

    Allocator::~Allocator()
    {
    #if !defined(NDEBUG)
      // leak detection
      for_each(m_dbgAllocInfos.begin(), m_dbgAllocInfos.end(), LeakWarning());  
    #endif
    #if defined( ALLOCATION_COUNTER )
      FILE *fh = fopen( "allocationMap.txt", "a" );
      if ( fh )
      {
        fprintf( fh, "allocations:  size    count\n" );
        size_t sum(0), small(0);
        size_t sumCount(0), smallCount(0);
        for ( std::map<size_t,size_t>::const_iterator it = m_allocSizeMap.begin() ; it != m_allocSizeMap.end() ; ++it )
        {
          fprintf( fh, "\t\t%Iu\t%Iu\n", it->first, it->second );
          sum += it->first * it->second;
          sumCount += it->second;
          if ( it->first <= maxBlockSize )
          {
            small += it->first * it->second;
            smallCount += it->second;
          }
        }
        fprintf( fh, "\nsum\t%Iu\t%Iu", sum, sumCount );
        fprintf( fh, "\nsmall\t%Iu\t%Iu\n", small, smallCount );
        fclose( fh );
      }
    #endif

    #if defined( MEMORY_PROFILER )
      newLoggingEnabled = false;
      FILE * fh = fopen( "memoryProfiler.txt", "w" );
      if ( fh )
      {
        std::multimap<size_t,std::pair<size_t,std::map<std::string,std::map<size_t,size_t> >::const_iterator> > sortedMap;
        size_t overallCalls = 0;
        size_t overallSize = 0;
        const std::map<std::string,std::map<size_t,size_t> > & newHistogram = getNewHistogram();
        for ( std::map<std::string,std::map<size_t,size_t> >::const_iterator it = newHistogram.begin() ; it != newHistogram.end() ; ++it )
        {
          size_t calls = 0;
          size_t size = 0;
          for ( std::map<size_t,size_t>::const_iterator jt = it->second.begin() ; jt != it->second.end() ; ++jt )
          {
            calls += jt->second;
            size += jt->first * jt->second;
          }
          sortedMap.insert( std::make_pair( size, std::make_pair( calls, it ) ) );
          overallSize += size;
          overallCalls += calls;
        }

        fprintf( fh, "calls: %Iu\tsize: %Iu\n\n", overallCalls, overallSize );
        for ( std::multimap<size_t,std::pair<size_t,std::map<std::string,std::map<size_t,size_t> >::const_iterator> >::const_reverse_iterator it = sortedMap.rbegin() ; it != sortedMap.rend() ; ++it )
        {
          fprintf( fh, "%s\n", it->second.second->first.c_str() );
          fprintf( fh, "\tcalls: %Iu\tsize: %Iu\n", it->second.first, it->first );
          for ( std::map<size_t,size_t>::const_reverse_iterator jt = it->second.second->second.rbegin() ; jt != it->second.second->second.rend() ; ++jt )
          {
            fprintf( fh, "\t\tcalls: %Iu\tsize: %Iu\n", jt->second, jt->first);
          }
        }
        fclose( fh );
      }
    #endif
    }
  } // namespace util
} // namespace dp

#if defined( MEMORY_PROFILER )
void logNew( size_t size )
{
  static bool inNew = false;

  if ( ! inNew && newLoggingEnabled )
  {
    inNew = true;
    std::vector<std::string> trace = dp::util::backtrace( 4, 1, getExclusionList() );
    DP_ASSERT( trace.empty() || ( trace.size() == 1 ) );
    getNewHistogram()[!trace.empty() ? trace[0] : "unknown location"][size]++;
    inNew = false;
  }
}

void * operator new( size_t size )
{
  logNew( size );
  void * ret = malloc( size );
  if ( ! ret )
  {
    throw std::bad_alloc();
  }
  return( ret );
}

void * operator new[]( size_t size )
{
  logNew( size );
  void * ret = malloc( size );
  if ( ! ret )
  {
    throw std::bad_alloc();
  }
  return( ret );
}

void operator delete( void * data )
{
  free( data );
}

void operator delete[]( void * data )
{
  free( data );
}
#endif
