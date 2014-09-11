// Copyright NVIDIA Corporation 2002-2012
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

//#define ALLOCATION_COUNTER

// placement operator new does not have a matching delete operator!
// 
// objects of type T that are constructed at explicit specified memory 
// locations by using the placement operator new() (i.g. new((void*)p) T())
// should be destructed by explicitly calling their destructor (p->~T())
#if defined( _MSC_VER )
#pragma warning(disable:4291) 
#endif

#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#if defined( ALLOCATION_COUNTER )
#include <map>
#endif
#include <dp/util/DPAssert.h>
#include <dp/util/Lock.h>

#if !defined(NDEBUG)
#include <dp/util/Backtrace.h>
#endif

#include <dp/util/Trace.h>
#include <dp/util/Singleton.h>

namespace dp
{
  namespace util
  {
    //! Chunk of memory for our low level memory managment of fixed size objects
    class Chunk
    {
      public:
        //! construct a chunk of blockSize bytes
        Chunk(size_t blockSize); 

        //! clean-up
        ~Chunk();
      
        //! provide comparison of chunks
        bool operator==(const Chunk& rhs) const;
        bool operator!=(const Chunk& rhs) const;
      
        //! memory management interface
        void * alloc();
        void dealloc(void * p);
      
        //! provide number of blocks available in this chunk
        unsigned char blocksAvailable() const;
        //! for cleanup purposes, provide check if chunk is entirely unused
        bool isUnused() const; 
        //! provide address checking
        bool isInsideBounds(void * p) const;

        //! helpers used with address checking
        void * lowerBound() const;
        void * upperBound() const;

      private:
        //! maximum number of blocks we can address
        enum { numBlocks = 255 };

        //! one time initialization
        void init();

        //! explicitly free memory
        void freeMemory(); 
      
        unsigned char * m_rawMem; // raw memory

        //! bounds
        size_t m_blockSize;
        size_t m_chunkSize;

        unsigned char m_firstAvailableBlock;  // addresses first memory block available
        unsigned char m_blocksAvailable;      // total amount of blocks available
    };

    //! Manages allocation requests for objects of a certain size only
    class FixedAllocator
    {
      public:
        //! default constructs a FixedAllocator object
        FixedAllocator(); 

        //! destructor - last chunk memory cleanup
        ~FixedAllocator();

        //! Allocate one memory block of size blockSize
        DP_UTIL_API void * alloc();

        //! Free the single memory block pointed to by \a p
        DP_UTIL_API void dealloc(void * p);

        //! one time initialization
        void init ( size_t blockSize );

      private:
        size_t m_blockSize;   // fixed block size

        typedef std::map<void*,Chunk*> Chunks;
        Chunks m_chunks; // array of chunks

        Chunk  * m_lastAlloc;   // last chunk taken for allocation
        Chunk  * m_lastDealloc; // last chunk taken for deallocation
        std::set<Chunk*>  m_availableChunks;  // set of chunks with free blocks
    };

  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //! Internal manager for memory allocations.
    /** This class is used as a \c Singleton by \c IAllocator, which is the base of all \c RCObject classes.
      * It manages the efficient allocation of small objects by using an array of \c FixedAllocator objects, one for
      * each small size up to \c maxBlockSize. Allocations larger than that size are redirected to the standard
      * allocation \c ::new.  */
    class Allocator
    {
      public:
        //! Default constructor
        /** Initializes the array of \c FixedAllocator objects. */
        DP_UTIL_API Allocator();

        //! Destructor
        /** In debug mode warnings on not deleted objects are emitted.  */
        DP_UTIL_API ~Allocator();

        static DP_UTIL_API Allocator* instance();

        //! Allocate continuous memory of size \a size.
        /** Forwards either to the \c FixedAllocator of the specified size, or to the global new operator. */
        void * alloc( size_t size   //!< size of memory to allocate
                    );

  #if !defined(NDEBUG)
        //! Allocate continuous memory of size \a size in debug mode.
        /** Forwards either to the \c FixedAllocator of the specified size, or to the global new operator.
          * In addition, it stores information for memory leak detection. */
        void * alloc( size_t size         //!<  size of memory to allocate
                    , const char * src    //!<  source file name where this memory is allocated
                    , unsigned int ln     //!<  line in the source file
                    );

        //! Dump all current memory allocations
        DP_UTIL_API void dumpAllocations(); 
        //! Push current set of memory allocations for use with dumpDiffAllocations
        void pushAllocations();
        //! pop last set of memory allocations
        void popAllocations();
        //! dump the difference between the head of the stack and the current memory allocs
        DP_UTIL_API void dumpDiffAllocations();
  #endif

        //! Deallocate previously allocated memory
        /** Forwards either to the \c FixedAllocator of the specified size, or to the global delete operator.
          * In debug mode, the corresponding information stored by alloc is erased. */
        void dealloc( void * p      //!<  pointer to the memory to deallocate
                    , size_t size   //!<  size of the allocated memory
                    );

        void enableLeakDetection();
      private:
        //! Helper allocation routine used with debug and non-debug mode
        void * palloc(size_t size);

      private:
        // forward to default new/delete operators if block size
        // exceeds the threshold given by maxBlockSize
        enum { maxBlockSize = 1024 };

        // provide one allocator for each single size <= maxBlockSize
        FixedAllocator m_allocTbl[maxBlockSize];

        Lock m_lock; // for synchronizing concurrent access

        // leak detection in debug mode
  #if !defined(NDEBUG)
        struct DbgAllocInfo
        {
          void  * p;            //  memory location
          size_t  size;         //  size of allocation
          std::string location; //  location of allocation

          // with this we can simply use a vector for our alloc infos
          bool operator==(void * _p) const { return p==_p; }
          bool operator!=(void * _p) const { return p!=_p; }
        };

        std::vector<DbgAllocInfo> m_dbgAllocInfos;   //  vector of allocation infos
        std::vector< std::vector<DbgAllocInfo> > m_dbgAllocInfoStack;

        //
        // Sort and equality are done by pointer, allocation size, and function name.
        //
        struct DbgLessThan
        {
          bool operator()( const DbgAllocInfo & lhs, const DbgAllocInfo & rhs ) const
          {
            return( ( lhs.p < rhs.p ) || 
                  ( ( lhs.p == rhs.p ) && ( ( lhs.size < rhs.size ) || 
                                          ( ( lhs.size == rhs.size ) && ( lhs.location < rhs.location ) ) ) ) );
          }
        };
      
        //  helper functor to put out a memory leak warning
        struct LeakWarning
        {
          void operator()(const DbgAllocInfo& allocInfo) const
          {
            std::stringstream str;

            str << "****Memory Leak Detected*** Source: ";
            str << allocInfo.location;
            str << ", Ptr: 0x";
            str << std::hex << allocInfo.p;
            str << " (" << std::dec << allocInfo.size << ") bytes."; 
            str << std::endl;

            dp::util::traceDebugOutput()(str.str().c_str());
          }
        };

        //  helper functor to put out a memory usage info
        struct UsageInfo
        {
          void operator()(const DbgAllocInfo& allocInfo) const
          {
            std::stringstream str;

            str << "Source: ";
            str << allocInfo.location;
            str << ", Ptr: 0x";
            str << std::hex << allocInfo.p;
            str << " (" << std::dec << allocInfo.size << ") bytes."; 
            str << std::endl;
            traceDebugOutput()(str.str().c_str());
          }
        };
  #endif
  #if defined( ALLOCATION_COUNTER )
        std::map<size_t,size_t> m_allocSizeMap;
  #endif

        bool          m_enableLeakDetection;
    };

    inline void * Allocator::alloc(size_t size)
    {
      Lock::Block lock(&m_lock);
      void *p = palloc(size);

  #if !defined(NDEBUG)
      if( m_enableLeakDetection )
      {
        std::vector<std::string> trace = backtrace(4, 1);
        DbgAllocInfo allocInfo;
        allocInfo.p = p;
        allocInfo.size = size;
        allocInfo.location = !trace.empty() ? trace[0] : "unknown location";
        m_dbgAllocInfos.push_back(allocInfo);
      }
  #endif

      return p;
    }

  #if !defined(NDEBUG)
    inline void Allocator::pushAllocations()
    {
      m_dbgAllocInfoStack.push_back( m_dbgAllocInfos );
    }

    inline void Allocator::popAllocations()
    {
      m_dbgAllocInfoStack.pop_back();
    }
  #endif

    inline void Allocator::enableLeakDetection()
    {
      m_enableLeakDetection = true;
    }

  #if !defined(NDEBUG)
    inline void * Allocator::alloc(size_t size, const char* src, unsigned int ln)
    {
      Lock::Block lock(&m_lock);
      void * p = palloc(size);

      // optionally collect allocation info for memory leak detection
      if( m_enableLeakDetection )
      {
        std::ostringstream location;
        location << src << ":" << ln;
        DbgAllocInfo allocInfo;
        allocInfo.p = p;
        allocInfo.size = size;
        allocInfo.location = location.str();
        m_dbgAllocInfos.push_back(allocInfo);
      }
      return p;
    }
  #endif

    inline void Allocator::dealloc(void *p, size_t size)
    {
      Lock::Block lock(&m_lock);
  #if !defined(NDEBUG)
      if( m_enableLeakDetection )
      {
        m_dbgAllocInfos.erase(std::remove(m_dbgAllocInfos.begin(), m_dbgAllocInfos.end(), p), m_dbgAllocInfos.end());
      }
  #endif
      if ( size <= maxBlockSize )
      {
        m_allocTbl[size-1].dealloc(p); 
      } 
      else
      {
  #if 0 && defined(WIN32)
          _aligned_free(p);
  #else
        free(p);
  #endif
      }
    }

    inline void * Allocator::palloc(size_t size)
    {
  #if defined( ALLOCATION_COUNTER )
      m_allocSizeMap[size]++;
  #endif
      // use suitable allocation for given size
      if ( size <= maxBlockSize )
      {
        return m_allocTbl[size-1].alloc();
      }
      else
      {
#if 0 // aligned mallocs not required at the moment
  #if defined(WIN32)
        return _aligned_malloc( size, ALIGNMENT_ORDER(4) );
  #else
        void *pointer = NULL;
        int ret = posix_memalign( &pointer, ALIGNMENT_ORDER(4), size);

        return (!ret) ? pointer : NULL;
  #endif
#else
        return malloc( size );
#endif
      }
    }
  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    //! An allocator interface
    /** The \c IAllocator interface provides overloads of the \c new and \c delete operators for heap allocation.
      * This overloads make use of a specialized memory manager, that is highly optimized for small object allocation.
      * For large objects, i.e. objects greater than 128 bytes, the \c IAllocator interface utilizes the default
      * memory manager.
      * \note Typically a user defined class utilizes this interface through public inheritance.
      */
    class IAllocator
    {
      public:
     
        //! Default constructor
        /** Default constructs an \c IAllocator object.
          */
        IAllocator() {}
        //! Destructor
        virtual ~IAllocator() {}

        //! Operator new overload
        /** Allocates a memory block of size bytes from heap.
          * Returns start address of allocated memory block. */
        void * operator new( size_t size // Size in bytes of demanded memory block.
                                    );
        //! Placement operator new overload
        /** Provides object creation at a specified memory address.
          *
          * p must point to a valid block of memory, large enough to hold
          * an object of size bytes.
          *
          * Return start address of the specified memory block. */
        void * operator new( size_t size // Size in bytes of demanded memory block.
                                    , void * p    // Start address of a valid memory block where the object will be created.
                                    );
        //! Operator delete overload
        /** Frees size bytes of memory at address p. */
        void operator delete( void * p    // Start address of memory block to be freed.
                                    , size_t size // Size of memory block to be freed.
                                    ); 

        //! Operator new[] overload
        /** Allocates a memory block of size bytes from heap.
          * Returns start address of allocated memory block. */
        void * operator new[]( size_t size );

        //! Operator delete[] overload
        /** Frees size bytes of memory at address p. */
        void operator delete[]( void * p, size_t size );
    };


    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // implementation following
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    /**
    * Chunk::init() 
    *
    * provides one time initialization
    */
    inline void Chunk::init()
    {
#if 0 // no aligned mallocs required at the moment
  #if defined(WIN32)
      m_rawMem = (unsigned char *)_aligned_malloc( m_chunkSize, ALIGNMENT_ORDER(4) );
  #else
      int ret = posix_memalign( reinterpret_cast<void**>(&m_rawMem), ALIGNMENT_ORDER(4), m_chunkSize);
      DP_ASSERT(!ret);
  #endif
#else
      m_rawMem = (unsigned char *)malloc( m_chunkSize );
#endif
      DP_ASSERT(m_rawMem);
    
      m_firstAvailableBlock = 0;
      m_blocksAvailable = numBlocks;

      unsigned char * p = m_rawMem;
      for ( unsigned char i=0; i!=numBlocks; p+=m_blockSize ) 
      {
        // code indices of next available blocks in first byte of each block 
        *p = ++i;
      }
    }

    /**
    * Chunk::freeMemory() 
    *
    * explicitly free chunk's raw memory
    */
    inline void Chunk::freeMemory()
    {
      // should not attempt to free memory of a chunk in use
      DP_ASSERT(isUnused());
  #if 0 && defined(WIN32)
      _aligned_free( m_rawMem );
  #else
      free( m_rawMem );
  #endif
    }

    /**
    * Chunk::alloc() 
    *
    * get one block from chunk; returns NULL if no blocks available
    */
    inline void * Chunk::alloc()
    {
      if ( !m_blocksAvailable ) 
      {
        // this needs to be handled in superior layers
        return NULL;
      }

      unsigned char * p = &m_rawMem[m_firstAvailableBlock*m_blockSize];
      // index of next block available is coded in first byte
      m_firstAvailableBlock = *p; 
      // one block less available in this chunk
      --m_blocksAvailable; 

      return (void *)p;
    }

    /**
    * Chunk::dealloc()
    *
    * deallocate a block pointed to by p
    */
    inline void Chunk::dealloc(void * p)
    {
      unsigned char * ptr = (unsigned char*)p;
    
      // range check
      DP_ASSERT(ptr>=m_rawMem);
      DP_ASSERT(ptr<&m_rawMem[m_chunkSize]);
      // alignment check
      DP_ASSERT(!((ptr-m_rawMem)%m_blockSize));
    
      *ptr = m_firstAvailableBlock;
      m_firstAvailableBlock = (unsigned char)((ptr - m_rawMem) / m_blockSize);

      // truncation check
      DP_ASSERT(m_firstAvailableBlock==((ptr-m_rawMem)/m_blockSize));
      ++m_blocksAvailable;
    }

    /**
    * Chunk::operator==()
    *
    * returns true if rhs equals this chunk
    */
    inline bool Chunk::operator==(const Chunk& rhs) const
    {
      // consider chunks to be equal if they point to same raw memory location
      DP_ASSERT(    ( m_rawMem != rhs.m_rawMem )
                  ||  (     ( m_blockSize           == rhs.m_blockSize )
                        &&  ( m_firstAvailableBlock == rhs.m_firstAvailableBlock )
                        &&  ( m_blocksAvailable     == rhs.m_blocksAvailable ) ) );
      return (  m_rawMem==rhs.m_rawMem );
    }

    /**
    * Chunk::operator!=()
    *
    * returns true if rhs not equals this chunk
    */
    inline bool Chunk::operator!=(const Chunk& rhs) const
    {
      return !operator==(rhs);
    }

    /** 
    * Chunk::blocksAvailable()
    * 
    * provides number of blocks available in this chunk 
    */
    inline unsigned char Chunk::blocksAvailable() const 
    { 
      return m_blocksAvailable; 
    }

    /**
    * Chunk::isInsideBounds() 
    *
    * returns true if the address given belongs to this chunk; false otherwise  
    */
    inline bool Chunk::isInsideBounds(void * p) const
    {
      return p>=lowerBound() && p<upperBound();
    }

    /** 
    * Chunk::isUnused()
    * 
    * for cleanup purposes, provide check if chunk is entirely unused 
    */
    inline bool Chunk::isUnused() const 
    { 
      return m_blocksAvailable==numBlocks; 
    }

    /** 
    * Chunk::lowerBound()
    * 
    * query lower bound of memory chunk 
    */
    inline void * Chunk::lowerBound() const 
    { 
      return (void*)m_rawMem; 
    }

    /** 
    * Chunk::upperBound()
    * 
    * query upper bound of memory chunk 
    */
    inline void * Chunk::upperBound() const 
    { 
      return (void*)&m_rawMem[m_chunkSize]; 
    }

    /**
    * FixedAllocator::init()
    *
    * one time initialization
    */
    inline void FixedAllocator::init(size_t blockSize)
    { 
      m_blockSize = blockSize; 
    }

    /**
    * IAllocator::operator new()
    *
    * overrides operator new()  
    */
    inline void * IAllocator::operator new(size_t size) 
    {
      return Allocator::instance()->alloc(size);
    }

    /**
    * IAllocator::operator new()
    *
    * overides placement operator new()
    */
    inline void * IAllocator::operator new(size_t /*size*/, void * p)
    {
      // as this is used when constructing an object at a explicit
      // specified memory location, that location shoud be a valid 
      // one - assert this!
      DP_ASSERT(p);
      return p;
    }

    /**
    * IAllocator::operator delete()
    *
    * overrides operator delete()  
    */
    inline void IAllocator::operator delete(void * p, size_t size)
    {
      Allocator::instance()->dealloc(p, size);
    }

    inline void * IAllocator::operator new[]( size_t size )
    {
      // Due to a bug in VC, we need to store the size of the array, as delete[] gets a wrong value there!
      size_t fakeSize = sizeof(size_t) + size;
      void * fakeP = Allocator::instance()->alloc( fakeSize );
      *(size_t *)fakeP = fakeSize;
      return( (char *)fakeP + sizeof(size_t) );
    }

    inline void IAllocator::operator delete[]( void * p, size_t /*size*/ )
    {
      // Due to a bug in VC, we need to store the size of the array, as delete[] gets a wrong value there!
      void * fakeP = (char *)p - sizeof(size_t);
      size_t fakeSize = *(size_t *)fakeP;
      Allocator::instance()->dealloc( fakeP, fakeSize );
    }

  } // namespace util
} // namespace dp

