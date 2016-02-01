// Copyright (c) 2010-2016, NVIDIA CORPORATION. All rights reserved.
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
/** @file */

#include <dp/sg/core/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Event.h>
#include <dp/sg/core/HandledObject.h>
#include <dp/util/BitMask.h>
#include <dp/util/Flags.h>
#include <dp/util/StridedIterator.h>
#include <dp/util/WeakPtr.h>
#include <cstring>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      class ReadLockImpl;
      class WriteLockImpl;

      /** \brief Interface for buffers used as storage for SceniX.
       *  \sa BufferHost, BufferGL
      **/
      class Buffer : public HandledObject
      {
      public:
        class Event : public core::Event
        {
        public:
          Event( Buffer const* buffer )
            : core::Event( core::Event::Type::BUFFER )
            , m_buffer( buffer)
          {
          }

          Buffer const* getBuffer() const { return m_buffer; }
        private:
          const Buffer* m_buffer;
        };

      public:
        DP_SG_CORE_API virtual ~Buffer();

      public:
        enum class MapMode
        {
          NONE      = 0,
          READ      = BIT0,
          WRITE     = BIT1,
          READWRITE = READ | WRITE
        };

        typedef dp::util::Flags<MapMode> MapModeMask;

        /** \brief copy the continuous memory from another buffer to self
        **/
        DP_SG_CORE_API virtual void setData( size_t dst_offset, size_t length, const void* src_data );
        DP_SG_CORE_API virtual void setData( size_t dst_offset, size_t length, const BufferSharedPtr &src_buffer , size_t src_offset );

        /** \brief copy the continuous memory to another buffer
        **/
        DP_SG_CORE_API virtual void getData( size_t src_offset, size_t length, void* dst_data ) const;
        DP_SG_CORE_API virtual void getData( size_t src_offset, size_t length, const BufferSharedPtr &dst_buffer , size_t dst_offset ) const;


        /** \brief Retrieve the size of the Buffer
         **/
        DP_SG_CORE_API virtual size_t getSize() const = 0;

        /** \brief Resize the Buffer. Any data currently stored in this buffer will be lost.
         \  \param New size for the buffer
        **/
        DP_SG_CORE_API virtual void setSize( size_t size ) = 0;

        /** \brief Resize the Buffer. This function keeps data currently stored in the buffer if possible.
         \  \param New size for the buffer
        **/
        DP_SG_CORE_API virtual void resize( size_t size );

        /** \brief Get whether buffer is managed by internal system or user
        **/
        DP_SG_CORE_API bool isManagedBySystem() const;

        /** \brief Object to acquire a thread-safe read access to the buffer's data.
         *  \sa WriteLock
        **/
        class DataReadLock {

        public:
          /**
           *  \brief Prepare thread-safe access to a range within buffer data. Only use the pointers the way the MapMode describes it!
              \param mode desired MapMode to the buffer
              \param offset in bytes to start map from
              \param length in bytes for the mapped range
          **/
          DataReadLock() {};
          DataReadLock( const BufferSharedPtr &buffer) : m_lock( ReadLockImpl::create( buffer ) ) {}
          DataReadLock( const BufferSharedPtr &buffer, size_t offset, size_t length)
                             : m_lock( ReadLockImpl::create( buffer , offset, length) ) {}
          DataReadLock( const DataReadLock &rhs ) { m_lock = rhs.m_lock; }
          DataReadLock &operator=( const DataReadLock &rhs ) { m_lock = rhs.m_lock; return *this;}

          void reset() { m_lock.reset(); }

          /** \brief Returns the pointer to the mapped range within the buffer.
          **/
          template <typename ValueType> const ValueType *getPtr() const { return reinterpret_cast<const ValueType *>(m_lock->m_ptr); }
          const void *getPtr() const { return m_lock->m_ptr; }

        protected:
          DEFINE_PTR_TYPES( ReadLockImpl );
          class ReadLockImpl
          {
          public:
            static ReadLockImplSharedPtr create( BufferSharedPtr const& buffer );
            static ReadLockImplSharedPtr create( BufferSharedPtr const& buffer, size_t offset, size_t length );
            ~ReadLockImpl();

            BufferSharedPtr m_buffer;
            const void *m_ptr;

          protected:
            ReadLockImpl( const BufferSharedPtr &buffer);
            ReadLockImpl( const BufferSharedPtr &buffer, size_t offset, size_t length);
          };

        private:
          ReadLockImplSharedPtr m_lock;
        };

        /** \brief Object to acquire a thread-safe write, read or read-write access to the buffer's data.
         *  \sa ReadLock
        **/
        class DataWriteLock {

        public:
          /**
           *  \brief Prepare thread-safe access to a range within buffer data. Only use the pointers the way the MapMode describes it!
              \param mapMode desired MapMode to the buffer.
              \param offset in bytes to start map from
              \param length in bytes for the mapped range
              \note Mapping with \c MapMode::WRITE must not be used to retrieve data, and when using \c MapMode::READ
              you must not write data. Both operations are legal on the C++ side as you get a non-const pointer
              when mapping, but are illegal for the system's integrity. Buffer implementations will have undefined behavior
              when you use the MapMode wrongly.
              \sa getPtr
          **/
          DataWriteLock() {};
          DataWriteLock( const BufferSharedPtr &buffer, Buffer::MapMode mapMode )
                              : m_lock(WriteLockImpl::create( buffer, mapMode ) ) { }
          DataWriteLock( const BufferSharedPtr &buffer, Buffer::MapMode mapMode, size_t offset, size_t length )
                              : m_lock(WriteLockImpl::create( buffer, mapMode, offset, length ) ) { }
          DataWriteLock( const DataWriteLock &rhs ) { m_lock = rhs.m_lock; }
          DataWriteLock &operator=( const DataWriteLock &rhs ) { m_lock = rhs.m_lock; return *this; }

          void reset() { m_lock.reset(); }

          /** \brief Returns the pointer to the mapped range within the buffer. Make sure you only use it according to MapMode.
          **/
          void*getPtr() const { return m_lock->m_ptr; }
          template <typename ValueType> ValueType *getPtr() const { return reinterpret_cast<ValueType *>(m_lock->m_ptr); }

        protected:
          DEFINE_PTR_TYPES( WriteLockImpl );

          class WriteLockImpl
          {
          public:
            static WriteLockImplSharedPtr create( const BufferSharedPtr &buffer, Buffer::MapMode mapMode );
            static WriteLockImplSharedPtr create( const BufferSharedPtr &buffer, Buffer::MapMode mapMode, size_t offset, size_t length );
            ~WriteLockImpl();

            BufferSharedPtr m_buffer;
            void *m_ptr;

          protected:
            WriteLockImpl( const BufferSharedPtr &buffer, Buffer::MapMode mapMode );
            WriteLockImpl( const BufferSharedPtr &buffer, Buffer::MapMode mapMode, size_t offset, size_t length );
          };

        private:
          WriteLockImplSharedPtr  m_lock;
        };

        // FIXME rename to WriteLockIterator
        template <typename ValueType>
        struct Iterator
        {
          typedef typename dp::util::StridedIterator<ValueType, DataWriteLock> Type;
        };

        // FIXME rename to ReadLockIterator
        template <typename ValueType>
        struct ConstIterator
        {
          typedef typename dp::util::StridedConstIterator<ValueType, DataReadLock> Type;
        };

        /** \brief Returns an StridedIterator to access the buffer. The buffer stays write-locked as long as an iterator constructed from
                   the returned one exist.
            \param mapMode Access mode to this buffer
            \param offset offset in bytes of the first element to access
            \param strideInBytes Stride between two elements in the buffer in bytes. A stride of 0 will use siezof(ValueType) as stride.
         **/
        template <typename ValueType>
        typename Buffer::Iterator<ValueType>::Type getIterator( MapMode mapMode, size_t offset = 0, int strideInBytes = 0);

        /** \brief Returns an StridedConstIterator to access the buffer. The buffer stays read locked as long as an iterator constructed from
                   the returned one exist.
            \param offset offset in bytes of the first element to access
            \param strideInBytes Stride between two elements in bytes. A stride of 0 will use siezof(ValueType) as stride.
         **/
        template <typename ValueType>
        typename ConstIterator<ValueType>::Type getConstIterator( size_t offset = 0, int strideInBytes = 0) const;

      protected:
        DP_SG_CORE_API Buffer();

        /**
         * \brief Retrieve pointer to a range within buffer data. Only use the pointers the way the MapMode describes it!
            \param mode desired MapMode to the buffer
            \param offset in bytes to start map from
            \param length in bytes for the mapped range
            \note Mapping with \c MapMode::WRITE must not be used to retrieve data, and when using \c MapMode::READ
            you must not write data. Both operations are legal on the C++ side as you get a non-const pointer
            when mapping, but are illegal for the system's integrity. Buffer implementations will have undefined behavior
            when you use the MapMode wrongly.
            \sa unmap
         **/
        DP_SG_CORE_API virtual void *map( MapMode mode ); // maps entire range
        DP_SG_CORE_API virtual void *map( MapMode mode, size_t offset, size_t length ) = 0;
        DP_SG_CORE_API virtual const void *mapRead() const; // map entire range on const object
        DP_SG_CORE_API virtual const void *mapRead( size_t offset, size_t length ) const = 0; // map on const object


        /** \Brief Unmap the previously mapped buffer
         *  \sa map
         **/
        DP_SG_CORE_API virtual void unmap( ) = 0;
        DP_SG_CORE_API virtual void unmapRead( ) const = 0; // unmap on const object

        void *lock( MapMode mapMode );
        void *lock( MapMode mode, size_t offset, size_t length );
        const void *lockRead() const;
        const void *lockRead( size_t offset, size_t length ) const;
        void unlock();
        void unlockRead() const;

      private:
        mutable int   m_lockCount;
        mutable void* m_mappedPtr;
        bool          m_managedBySystem;
      };

      inline Buffer::MapModeMask operator|( Buffer::MapMode bit0, Buffer::MapMode bit1 )
      {
        return Buffer::MapModeMask( bit0 ) | bit1;
      }

      inline bool Buffer::isManagedBySystem() const
      {
        return m_managedBySystem;
      }

      inline void* Buffer::map(MapMode mode)
      {
        return map( mode, 0, getSize() );
      }

      inline const void* Buffer::mapRead() const
      {
        return mapRead( 0, getSize() );
      }

      inline void Buffer::resize( size_t newSize )
      {
        size_t oldSize = getSize();
        if ( oldSize && newSize != oldSize )
        {
          size_t tmpSize = std::min( getSize(), newSize );
          char * tmpData = (char *)malloc( tmpSize );
          const char *oldData = reinterpret_cast<const char *>(mapRead());
          memcpy( tmpData, oldData, tmpSize );
          unmapRead();

          setSize( newSize );

          char *newData = reinterpret_cast<char *>(map( MapMode::WRITE ));
          memcpy( newData, tmpData, tmpSize );
          unmap();

          free( tmpData );
        }
        else
        {
          setSize( newSize );
        }
      }

      inline void *Buffer::lock( Buffer::MapMode mapMode )
      {
        if (!m_lockCount)
        {
          m_mappedPtr = map( mapMode );
        }
        ++m_lockCount;
        return m_mappedPtr;
      }

      inline void *Buffer::lock( Buffer::MapMode mapMode, size_t offset, size_t length )
      {
        if (!m_lockCount)
        {
          m_mappedPtr = map( mapMode, 0, getSize() );
        }
        ++m_lockCount;
        return (char *)m_mappedPtr + offset;
      }

      inline const void *Buffer::lockRead() const
      {
        if (!m_lockCount)
        {
          m_mappedPtr = const_cast<void*>(mapRead( ));
        }
        ++m_lockCount;
        return m_mappedPtr;
      }

      inline const void *Buffer::lockRead( size_t offset, size_t length ) const
      {
        if (!m_lockCount)
        {
          m_mappedPtr = const_cast<void*>(mapRead( 0, getSize() ));
        }
        ++m_lockCount;
        return (const char *)m_mappedPtr + offset;
      }

      inline void Buffer::unlock()
      {
        --m_lockCount;
        if (!m_lockCount)
        {
          unmap();
          m_mappedPtr = nullptr;
        }
      }

      inline void Buffer::unlockRead() const
      {
        --m_lockCount;
        if (!m_lockCount)
        {
          unmapRead();
          m_mappedPtr = nullptr;
        }
      }

      inline Buffer::DataReadLock::ReadLockImplSharedPtr Buffer::DataReadLock::ReadLockImpl::create( BufferSharedPtr const& buffer )
      {
        return( std::shared_ptr<ReadLockImpl>( new ReadLockImpl( buffer ) ) );
      }

      inline Buffer::DataReadLock::ReadLockImplSharedPtr Buffer::DataReadLock::ReadLockImpl::create( BufferSharedPtr const& buffer, size_t offset, size_t length )
      {
        return( std::shared_ptr<ReadLockImpl>( new ReadLockImpl( buffer, offset, length ) ) );
      }

      inline Buffer::DataReadLock::ReadLockImpl::ReadLockImpl( const BufferSharedPtr &buffer )
        : m_buffer( buffer )
        , m_ptr(0)
      {
        DP_ASSERT( buffer );
        m_ptr = m_buffer->lockRead( );
      }

      inline Buffer::DataReadLock::ReadLockImpl::ReadLockImpl( const BufferSharedPtr &buffer , size_t offset, size_t length)
        : m_buffer( buffer )
        , m_ptr(0)
      {
        DP_ASSERT( buffer );
        m_ptr = m_buffer->lockRead( offset, length );
      }

      inline Buffer::DataReadLock::ReadLockImpl::~ReadLockImpl()
      {
        m_buffer->unlockRead();
      }

      inline Buffer::DataWriteLock::WriteLockImplSharedPtr Buffer::DataWriteLock::WriteLockImpl::create( BufferSharedPtr const& buffer, Buffer::MapMode mapMode )
      {
        return( std::shared_ptr<WriteLockImpl>( new WriteLockImpl( buffer, mapMode ) ) );
      }

      inline Buffer::DataWriteLock::WriteLockImplSharedPtr Buffer::DataWriteLock::WriteLockImpl::create( BufferSharedPtr const& buffer, Buffer::MapMode mapMode, size_t offset, size_t length )
      {
        return( std::shared_ptr<WriteLockImpl>( new WriteLockImpl( buffer, mapMode, offset, length ) ) );
      }

      inline Buffer::DataWriteLock::WriteLockImpl::WriteLockImpl( const BufferSharedPtr &buffer, Buffer::MapMode mapMode )
        : m_buffer( buffer )
        , m_ptr(0)
      {
        DP_ASSERT( buffer );
        m_ptr = m_buffer->lock( mapMode );
      }

      inline Buffer::DataWriteLock::WriteLockImpl::WriteLockImpl( const BufferSharedPtr &buffer, Buffer::MapMode mapMode,
                                                              size_t offset, size_t length)
        : m_buffer( buffer )
        , m_ptr(0)
      {
        DP_ASSERT( buffer );
        m_ptr = m_buffer->lock( mapMode, offset, length );
      }

      inline Buffer::DataWriteLock::WriteLockImpl::~WriteLockImpl()
      {
        m_buffer->unlock();
      }

      template <typename ValueType>
      typename Buffer::Iterator<ValueType>::Type Buffer::getIterator( MapMode mapMode, size_t offset, int strideInBytes )
      {
        DataWriteLock writeLock( this->getSharedPtr<Buffer>(), mapMode );
        char *basePtr = reinterpret_cast<char*>(writeLock.getPtr()) + offset;
        return typename Buffer::Iterator<ValueType>::Type( reinterpret_cast<ValueType *>(basePtr), strideInBytes ? strideInBytes : sizeof(ValueType), writeLock );
      }

      /** \brief Returns an StridedConstIterator to access the buffer. The buffer stays write-locked as long as an iterator constructed from
                 the returned one exist.
       **/
      template <typename ValueType>
      typename Buffer::ConstIterator<ValueType>::Type Buffer::getConstIterator( size_t offset, int strideInBytes ) const
      {
        DataReadLock readLock( this->getSharedPtr<Buffer>() );
        const char *basePtr = reinterpret_cast<const char*>(readLock.getPtr()) + offset;
        return ConstIterator<ValueType>::Type( reinterpret_cast<const ValueType *>(basePtr), strideInBytes ? strideInBytes : sizeof(ValueType), readLock );
      }


    } // namespace core
  } // namespace sg
} // namespace dp

