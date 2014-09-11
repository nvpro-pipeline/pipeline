// Copyright NVIDIA Corporation 2012
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

#include <dp/util/Config.h>

#if defined(DP_OS_WINDOWS)
  #include <windows.h>
#elif defined(DP_OS_LINUX)
  #include <pthread.h>
  #ifndef PTHREAD_MUTEX_ERRORCHECK
    #define PTHREAD_MUTEX_ERRORCHECK PTHREAD_MUTEX_ERRORCHECK_NP
  #endif
#endif

namespace dp
{
  namespace util
  {

    /// Lock class implementing a critical region that only one thread can enter at a time.
    class Lock
    {

    public:

      /// Constructor.
      #if defined(DP_OS_WINDOWS)
        inline Lock() { InitializeCriticalSection(&m_critical_section); };
      #elif defined(DP_OS_LINUX)
        inline Lock() { pthread_mutex_init(&m_mutex, NULL); };
      #endif

      /// Destructor.
      #if defined(DP_OS_WINDOWS)
        inline ~Lock() { DeleteCriticalSection(&m_critical_section); };
      #elif defined(DP_OS_LINUX)
        inline ~Lock() { pthread_mutex_destroy(&m_mutex); };
      #endif

      /// Utility class to acquire a lock that is released by the destructor.
      class Block {

      public:

        /// Constructor.
        ///
        /// \param lock   If not \c NULL, this lock is acquired. If \c NULL, #set() can be used to
        ///               explicitly acquire a lock later.
        explicit Block(Lock* lock = 0);

        /// Destructor.
        ///
        /// Releases the lock (if it is acquired).
        ~Block();

        /// Acquires a lock.
        ///
        /// Releases the current lock (if it is set) and acquires the given lock. Useful to acquire
        /// a different lock, or to acquire a lock if no lock was acquired in the constructor.
        ///
        /// \param lock   The new lock to acquire.
        void set(Lock* lock);

        /// Releases the lock.
        ///
        /// Useful to release the lock before the destructor is called.
        void release();

      private:

        // The lock associated with this helper class.
        Lock* m_lock;
      };

    protected:

      /// Lock the lock.
      #if defined(DP_OS_WINDOWS)
        void lock() { EnterCriticalSection(&m_critical_section); };
      #elif defined(DP_OS_LINUX)
        void lock() { pthread_mutex_lock(&m_mutex); };
      #endif

      /// Unlock the lock.
      #if defined(DP_OS_WINDOWS)
        void unlock() { LeaveCriticalSection(&m_critical_section); };
      #elif defined(DP_OS_LINUX)
        void unlock() { pthread_mutex_unlock(&m_mutex); };
      #endif

    private:

      // This class is non-copyable.
      Lock(Lock const &);

      // This class is non-assignable.
      Lock& operator= (Lock const &);

      // The mutex implementing the lock.
      #if defined(DP_OS_WINDOWS)
        CRITICAL_SECTION m_critical_section;
      #elif defined(DP_OS_LINUX)
        pthread_mutex_t m_mutex;
      #endif
    };

    inline Lock::Block::Block(Lock* lock)
    {
      m_lock = lock;
      if (m_lock)
      {
        m_lock->lock();
      }
    }

    inline Lock::Block::~Block()
    {
      if (m_lock)
      {
        m_lock->unlock();
      }
    }

    inline void Lock::Block::set(Lock* lock)
    {
      if (m_lock)
      {
        m_lock->unlock();
      }

      m_lock = lock;

      if (m_lock)
      {
        m_lock->lock();
      }
    }

    inline void Lock::Block::release()
    {
      if ( m_lock )
      {
        m_lock->unlock();
        m_lock = 0;
      }
    }

  }  // namespace util
}  // namespace dp
