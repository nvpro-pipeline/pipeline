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

#include <dp/util/Types.h>

//
// Check what version of implementation we can use
//

#if defined(DP_COMPILER_MSVC) && ( defined(DP_ARCH_X86) || defined(DP_ARCH_X86_64) )
#  define DP_ATOM_INTERLOCKED
#pragma warning(push)
#pragma warning(disable : 4985 )
#include <math.h> // math.h is actually not required. It's there to avoid later 4985 warnings issued due to an msvc 2008 compiler bug.
#include <intrin.h>
#pragma warning(pop)
#  pragma intrinsic (_InterlockedExchangeAdd)
#  pragma intrinsic (_InterlockedCompareExchange)
#elif (defined(DP_COMPILER_GCC) || defined(DP_COMPILER_ICC)) && defined(DP_ARCH_X86)
#  define DP_ATOM_GCC_ASM_X86
#else
#  define DP_ATOM_GENERIC
#  include <dp/util/Lock.h>
#endif

namespace dp {
  namespace util {

    /// 32-bit unsigned counter with atomic arithmetic, increments, and decrements.
    ///
    class Atom32
    {
    public:
        /// Default constructor initializes the counter to zero.
        Atom32() : m_atom(0) {}

        /// Constructor initializes the counter to \c val.
        Atom32(const Uint32 val) : m_atom(val) {}

        /// Assigns \c val to the counter.
        Uint32 operator = (const Uint32 val) {
            m_atom = val;
            return val;
        }

        /// Adds \c val to counter.
        Uint32 operator += (const Uint32 val);

        /// Subtracts \c val from counter.
        Uint32 operator -= (const Uint32 val);

        /// Increments counter by one.
        Uint32 operator ++();

        /// Increments counter by one.
        Uint32 operator ++(int);

        /// Decrements counter by one.
        Uint32 operator --();

        /// Decrements counter by one.
        Uint32 operator --(int);

        /// Implicit conversion operator to \c Uint32.
        operator Uint32 () const {
            return m_atom;
        }

        /// Assign \c val to counter and return the old value of counter.
        Uint32 swap(const Uint32 val);

    private:
        volatile Uint32 m_atom;

    #if defined(DP_ATOM_GENERIC)
        dp::util::Lock m_lock;
    #endif
    };

    #if defined(DP_ATOM_INTERLOCKED)

    //
    // operator +=
    //
    __forceinline Uint32 Atom32::operator += ( const Uint32 rhs) {
        return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),rhs) + rhs;
    }

    //
    // operator -=
    //
    __forceinline Uint32 Atom32::operator -= ( const Uint32 rhs) {
        return _InterlockedExchangeAdd(
            reinterpret_cast<volatile long *>(&m_atom), -static_cast<const Int32>(rhs)) - rhs;
    }

    //
    // operator ++
    //
    __forceinline Uint32 Atom32::operator ++() {
        return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),1L) + 1L;
    }

    //
    // operator --
    //
    __forceinline Uint32 Atom32::operator --() {
        return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),-1L) - 1L;
    }

    //
    // operator ++
    //
    __forceinline Uint32 Atom32::operator ++(/**/int) {
        return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),1L);
    }

    //
    // operator --
    //
    __forceinline Uint32 Atom32::operator --(/**/int) {
        return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),-1L);
    }

    //
    // atomic value swapper
    //
    __forceinline Uint32 Atom32::swap( const Uint32 rhs) {
        return _InterlockedExchange(reinterpret_cast<volatile long *>(&m_atom),rhs);
    }

    #elif defined(DP_ATOM_GCC_ASM_X86)

    //
    // operator +=
    //
    inline Uint32 Atom32::operator += ( const Uint32 rhs) {
        Uint32 retval;
        asm volatile (
            "movl %2,%0\n"
            "lock; xaddl %0,%1\n"
            "addl %2,%0\n"
            : "=&r" (retval), "+m" (m_atom)
            : "r" (rhs)
            : "cc"
            );
        return retval;
    }

    //
    // operator -=
    //
    inline Uint32 Atom32::operator -= ( const Uint32 rhs) {
        Uint32 retval;
        asm volatile (
            "neg %2\n"
            "movl %2,%0\n"
            "lock; xaddl %0,%1\n"
            "addl %2,%0\n"
            : "=&r" (retval), "+m" (m_atom)
            : "r" (rhs)
            : "cc", "%2"
            );
        return retval;
    }

    //
    // operator ++
    //
    inline Uint32 Atom32::operator ++() {
        Uint32 retval;
        asm volatile (
            "movl $1,%0\n"
            "lock; xaddl %0,%1\n"
            "addl $1,%0\n"
            : "=&r" (retval), "+m" (m_atom)
            :
            : "cc"
            );
        return retval;
    }

    //
    // operator --
    //
    inline Uint32 Atom32::operator --() {
        Uint32 retval;
        asm volatile (
            "movl $-1,%0\n"
            "lock; xaddl %0,%1\n"
            "addl $-1,%0\n"
            : "=&r" (retval), "+m" (m_atom)
            :
            : "cc"
            );
        return retval;
    }

    //
    // operator ++
    //
    inline Uint32 Atom32::operator ++(/**/int) {
        Uint32 retval;
        asm volatile (
            "movl $1,%0\n"
            "lock; xaddl %0,%1\n"
            : "=&r" (retval), "+m" (m_atom)
            :
            : "cc"
            );
        return retval;
    }

    //
    // operator --
    //
    inline Uint32 Atom32::operator --(/**/int) {
        Uint32 retval;
        asm volatile (
            "movl $-1,%0\n"
            "lock; xaddl %0,%1\n"
            : "=&r" (retval), "+m" (m_atom)
            :
            : "cc"
            );
        return retval;
    }

    //
    // atomic value swapper
    //
    inline Uint32 Atom32::swap( const Uint32 rhs) {
        Uint32 retval;
        asm volatile (
        "0:\n"
            "movl %1,%0\n"
            "lock; cmpxchg %2,%1\n"
            "jnz 0b\n"
            : "=&a" (retval), "+m" (m_atom)
            : "r" (rhs)
            : "cc"
            );
        return retval;
    }

    #elif defined(DP_ATOM_GENERIC)

    //
    // operator +=
    //
    inline Uint32 Atom32::operator += ( const Uint32 rhs) {
        dp::util::Lock::Block block(&m_lock);
        return m_atom += rhs;
    }

    //
    // operator -=
    //
    inline Uint32 Atom32::operator -= ( const Uint32 rhs) {
        dp::util::Lock::Block block(&m_lock);
        return m_atom -= rhs;
    }

    //
    // operator ++
    //
    inline Uint32 Atom32::operator ++() {
        dp::util::Lock::Block block(&m_lock);
        return ++m_atom;
    }

    //
    // operator --
    //
    inline Uint32 Atom32::operator --() {
        dp::util::Lock::Block block(&m_lock);
        return --m_atom;
    }

    //
    // operator ++
    //
    inline Uint32 Atom32::operator ++(/**/int) {
        dp::util::Lock::Block block(&m_lock);
        return m_atom++;
    }

    //
    // operator --
    //
    inline Uint32 Atom32::operator --(/**/int) {
        dp::util::Lock::Block block(&m_lock);
        return m_atom--;
    }

    //
    // atomic value swap operation
    //
    inline Uint32 Atom32::swap( const Uint32 rhs) {
        dp::util::Lock::Block block(&m_lock);
        Uint32 retval = m_atom;
        m_atom = rhs;
        return retval;
    }

    #else // DP_ATOM_GENERIC
      #error One of DP_ATOM_INTERLOCKED, DP_ATOM_GCC_ASM_X86 or DP_ATOM_GENERIC must be selected
    #endif

    #undef DP_ATOM_INTERLOCKED
    #undef DP_ATOM_GCC_ASM_X86
    #undef DP_ATOM_GENERIC

  } // namespace util
} // namespace dp
