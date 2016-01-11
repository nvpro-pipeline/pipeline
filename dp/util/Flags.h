// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <stdint.h>

namespace dp
{
  namespace util
  {

      template <typename BitType, typename MaskType = uint32_t>
      class Flags
      {
        public:
          Flags()
            : m_mask(0)
          {}

          Flags( BitType bit )
            : m_mask(static_cast<uint32_t>(bit))
          {}

          Flags( Flags<BitType> const& rhs )
            : m_mask(rhs.m_mask)
          {}

          Flags( std::initializer_list<BitType> const& rhs )
            : m_mask(0)
          {
            // TODO: check if compiler can unroll this loop
            for ( auto it = rhs.begin() ; it != rhs.end() ; ++it )
            {
              m_mask |= static_cast<uint32_t>(*it);
            }
          }

          Flags<BitType> & operator=( Flags<BitType> const& rhs )
          {
            m_mask = rhs.m_mask;
            return *this;
          }

          Flags<BitType> & operator|=( Flags<BitType> const& rhs )
          {
            m_mask |= rhs.m_mask;
            return *this;
          }

          Flags<BitType> & operator&=( Flags<BitType> const& rhs )
          {
            m_mask &= rhs.m_mask;
            return *this;
          }

          Flags<BitType> & operator^=( Flags<BitType> const& rhs )
          {
            m_mask ^= rhs.m_mask;
            return *this;
          }

          Flags<BitType> operator|( Flags<BitType> const& rhs ) const
          {
            Flags<BitType> result(*this);
            result |= rhs;
            return result;
          }

          Flags<BitType> operator&( Flags<BitType> const& rhs ) const
          {
            Flags<BitType> result(*this);
            result &= rhs;
            return result;
          }

          Flags<BitType> operator^( Flags<BitType> const& rhs ) const
          {
            Flags<BitType> result(*this);
            result ^= rhs;
            return result;
          }

          bool operator!() const
          {
            return !m_mask;
          }

          bool operator==( Flags<BitType> const& rhs ) const
          {
            return m_mask == rhs.m_mask;
          }

          bool operator!=( Flags<BitType> const& rhs ) const
          {
            return m_mask != rhs.m_mask;
          }

          operator bool() const
          {
            return !!m_mask;
          }

        private:
          MaskType  m_mask;
      };

      template <typename BitType>
      dp::util::Flags<BitType> operator|( BitType bit, dp::util::Flags<BitType> const& flags )
      {
        return flags | bit;
      }

      template <typename BitType>
      dp::util::Flags<BitType> operator&( BitType bit, dp::util::Flags<BitType> const& flags )
      {
        return flags & bit;
      }

      template <typename BitType>
      dp::util::Flags<BitType> operator^( BitType bit, dp::util::Flags<BitType> const& flags )
      {
        return flags ^ bit;
      }

  } // namespace util
} // namespace dp
