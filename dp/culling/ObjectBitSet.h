// Copyright NVIDIA Corporation 2012-2013
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

#include <dp/culling/Manager.h>
#include <dp/util/Types.h>

namespace dp
{
  namespace culling
  {
    HANDLE_TYPES( GroupBitSet );
    HANDLE_TYPES( ObjectBitSet );

    /************************************************************************/
    /* Keep this class always aligned to 16 bytes!                          */
    /************************************************************************/
    class ObjectBitSet : public Object
    {
    public:
      DP_CULLING_API static ObjectBitSetHandle create( SmartPayload const& userData );

      void setTransformIndex( size_t transformIndex );
      size_t getTransformIndex() const;

      void setLowerLeft( dp::math::Vec4f const& lowerLeft );
      dp::math::Vec4f const& getLowerLeft( ) const;

      void setExtent( dp::math::Vec4f const& lowerLeft );
      dp::math::Vec4f const& getExtent( ) const;

      void setUserData( SmartPayload const& userData );
      SmartPayload const& getUserData( ) const;

      void setGroupIndex( size_t groupIndex );
      size_t getGroupIndex() const;

      void setGroup( GroupBitSetHandle const & group );
      GroupBitSetHandle getGroup( ) const;

    protected:
      DP_CULLING_API ObjectBitSet( SmartPayload const& userData );

    protected:
      DP_ALIGN(16)                              // align to 16 bytes to allow usage with SSE
      dp::math::Vec4f         m_lowerLeft;
      dp::math::Vec4f         m_extent;
      size_t                  m_transformIndex;
      SmartPayload            m_userData;
      size_t                  m_groupIndex;
      GroupBitSet*            m_group;
    };

    inline void ObjectBitSet::setTransformIndex( size_t transformIndex )
    {
      m_transformIndex = transformIndex;
    }

    inline size_t ObjectBitSet::getTransformIndex() const
    {
      return m_transformIndex;
    }

    inline void ObjectBitSet::setLowerLeft( dp::math::Vec4f const& lowerLeft )
    {
      m_lowerLeft = lowerLeft;
    }

    inline dp::math::Vec4f const& ObjectBitSet::getLowerLeft( ) const
    {
      return m_lowerLeft;
    }

    inline void ObjectBitSet::setExtent( dp::math::Vec4f const& extent )
    {
      m_extent = extent;
    }

    inline dp::math::Vec4f const& ObjectBitSet::getExtent( ) const
    {
      return m_extent;
    }

    inline void ObjectBitSet::setUserData( SmartPayload const& userData )
    {
      m_userData = userData;
    }

    inline SmartPayload const& ObjectBitSet::getUserData( ) const
    {
      return m_userData;
    }

    inline void ObjectBitSet::setGroupIndex( size_t groupIndex )
    {
      m_groupIndex = groupIndex;
    }

    inline size_t ObjectBitSet::getGroupIndex() const
    {
      return m_groupIndex;
    }

  } // namespace culling
} // namespace dp
