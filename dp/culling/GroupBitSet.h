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

#include <dp/culling/ObjectBitSet.h>
#include <boost/shared_array.hpp>
#include <dp/util/BitArray.h>
#include <dp/util/Observer.h>

namespace dp
{
  namespace culling
  {
    HANDLE_TYPES( GroupBitSet );

    class GroupBitSet : public Group, public dp::util::Subject
    {
    public:
      //! \brief Object index remap event. Object moved from oldIndex to newIndex in the group array.
      class Event : public dp::util::Event
      {
      public:
        Event( size_t oldIndex, size_t newIndex )
          : m_oldIndex( oldIndex )
          , m_newIndex( newIndex )
        {
        }

        size_t getOldIndex() const { return m_oldIndex; }
        size_t getNewIndex() const { return m_newIndex; }

      private:
        size_t m_oldIndex;
        size_t m_newIndex;
      };

      DP_CULLING_API static GroupBitSetHandle create();
      DP_CULLING_API virtual ~GroupBitSet();

      DP_CULLING_API void addObject( const ObjectBitSetHandle& object );
      DP_CULLING_API void removeObject( const ObjectBitSetHandle& object );
      DP_CULLING_API void clearObjects();
      const ObjectBitSetHandle& getObject( size_t index ) const;
      size_t getObjectCount() const;
      DP_CULLING_API void setMatrices( const void* matrices, size_t numberOfMatrices, size_t stride );
      void const* getMatrices() const;
      size_t getMatricesStride() const;
      size_t getMatricesCount() const;
      void markMatrixDirty( size_t index );

      size_t getObjectIncarnation() const;

      void setBoundingBoxDirty( bool dirty );
      bool isBoundingBoxDirty() const;

      void setOBBDirty( bool dirty );
      bool isOBBDirty() const;

      dp::math::Box3f const& getBoundingBox() const;
      void setBoundingBox( dp::math::Box3f const& boundingBox );

    protected:
      DP_CULLING_API GroupBitSet();

    protected:
      dp::util::BitArray              m_dirtyMatrices;
      bool                            m_inputChanged;
      bool                            m_matricesChanged;
      bool                            m_obbDirty;
      size_t                          m_objectIncarnation; // incremented on add/removeObject, TODO replace by observer
      std::vector<ObjectBitSetHandle> m_objects;

    private:
      bool m_boundingBoxDirty;
      dp::math::Box3f    m_boundingBox;

      const void* m_matrices;
      size_t      m_matricesCount;
      size_t      m_matricesStride; 
    };

    /************************************************************************/
    /* Inline functions                                                     */
    /************************************************************************/
    inline ObjectBitSetHandle const& GroupBitSet::getObject( size_t index ) const
    {
      DP_ASSERT( index < m_objects.size() );
      return m_objects[index];
    }

    inline size_t GroupBitSet::getObjectCount() const
    {
      return m_objects.size();
    }

    inline void const* GroupBitSet::getMatrices() const
    {
      return m_matrices;
    }

    inline size_t GroupBitSet::getMatricesStride() const
    {
      return m_matricesStride;
    }

    inline size_t GroupBitSet::getMatricesCount() const
    {
      return m_matricesCount;
    }

    inline size_t GroupBitSet::getObjectIncarnation() const
    {
      return m_objectIncarnation;
    }

    inline void GroupBitSet::markMatrixDirty( size_t index )
    {
      // it is legal to call markMatrixDirty with index >= m_matricesCount. In this case
      // the matrices haven't been updated yet.
      if ( index < m_matricesCount )
      {
        m_dirtyMatrices.enableBit( index );
        m_boundingBoxDirty = true;
        m_obbDirty = true;
      }
    }

    inline void GroupBitSet::setBoundingBoxDirty( bool dirty )
    {
      m_boundingBoxDirty = dirty;
    }

    inline bool GroupBitSet::isBoundingBoxDirty() const
    {
      return m_boundingBoxDirty;
    }

    inline void GroupBitSet::setOBBDirty( bool dirty )
    {
      m_obbDirty = dirty;
    }

    inline bool GroupBitSet::isOBBDirty() const
    {
      return m_obbDirty;
    }

    inline dp::math::Box3f const& GroupBitSet::getBoundingBox() const
    {
      return m_boundingBox;
    }

    inline void GroupBitSet::setBoundingBox( dp::math::Box3f const& boundingBox )
    {
      m_boundingBox = boundingBox;
    }


  } // namespace culling
} // namespace dp
