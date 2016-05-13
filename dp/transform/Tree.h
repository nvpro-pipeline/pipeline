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

#include <dp/transform/Config.h>
#include <dp/util/BitArray.h>
#include <dp/util/Observer.h>
#include <dp/math/Matmnt.h>

namespace dp
{
  namespace transform
  {

    DEFINE_PTR_TYPES(TransformTree);

    typedef uint32_t Index;

    class Tree : public dp::util::Subject
    {
    public:
      typedef std::vector<dp::math::Mat44f> Transforms;

      /** \brief EventWorldMatricesChanged is triggered after compute() to notify observers which world matrices have been changed **/
      class EventWorldMatricesChanged : public dp::util::Event
      {
      public:
        EventWorldMatricesChanged(dp::util::BitArray const & dirtyWorldMatrices)
          : m_dirtyWorldMatrices(dirtyWorldMatrices)
        {
        }

        dp::util::BitArray const & getDirtyWorldMatrices() const { return m_dirtyWorldMatrices; }

      private:
        dp::util::BitArray const & m_dirtyWorldMatrices;
      };

      DP_TRANSFORM_API Tree();
      DP_TRANSFORM_API virtual ~Tree();

      /** \brief Add a new Transform to the Tree.
          \param parentIndex The parent transform of the Tree. Use \sa Tree::getRoot() if a root transform is required.
          \param matrix The local matrix of the transform.
      **/
      DP_TRANSFORM_API Index addTransform(Index parentIndex, dp::math::Mat44f const & matrix);

      /** \brief Remove a single Transform from the Tree. The children of the deleted transform will
                 not be deleted resulting in orphaned subtrees.
          \param transformIndex Index to the transform to delete.
      **/
      DP_TRANSFORM_API void removeTransform(Index transformIndex);

      //! \brief Recompute the values in the transform tree
      DP_TRANSFORM_API virtual void compute(dp::math::Mat44f const & camera);

      dp::math::Mat44f const & getWorldMatrix(Index index) const { return m_matricesWorld[index]; }
      dp::math::Mat44f const * getWorldMatrices() const { return m_matricesWorld.data(); }
      size_t            getTransformCount() const { return m_level.size(); }
      dp::util::BitArray const & getDirtyWorldMatrices() const { return m_dirtyWorldMatrices; }

      void updateLocalMatrix(Index index, dp::math::Mat44f const & matrix) { m_matricesLocal[index] = matrix; m_dirtyTransforms.enableBit(index); }

      //! \brief Get index of the virtual root node
      Index getRoot() const { return 0; }

    protected:
      //! \brief Resize data structures to new size
      DP_TRANSFORM_API virtual void resizeDataStructures(size_t newSize);

      DP_TRANSFORM_API void notifyTransformsChanged(dp::util::BitArray const & dirtyWorldMatrices);

      DP_TRANSFORM_API Index allocateIndex();
      DP_TRANSFORM_API void freeIndex(Index transformIndex);

      bool isValidIndex(Index transformIndex)
      {
        return transformIndex < m_freeTransforms.getSize() && !m_freeTransforms.getBit(transformIndex);
      }

      dp::util::BitArray m_freeTransforms;     // free if bit is true, occupied otherwise
      dp::util::BitArray m_dirtyTransforms;    // true if a transform has been changed
      dp::util::BitArray m_dirtyWorldMatrices; // a bitarray which specifies which world matrices has changed during the last compute iteration

      Transforms m_matricesLocal;
      Transforms m_matricesWorld;

      bool  m_transformsManaged;
      Index m_maxUsedTransform; // max ever used transformIndex

      struct TransformListEntry {
        unsigned int parent;    // index to parent transform in transform array
        unsigned int transform; // index to transform in transform array
      };

      typedef std::vector<TransformListEntry> TransformListEntries;

      struct TransformLevel {
        TransformListEntries transformListEntries;
      };

      typedef std::vector<TransformLevel> TransformLevels;
      TransformLevels m_transformLevels;

      std::vector<uint32_t> m_level; // level per node
    };

  } // namespace transform
} // namespace dp
