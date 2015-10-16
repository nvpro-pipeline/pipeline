// Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/xbar/xbar.h>
#include <dp/math/Matmnt.h>
#include <dp/util/BitArray.h>
#include <dp/sg/core/CoreTypes.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      DEFINE_PTR_TYPES(TransformObserver);
      DEFINE_PTR_TYPES(TransformTree);

      typedef uint32_t TransformIndex;

      class TransformTree : public dp::util::Subject
      {
      public:
        struct Transform {
          dp::math::Mat44f local;
          dp::math::Mat44f world;
        };

        typedef std::vector<Transform> Transforms;

        /** \brief EventTransform used to notify about changed within transforms **/
        class EventTransform : public dp::util::Event
        {
        public:
          EventTransform(TransformIndex index, dp::sg::xbar::TransformTree::Transform const & transform)
            : m_index(index)
            , m_transform(transform)
          {
          }

          TransformIndex getIndex() const { return m_index; }
          dp::sg::xbar::TransformTree::Transform const& getTransform() const { return m_transform; }

        private:
          TransformIndex                                 m_index;
          dp::sg::xbar::TransformTree::Transform const & m_transform;
        };

        TransformTree();
        ~TransformTree();

        TransformIndex addTransform(TransformIndex parentIndex, dp::sg::core::TransformSharedPtr const & transform);
        void removeTransform(TransformIndex transformIndex);

        TransformIndex addBillboard(TransformIndex parentIndex, dp::sg::core::BillboardSharedPtr const & billboard);
        void removeBillboard(TransformIndex transformIndex);

        //! \brief Recompute the values in the transform tree
        void compute(dp::sg::core::CameraSharedPtr const & camera);

        dp::math::Mat44f const & getWorldMatrix(TransformIndex transformIndex) const { return m_transforms[transformIndex].world; }
        Transforms const & getTransforms() const { return m_transforms; }
        dp::util::BitArray const & getDirtyWorldMatrices() const { return m_dirtyWorldMatrices; }

        //! \brief Get index of the virtual root node
        TransformIndex getSentinel() const { return 0; }

      private:
        //! \brief Resize data structures to new size
        void resizeDataStructures(size_t newSize);

        void notifyTransformUpdated(TransformIndex index, dp::sg::xbar::TransformTree::Transform const & transform);

        TransformIndex allocateIndex();
        void freeIndex(TransformIndex transformIndex);
        bool isValidIndex(TransformIndex transformIndex)
        {
          return transformIndex < m_transformFreeVector.getSize() && !m_transformFreeVector.getBit(transformIndex);
        }

        dp::util::BitArray m_transformFreeVector; // free if bit is true, occupied otherwise
        dp::util::BitArray m_dirtyTransforms; // true if a transform has been changed
        dp::util::BitArray m_dirtyWorldMatrices; // a bitarray which specifies which world matrices has changed during the last compute iteration

        Transforms m_transforms; // array with all transforms

        struct TransformListEntry {
          unsigned int parent;    // index to parent transform in transform array
          unsigned int transform; // index to transform in transform array
        };

        struct BillboardListEntry {
          unsigned int parent;    // index to parent transform in transform array
          unsigned int transform; // index to transform in transform array
          dp::sg::core::BillboardSharedPtr billboard; // billboard to use for computation
        };

        typedef std::vector<TransformListEntry> TransformListEntries;
        typedef std::vector<BillboardListEntry> BillboardListEntries;

        struct TransformLevel {
          TransformListEntries transformListEntries;
          BillboardListEntries billboardListEntries;
        };

        typedef std::vector<TransformLevel> TransformLevels;
        TransformLevels m_transformLevels;

        // internal transform information
        struct TransformInfo {
          dp::sg::core::ObjectSharedPtr object;
          uint32_t                      level;
        };

        typedef std::vector<TransformInfo> TransformInfos;
        TransformInfos m_transformInfos;

        TransformObserverSharedPtr m_transformObserver;
        bool m_firstCompute;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
