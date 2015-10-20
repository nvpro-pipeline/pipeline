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

#include <dp/sg/xbar/TransformTree.h>
#include <dp/sg/xbar/inc/TransformObserver.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/Billboard.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      namespace
      {
        const size_t VectorGrowth = 65536;
      }

      TransformTree::TransformTree()
        : m_firstCompute(true)
      {
        resizeDataStructures(VectorGrowth);

        m_transformFreeVector.disableBit(0);
        m_transforms[0].local = dp::math::cIdentity44f;
        m_transforms[0].world = dp::math::cIdentity44f;

        m_transformInfos[0].level = -1;

        m_transformObserver = TransformObserver::create(m_dirtyTransforms);
      }

      TransformTree::~TransformTree()
      {
      }

      TransformIndex TransformTree::addTransform(TransformIndex parentIndex, dp::sg::core::TransformSharedPtr const & transform)
      {
        if (!isValidIndex(parentIndex))
        {
          throw std::runtime_error("TransformTree::addTransform: Parent does not exist");
        }

        TransformIndex newIndex = allocateIndex();
        m_transformInfos[newIndex].object = transform;
        m_transformInfos[newIndex].level = m_transformInfos[parentIndex].level + 1;
        m_dirtyTransforms.enableBit(newIndex);

        m_transformObserver->attach(transform, newIndex);

        if (m_transformLevels.size() <= m_transformInfos[newIndex].level)
        {
          m_transformLevels.resize(m_transformInfos[newIndex].level + 1);
        }

        // TODO add parent info or entry to this list in transform info?
        m_transformLevels[m_transformInfos[newIndex].level].transformListEntries.push_back(TransformListEntry { parentIndex, newIndex });

        return newIndex;
      }

      void TransformTree::removeTransform(TransformIndex transformIndex)
      {
        if (!isValidIndex(transformIndex))
        {
          throw std::runtime_error("TransformTree::removeTransform: Transform does not exist");
        }

        // TODO store position in array in offset for efficient removal
        TransformLevel &level = m_transformLevels[m_transformInfos[transformIndex].level];
        for (size_t index = 0;index < level.transformListEntries.size();++index)
        {
          if (level.transformListEntries[index].transform == transformIndex)
          {
            level.transformListEntries[index] = level.transformListEntries.back();
            level.transformListEntries.pop_back();
          }
        }

        m_transformObserver->detach(transformIndex);

        freeIndex(transformIndex);

        // this is slower than necessary depending on what should be cleared
        m_transformInfos[transformIndex] = TransformInfo();

      }

      TransformIndex TransformTree::addBillboard(TransformIndex parentIndex, dp::sg::core::BillboardSharedPtr const & billboard)
      {
        if (!isValidIndex(parentIndex))
        {
          throw std::runtime_error("TransformTree::addBillboard: Parent does not exist");
        }

        TransformIndex newIndex = allocateIndex();
        m_transformInfos[newIndex].object = billboard;
        m_transformInfos[newIndex].level = m_transformInfos[parentIndex].level + 1;

        if (m_transformLevels.size() <= m_transformInfos[newIndex].level)
        {
          m_transformLevels.resize(m_transformInfos[newIndex].level + 1);
        }

        // TODO add parent info or entry to this list in transform info?
        m_transformLevels[m_transformInfos[newIndex].level].billboardListEntries.push_back(BillboardListEntry{ parentIndex, newIndex });

        return newIndex;
      }

      void TransformTree::removeBillboard(TransformIndex billboardIndex)
      {
        if (!isValidIndex(billboardIndex))
        {
          throw std::runtime_error("TransformTree::removeTransform: Transform does not exist");
        }

        // TODO store position in array in offset for efficient removal
        TransformLevel &level = m_transformLevels[m_transformInfos[billboardIndex].level];
        for (size_t index = 0; index < level.billboardListEntries.size(); ++index)
        {
          if (level.billboardListEntries[index].transform == billboardIndex)
          {
            level.billboardListEntries[index] = level.billboardListEntries.back();
            level.billboardListEntries.pop_back();
          }
        }

        freeIndex(billboardIndex);

        // this is slower than necessary depending on what should be cleared
        m_transformInfos[billboardIndex] = TransformInfo();
      }

      TransformIndex TransformTree::allocateIndex()
      {
        TransformIndex newIndex = checked_cast<TransformIndex>(m_transformFreeVector.countLeadingZeroes());
        if (newIndex == m_transformFreeVector.getSize()) {
          resizeDataStructures(m_transformFreeVector.getSize() + VectorGrowth);
        }

        m_transformFreeVector.disableBit(newIndex);

        return newIndex;
      }

      void TransformTree::freeIndex(TransformIndex transformIndex)
      {
        m_transformFreeVector.enableBit(transformIndex);
      }

      void TransformTree::resizeDataStructures(size_t newSize)
      {
        m_transformFreeVector.resize(newSize, true);
        m_dirtyTransforms.resize(newSize, false);
        m_dirtyWorldMatrices.resize(newSize, false);
        m_transforms.resize(newSize);
        m_transformInfos.resize(newSize);
      }

      void TransformTree::compute(dp::sg::core::CameraSharedPtr const & camera)
      {
        m_dirtyWorldMatrices.clear();
        if (m_firstCompute) {
          // after first iteration world matrix 0 is dirty
          m_dirtyWorldMatrices.enableBit(0);
          m_firstCompute = false;
        }
        m_dirtyTransforms.traverseBits([&](size_t index)
        {
            m_transforms[index].local = m_transformInfos[index].object.inplaceCast<dp::sg::core::Transform>()->getMatrix();
            m_dirtyWorldMatrices.enableBit(index);
        } );

        for (TransformLevel const &transformLevel : m_transformLevels)
        {
          // update billboards
          for (BillboardListEntry const &billboardEntry : transformLevel.billboardListEntries)
          {
            dp::math::Mat44f parentMatrix = m_transforms[billboardEntry.parent].world;
            parentMatrix.invert();
            dp::math::Trafo t = m_transformInfos[billboardEntry.transform].object.inplaceCast<dp::sg::core::Billboard>()->getTrafo(camera, parentMatrix);
            m_transforms[billboardEntry.transform].local = t.getMatrix();

            m_transforms[billboardEntry.transform].world = m_transforms[billboardEntry.transform].local * m_transforms[billboardEntry.parent].world;
            m_dirtyWorldMatrices.enableBit(billboardEntry.transform);
          }

          // update transforms
          for (TransformListEntry const &transformEntry : transformLevel.transformListEntries)
          {
            if (m_dirtyWorldMatrices.getBit(transformEntry.parent) || m_dirtyWorldMatrices.getBit(transformEntry.transform))
            {
              m_transforms[transformEntry.transform].world = m_transforms[transformEntry.transform].local * m_transforms[transformEntry.parent].world;
              m_dirtyWorldMatrices.enableBit(transformEntry.transform);
            }
          }
        }

        notify(EventTransform(m_dirtyWorldMatrices));

        m_dirtyTransforms.clear();
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
