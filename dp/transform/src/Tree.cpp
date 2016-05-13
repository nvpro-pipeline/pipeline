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

#include <dp/transform/Tree.h>

namespace dp
{
  namespace transform
  {

    namespace
    {
      const size_t VectorGrowth = 65536;
    }

    Tree::Tree()
    {
      resizeDataStructures(VectorGrowth);

      m_freeTransforms.disableBit(0);
      m_dirtyTransforms.enableBit(0);
      m_matricesLocal[0] = dp::math::cIdentity44f;
      m_matricesWorld[0] = dp::math::cIdentity44f;

      m_level[0] = -1;
    }

    Tree::~Tree()
    {
    }

    Index Tree::addTransform(Index parentIndex, dp::math::Mat44f const & matrix)
    {
      if (!isValidIndex(parentIndex))
      {
        throw std::runtime_error("Tree::addTransform: Parent does not exist");
      }

      Index newIndex = allocateIndex();

      m_level[newIndex] = m_level[parentIndex] + 1;
      m_dirtyTransforms.enableBit(newIndex);

      if (m_transformLevels.size() <= m_level[newIndex])
      {
        m_transformLevels.resize(m_level[newIndex] + 1);
      }

      // TODO add parent info or entry to this list in transform info?
      m_transformLevels[m_level[newIndex]].transformListEntries.push_back(TransformListEntry{ parentIndex, newIndex });
      m_matricesLocal[newIndex] = matrix;

      return newIndex;
    }

    void Tree::removeTransform(Index index)
    {
      if (!isValidIndex(index))
      {
        throw std::runtime_error("Tree::removeTransform: Transform does not exist");
      }

      // TODO store position in array in offset for efficient removal
      TransformLevel &level = m_transformLevels[m_level[index]];
      for (size_t entry = 0; entry < level.transformListEntries.size(); ++entry)
      {
        if (level.transformListEntries[entry].transform == index)
        {
          level.transformListEntries[entry] = level.transformListEntries.back();
          level.transformListEntries.pop_back();
          break;
        }
      }

      freeIndex(index);
    }

    Index Tree::allocateIndex()
    {
      Index newIndex = checked_cast<Index>(m_freeTransforms.countLeadingZeroes());
      if (newIndex == m_freeTransforms.getSize()) {
        resizeDataStructures(m_freeTransforms.getSize() + VectorGrowth);
      }

      m_freeTransforms.disableBit(newIndex);

      return newIndex;
    }

    void Tree::freeIndex(Index Index)
    {
      m_freeTransforms.enableBit(Index);
    }

    void Tree::resizeDataStructures(size_t newSize)
    {
      m_matricesLocal.resize(newSize);
      m_matricesWorld.resize(newSize);

      m_freeTransforms.resize(newSize, true);
      m_dirtyTransforms.resize(newSize, false);
      m_dirtyWorldMatrices.resize(newSize, false);
      m_level.resize(newSize);
    }

    void Tree::notifyTransformsChanged(dp::util::BitArray const &dirtyWorldMatrices)
    {
      notify(EventWorldMatricesChanged(dirtyWorldMatrices));
    }

    void Tree::compute(dp::math::Mat44f const & camera)
    {
      for (TransformLevel const &transformLevel : m_transformLevels)
      {
#if 0
        // update billboards
        for (BillboardListEntry const &billboardEntry : transformLevel.billboardListEntries)
        {
          dp::math::Mat44f parentMatrix = m_transforms[billboardEntry.parent].world;
          parentMatrix.invert();
          dp::math::Trafo t = m_transformInfos[billboardEntry.transform].object.inplaceCast<dp::sg::core::Billboard>()->getTrafo(camera, parentMatrix);
          m_transforms[billboardEntry.transform].local = t.getMatrix();

          m_transforms[billboardEntry.transform].world = m_transforms[billboardEntry.transform].local * m_transforms[billboardEntry.parent].world;
          m_dirtyWorldMatrices.enableBit(billboardEntry.transform);
          notifyTransformUpdated(billboardEntry.transform, m_transforms[billboardEntry.transform]);
        }
#endif

        // update transforms
        for (TransformListEntry const &transformEntry : transformLevel.transformListEntries)
        {
          if (m_dirtyWorldMatrices.getBit(transformEntry.parent) || m_dirtyTransforms.getBit(transformEntry.transform))
          {
            m_matricesWorld[transformEntry.transform] = m_matricesLocal[transformEntry.transform] * m_matricesWorld[transformEntry.parent];
            m_dirtyWorldMatrices.enableBit(transformEntry.transform);
          }
        }
      }
      notifyTransformsChanged(m_dirtyWorldMatrices);

      m_dirtyTransforms.clear();
      m_dirtyWorldMatrices.clear();
    }

  } // namespace sg
} // namespace dp
