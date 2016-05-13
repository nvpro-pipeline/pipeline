// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/sg/core/Camera.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      TransformTree::TransformTree()
      {
        m_transformObserver = TransformObserver::create(m_dirtyTransforms);
        resizeDataStructures(m_tree.getTransformCount());
      }

      TransformTree::~TransformTree()
      {
      }

      TransformIndex TransformTree::addTransform(TransformIndex parentIndex, dp::sg::core::TransformSharedPtr const & transform)
      {
        dp::transform::Index index = m_tree.addTransform(parentIndex, transform->getTrafo().getMatrix());
        resizeDataStructures(m_tree.getTransformCount()); // synchronize local data structures to tree data structures

        m_objects[index] = transform;

        m_transformObserver->attach(transform, index); // observe transform

        return index;
      }

      void TransformTree::removeTransform(TransformIndex transformIndex)
      {
        m_tree.removeTransform(transformIndex);
        m_transformObserver->detach(transformIndex);
        m_objects[transformIndex].reset();
      }

      TransformIndex TransformTree::addBillboard(TransformIndex parentIndex, dp::sg::core::BillboardSharedPtr const & billboard)
      {
        // Billboards are not used anymore, but add a lot of complexity. Replace them by identity matrices for now.
        TransformIndex index = m_tree.addTransform(parentIndex, dp::math::cIdentity44f);
        resizeDataStructures(m_tree.getTransformCount()); // synchronize local data structures to tree data structures
        m_objects[index] = billboard;
        return index;
      }

      void TransformTree::removeBillboard(TransformIndex billboardIndex)
      {
        m_tree.removeTransform(billboardIndex);
        m_objects[billboardIndex].reset();
      }

      void TransformTree::resizeDataStructures(size_t newSize)
      {
        if (newSize != m_objects.size())
        {
          m_objects.resize(newSize);
          m_dirtyTransforms.resize(newSize);
        }
      }

      void TransformTree::compute(dp::sg::core::CameraSharedPtr const & camera)
      {
        m_dirtyTransforms.traverseBits([&](size_t index)
        {
          m_tree.updateLocalMatrix(static_cast<dp::transform::Index>(index), std::static_pointer_cast<dp::sg::core::Transform>(m_objects[index])->getMatrix());
        } );

        m_tree.compute(camera->getViewToWorldMatrix());
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
