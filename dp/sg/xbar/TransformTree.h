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

#pragma once

#include <dp/sg/xbar/xbar.h>
#include <dp/math/Matmnt.h>
#include <dp/util/BitArray.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/transform/Tree.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      DEFINE_PTR_TYPES(TransformObserver);
      DEFINE_PTR_TYPES(TransformTree);

      typedef uint32_t TransformIndex;

      class TransformTree
      {
      public:
        struct Transform {
          dp::math::Mat44f local;
          dp::math::Mat44f world;
        };

        typedef std::vector<Transform> Transforms;

        TransformTree();
        virtual ~TransformTree();

        TransformIndex addTransform(TransformIndex parentIndex, dp::sg::core::TransformSharedPtr const & transform);
        void removeTransform(TransformIndex transformIndex);

        TransformIndex addBillboard(TransformIndex parentIndex, dp::sg::core::BillboardSharedPtr const & billboard);
        void removeBillboard(TransformIndex transformIndex);

        //! \brief Recompute the values in the transform tree
        void compute(dp::sg::core::CameraSharedPtr const & camera);

        dp::transform::Tree & getTree() { return m_tree; }

      private:
        //! \brief Resize data structures to new size
        void resizeDataStructures(size_t newSize);

        // internal transform information
        struct TransformInfo {
          dp::sg::core::ObjectSharedPtr object;
        };

        // per object data structures
        typedef std::vector<dp::sg::core::ObjectSharedPtr> Objects;

        Objects m_objects;
        dp::util::BitArray m_dirtyTransforms; // true if a transform has been changed

        // observer
        TransformObserverSharedPtr m_transformObserver;

        // data structure which keeps all transforms
        dp::transform::Tree m_tree;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
