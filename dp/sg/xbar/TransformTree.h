// Copyright NVIDIA Corporation 2011-2012
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
#include <dp/sg/xbar/Tree.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/math/Matmnt.h>
#include <dp/util/BitMask.h>
#include <dp/util/Config.h>
#include <dp/util/WeakPtr.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      typedef TreeNodeBaseClass::NodeIndex TransformTreeIndex;

      struct TransformTreeNode : public TreeNodeBaseClass
      {
        enum DirtyBits
        {
          DEFAULT_DIRTY = BIT(0) // default dirty flag: update transform and bits
        };

        enum NodeBits
        {
          ISMIRRORTRANSFORM = BIT(0)
        };

        TransformTreeNode()
          : m_localBits( 0 )
          , m_worldBits( 0 )
          , m_objectTreeIndex( ~0 )
        {

        }

        TransformTreeNode( const TransformTreeNode& rhs )
          : TreeNodeBaseClass( rhs )

          , m_transform( rhs.m_transform )
          , m_billboard( rhs.m_billboard )
          , m_localMatrix( rhs.m_localMatrix )
          , m_worldMatrix( rhs.m_worldMatrix )
          , m_localBits( rhs.m_localBits )
          , m_worldBits ( rhs.m_worldBits )
          , m_objectTreeIndex( rhs.m_objectTreeIndex )
        {

        }

        TransformTreeNode& operator=( const TransformTreeNode& rhs )
        {
          TreeNodeBaseClass::operator=( rhs );

          m_transform = rhs.m_transform;
          m_billboard = rhs.m_billboard;
          m_localMatrix = rhs.m_localMatrix;
          m_worldMatrix = rhs.m_worldMatrix;
          m_localBits = rhs.m_localBits;
          m_worldBits  = rhs.m_worldBits;
          m_objectTreeIndex = rhs.m_objectTreeIndex;

          return *this;
        }


        inline void setLocalBits( unsigned int bits, bool value )
        {
          m_localBits = value ? m_localBits |  bits
            : m_localBits & ~bits;
        }

        inline void setWorldBits( unsigned int bits, bool value )
        {
          m_worldBits = value ? m_worldBits |  bits
            : m_worldBits & ~bits;
        }

        dp::sg::core::TransformWeakPtr m_transform;
        dp::sg::core::BillboardWeakPtr m_billboard;
        dp::math::Mat44f               m_localMatrix;
        dp::math::Mat44f               m_worldMatrix;
        unsigned int                   m_localBits;
        unsigned int                   m_worldBits;
        ObjectTreeIndex                m_objectTreeIndex;
#if defined(DP_ARCH_ARM_32) // alignment for culling
        unsigned int                   m_dummy;
#endif
      };

      class TransformTree : public TreeBaseClass< TransformTreeNode, TransformTreeIndex >
      {};
      typedef std::set< TransformTreeIndex > TransformTreeIndexSet;

    } // namespace xbar
  } // namespace sg
} // namespace dp
