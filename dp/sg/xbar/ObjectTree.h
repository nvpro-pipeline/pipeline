// Copyright (c) 2011-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/math/Boxnt.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/xbar/Tree.h>
#include <dp/sg/xbar/TreeResourceGroup.h>
#include <dp/util/BitMask.h>
#include <dp/util/BitArray.h>
#include <dp/sg/core/Object.h>
#include <set>
#include <vector>
#include <map>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      typedef TreeNodeBaseClass::NodeIndex ObjectTreeIndex;
      typedef unsigned int                 TransformIndex;

      DEFINE_PTR_TYPES( ClipPlaneInstance );

      typedef TreeResourceGroup<ClipPlaneInstance>    ClipPlaneGroup;
      typedef dp::util::SharedPtr<ClipPlaneGroup>     SmartClipPlaneGroup;

      struct ObjectTreeNode : public TreeNodeBaseClass
      {
        enum DirtyBits
        {
          DEFAULT_DIRTY = BIT(0),  // default dirty flag: update hints, mask and active flag
        };

        ObjectTreeNode()
          : m_localHints( 0 )
          , m_worldHints( 0 )
          , m_localActive( true )
          , m_worldActive( true )
          , m_isDrawable( false )
          , m_isTransform( false )
          , m_isBillboard( false )
          , m_localMask( ~0 )
          , m_worldMask( ~0 )
        {}


        ~ObjectTreeNode()
        {
        }

        dp::sg::core::ObjectSharedPtr m_object;         // the node's object in the tree

        // new transform hierarchy
        TransformIndex              m_transform;       // id in transform array
        ObjectTreeIndex             m_transformParent; // object index of parent in transform hierarchy

        unsigned int                m_localHints;     // the hints the corresponding node/primitive provides
        unsigned int                m_worldHints;     // the resulting hints for this node
        unsigned int                m_localMask;      // mask of the node's object
        unsigned int                m_worldMask;      // resulting mask
        // TODO try bitmask for all those booleans and check if it has impact on performance
        bool                        m_localActive;    // false iff node is hidden due to Switch/LOD/FBA (can only be child of one at a time -> bool)
        bool                        m_worldActive;    // false iff node is hidden due to !m_localActive or !parent.m_localActive
        bool                        m_isDrawable;
        bool                        m_isTransform;    // object is any kind of transform
        bool                        m_isBillboard;    // object is billboard

        SmartClipPlaneGroup         m_clipPlaneGroup;
      };

      class ObjectTree : public TreeBaseClass< ObjectTreeNode, ObjectTreeIndex >
      {
      public:
        std::map< ObjectTreeIndex, dp::sg::core::SwitchWeakPtr > m_switchNodes;
        std::map< ObjectTreeIndex, dp::sg::core::LODWeakPtr >    m_LODs;
      };

      typedef std::set< ObjectTreeIndex > ObjectTreeIndexSet;

      typedef std::vector< dp::sg::core::SwitchWeakPtr > SwitchVector;

    } // namespace xbar
  } // namespace sg
} // namespace dp
