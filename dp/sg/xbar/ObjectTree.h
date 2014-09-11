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

#include <dp/math/Boxnt.h>
#include <dp/sg/xbar/Tree.h>
#include <dp/sg/xbar/TreeResourceGroup.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      typedef TreeNodeBaseClass::NodeIndex TransformTreeIndex;
      typedef TreeNodeBaseClass::NodeIndex ObjectTreeIndex;

      class LightInstance;
      class ClipPlaneInstance;
      typedef dp::util::SmartPtr< LightInstance >     LightInstanceSharedPtr;
      typedef dp::util::SmartPtr< ClipPlaneInstance > ClipPlaneInstanceSharedPtr;

      typedef TreeResourceGroup<ClipPlaneInstance>    ClipPlaneGroup;
      typedef dp::util::SmartPtr<ClipPlaneGroup>      SmartClipPlaneGroup;

      struct ObjectTreeNode : public TreeNodeBaseClass
      {
        enum DirtyBits
        {
          DEFAULT_DIRTY = BIT(0),  // default dirty flag: update hints, mask and active flag
        };

        ObjectTreeNode()
          : m_transformIndex( 0 )
          , m_object( nullptr )
          , m_localHints( 0 )
          , m_worldHints( 0 )
          , m_localActive( true )
          , m_worldActive( true )
          , m_isDrawable( false )
          , m_localMask( ~0 )
          , m_worldMask( ~0 )
        {}

        ObjectTreeNode( const ObjectTreeNode &rhs )
          : TreeNodeBaseClass( rhs )
          , m_transformIndex( rhs.m_transformIndex )
          , m_object( rhs.m_object )
          , m_localHints( rhs.m_localHints )
          , m_worldHints( rhs.m_worldHints )
          , m_localActive( rhs.m_localActive )
          , m_worldActive( rhs.m_worldActive )
          , m_isDrawable( rhs.m_isDrawable )
          , m_localMask( rhs.m_localMask )
          , m_worldMask( rhs.m_worldMask )
          , m_clipPlaneGroup( rhs.m_clipPlaneGroup )
        {
        }

        ObjectTreeNode& operator=( const ObjectTreeNode &rhs )
        {
          TreeNodeBaseClass::operator=(rhs);

          m_transformIndex = rhs.m_transformIndex;
          m_object = rhs.m_object;
          m_localHints = rhs.m_localHints;
          m_worldHints = rhs.m_worldHints;
          m_localActive = rhs.m_localActive;
          m_worldActive = rhs.m_worldActive;
          m_isDrawable  = rhs.m_isDrawable;
          m_localMask = rhs.m_localMask;
          m_worldMask = rhs.m_worldMask;
          m_clipPlaneGroup = rhs.m_clipPlaneGroup;

          return *this;
        }

        ~ObjectTreeNode()
        {
        }

        TransformTreeIndex          m_transformIndex; // index of the current parent transform
        dp::sg::core::ObjectWeakPtr m_object;         // the node's object in the tree
        unsigned int                m_localHints;     // the hints the corresponding node/primitive provides
        unsigned int                m_worldHints;     // the resulting hints for this node
        unsigned int                m_localMask;      // mask of the node's object
        unsigned int                m_worldMask;      // resulting mask
        bool                        m_localActive;    // false iff node is hidden due to Switch/LOD/FBA (can only be child of one at a time -> bool)
        bool                        m_worldActive;    // false iff node is hidden due to !m_localActive or !parent.m_localActive
        bool                        m_isDrawable;
                                    
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
