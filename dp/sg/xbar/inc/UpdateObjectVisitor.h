// Copyright (c) 2010-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/xbar/SceneTree.h>
#include <vector>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      class UpdateObjectVisitor
      {
      public:
        UpdateObjectVisitor( ObjectTree& tree, SceneTree *sceneTree )
          : m_objectTree(tree)
          , m_sceneTree( sceneTree )
        {
        }

        struct Data {};

        bool preTraverse( ObjectTreeIndex index, const Data& data )
        {
          ObjectTreeNode& current = m_objectTree[index];

          unsigned int newMask;
          bool         newActive;

          // need a parent to propagate info downwards
          if( current.m_parentIndex != ~0 )
          {
            const ObjectTreeNode& curParent = m_objectTree[current.m_parentIndex];

            // propagate dirty bits downward
            current.m_dirtyBits |= curParent.m_dirtyBits;

            if( current.m_dirtyBits & ObjectTreeNode::DEFAULT_DIRTY )
            {
              // compute local information
              // propagate all hints downwards, invalidate DrawableInstance3 hints
              unsigned int newHints = current.m_localHints | curParent.m_worldHints;
              // only propagate mask bits downward that are set everywhere
              newMask =  current.m_localMask & curParent.m_worldMask;
              // propagate active flag downwards
              newActive = current.m_localActive && curParent.m_worldActive;

              // update di if one is there
              if( current.m_isDrawable)
              {
                if( current.m_worldHints != newHints )
                {
                  if ( newHints & dp::sg::core::Object::DP_SG_HINT_ALWAYS_INVISIBLE )
                  {
                    newActive = false;
                  }
                  else if ( newHints & dp::sg::core::Object::DP_SG_HINT_ALWAYS_VISIBLE )
                  {
                    newActive = true;
                  }
                }
              }

              current.m_worldHints = newHints;
            }
          }
          else
          {
            current.m_worldHints  = current.m_localHints;
            newMask   = current.m_localMask;
            newActive = current.m_localActive;
          }

          if( current.m_worldMask != newMask )
          {
            current.m_worldMask = newMask;
            if ( current.m_isDrawable )
            {
              m_sceneTree->notify( SceneTree::Event( index, current, SceneTree::Event::Type::TRAVERSAL_MASK_CHANGED ) );
            }
          }

          if( current.m_worldActive != newActive )
          {
            current.m_worldActive = newActive;
            if ( current.m_isDrawable )
            {
              m_sceneTree->notify( SceneTree::Event( index, current, SceneTree::Event::Type::ACTIVE_CHANGED ) );
            }
          }

          return true;
        };

        void postTraverse( ObjectTreeIndex index, const Data& data )
        {
          // TODO hack! need better dirty handling
          //m_objectTree[index].m_dirtyBits &= ~( ObjectTreeNode::DEFAULT_DIRTY );
          m_objectTree[index].m_dirtyBits = 0;
        };

      protected:
        ObjectTree& m_objectTree;
        SceneTree *m_sceneTree;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
