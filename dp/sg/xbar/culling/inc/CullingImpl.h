// Copyright NVIDIA Corporation 2013
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

#include <dp/sg/xbar/culling/Culling.h>
#include <dp/culling/Manager.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      namespace culling
      {

        class CullingImpl : public Culling, dp::util::Observer
        {
        public:
          CullingImpl( SceneTreeSharedPtr const & sceneTree, dp::culling::Mode cullingMode );
          virtual ~CullingImpl();

          virtual ResultHandle resultCreate();
          virtual bool resultIsVisible( ResultHandle const & result, ObjectTreeIndex objectTreeIndex ) const;
          virtual std::vector<dp::sg::xbar::ObjectTreeIndex> const & resultGetChangedIndices( ResultHandle const & result ) const;
          virtual void cull( ResultHandle const & result, dp::math::Mat44f const & world2ViewProjection );
          virtual dp::math::Box3f getBoundingBox();

        protected:

          // observer framework
          virtual void onNotify( dp::util::Event const & event, dp::util::Payload * payload );
          virtual void onDestroyed( dp::util::Subject const & subject, dp::util::Payload * payload );

        private:
          //! \brief Add the object at the given tree location to the culling group
          void addObject(ObjectTreeIndex objectTreeIndex );

          //! \brief Update bounding box for the given ObjectTreeIndex
          void updateBoundingBox( ObjectTreeIndex objectTreeIndex );

        private:
          SceneTreeSharedPtr const m_sceneTree;

          // culling data

          //! \brief Payload class which assigns an ObjectTreeIndex to each culling object.
          class Payload : public dp::util::RCObject
          {
          public:
            Payload( ObjectTreeIndex objectTreeIndex )
              : m_objectTreeIndex( objectTreeIndex )
            {
            }

            ~Payload()
            {
            }

            ObjectTreeIndex getObjectTreeIndex() const { return m_objectTreeIndex; }

          private:
            ObjectTreeIndex m_objectTreeIndex;
          };

          std::unique_ptr<dp::culling::Manager>  m_culling;
          dp::culling::GroupHandle               m_cullingGroup;
          std::vector<dp::culling::ObjectHandle> m_objects;
        };

      } // namespace culling
    } // namespace xbar
  } // namespace sg
} // namespace dp
