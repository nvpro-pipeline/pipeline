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

#include <dp/sg/xbar/culling/Config.h>
#include <dp/culling/Manager.h>
#include <dp/util/SmartPtr.h>
#include <dp/sg/xbar/SceneTree.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      namespace culling
      {

        class Culling;
        typedef dp::util::SmartPtr<Culling> CullingSharedPtr;

        class Result;
        typedef dp::util::SmartPtr<Result> ResultHandle;


        class Result : public dp::util::RCObject
        {
        public:
          DP_SG_XBAR_CULLING_API virtual ~Result();
        };

        /** \brief This class provides culling on all GeoNodes in a SceneTree. It supports multiple viewports at the same time through the ResultHandle.
                   After calling cull the function resultGetChangedIndices returns a list of ObjectTree indices with changed visibility. It is possible
                   to support multiple viewports by creating multiple results.
        **/
        class Culling : public dp::util::RCObject
        {
        public:
          DP_SG_XBAR_CULLING_API virtual ~Culling();

          /** \brief Create a new Culling object for the given SceneTree **/
          DP_SG_XBAR_CULLING_API static CullingSharedPtr create( SceneTreeSharedPtr const& sceneTree, dp::culling::Mode cullingMode );

          /** \brief Create a culling result. This handle contains the visibility of all objects in the SceneTree **/
          DP_SG_XBAR_CULLING_API virtual ResultHandle resultCreate() = 0;

          /** \brief Check if an object was visible during the last cull call **/
          DP_SG_XBAR_CULLING_API virtual bool resultIsVisible( ResultHandle const & result, ObjectTreeIndex objectTreeIndex ) const = 0;

          /** \brief Get the list of ObjectTree indices whose visibility has changed during the last cull call **/
          DP_SG_XBAR_CULLING_API virtual std::vector<dp::sg::xbar::ObjectTreeIndex> const & resultGetChangedIndices( ResultHandle const & ) const = 0;

          /** \brief Cull the SceneTree against the given world2ViewProjection matrix and update the given result **/
          DP_SG_XBAR_CULLING_API virtual void cull( ResultHandle const& result, dp::math::Mat44f const & world2ViewProjection ) = 0;

          /** \brief Calculate the bounding box of the SceneTree. Currently all active and inactive objects are used to calculate the result **/
          DP_SG_XBAR_CULLING_API virtual dp::math::Box3f getBoundingBox( ) = 0;
        };

      } // namespace culling
    } // namespace xbar
  } // namespace sg
} // namespace dp
