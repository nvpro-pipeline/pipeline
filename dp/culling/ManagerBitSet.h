// Copyright NVIDIA Corporation 2012
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

#include <dp/culling/Manager.h>

namespace dp
{
  namespace culling
  {

    class ManagerBitSet : public Manager
    {
    public:
      DP_CULLING_API ManagerBitSet();
      DP_CULLING_API virtual ~ManagerBitSet();
      DP_CULLING_API virtual void objectSetBoundingBox( const ObjectHandle& object, const dp::math::Box3f& boundingBox );
      DP_CULLING_API virtual void objectSetTransformIndex( const ObjectHandle& object, size_t index );
      DP_CULLING_API virtual void objectSetUserData( const ObjectHandle& object, SmartPayload const& userData );
      DP_CULLING_API virtual SmartPayload const& objectGetUserData( const ObjectHandle& object );

      DP_CULLING_API virtual void groupAddObject( const GroupHandle& group, const ObjectHandle& object );
      DP_CULLING_API virtual ObjectHandle groupGetObject( const GroupHandle& group, size_t index );
      DP_CULLING_API virtual void groupRemoveObject( const GroupHandle& group, const ObjectHandle& object );
      DP_CULLING_API virtual size_t groupGetCount( const GroupHandle& group );
      DP_CULLING_API virtual void groupSetMatrices( GroupHandle const& group, void const* matrices, size_t numberOfMatrices, size_t stride );
      DP_CULLING_API virtual void groupMatrixChanged( GroupHandle const& group, size_t index );

      DP_CULLING_API virtual std::vector<ObjectHandle> const & resultGetChanged( ResultHandle const & result );
      DP_CULLING_API virtual bool resultObjectIsVisible( ResultHandle const& result, ObjectHandle const& object );

      DP_CULLING_API virtual dp::math::Box3f getBoundingBox( const GroupHandle& group ) const;
      DP_CULLING_API virtual dp::math::Box3f calculateBoundingBox( const GroupHandle& group ) const;
    };

  } // namespace culling
} // namespace dp
