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

#include <dp/culling/Config.h>
#include <dp/culling/ManagerBitSet.h>

namespace dp
{
  namespace culling
  {
    namespace cpu
    {

#if 0
      class ManagerImpl : public Manager
      {
      public:
        virtual ObjectSharedPtr objectCreate( PayloadSharedPtr const& userData );
        virtual void objectSetBoundingBox( const ObjectSharedPtr& object, const dp::math::Box3f& boundingBox );
        virtual void objectSetTransformIndex( const ObjectSharedPtr& object, size_t index );
        virtual void objectSetUserData( const ObjectSharedPtr& object, PayloadSharedPtr const& userData );
        virtual bool objectIsVisible( const ObjectSharedPtr& object );

        virtual PayloadSharedPtr const& objectGetUserData( const ObjectSharedPtr& object );

        virtual GroupSharedPtr groupCreate();
        virtual void groupAddObject( const GroupSharedPtr& group, const ObjectSharedPtr& object );
        virtual ObjectSharedPtr groupGetObject( const GroupSharedPtr& group, size_t index );
        virtual void groupRemoveObject( const GroupSharedPtr& group, const ObjectSharedPtr& object );
        virtual size_t groupGetCount( const GroupSharedPtr& group );
        virtual void groupSetMatrices( const GroupSharedPtr& group, void const* matrices, size_t numberOfMatrices, size_t stride );
        virtual void groupMatrixChanged( GroupSharedPtr const& group, size_t index );
        virtual ResultSharedPtr groupCreateResult( GroupSharedPtr const& group );

        virtual GroupSharedPtr resultGetChanged( const ResultSharedPtr& result );
        virtual bool resultObjectIsVisible( ResultSharedPtr const& result, ObjectSharedPtr const& object );

        virtual void cull( const GroupSharedPtr& group, const ResultSharedPtr& result, const dp::math::Mat44f& viewProjection );

        virtual dp::math::Box3f getBoundingBox( const GroupSharedPtr& group ) const;
#else
      class ManagerImpl : public Manager
      {
      public:
        ManagerImpl();
        virtual ~ManagerImpl();
        virtual ObjectSharedPtr objectCreate( PayloadSharedPtr const& userData );
        virtual GroupSharedPtr groupCreate();
        virtual ResultSharedPtr groupCreateResult( GroupSharedPtr const& group );

        virtual void cull( const GroupSharedPtr& group, const ResultSharedPtr& result, const dp::math::Mat44f& viewProjection );
      };
#endif

    } // namespace cpu
  } // namespace culling
} // namespace dp
