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
        virtual ObjectHandle objectCreate( const dp::util::SmartRCObject& userData );
        virtual void objectSetBoundingBox( const ObjectHandle& object, const dp::math::Box3f& boundingBox );
        virtual void objectSetTransformIndex( const ObjectHandle& object, size_t index );
        virtual void objectSetUserData( const ObjectHandle& object, const dp::util::SmartRCObject& userData );
        virtual bool objectIsVisible( const ObjectHandle& object );

        virtual const dp::util::SmartRCObject& objectGetUserData( const ObjectHandle& object );

        virtual GroupHandle groupCreate();
        virtual void groupAddObject( const GroupHandle& group, const ObjectHandle& object );
        virtual ObjectHandle groupGetObject( const GroupHandle& group, size_t index );
        virtual void groupRemoveObject( const GroupHandle& group, const ObjectHandle& object );
        virtual size_t groupGetCount( const GroupHandle& group );
        virtual void groupSetMatrices( const GroupHandle& group, void const* matrices, size_t numberOfMatrices, size_t stride );
        virtual void groupMatrixChanged( GroupHandle const& group, size_t index );
        virtual ResultHandle groupCreateResult( GroupHandle const& group );

        virtual GroupHandle resultGetChanged( const ResultHandle& result );
        virtual bool resultObjectIsVisible( ResultHandle const& result, ObjectHandle const& object );

        virtual void cull( const GroupHandle& group, const ResultHandle& result, const dp::math::Mat44f& viewProjection );

        virtual dp::math::Box3f getBoundingBox( const GroupHandle& group ) const;
#else
      class ManagerImpl : public Manager
      {
      public:
        ManagerImpl();
        virtual ~ManagerImpl();
        virtual ObjectHandle objectCreate( const dp::util::SmartRCObject& userData );
        virtual GroupHandle groupCreate();
        virtual ResultHandle groupCreateResult( GroupHandle const& group );

        virtual void cull( const GroupHandle& group, const ResultHandle& result, const dp::math::Mat44f& viewProjection );
      };
#endif

    } // namespace cpu
  } // namespace culling
} // namespace dp
