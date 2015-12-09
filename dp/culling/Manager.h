// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/math/Matmnt.h>
#include <dp/math/Boxnt.h>
#include <dp/util/SharedPtr.h>

namespace dp
{
  namespace culling
  {
    DEFINE_PTR_TYPES( Object );
    class Object
    {
      public:
        ObjectSharedPtr create()
        {
          return( std::shared_ptr<Object>( new Object() ) );
        }

        virtual ~Object() {}

      protected:
        Object() {}
    };

    DEFINE_PTR_TYPES( Group );
    class Group : public std::enable_shared_from_this<Group>
    {
      public:
        GroupSharedPtr create()
        {
          return( std::shared_ptr<Group>( new Group() ) );
        }

        virtual ~Group() {}

      protected:
        Group() {}
    };

    DEFINE_PTR_TYPES( Result );
    class Result
    {
      public:
        ResultSharedPtr create()
        {
          return( std::shared_ptr<Result>( new Result() ) );
        }

        virtual ~Result() {}

      protected:
        Result() {}
    };

    DEFINE_PTR_TYPES( Payload );
    class Payload
    {
      public:
        virtual ~Payload() {}

      protected:
        Payload() {}
    };


    enum class Mode
    {
        CPU
      , OPENGL_COMPUTE
      , CUDA
      , AUTO // figure out which culling is best automatically
    };

    /** \brief This is the culling interface for all frustum culling methods in devtech platform.
               An object is a representation of an instance to cull.
               A group is a collection of objects. One object can added to one group at a time.
               A result manages the visibility of a list of objects in a group. It supports querying a list of objects
               whose visiblity has changed since the last view.
               **/
    class Manager
    {
    public:
      DP_CULLING_API virtual ~Manager();

      DP_CULLING_API virtual ObjectSharedPtr objectCreate( PayloadSharedPtr const & userData ) = 0;
      DP_CULLING_API virtual void objectSetBoundingBox( ObjectSharedPtr const & object, dp::math::Box3f const & boundingBox ) = 0 ;
      DP_CULLING_API virtual void objectSetTransformIndex( ObjectSharedPtr const & object, size_t index ) = 0;
      DP_CULLING_API virtual void objectSetUserData( ObjectSharedPtr const & object, PayloadSharedPtr const & userData ) = 0;
      DP_CULLING_API virtual PayloadSharedPtr const & objectGetUserData( ObjectSharedPtr const & object ) = 0;

      DP_CULLING_API virtual GroupSharedPtr groupCreate() = 0;
      DP_CULLING_API virtual void groupAddObject( GroupSharedPtr const & group, const ObjectSharedPtr& object ) = 0;
      DP_CULLING_API virtual ObjectSharedPtr groupGetObject( GroupSharedPtr const & group, size_t index ) = 0;
      DP_CULLING_API virtual void groupRemoveObject( GroupSharedPtr const & group, ObjectSharedPtr const & object ) = 0;
      DP_CULLING_API virtual size_t groupGetCount( GroupSharedPtr const & group ) = 0;
      DP_CULLING_API virtual void groupSetMatrices( GroupSharedPtr const & group, void const * matrices, size_t numberOfMatrices, size_t stride ) = 0;
      DP_CULLING_API virtual void groupMatrixChanged( GroupSharedPtr const & group, size_t index ) = 0;
      DP_CULLING_API virtual ResultSharedPtr groupCreateResult( GroupSharedPtr const & group ) = 0;

      /** \brief Get a list of objects whose visiblity is changed between the last and current draw call **/
      DP_CULLING_API virtual std::vector<ObjectSharedPtr> const & resultGetChanged( ResultSharedPtr const& result ) = 0;

      /** \brief Query if an object is visible within a result. **/
      DP_CULLING_API virtual bool resultObjectIsVisible( ResultSharedPtr const& result, ObjectSharedPtr const& object ) = 0;

      /** \brief Cull a given group and store the result in the given result.
          \param group The group which contains the objects to cull
          \param result The result object which stores the result for the given group. The result must match the given group.
          \param viewProjection The camera/projection matrix
      **/
      DP_CULLING_API virtual void cull( GroupSharedPtr const & group, ResultSharedPtr const & result, dp::math::Mat44f const & viewProjection ) = 0;

      /** \brief Compute the bounding box for the given group **/
      DP_CULLING_API virtual dp::math::Box3f getBoundingBox( GroupSharedPtr const & group ) const = 0;
    };

  } // namespace culling
} // namespace dp
