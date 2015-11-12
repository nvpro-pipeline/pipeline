// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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
/** @file */

#include <dp/util/Reflection.h>

// required declaration
namespace dp
{
  namespace math
  {
    class Trafo;

    // Vector
    template<unsigned int n, typename T> class Vecnt;
    typedef Vecnt<2,float>  Vec2f;
    typedef Vecnt<2,double> Vec2d;
    typedef Vecnt<3,float>  Vec3f;
    typedef Vecnt<3,double> Vec3d;
    typedef Vecnt<4,float>  Vec4f;
    typedef Vecnt<4,double> Vec4d;
    typedef Vecnt<2,int> Vec2i;
    typedef Vecnt<3,int> Vec3i;
    typedef Vecnt<4,int> Vec4i;
    typedef Vecnt<2,unsigned int> Vec2ui;
    typedef Vecnt<3,unsigned int> Vec3ui;
    typedef Vecnt<4,unsigned int> Vec4ui;


    // Matrix
    template<unsigned int m, unsigned int n, typename T> class Matmnt;
    typedef Matmnt<3,3,float>   Mat33f;
    typedef Matmnt<3,3,double>  Mat33d;
    typedef Matmnt<4,4,float>   Mat44f;
    typedef Matmnt<4,4,double>  Mat44d;

    // Quaternions
    template<typename T> class Quatt;
    typedef Quatt<float>   Quatf;
    typedef Quatt<double>  Quatd;
  }
}

namespace dp
{
  namespace sg
  {
    namespace core
    {
      DEFINE_PTR_TYPES( Billboard );
      DEFINE_PTR_TYPES( Buffer );
      DEFINE_PTR_TYPES( BufferHost );
      DEFINE_PTR_TYPES( Camera );
      DEFINE_PTR_TYPES( ClipPlane );
      DEFINE_PTR_TYPES( FrustumCamera );
      DEFINE_PTR_TYPES( GeoNode );
      DEFINE_PTR_TYPES( Group );
      DEFINE_PTR_TYPES( HandledObject );
      DEFINE_PTR_TYPES( IndexSet );
      DEFINE_PTR_TYPES( LightSource );
      DEFINE_PTR_TYPES( LOD );
      DEFINE_PTR_TYPES( MatrixCamera );
      DEFINE_PTR_TYPES( Node );
      DEFINE_PTR_TYPES( Object );
      DEFINE_PTR_TYPES( ParallelCamera );
      DEFINE_PTR_TYPES( ParameterGroupData );
      DEFINE_PTR_TYPES( Path );
      DEFINE_PTR_TYPES( PerspectiveCamera );
      DEFINE_PTR_TYPES( PipelineData );
      DEFINE_PTR_TYPES( Primitive );
      DEFINE_PTR_TYPES( Sampler );
      DEFINE_PTR_TYPES( Scene );
      DEFINE_PTR_TYPES( Switch );
      DEFINE_PTR_TYPES( Texture );
      DEFINE_PTR_TYPES( TextureFile );
      DEFINE_PTR_TYPES( TextureHost );
      DEFINE_PTR_TYPES( Transform );
      DEFINE_PTR_TYPES( VertexAttributeSet );

    } // namespace core
  } // namespace sg
} // namespace dp


namespace dp
{
  namespace util
  {

    SHARED_OBJECT_TRAITS_BASE( dp::sg::core::HandledObject );
    SHARED_OBJECT_TRAITS_BASE( dp::sg::core::Path );

    SHARED_OBJECT_TRAITS( dp::sg::core::Object,              dp::sg::core::HandledObject );
    SHARED_OBJECT_TRAITS( dp::sg::core::Camera,              dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::FrustumCamera,       dp::sg::core::Camera );
    SHARED_OBJECT_TRAITS( dp::sg::core::ParallelCamera,      dp::sg::core::FrustumCamera );
    SHARED_OBJECT_TRAITS( dp::sg::core::PerspectiveCamera,   dp::sg::core::FrustumCamera );
    SHARED_OBJECT_TRAITS( dp::sg::core::MatrixCamera,        dp::sg::core::Camera );
    SHARED_OBJECT_TRAITS( dp::sg::core::Node,                dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::Group,               dp::sg::core::Node );
    SHARED_OBJECT_TRAITS( dp::sg::core::Billboard,           dp::sg::core::Group );
    SHARED_OBJECT_TRAITS( dp::sg::core::LOD,                 dp::sg::core::Group );
    SHARED_OBJECT_TRAITS( dp::sg::core::Transform,           dp::sg::core::Group );
    SHARED_OBJECT_TRAITS( dp::sg::core::Switch,              dp::sg::core::Group );
    SHARED_OBJECT_TRAITS( dp::sg::core::LightSource,         dp::sg::core:: Node );
    SHARED_OBJECT_TRAITS( dp::sg::core::ClipPlane,           dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::GeoNode,             dp::sg::core::Node );
    SHARED_OBJECT_TRAITS( dp::sg::core::Primitive,           dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::VertexAttributeSet,  dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::IndexSet,            dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::ParameterGroupData,  dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::PipelineData,        dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::Scene,               dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::Buffer,              dp::sg::core::HandledObject );
    SHARED_OBJECT_TRAITS( dp::sg::core::BufferHost,          dp::sg::core::Buffer );
    SHARED_OBJECT_TRAITS( dp::sg::core::Sampler,             dp::sg::core::Object );
    SHARED_OBJECT_TRAITS( dp::sg::core::Texture,             dp::sg::core::HandledObject );
    SHARED_OBJECT_TRAITS( dp::sg::core::TextureFile,         dp::sg::core::Texture );
    SHARED_OBJECT_TRAITS( dp::sg::core::TextureHost,         dp::sg::core::Texture );


    /************************************************************************/
    /* Reflection                                                           */
    /************************************************************************/
    /*! \brief Specialization of the TypedPropertyEnum template for type SceneSharedPtr. */
    template <> struct TypedPropertyEnum< dp::sg::core::SceneSharedPtr > {
      enum { type = Property::TYPE_SCENE };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type Texture. */
    template <> struct TypedPropertyEnum< dp::sg::core::TextureSharedPtr> {
      enum { type = Property::TYPE_TEXTURE };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type NodeSharedPtr. */
    template <> struct TypedPropertyEnum< dp::sg::core::NodeSharedPtr > {
      enum { type = Property::TYPE_NODE };
    };

  } // namespace util
} // namespace dp
