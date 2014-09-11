// Copyright NVIDIA Corporation 2002-2007
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

#include <dp/sg/core/SharedPtr.h>
#include <dp/util/Reflection.h>
#include <dp/util/SmartPtr.h>

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

/*! \brief Macro to define the three standard types for a type T.
 *  \remark For convenience, for each class T, we define the types TSharedPtr, TWeakPtr, and TLock */
#define CORE_TYPES(T)                               \
  typedef dp::sg::core::SharedPtr<T>  T##SharedPtr; \
  typedef T*                          T##WeakPtr;

/*! \brief Macro to define ObjectType and our four standard types of a base type T as part of a templated struct.
 *  \remark Using this struct, the standard types Handle, SharedPtr, WeakPtr and Lock, as well as
 *  the ObjectType itself, are easily available within a template context. */
#define OBJECT_TRAITS_BASE(T)                     \
template <> struct ObjectTraits<T>                \
{                                                 \
  typedef T                           ObjectType; \
  typedef dp::sg::core::SharedPtr<T>  SharedPtr;  \
  typedef T*                          WeakPtr;    \
}

/*! \brief Macro to define ObjectType and our five standard types of a type T, with base type BT, as part of a templated struct.
 *  \remark Using this struct, the standard types Handle, SharedPtr, WeakPtr and Lock, as well as
 *  the ObjectType itself, are easily available within a template context. */
#define OBJECT_TRAITS(T, BT)                      \
template <> struct ObjectTraits<T>                \
{                                                 \
  typedef T                           ObjectType; \
  typedef BT                          Base;       \
  typedef dp::sg::core::SharedPtr<T>  SharedPtr;  \
  typedef T*                          WeakPtr;    \
}

namespace dp
{
  namespace sg
  {
    namespace core
    {

      class HandledObject;

      // Object types
      class Object;
        // cameras ...
        class Camera;
          class FrustumCamera;
            class ParallelCamera;
            class PerspectiveCamera;
          class MatrixCamera;
        // nodes and node components ...
        class Node;
          // ... groups
          class Group;
            class Billboard;
            class LOD;
            class Transform;
            class Switch;
          // ... lights
          class LightSource;
          // ... clip plane
          class ClipPlane;
          // ... geometry
          class GeoNode;

        // Primitives
        class Primitive;

        // VertexAttributes
        class VertexAttributeSet;

        // Indices
        class IndexSet;

        // Effects
        class EffectData;
        class ParameterGroupData;
        
        // Sampler
        class Sampler;

      // additionally required declarations
      class Scene;
      class VertexAttribute;

      // buffer types
      class Buffer;
      class BufferHost;

      // texture types
      class Texture;
      class TextureFile;
      class TextureHost;

      // Handle types
      CORE_TYPES( HandledObject );

      CORE_TYPES( Object );
      CORE_TYPES( Camera );
      CORE_TYPES( FrustumCamera );
      CORE_TYPES( ParallelCamera );
      CORE_TYPES( PerspectiveCamera );
      CORE_TYPES( MatrixCamera );
      CORE_TYPES( Node );
      CORE_TYPES( Group );
      CORE_TYPES( Billboard );
      CORE_TYPES( LOD );
      CORE_TYPES( Transform );
      CORE_TYPES( Switch );
      CORE_TYPES( LightSource );
      CORE_TYPES( ClipPlane );
      CORE_TYPES( GeoNode );
      CORE_TYPES( Primitive );
      CORE_TYPES( VertexAttributeSet );
      CORE_TYPES( IndexSet );
      CORE_TYPES( EffectData );
      CORE_TYPES( ParameterGroupData );
      CORE_TYPES( Scene );
      CORE_TYPES( Buffer );
      CORE_TYPES( BufferHost );
      CORE_TYPES( Sampler );
      CORE_TYPES( Texture );
      CORE_TYPES( TextureFile );
      CORE_TYPES( TextureHost );

      OBJECT_TRAITS_BASE( HandledObject );

      OBJECT_TRAITS( Object, HandledObject );
      OBJECT_TRAITS( Camera, Object );
      OBJECT_TRAITS( FrustumCamera, Camera );
      OBJECT_TRAITS( ParallelCamera, FrustumCamera );
      OBJECT_TRAITS( PerspectiveCamera, FrustumCamera );
      OBJECT_TRAITS( MatrixCamera, Camera );
      OBJECT_TRAITS( Node, Object );
      OBJECT_TRAITS( Group, Node );
      OBJECT_TRAITS( Billboard, Group );
      OBJECT_TRAITS( LOD, Group );
      OBJECT_TRAITS( Transform, Group );
      OBJECT_TRAITS( Switch, Group );
      OBJECT_TRAITS( LightSource, Node );
      OBJECT_TRAITS( ClipPlane, Object );
      OBJECT_TRAITS( GeoNode, Node );
      OBJECT_TRAITS( Primitive, Object );
      OBJECT_TRAITS( VertexAttributeSet, Object );
      OBJECT_TRAITS( IndexSet, Object );
      OBJECT_TRAITS( EffectData, Object );
      OBJECT_TRAITS( ParameterGroupData, Object );
      OBJECT_TRAITS( Scene, Object );
      OBJECT_TRAITS( Buffer, HandledObject );
      OBJECT_TRAITS( BufferHost, Buffer );
      OBJECT_TRAITS( Sampler, Object );
      OBJECT_TRAITS( Texture, HandledObject );
      OBJECT_TRAITS( TextureFile, Texture );
      OBJECT_TRAITS( TextureHost, Texture );

    } // namespace core
  } // namespace sg
} // namespace dp


namespace dp
{
  namespace util
  {

    /************************************************************************/
    /* Reflection                                                           */
    /************************************************************************/
    /*! \brief Specialization of the TypedPropertyEnum template for type SceneSharedPtr. */
    template <> struct TypedPropertyEnum< dp::sg::core::SceneSharedPtr > {
      enum { type = Property::TYPE_SCENE };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type VertexAttribute. */
    template <> struct TypedPropertyEnum< dp::sg::core::VertexAttribute> {
      enum { type = Property::TYPE_VERTEX_ATTRIBUTE };
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
