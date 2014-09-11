// Copyright NVIDIA Corporation 2009-2011
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


/*
\brief Generator to generate a series of testing scenes
*/

#pragma once

#include <dp/sg/generator/Config.h>

#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/fx/EffectSpec.h>
#include <dp/math/Vecnt.h>
#include <dp/math/Quatt.h>

namespace dp
{
  namespace sg
  {
    namespace generator
    {

      //! Generate a quadmesh with m x n quads, m rows and n columns
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createQuadSet( unsigned int m, unsigned int n, const float size = 1.0f, const float gap  = 0.5f );

      //! Generate a quad strip with n quads
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createQuadStrip( unsigned int n, float height = 1.0f , float radius = 1.0f );

      //! Generate a triangle mesh with m x n triangles, m rows and n columns
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTriSet( unsigned int m, unsigned int n, const float size = 1.0f, const float gap  = 0.5f );

      //! Generate a half-circle trifan with n segments
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTriFan( unsigned int n, const float radius = 1.0f, const float elevation = 0.0f );

      //! Generate \a rows rows and \a columns columns of tri strips with the width \a width and the height \a height of one triangle pair
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTriStrip( unsigned int rows, unsigned int columns, float width = 1.0f, float height = 1.0f );

      //! Generate a GeoNode with a TriPatches4 with n x m triangles, m rows and n columns, each of size \a size, separated by \a offset
      DP_SG_GENERATOR_API dp::sg::core::GeoNodeSharedPtr createTriPatches4( const std::vector<std::string> & searchPaths
                                              , unsigned int n, unsigned int m
                                              , const dp::math::Vec3f & size = dp::math::Vec3f( 4.0f, 4.0f, 4.0f )
                                              , const dp::math::Vec2f & offset = dp::math::Vec2f( 4.0f, 4.0f ) );

      //! Generate a GeoNode with a QuadPatches4x4 with n x m cylinders, seperated by \a offset
      DP_SG_GENERATOR_API dp::sg::core::GeoNodeSharedPtr createQuadPatches4x4( const std::vector<std::string> & searchPaths
                                                 , unsigned int n, unsigned int m
                                                 , const dp::math::Vec2f & offset = dp::math::Vec2f( 4.0f, 4.0f ) );

      //! Generate a unit cube inside the range [-1, 1]
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createCube();

      //! Generate a tetrahedron
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTetrahedron();

      //! Generate an octahedron
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createOctahedron();

      //! Generate a dodecahedron
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createDodecahedron();

      //! Generate an icosahedron
      // supported attributes: vertex, normal
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createIcosahedron();

      //! Generate a sphere around (0,0,0) with a radius of \a radius, m edges in longitudinal and n edges in latitudinal direction, both m and n should be at least 3
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createSphere( unsigned int m , unsigned int n , float radius = 1.0f );

      //! Generate a cylinder with a radius of size \a radius, m edges in longitudinal and n edges in latitudinal direction, both m and n should be at least 3
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createCylinder( float r, float h, unsigned int hdivs, unsigned int thdivs, bool bOuter = true, bool bcaps = true );

      //! Generate a torus with m edges in longitudinal and n edges in latitudinal direction, an inner radius of \a innerRadius and an outer radius of \a outerRadius
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTorus( unsigned int m, unsigned int n, float innerRadius = 1.0f, float outerRadius = 0.5f );

      //! Generate a tessellated plane with \a subdiv subdivisions and the possibility to apply a transformation matrix transf
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTessellatedPlane( unsigned int subdiv
                                                     , const dp::math::Mat44f &transf = dp::math::Mat44f( dp::util::makeArray( 1.0f, 0.0f, 0.0f, 0.0f
                                                                                                                       , 0.0f, 1.0f, 0.0f, 0.0f
                                                                                                                       , 0.0f, 0.0f, 1.0f, 0.0f
                                                                                                                       , 0.0f, 0.0f, 0.0f, 1.0f ) ) );

      /*! Create an XY aligned plane that could conveniantly be used to make pixel-aligned rectangles: ( \a x0, \a y0 ) - the bottom left corner of the rect, 
      and the width \a width in pixels, and the height \a height in pixels of the rect. \a wext and \a hext can be set to specify how far
      the texture coordinates extend at the top right corner of the rect*/
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createPlane( float x0, float y0, float width, float height, float wext = 1.0f, float hext = 1.0f);

      //! Generate a tessellated box out of tessellated planes with \a subdiv subdivisions
      // supported attributes: vertex, normal, texcoord0 (2D)
      DP_SG_GENERATOR_API dp::sg::core::PrimitiveSharedPtr createTessellatedBox( unsigned int subdiv );

      //! Generate a texture of size 8x8 with a colored checker pattern
      DP_SG_GENERATOR_API dp::sg::core::EffectDataSharedPtr createTexture();

      //! Generate a texture of size n x n with grey and alpha values
      DP_SG_GENERATOR_API dp::sg::core::EffectDataSharedPtr createAlphaTexture( unsigned int n=64 );

      //! Generate a GeoNode from a shape \a drawable
      DP_SG_GENERATOR_API dp::sg::core::GeoNodeSharedPtr createGeoNode( const dp::sg::core::PrimitiveSharedPtr &drawable );

      //! Generate a GeoNode from a shape \a drawable and a material effect \a materialEffect
      DP_SG_GENERATOR_API dp::sg::core::GeoNodeSharedPtr createGeoNode( const dp::sg::core::PrimitiveSharedPtr &drawable, const dp::sg::core::EffectDataSharedPtr & materialEffect );

      //! Generate a transformation for a node \a node with a translation vector \a translation as well as an orientation vector \a orientation
      DP_SG_GENERATOR_API dp::sg::core::TransformSharedPtr createTransform( const dp::sg::core::NodeSharedPtr &node,
                                                                    const dp::math::Vec3f &translation = dp::math::Vec3f( 0.0f, 0.0f, 0.0f ),
                                                                    const dp::math::Quatf &orientation = dp::math::Quatf( dp::math::Vec3f( 0.0f, 1.0f, 0.0f ) , 0.0f ),
                                                                    const dp::math::Vec3f &scaling = dp::math::Vec3f( 1.0f, 1.0f, 1.0f ) );

      //! Generate a transformation for a node that maps x, y coordinates as if they are direct window space coordinates
      DP_SG_GENERATOR_API dp::sg::core::TransformSharedPtr imitateRaster( const dp::sg::core::NodeSharedPtr &node, unsigned int width, unsigned int height);

      //! Generate a transformation that maps x, y coordinates as if they are direct window space coordinates
      DP_SG_GENERATOR_API dp::sg::core::TransformSharedPtr imitateRaster( unsigned int width, unsigned int height );

      //! Sets the point of view of the camera of the given view state
      DP_SG_GENERATOR_API void setCameraPOV(float x, float y, float z, dp::sg::ui::ViewStateSharedPtr const& viewState);

      //! Sets the direction of the camera of the given view state with up vector always point in positive y direction
      DP_SG_GENERATOR_API void setCameraDirNoRoll(float x, float y, float z, dp::sg::ui::ViewStateSharedPtr const& viewState);

      //! Sets the direction of the camera of the given view state
      DP_SG_GENERATOR_API void setCameraDir(float x, float y, float z, dp::sg::ui::ViewStateSharedPtr const& viewState);

      //! "Gridify" a node by creating a Group with grid.x by grid.y by grid.z Transforms, each holding a clone of the provided Node
      DP_SG_GENERATOR_API dp::sg::core::GroupSharedPtr replicate( dp::sg::core::NodeSharedPtr const& node
                                                                , dp::math::Vec3ui const& gridSize
                                                                , dp::math::Vec3f const& gridSpacing = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                , bool clone = true );
    } // namespace generator
  } // namespace sg
} // namespace dp

