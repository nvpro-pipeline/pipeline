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

#include <dp/fx/EffectLibrary.h>
#include <dp/math/Vecnt.h>
#include <dp/sg/core/CoreTypes.h>

SHARED_PTR_TYPES( Animator );

class Animator
{
public:
  virtual ~Animator();
  virtual void update( float time ) = 0;
};

SHARED_PTR_TYPES( AnimatedScene );

class AnimatedScene
{
public:
  static AnimatedSceneSharedPtr create( const dp::math::Vec2f& gridSize, const dp::math::Vec2i& objectCount );
  virtual ~AnimatedScene();

  dp::sg::core::SceneSharedPtr getScene() const;

  void update( float time );
  void updateTransforms( float time );

protected:
  AnimatedScene( const dp::math::Vec2f& gridSize, const dp::math::Vec2i& objectCount );

  void setColor( const dp::math::Vec3f& color, size_t x, size_t y );
  void createGrid();
  dp::sg::core::PrimitiveSharedPtr createPrimitive( size_t x, size_t y );
  dp::sg::core::EffectDataSharedPtr createMaterial( size_t x, size_t y );
  dp::sg::core::TransformSharedPtr createTransform( size_t x, size_t y );

  dp::math::Vec2f m_gridSize;
  dp::math::Vec2i m_objectCount;

  dp::fx::SmartEffectSpec  m_effectSpec;
  dp::sg::core::PrimitiveSharedPtr m_primitive;
  dp::sg::core::GroupSharedPtr     m_root;
  dp::sg::core::SceneSharedPtr     m_scene;

  // for updating colors;
  dp::fx::EffectSpec::iterator         m_itColors;

  typedef std::vector< dp::sg::core::PrimitiveSharedPtr >  Primitives;
  typedef std::vector< dp::sg::core::EffectDataSharedPtr > Materials;
  typedef std::vector< dp::sg::core::TransformSharedPtr >  Transforms;
  typedef std::vector< dp::sg::core::GeoNodeSharedPtr>     GeoNodes;
  typedef std::vector< AnimatorSharedPtr >                 Animators;

  Primitives m_primitives;
  Materials  m_materials;
  Transforms m_transforms;
  GeoNodes   m_geoNodes;
  Animators  m_animators;

  // effect templates for cloning to ensure that TextureFile 
  dp::sg::core::EffectDataSharedPtr m_carpaint;
  dp::sg::core::EffectDataSharedPtr m_phong;
  dp::sg::core::EffectDataSharedPtr m_standard_material;
  dp::sg::core::EffectDataSharedPtr m_thinglass;
};