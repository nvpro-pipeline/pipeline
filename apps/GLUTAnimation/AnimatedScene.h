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

#include <dp/fx/EffectLibrary.h>
#include <dp/math/Vecnt.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/animation/LinkManager.h>

DEFINE_PTR_TYPES( Animator );

class Animator
{
public:
  virtual ~Animator();
  virtual void update( float time ) = 0;
};

DEFINE_PTR_TYPES(AnimationTime);

class AnimationTime : public dp::sg::core::Object
{
public:
  AnimationTime();
  void setTime(float time) { m_time = time; }
  float getTime() const { return m_time; }

  dp::sg::core::HandledObjectSharedPtr clone() const
  {
    return(std::shared_ptr<AnimationTime>(new AnimationTime(*this)));
  }

public:
  REFLECTION_INFO(AnimationTime);
  BEGIN_DECLARE_STATIC_PROPERTIES
    DECLARE_STATIC_PROPERTY(Time);
  END_DECLARE_STATIC_PROPERTIES

private:
  float m_time;
};

class AnimatorBase : public dp::sg::core::Object
{
public:
  AnimatorBase();

  dp::sg::core::HandledObjectSharedPtr clone() const
  {
    return(std::shared_ptr<AnimatorBase>(new AnimatorBase(*this)));
  }
};

DEFINE_PTR_TYPES(AnimatorColor);

class AnimatorColor : public AnimatorBase
{
public:
  static AnimatorColorSharedPtr create(int x, int y, dp::math::Vec2i const & objectCount);

  float getTime() const { return m_time; }
  void setTime(float time) { m_time = time; }
  dp::math::Vec3f getColor3() const;
  dp::math::Vec4f getColor4() const;

  dp::sg::core::HandledObjectSharedPtr clone() const
  {
    return(std::shared_ptr<AnimatorColor>(new AnimatorColor(*this)));
  }

protected:
  AnimatorColor(int x, int y, dp::math::Vec2i const & objectCount);

public:
  REFLECTION_INFO(AnimatorColor);
  BEGIN_DECLARE_STATIC_PROPERTIES
    DECLARE_STATIC_PROPERTY(Time);
    DECLARE_STATIC_PROPERTY(Color3);
    DECLARE_STATIC_PROPERTY(Color4);
  END_DECLARE_STATIC_PROPERTIES


private:
  float m_time;

  int m_x;
  int m_y;
  dp::math::Vec2i m_objectCount;
};



DEFINE_PTR_TYPES( AnimatedScene );

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
  void linkAnimationColor2(AnimatorColorSharedPtr const & animation, dp::sg::core::EffectDataSharedPtr const & effect, char const *group, char const *parameter);

  void setColor( const dp::math::Vec3f& color, size_t x, size_t y );
  void createGrid();
  dp::sg::core::PrimitiveSharedPtr createPrimitive( size_t x, size_t y );
  dp::sg::core::EffectDataSharedPtr createMaterial( size_t x, size_t y );
  dp::sg::core::TransformSharedPtr createTransform( size_t x, size_t y );

  dp::math::Vec2f m_gridSize;
  dp::math::Vec2i m_objectCount;

  dp::sg::animation::LinkManager m_linkManager;
  AnimationTimeSharedPtr         m_animationTime;
  std::vector<AnimatorColorSharedPtr>    m_colorAnimators;

  dp::fx::EffectSpecSharedPtr       m_effectSpec;
  dp::sg::core::PrimitiveSharedPtr  m_primitive;
  dp::sg::core::GroupSharedPtr      m_root;
  dp::sg::core::SceneSharedPtr      m_scene;

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