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


#include <AnimatedScene.h>
#include <dp/DP.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Transform.h>

#include <dp/sg/generator/MeshGenerator.h>
#include <boost/make_shared.hpp>

#include <cmath>

AnimationTime::AnimationTime()
  : m_time(0.0f)
{
}

DEFINE_STATIC_PROPERTY(AnimationTime, Time);

BEGIN_REFLECTION_INFO(AnimationTime)
INIT_STATIC_PROPERTY_RW(AnimationTime, Time, float, SEMANTIC_VALUE, value, value);
END_REFLECTION_INFO

Animator::~Animator()
{
}


AnimatorBase::AnimatorBase()
{
}


DEFINE_STATIC_PROPERTY(AnimatorColor, Time);
DEFINE_STATIC_PROPERTY(AnimatorColor, Color3);
DEFINE_STATIC_PROPERTY(AnimatorColor, Color4);

BEGIN_REFLECTION_INFO(AnimatorColor)
INIT_STATIC_PROPERTY_RW(AnimatorColor, Time, float, SEMANTIC_VALUE, value, value);
INIT_STATIC_PROPERTY_RO(AnimatorColor, Color3, dp::math::Vec3f, SEMANTIC_VALUE, value);
INIT_STATIC_PROPERTY_RO(AnimatorColor, Color4, dp::math::Vec4f, SEMANTIC_VALUE, value);
END_REFLECTION_INFO


AnimatorColorSharedPtr AnimatorColor::create(int x, int y, dp::math::Vec2i const & objectCount)
{
  return(std::shared_ptr<AnimatorColor>(new AnimatorColor(x, y, objectCount)));
}

AnimatorColor::AnimatorColor(int x, int y, dp::math::Vec2i const & objectCount)
  : m_x(x)
  , m_y(y)
  , m_objectCount(objectCount)
{
}

dp::math::Vec3f AnimatorColor::getColor3() const
{
  float sx;
  float ix;
  sx = std::modf((float(m_x) / m_objectCount[0]) + m_time, &ix);
  float sy = float(m_y) / m_objectCount[1];

  return dp::math::Vec3f(sx, sy, sx * sy);
}

dp::math::Vec4f AnimatorColor::getColor4() const
{
  float sx;
  float ix;
  sx = std::modf((float(m_x) / m_objectCount[0]) + m_time, &ix);
  float sy = float(m_y) / m_objectCount[1];

  return dp::math::Vec4f(sx, sy, sx * sy, 1.0f);
}

/************************************************************************/
/* AnimatedScene                                                        */
/************************************************************************/
AnimatedSceneSharedPtr AnimatedScene::create( const dp::math::Vec2f& gridSize, const dp::math::Vec2i& objectCount )
{
  return( std::shared_ptr<AnimatedScene>( new AnimatedScene( gridSize, objectCount ) ) );
}

AnimatedScene::AnimatedScene( const dp::math::Vec2f& gridSize, const dp::math::Vec2i& objectCount )
  : m_gridSize( gridSize )
  , m_objectCount( objectCount )
{
  m_root = dp::sg::core::Group::create();
  m_scene = dp::sg::core::Scene::create();
  m_scene->setRootNode( m_root );
  m_effectSpec = dp::sg::core::getStandardMaterialSpec();
  m_primitive = dp::sg::generator::createSphere( 5, 5, 1.5 );
  //m_primitive = dp::sg::generator::createCube();

  m_itColors = m_effectSpec->findParameterGroupSpec( std::string("standardMaterialParameters") );

  dp::util::FileFinder fileFinder;
  fileFinder.addSearchPath( dp::home() + "/media/dpfx" );
  fileFinder.addSearchPath( dp::home() + "/media/textures" );
  dp::fx::EffectLibrary::instance()->loadEffects( dp::home() + "/media/effects/xml/carpaint.xml", fileFinder );
  dp::fx::EffectLibrary::instance()->loadEffects( dp::home() + "/media/effects/xml/phong.xml", fileFinder );
  dp::fx::EffectLibrary::instance()->loadEffects( dp::home() + "/media/effects/xml/standard_material.xml", fileFinder );
  dp::fx::EffectLibrary::instance()->loadEffects( dp::home() + "/media/effects/xml/thinglass.xml", fileFinder );

  m_carpaint          = dp::sg::core::PipelineData::create( dp::fx::EffectLibrary::instance()->getEffectData("carpaint") );
  m_phong             = dp::sg::core::PipelineData::create( dp::fx::EffectLibrary::instance()->getEffectData("phong") );
  m_standard_material = dp::sg::core::PipelineData::create( dp::fx::EffectLibrary::instance()->getEffectData("standardMaterial") );
  m_thinglass         = dp::sg::core::PipelineData::create( dp::fx::EffectLibrary::instance()->getEffectData("thinglass") );

  m_animationTime = std::shared_ptr<AnimationTime>(new AnimationTime());

  DP_ASSERT( m_carpaint );
  DP_ASSERT( m_phong );
  DP_ASSERT( m_standard_material );
  DP_ASSERT( m_thinglass );

  createGrid( );
}

AnimatedScene::~AnimatedScene()
{

}

dp::sg::core::SceneSharedPtr AnimatedScene::getScene() const
{
  return m_scene;
}

dp::sg::core::PrimitiveSharedPtr AnimatedScene::createPrimitive( size_t x, size_t y )
{
  return m_primitive;
}

void AnimatedScene::linkAnimationColor2(AnimatorColorSharedPtr const & animation, dp::sg::core::PipelineDataSharedPtr const & effect, char const *group, char const *parameter)
{
  m_linkManager.link(m_animationTime, AnimationTime::PID_Time, animation, AnimatorColor::PID_Time);

  dp::fx::EffectSpec::iterator esIt = effect->getEffectSpec()->findParameterGroupSpec(group);
  dp::fx::ParameterGroupSpec::iterator itParameter = (*esIt)->findParameterSpec(parameter);
  dp::util::PropertyId pidParameter = effect->getParameterGroupData(esIt)->getProperty(parameter);

  if ((itParameter->first.getType() & dp::fx::PT_SCALAR_MODIFIER_MASK) == dp::fx::PT_VECTOR3)
  {
    m_linkManager.link(animation, AnimatorColor::PID_Color3, effect->getParameterGroupData(esIt), pidParameter);
  }
  else if ((itParameter->first.getType() & dp::fx::PT_SCALAR_MODIFIER_MASK) == dp::fx::PT_VECTOR4)
  {
    m_linkManager.link(animation, AnimatorColor::PID_Color4, effect->getParameterGroupData(esIt), pidParameter);
  }
  else {
    DP_ASSERT(!"unsupported animated color parameter");
  }
}

dp::sg::core::PipelineDataSharedPtr AnimatedScene::createMaterial( size_t x, size_t y )
{
  dp::sg::core::PipelineDataSharedPtr effect;
  size_t index = (y * m_objectCount[0]) + x;
  switch (index % 5)
  {
  case 0:
    {
      effect = dp::sg::core::PipelineData::create( m_effectSpec );

      dp::sg::core::ParameterGroupDataSharedPtr parameterGroupDataSharedPtr = dp::sg::core::ParameterGroupData::create( *m_itColors );
      effect->setParameterGroupData( parameterGroupDataSharedPtr );
      linkAnimationColor2(AnimatorColor::create(int(x), int(y), m_objectCount), effect, "standardMaterialParameters", "frontDiffuseColor");
    }
    break;
  case 1:
    effect = m_carpaint.clone();
    linkAnimationColor2(AnimatorColor::create(int(x), int(y), m_objectCount), effect, "carpaint_parameters", "diffuse");
    break;
  case 2:
    effect = m_phong.clone();
    linkAnimationColor2(AnimatorColor::create(int(x), int(y), m_objectCount), effect, "phongParameters", "diffuseColor");
    break;
  case 3:
    effect = m_standard_material.clone();
    linkAnimationColor2(AnimatorColor::create(int(x), int(y), m_objectCount), effect, "standardMaterialParameters", "frontDiffuseColor");
    break;
  case 4:
    effect = m_thinglass.clone();
    linkAnimationColor2(AnimatorColor::create(int(x), int(y), m_objectCount), effect, "thinglass_parameters", "transparentColor");
    break;
  }

  return effect;
}

dp::sg::core::TransformSharedPtr AnimatedScene::createTransform( size_t x, size_t y )
{
  float posX = float(x) / float(m_objectCount[0]) * m_gridSize[0];
  float posY = float(y) / float(m_objectCount[1]) * m_gridSize[1];
  dp::sg::core::TransformSharedPtr transform = dp::sg::core::Transform::create();
  transform->setTranslation( dp::math::Vec3f( posX, posY, 0.0f) );
  return transform;
}

void AnimatedScene::createGrid()
{
  size_t objectCount = m_objectCount[0] * m_objectCount[1];
  m_primitives.resize( objectCount );
  m_materials.resize( objectCount );
  m_geoNodes.resize( objectCount );
  m_transforms.resize( objectCount );
  m_animators.resize( objectCount );

  size_t index = 0;
  for ( size_t y = 0; y < m_objectCount[1]; ++y )
  {
    for ( size_t x = 0; x < m_objectCount[0]; ++x )
    {
      m_primitives[index] = createPrimitive( x, y );
      m_materials[index] = createMaterial( x, y );
      m_geoNodes[index] = dp::sg::core::GeoNode::create();
      {
        m_geoNodes[index]->setMaterialPipeline( m_materials[index] );
        m_geoNodes[index]->setPrimitive( m_primitives[index] );
      }
      m_transforms[index] = createTransform( x, y );
      {
        m_transforms[index]->addChild( m_geoNodes[index] );
      }
      m_root->addChild( m_transforms[index] );

      ++index;
    }
  }
}

void AnimatedScene::update( float time )
{
  m_animationTime->setTime(time);
  m_linkManager.processLinks();
}

void AnimatedScene::updateTransforms( float time )
{
  size_t index = 0;
  float animationTime = sin(time) * 10;
  for ( size_t y = 0; y < m_objectCount[1]; ++y )
  {
    for ( size_t x = 0; x < m_objectCount[0]; ++x )
    {
      float fx = float(x) / m_objectCount[0] + (animationTime / 10.0f);
      float fy = float(y) / m_objectCount[1] + sin(animationTime);
      float height = sin(fx*fx) * cos(fx*fy*cos(animationTime));
      dp::math::Vec3f translation = m_transforms[index]->getTranslation();
      translation[2] = height * (m_gridSize[0] + m_gridSize[1]) * 0.1f;
      m_transforms[index]->setTranslation(translation);
      ++index;
    }
  }
}

