// Copxright NVIDIA Corporation 2012
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
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Transform.h>

#include <dp/sg/generator/MeshGenerator.h>
#include <boost/make_shared.hpp>

#include <cmath>

Animator::~Animator()
{
}

class AnimatorBase : public Animator
{
public:
  AnimatorBase( dp::sg::core::EffectDataSharedPtr const & effectData, std::string const & parameterGroup, std::string const & parameter );

protected:
  dp::sg::core::ParameterGroupDataSharedPtr m_parameterGroupData;
  dp::fx::ParameterGroupSpec::iterator m_itParameter;
};

AnimatorBase::AnimatorBase( dp::sg::core::EffectDataSharedPtr const & effectData, std::string const & parameterGroup, std::string const & parameter )
{
  dp::fx::EffectSpec::iterator esIt = effectData->getEffectSpec()->findParameterGroupSpec( parameterGroup );
  m_parameterGroupData = effectData->getParameterGroupData( esIt );
  m_itParameter = (*esIt)->findParameterSpec(parameter);
}


class AnimatorColor : public AnimatorBase
{
public:
  AnimatorColor( dp::sg::core::EffectDataSharedPtr &effect, std::string const & parameterGroup, std::string const & parameter, int x, int y, dp::math::Vec2i const & objectCount );
  virtual void update( float time );

private:
  int m_x;
  int m_y;
  dp::math::Vec2i m_objectCount;
};

AnimatorColor::AnimatorColor( dp::sg::core::EffectDataSharedPtr &effect, std::string const & parameterGroup, std::string const & parameter, int x, int y, dp::math::Vec2i const & objectCount )
  : AnimatorBase( effect, parameterGroup, parameter )
  , m_x(x)
  , m_y(y)
  , m_objectCount( objectCount )
{
}

void AnimatorColor::update( float time )
{
  float sx;
  float ix;
  sx = std::modf( (float(m_x) / m_objectCount[0]) + time, &ix);
  float sy = float(m_y) / m_objectCount[1];
  dp::math::Vec3f color( sx, sy, sx * sy );

  m_parameterGroupData->setParameter( m_itParameter, color );
}

/************************************************************************/
/* Animator Bumpiness                                  */
/************************************************************************/
class AnimatorBumpiness : public AnimatorBase
{
public:
  AnimatorBumpiness( dp::sg::core::EffectDataSharedPtr &effect, std::string const & parameterGroup, std::string const & parameter, int x, int y, dp::math::Vec2i const & objectCount );
  virtual void update( float time );

private:
  int m_x;
  int m_y;
  dp::math::Vec2i m_objectCount;
};

AnimatorBumpiness::AnimatorBumpiness( dp::sg::core::EffectDataSharedPtr &effect, std::string const & parameterGroup, std::string const & parameter, int x, int y, dp::math::Vec2i const & objectCount )
  : AnimatorBase( effect, parameterGroup, parameter )
  , m_x(x)
  , m_y(y)
  , m_objectCount( objectCount )
{
}

void AnimatorBumpiness::update( float time )
{
  float sx;
  sx = 5 * sin( (2 * float(m_x) / m_objectCount[0] + float(m_y) / m_objectCount[1] + time / 5.0f) * 6.28f) + 5;

  m_parameterGroupData->setParameter( m_itParameter, sx );
}

/************************************************************************/
/* AnimatedScene                                                        */
/************************************************************************/
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

  std::vector<std::string> searchPaths;
  searchPaths.push_back( dp::home() + "/media/effects/mdl" );
  searchPaths.push_back( dp::home() + "/media/textures" );
  dp::fx::EffectLibrary::instance()->loadEffects( "material_catalog/plastic/rubber_studded_black.xml", searchPaths );
  dp::fx::EffectLibrary::instance()->loadEffects( "material_catalog/plastic/resin_polyurethane_coated.xml", searchPaths );
  dp::fx::EffectLibrary::instance()->loadEffects( "material_catalog/wood/mahogany_floorboards.xml", searchPaths );
  dp::fx::EffectLibrary::instance()->loadEffects( "material_catalog/wood/mahogany_floorboards.xml", searchPaths );
  dp::fx::EffectLibrary::instance()->loadEffects( "material_catalog/metal/steel_milled_concentric.xml", searchPaths );
  
  m_rubber_studded_black = dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectData("rubber_studded_black") );
  m_resin_polyurethane_coated = dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectData("resin_polyurethane_coated") );
  m_mahogany_floorboards = dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectData("mahogany_floorboards") );
  m_steel_milled_concentric = dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectData("steel_milled_concentric") );

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

dp::sg::core::EffectDataSharedPtr AnimatedScene::createMaterial( size_t x, size_t y )
{
  dp::sg::core::EffectDataSharedPtr effect;
  size_t index = (y * m_objectCount[0]) + x;
  switch (index % 5)
  {
  case 0:
    {
      effect = dp::sg::core::EffectData::create( m_effectSpec );

      dp::sg::core::ParameterGroupDataSharedPtr parameterGroupDataSharedPtr = dp::sg::core::ParameterGroupData::create( *m_itColors );
      effect->setParameterGroupData( parameterGroupDataSharedPtr );
      m_animators[( y * m_objectCount[0] + x)] = boost::make_shared<AnimatorColor>(effect, "standardMaterialParameters", "frontDiffuseColor", int(x), int(y), m_objectCount );
    }
    break;
  case 1:
    effect = m_rubber_studded_black.clone();

    m_animators[index] = boost::make_shared<AnimatorColor>(effect, "rubber_studded_blackFragmentParameters", "base_color", int(x), int(y), m_objectCount );
    break;
  case 2:
    effect = m_resin_polyurethane_coated.clone();

    m_animators[index] = boost::make_shared<AnimatorColor>(effect, "resin_polyurethane_coatedFragmentParameters", "resin_color", int(x), int(y), m_objectCount );
    break;
  case 3:
    effect = m_mahogany_floorboards.clone();

    m_animators[index] = boost::make_shared<AnimatorBumpiness>(effect, "mahogany_floorboardsFragmentParameters", "bump_amount", int(x), int(y), m_objectCount );
    break;
  case 4:
    effect = m_steel_milled_concentric.clone();

    m_animators[index] = boost::make_shared<AnimatorBumpiness>(effect, "steel_milled_concentricFragmentParameters", "material_ior", int(x), int(y), m_objectCount );
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
        m_geoNodes[index]->setMaterialEffect( m_materials[index] );
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
  for ( Animators::iterator it = m_animators.begin(); it != m_animators.end(); ++it )
  {
    (*it)->update( time );
  }
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

