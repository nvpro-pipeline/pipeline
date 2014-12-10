// Copyright NVIDIA Corporation 2002-2005
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


#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include "VRMLTypes.h"

using namespace vrml;
using std::string;

AnchorSharedPtr Anchor::create()
{
  return( std::shared_ptr<Anchor>( new Anchor() ) );
}

const string & Anchor::getType( void ) const
{
  static string type = "Anchor";
  return( type );
}

AppearanceSharedPtr Appearance::create()
{
  return( std::shared_ptr<Appearance>( new Appearance() ) );
}

Appearance::Appearance()
{
}

Appearance::~Appearance()
{
}

const string & Appearance::getType( void ) const
{
  static string type = "Appearance";
  return( type );
}

AudioClipSharedPtr AudioClip::create()
{
  return( std::shared_ptr<AudioClip>( new AudioClip() ) );
}

AudioClip::AudioClip()
: loop(false)
, pitch(1.0f)
, startTime(0.0)
, stopTime(0.0)
{
}

const string & AudioClip::getType( void ) const
{
  static string type = "AudioClip";
  return( type );
}

BackgroundSharedPtr Background::create()
{
  return( std::shared_ptr<Background>( new Background() ) );
}

Background::Background()
{
  skyColor.push_back( SFColor( 0.0f, 0.0f, 0.0f ) );
}

const string & Background::getType( void ) const
{
  static string type = "Background";
  return( type );
}

BillboardSharedPtr Billboard::create()
{
  return( std::shared_ptr<Billboard>( new Billboard() ) );
}

Billboard::Billboard()
: axisOfRotation(0.0f, 1.0f, 0.0f )
{
}

Billboard::~Billboard()
{
}

const string & Billboard::getType( void ) const
{
  static string type = "Billboard";
  return( type );
}

BoxSharedPtr Box::create()
{
  return( std::shared_ptr<Box>( new Box() ) );
}

Box::Box()
: size(2.0f, 2.0f, 2.0f)
{
}

Box::~Box()
{
}

const string & Box::getType( void ) const
{
  static string type = "Box";
  return( type );
}

CollisionSharedPtr Collision::create()
{
  return( std::shared_ptr<Collision>( new Collision() ) );
}

Collision::Collision()
: collide(true)
{
}

Collision::~Collision()
{
}

const string & Collision::getType( void ) const
{
  static string type = "Collision";
  return( type );
}

ColorSharedPtr Color::create()
{
  return( std::shared_ptr<Color>( new Color() ) );
}

Color::Color()
: interpreted(false)
, set_color(NULL)
{
}

const string & Color::getType( void ) const
{
  static string type = "Color";
  return( type );
}

ColorInterpolatorSharedPtr ColorInterpolator::create()
{
  return( std::shared_ptr<ColorInterpolator>( new ColorInterpolator() ) );
}

const string & ColorInterpolator::getType( void ) const
{
  static string type = "ColorInterpolator";
  return( type );
}

ConeSharedPtr Cone::create()
{
  return( std::shared_ptr<Cone>( new Cone() ) );
}

Cone::Cone()
: bottom(true)
, bottomRadius(1.0f)
, height(2.0f)
, side(true)
{
}

const string & Cone::getType( void ) const
{
  static string type = "Cone";
  return( type );
}

CoordinateSharedPtr Coordinate::create()
{
  return( std::shared_ptr<Coordinate>( new Coordinate() ) );
}

Coordinate::Coordinate()
: interpreted(false)
{
}

Coordinate::~Coordinate()
{
}

const string & Coordinate::getType( void ) const
{
  static string type = "Coordinate";
  return( type );
}

CoordinateInterpolatorSharedPtr CoordinateInterpolator::create()
{
  return( std::shared_ptr<CoordinateInterpolator>( new CoordinateInterpolator() ) );
}

const string & CoordinateInterpolator::getType( void ) const
{
  static string type = "CoordinateInterpolator";
  return( type );
}

CylinderSharedPtr Cylinder::create()
{
  return( std::shared_ptr<Cylinder>( new Cylinder() ) );
}

Cylinder::Cylinder()
: bottom(true)
, height(2.0f)
, radius(1.0f)
, side(true)
, top(true)
{
}

const string & Cylinder::getType( void ) const
{
  static string type = "Cylinder";
  return( type );
}

CylinderSensorSharedPtr CylinderSensor::create()
{
  return( std::shared_ptr<CylinderSensor>( new CylinderSensor() ) );
}

CylinderSensor::CylinderSensor()
: autoOffset(true)
, diskAngle(0.262f)
, maxAngle(-1.0f)
, minAngle(0.0f)
, offset(0.0f)
{
}

const string & CylinderSensor::getType( void ) const
{
  static string type = "CylinderSensor";
  return( type );
}

DirectionalLightSharedPtr DirectionalLight::create()
{
  return( std::shared_ptr<DirectionalLight>( new DirectionalLight() ) );
}

DirectionalLight::DirectionalLight()
: direction(0.0f,0.0f,-1.0f)
{
}

DirectionalLight::~DirectionalLight()
{
}

const string & DirectionalLight::getType( void ) const
{
  static string type = "DirectionalLight";
  return( type );
}

ElevationGridSharedPtr ElevationGrid::create()
{
  return( std::shared_ptr<ElevationGrid>( new ElevationGrid() ) );
}

ElevationGrid::ElevationGrid()
: ccw(true)
, colorPerVertex(true)
, creaseAngle(0.0f)
, normalPerVertex(true)
, solid(true)
, xDimension(0)
, xSpacing(0.0f)
, zDimension(0)
, zSpacing(0.0f)
{
}

ElevationGrid::~ElevationGrid()
{
}

const string & ElevationGrid::getType( void ) const
{
  static string type = "ElevationGrid";
  return( type );
}

ExtrusionSharedPtr Extrusion::create()
{
  return( std::shared_ptr<Extrusion>( new Extrusion() ) );
}

Extrusion::Extrusion()
: beginCap(true)
, ccw(true)
, convex(true)
, creaseAngle(0.0f)
, endCap(true)
, solid(true)
{
  crossSection.push_back( SFVec2f(  1.0f,  1.0f ) );
  crossSection.push_back( SFVec2f(  1.0f, -1.0f ) );
  crossSection.push_back( SFVec2f( -1.0f, -1.0f ) );
  crossSection.push_back( SFVec2f( -1.0f,  1.0f ) );
  crossSection.push_back( SFVec2f(  1.0f,  1.0f ) );
  orientation.push_back( SFRotation( SFVec3f( 0.0f, 0.0f, 1.0f ), 0.0f ) );
  scale.push_back( SFVec2f( 1.0f, 1.0f ) );
  spine.push_back( SFVec3f( 0.0f, 0.0f, 0.0f ) );
  spine.push_back( SFVec3f( 0.0f, 1.0f, 0.0f ) );
}

const string & Extrusion::getType( void ) const
{
  static string type = "Extrusion";
  return( type );
}

FogSharedPtr Fog::create()
{
  return( std::shared_ptr<Fog>( new Fog() ) );
}

Fog::Fog()
: color(1.0f,1.0f,1.0f)
, fogType("LINEAR")
, visibilityRange(0.0f)
{
}

const string & Fog::getType( void ) const
{
  static string type = "Fog";
  return( type );
}

FontStyleSharedPtr FontStyle::create()
{
  return( std::shared_ptr<FontStyle>( new FontStyle() ) );
}

FontStyle::FontStyle()
: horizontal(true)
, leftToRight(true)
, size(1.0f)
, spacing(1.0f)
, style("PLAIN")
, topToBottom(true)
{
  family.push_back( "SERIF" );
  justify.push_back( "BEGIN" );
}

const string & FontStyle::getType( void ) const
{
  static string type = "FontStyle";
  return( type );
}

const string & Geometry::getType( void ) const
{
  static string type = "Geometry";
  return( type );
}

GroupSharedPtr Group::create()
{
  return( std::shared_ptr<Group>( new Group() ) );
}

Group::Group()
{
}

Group::~Group()
{
}

const string & Group::getType( void ) const
{
  static string type = "Group";
  return( type );
}

ImageTextureSharedPtr ImageTexture::create()
{
  return( std::shared_ptr<ImageTexture>( new ImageTexture() ) );
}

ImageTexture::ImageTexture( void )
{
}

ImageTexture::~ImageTexture()
{
}

const string & ImageTexture::getType( void ) const
{
  static string type = "ImageTexture";
  return( type );
}

IndexedFaceSetSharedPtr IndexedFaceSet::create()
{
  return( std::shared_ptr<IndexedFaceSet>( new IndexedFaceSet() ) );
}

IndexedFaceSet::IndexedFaceSet()
: ccw(true)
, colorPerVertex(true)
, convex(true)
, creaseAngle(0.0f)
, normalPerVertex(true)
, solid(true)
{
}

IndexedFaceSet::~IndexedFaceSet()
{
}

const string & IndexedFaceSet::getType( void ) const
{
  static string type = "IndexedFaceSet";
  return( type );
}

IndexedLineSetSharedPtr IndexedLineSet::create()
{
  return( std::shared_ptr<IndexedLineSet>( new IndexedLineSet() ) );
}

IndexedLineSet::IndexedLineSet()
: colorPerVertex(true)
{
}

IndexedLineSet::~IndexedLineSet()
{
}

const string & IndexedLineSet::getType( void ) const
{
  static string type = "IndexedLineSet";
  return( type );
}

Interpolator::Interpolator()
: interpreted(false)
{
}

Interpolator::~Interpolator()
{
}

const string & Interpolator::getType( void ) const
{
  static string type = "Interpolator";
  return( type );
}

Light::Light()
: ambientIntensity(0.0f)
, color(1.0f,1.0f,1.0f)
, intensity(1.0f)
, on(true)
{
}

const string & Light::getType( void ) const
{
  static string type = "Light";
  return( type );
}

InlineSharedPtr Inline::create()
{
  return( std::shared_ptr<Inline>( new Inline() ) );
}

Inline::Inline()
: bboxCenter(0.0f,0.0f,0.0f)
, bboxSize(-1.0f,-1.0f,-1.0f)
{
}

Inline::~Inline()
{
}

const string & Inline::getType( void ) const
{
  static string type = "Inline";
  return( type );
}

LODSharedPtr LOD::create()
{
  return( std::shared_ptr<LOD>( new LOD() ) );
}

LOD::LOD()
: center(0.0f,0.0f,0.0f)
{
}

LOD::~LOD()
{
}

const string & LOD::getType( void ) const
{
  static string type = "LOD";
  return( type );
}

MaterialSharedPtr Material::create()
{
  return( std::shared_ptr<Material>( new Material() ) );
}

Material::Material()
: ambientIntensity(0.2f)
, diffuseColor(0.8f,0.8f,0.8f)
, emissiveColor(0.0f,0.0f,0.0f)
, shininess(0.2f)
, specularColor(0.0f,0.0f,0.0f)
, transparency(0.0f)
{
}

Material::~Material()
{
}

const string & Material::getType( void ) const
{
  static string type = "Material";
  return( type );
}

MovieTextureSharedPtr MovieTexture::create()
{
  return( std::shared_ptr<MovieTexture>( new MovieTexture() ) );
}

MovieTexture::MovieTexture()
: loop(false)
, speed(1.0f)
, startTime(0.0)
, stopTime(0.0)
{
}

const string & MovieTexture::getType( void ) const
{
  static string type = "MovieTexture";
  return( type );
}

NavigationInfoSharedPtr NavigationInfo::create()
{
  return( std::shared_ptr<NavigationInfo>( new NavigationInfo() ) );
}

NavigationInfo::NavigationInfo()
: headlight(true)
, speed(1.0f)
, visibilityLimit(0.0f)
{
  avatarSize.push_back( 0.25f );
  avatarSize.push_back( 1.6f );
  avatarSize.push_back( 0.75f );
  type.push_back( "WALK" );
}

const string & NavigationInfo::getType( void ) const
{
  static string type = "Material";
  return( type );
}

NormalSharedPtr Normal::create()
{
  return( std::shared_ptr<Normal>( new Normal() ) );
}

Normal::Normal()
: interpreted(false)
{
}

Normal::~Normal()
{
}

const string & Normal::getType( void ) const
{
  static string type = "Normal";
  return( type );
}

NormalInterpolatorSharedPtr NormalInterpolator::create()
{
  return( std::shared_ptr<NormalInterpolator>( new NormalInterpolator() ) );
}

const string & NormalInterpolator::getType( void ) const
{
  static string type = "NormalInterpolator";
  return( type );
}

ObjectSharedPtr Object::create()
{
  return( std::shared_ptr<Object>( new Object() ) );
}

const string & Object::getType( void ) const
{
  static string type = "Object";
  return( type );
}

OrientationInterpolatorSharedPtr OrientationInterpolator::create()
{
  return( std::shared_ptr<OrientationInterpolator>( new OrientationInterpolator() ) );
}

const string & OrientationInterpolator::getType( void ) const
{
  static string type = "OrientationInterpolator";
  return( type );
}

PixelTextureSharedPtr PixelTexture::create()
{
  return( std::shared_ptr<PixelTexture>( new PixelTexture() ) );
}

PixelTexture::PixelTexture()
{
  image.width         = 0;
  image.height        = 0;
  image.numComponents = 0;
  image.pixelsValues  = NULL;
}

const string & PixelTexture::getType( void ) const
{
  static string type = "PixelTexture";
  return( type );
}

PlaneSensorSharedPtr PlaneSensor::create()
{
  return( std::shared_ptr<PlaneSensor>( new PlaneSensor() ) );
}

PlaneSensor::PlaneSensor()
: autoOffset(true)
, maxPosition(-1.0f,-1.0f)
, minPosition(0.0f,0.0f)
, offset(0.0f,0.0f,0.0f)
{
}

const string & PlaneSensor::getType( void ) const
{
  static string type = "PlaneSensor";
  return( type );
}

PointLightSharedPtr PointLight::create()
{
  return( std::shared_ptr<PointLight>( new PointLight() ) );
}

PointLight::PointLight()
: attenuation(1.0f,0.0f,0.0f)
, location(0.0f,0.0f,0.0f)
, radius(100.0f)
{
}

PointLight::~PointLight()
{
}

const string & PointLight::getType( void ) const
{
  static string type = "PointLight";
  return( type );
}

PointSetSharedPtr PointSet::create()
{
  return( std::shared_ptr<PointSet>( new PointSet() ) );
}

PointSet::PointSet()
{
}

PointSet::~PointSet()
{
}

const string & PointSet::getType( void ) const
{
  static string type = "PointSet";
  return( type );
}

PositionInterpolatorSharedPtr PositionInterpolator::create()
{
  return( std::shared_ptr<PositionInterpolator>( new PositionInterpolator() ) );
}

const string & PositionInterpolator::getType( void ) const
{
  static string type = "PositionInterpolator";
  return( type );
}

ProximitySensorSharedPtr ProximitySensor::create()
{
  return( std::shared_ptr<ProximitySensor>( new ProximitySensor() ) );
}

ProximitySensor::ProximitySensor()
: center(0.0f,0.0f,0.0f)
, size(0.0f,0.0f,0.0f)
{
}

const string & ProximitySensor::getType( void ) const
{
  static string type = "ProximitySensor";
  return( type );
}

ScriptSharedPtr Script::create()
{
  return( std::shared_ptr<Script>( new Script() ) );
}

Script::Script()
: directOutput(false)
, mustEvaluate(false)
{
}

const string & Script::getType( void ) const
{
  static string type = "Script";
  return( type );
}

Sensor::Sensor()
: enabled(true)
{
}

const string & Sensor::getType( void ) const
{
  static string type = "Sensor";
  return( type );
}

SFImage::SFImage()
: width(0)
, height(0)
, numComponents(0)
, pixelsValues(NULL)
{
}

SFImage::~SFImage()
{
  if ( pixelsValues )
  {
    delete [] pixelsValues;
  }
}

SFRotation::SFRotation( const SFVec3f &axis, SFFloat angle )
{
  float l = length( axis );
  SFVec3f a;
  if ( l <= FLT_EPSILON )
  {
    a = SFVec3f( 0.0f, 0.0f, 1.0f );
  }
  else
  {
    a = axis / l;
  }
  setVec( *this, a[0], a[1], a[2], angle );
}

ScalarInterpolatorSharedPtr ScalarInterpolator::create()
{
  return( std::shared_ptr<ScalarInterpolator>( new ScalarInterpolator() ) );
}

const string & ScalarInterpolator::getType( void ) const
{
  static string type = "ScalarInterpolator";
  return( type );
}

ShapeSharedPtr Shape::create()
{
  return( std::shared_ptr<Shape>( new Shape() ) );
}

Shape::Shape()
{
}

Shape::~Shape()
{
}

const string & Shape::getType( void ) const
{
  static string type = "Shape";
  return( type );
}

SoundSharedPtr Sound::create()
{
  return( std::shared_ptr<Sound>( new Sound() ) );
}

Sound::Sound()
: direction(0.0f,0.0f,1.0f)
, intensity(1.0f)
, location(0.0f,0.0f,0.0f)
, maxBack(10.0f)
, maxFront(10.0f)
, minBack(1.0f)
, minFront(1.0f)
, priority(0.0f)
, spatialize(true)
{
}

Sound::~Sound()
{
}

const string & Sound::getType( void ) const
{
  static string type = "Sound";
  return( type );
}

SphereSharedPtr Sphere::create()
{
  return( std::shared_ptr<Sphere>( new Sphere() ) );
}

Sphere::Sphere()
: radius(1.0f)
{
}

const string & Sphere::getType( void ) const
{
  static string type = "Sphere";
  return( type );
}

SphereSensorSharedPtr SphereSensor::create()
{
  return( std::shared_ptr<SphereSensor>( new SphereSensor() ) );
}

SphereSensor::SphereSensor()
: autoOffset(true)
, offset(SFVec3f(0.0f,1.0f,0.0f),0.0f)
{
}

const string & SphereSensor::getType( void ) const
{
  static string type = "SphereSensor";
  return( type );
}

SpotLightSharedPtr SpotLight::create()
{
  return( std::shared_ptr<SpotLight>( new SpotLight() ) );
}

SpotLight::SpotLight()
: attenuation(1.0f,0.0f,0.0f)
, beamWidth(1.570796f)
, cutOffAngle(0.785398f)
, direction(0.0f,0.0f,-1.0f)
, location(0.0f,0.0f,0.0f)
, radius(100.0f)
{
}

SpotLight::~SpotLight()
{
}

const string & SpotLight::getType( void ) const
{
  static string type = "SpotLight";
  return( type );
}

SwitchSharedPtr Switch::create()
{
  return( std::shared_ptr<Switch>( new Switch() ) );
}

Switch::Switch()
: whichChoice(-1)
{
}

Switch::~Switch()
{
}

const string & Switch::getType( void ) const
{
  static string type = "Switch";
  return( type );
}

TextSharedPtr Text::create()
{
  return( std::shared_ptr<Text>( new Text() ) );
}

Text::Text()
: maxExtent(0.0f)
{
}

Text::~Text()
{
}

const string & Text::getType( void ) const
{
  static std::string type = "Text";
  return( type );
}

TextureSharedPtr Texture::create()
{
  return( std::shared_ptr<Texture>( new Texture() ) );
}

Texture::Texture()
: repeatS(true)
, repeatT(true)
{
}

const string & Texture::getType( void ) const
{
  static string type = "Texture";
  return( type );
}

TextureCoordinateSharedPtr TextureCoordinate::create()
{
  return( std::shared_ptr<TextureCoordinate>( new TextureCoordinate() ) );
}

const string & TextureCoordinate::getType( void ) const
{
  static string type = "TextureCoordinate";
  return( type );
}

TextureTransformSharedPtr TextureTransform::create()
{
  return( std::shared_ptr<TextureTransform>( new TextureTransform() ) );
}

TextureTransform::TextureTransform()
: center(0.0f,0.0f)
, rotation(0.0f)
, scale(1.0f,1.0f)
, translation(0.0f,0.0f)
{
}

const string & TextureTransform::getType( void ) const
{
  static string type = "TextureTransform";
  return( type );
}

TimeSensor::TimeSensor()
: cycleInterval(1.0)
, loop(false)
, startTime(0.0)
, stopTime(0.0)
{
}

TimeSensorSharedPtr TimeSensor::create()
{
  return( std::shared_ptr<TimeSensor>( new TimeSensor() ) );
}

const string & TimeSensor::getType( void ) const
{
  static string type = "TimeSensor";
  return( type );
}

TouchSensorSharedPtr TouchSensor::create()
{
  return( std::shared_ptr<TouchSensor>( new TouchSensor() ) );
}

const string & TouchSensor::getType( void ) const
{
  static string type = "TouchSensor";
  return( type );
}

TransformSharedPtr Transform::create()
{
  return( std::shared_ptr<Transform>( new Transform() ) );
}

Transform::Transform()
: center(0.0f,0.0f,0.0f)
, rotation(SFVec3f(0.0f,0.0f,1.0f),0.0f)
, scale(1.0f,1.0f,1.0f)
, scaleOrientation(SFVec3f(0.0f,0.0f,1.0f),0.0f)
, translation(0.0f,0.0f,0.0f)
{
}

Transform::~Transform()
{
}

const string & Transform::getType( void ) const
{
  static string type = "Transform";
  return( type );
}

ViewpointSharedPtr Viewpoint::create()
{
  return( std::shared_ptr<Viewpoint>( new Viewpoint() ) );
}

Viewpoint::Viewpoint()
: fieldOfView(0.785398f)
, jump(true)
, orientation(SFVec3f(0.0f,0.0f,1.0f),0.0f)
, position(0.0f,0.0f,10.0f)
{
}

Viewpoint::~Viewpoint()
{
}

const string & Viewpoint::getType( void ) const
{
  static string type = "Viewpoint";
  return( type );
}

VisibilitySensorSharedPtr VisibilitySensor::create()
{
  return( std::shared_ptr<VisibilitySensor>( new VisibilitySensor() ) );
}

VisibilitySensor::VisibilitySensor()
: center(0.0f,0.0f,0.0f)
, size(0.0f,0.0f,0.0f)
{
}

const string & VisibilitySensor::getType( void ) const
{
  static string type = "VisibilitySensor";
  return( type );
}

WorldInfoSharedPtr WorldInfo::create()
{
  return( std::shared_ptr<WorldInfo>( new WorldInfo() ) );
}

const string & WorldInfo::getType( void ) const
{
  static string type = "WorldInfo";
  return( type );
}

ROUTE::ROUTE()
{
}

ROUTE::~ROUTE()
{
  DP_ASSERT( fromNode );
  DP_ASSERT( toNode );
}
