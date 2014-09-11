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


#pragma once
/** \file */

#include <dp/math/Quatt.h>
#include <dp/math/Vecnt.h>
#include <dp/sg/core/Object.h>

namespace vrml
{  

  //  it's not neccessary to document all these helper classes !
#if ! defined( DOXYGEN_IGNORE )

  class Object : public dp::util::RCObject
  {
    public:
      virtual ~Object() {}
      virtual const std::string & getType() const;
      const std::string& getName() const { return name; } 
      void setName(const std::string& n) { name = n; }
    private:
      std::string name;
  };

  template<typename T> 
  inline bool isTypeOf( Object *p )
  {
    return( dynamic_cast<T *>( p ) != NULL );
  }

  typedef bool                            SFBool;
  typedef dp::math::Vec3f                 SFColor;
  typedef std::vector<SFColor>            MFColor;
  typedef float                           SFFloat;
  typedef std::vector<SFFloat>            MFFloat;
  typedef char                            SFInt8;
  typedef int                             SFInt32;
  typedef std::vector<SFInt32>            MFInt32;
  typedef dp::util::SmartPtr<vrml::Object>  SFNode;
  typedef std::vector<SFNode>             MFNode;
  typedef std::string                     SFString;
  typedef std::vector<SFString>           MFString;
  typedef double                          SFTime;
  typedef std::vector<SFTime>             MFTime;
  typedef dp::math::Vec2f                 SFVec2f;
  typedef std::vector<SFVec2f>            MFVec2f;
  typedef dp::math::Vec3f                 SFVec3f;
  typedef std::vector<SFVec3f>            MFVec3f;

  class SFImage
  {
    public:
      SFImage();
      ~SFImage();

    public:
      SFInt32   width;
      SFInt32   height;
      SFInt32   numComponents;
      SFInt8  * pixelsValues;
  };

  class SFRotation : public dp::math::Vec4f
  {
    public:
      SFRotation()  {}
      SFRotation( const SFVec3f &axis, SFFloat angle );
      SFRotation( const dp::math::Vec4f &v )
        : dp::math::Vec4f( v )
      {
      }
  };
  typedef std::vector<SFRotation>  MFRotation;

  inline SFRotation lerp( float alpha, const SFRotation &r0, const SFRotation &r1 )
  {
    dp::math::Quatf q0(dp::math::Vec3f(r0[0],r0[1],r0[2]),r0[3]);
    dp::math::Quatf q1(dp::math::Vec3f(r1[0],r1[1],r1[2]),r1[3]);
    dp::math::Quatf q = dp::math::lerp( alpha, q0, q1 );
    dp::math::Vec3f axis;
    float angle;
    decompose( q, axis, angle );
    return( SFRotation( axis, angle ) );
  }


  class Geometry : public Object
  {
    public:
      virtual ~Geometry() {}
      virtual const std::string & getType( void ) const;
  };

  class Group : public Object
  {
    public:
      Group();
      virtual ~Group();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f bboxCenter;
      SFVec3f bboxSize;
      MFNode  children;

      dp::sg::core::GroupSharedPtr  pGroup;
  };

  class Sensor : public Object
  {
    public:
      Sensor();
      virtual ~Sensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool  enabled;
  };

  class TimeSensor : public Sensor
  {
    public:
      TimeSensor();
      virtual ~TimeSensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFTime  cycleInterval;
      SFBool  loop;
      SFTime  startTime;
      SFTime  stopTime;
  };

  class Interpolator : public Object
  {
    public:
      Interpolator();
      virtual ~Interpolator();
      virtual const std::string & getType( void ) const;

    public:
      MFFloat                       key;
      bool                          interpreted;
      dp::util::SmartPtr<TimeSensor>  set_fraction;
      std::vector<unsigned int>     steps;
  };

  class Light : public Object
  {
    public:
      Light();
      virtual ~Light()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFFloat ambientIntensity;
      SFColor color;
      SFFloat intensity;
      SFBool  on;

      dp::sg::core::LightSourceSharedPtr lightSource;
  };

  class Texture : public Object
  {
    public:
      Texture();
      virtual ~Texture()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool  repeatS;
      SFBool  repeatT;
  };

  class Anchor : public Group
  {
    public:
      virtual ~Anchor() {}
      virtual const std::string & getType( void ) const;

    public:
      SFString  description;
      MFString  parameter;
      MFString  url;
  };

  class Appearance : public Object
  {
    public:
      Appearance();
      virtual ~Appearance();
      virtual const std::string & getType( void ) const;

    public:
      SFNode  material;
      SFNode  texture;
      SFNode  textureTransform;

      dp::sg::core::EffectDataSharedPtr materialEffect;
  };

  class AudioClip : public Object
  {
    public:
      AudioClip();
      virtual ~AudioClip()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFString  description;
      SFBool    loop;
      SFFloat   pitch;
      SFTime    startTime;
      SFTime    stopTime;
      MFString  url;
  };

  class Background : public Object
  {
    public:
      Background();
      virtual ~Background() {}
      virtual const std::string & getType( void ) const;

    public:
      MFFloat   groundAngle;
      MFColor   groundColor;
      MFString  backUrl;
      MFString  bottomUrl;
      MFString  frontUrl;
      MFString  leftUrl;
      MFString  rightUrl;
      MFString  topUrl;
      MFFloat   skyAngle;
      MFColor   skyColor;
  };

  class Billboard : public Group
  {
    public:
      Billboard();
      virtual ~Billboard();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f axisOfRotation;

      dp::sg::core::BillboardSharedPtr  pBillboard;
  };

  class Box : public Geometry
  {
    public:
      Box();
      virtual ~Box();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f size;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  class Collision : public Group
  {
    public:
      Collision();
      virtual ~Collision();
      virtual const std::string & getType( void ) const;

    public:
      SFBool  collide;
      SFNode  proxy;
  };

  class ColorInterpolator : public Interpolator
  {
    public:
      virtual ~ColorInterpolator()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFColor keyValue;
  };

  class Color : public Object
  {
    public:
      Color();
      virtual ~Color() {}
      virtual const std::string & getType( void ) const;

    public:
      MFColor                             color;
      bool                                interpreted;
      dp::util::SmartPtr<ColorInterpolator> set_color;
  };

  class Cone : public Geometry
  {
    public:
      Cone();
      virtual ~Cone() {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool  bottom;
      SFFloat bottomRadius;
      SFFloat height;
      SFBool  side;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  class CoordinateInterpolator : public Interpolator
  {
    public:
      virtual ~CoordinateInterpolator()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFVec3f keyValue;
  };

  class Coordinate : public Object
  {
    public:
      Coordinate();
      virtual ~Coordinate();
      virtual const std::string & getType( void ) const;

    public:
      MFVec3f                                   point;
      bool                                      interpreted;
      dp::util::SmartPtr<CoordinateInterpolator>  set_point;
  };

  class Cylinder : public Geometry
  {
    public:
      Cylinder();
      virtual ~Cylinder() {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool  bottom;
      SFFloat height;
      SFFloat radius;
      SFBool  side;
      SFBool  top;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  class CylinderSensor : public Sensor
  {
    public:
      CylinderSensor();
      virtual ~CylinderSensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool  autoOffset;
      SFFloat diskAngle;
      SFFloat maxAngle;
      SFFloat minAngle;
      SFFloat offset;
  };

  class DirectionalLight : public Light
  {
    public:
      DirectionalLight();
      virtual ~DirectionalLight();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f direction;
  };

  class ElevationGrid : public Geometry
  {
    public:
      ElevationGrid();
      virtual ~ElevationGrid();
      virtual const std::string & getType( void ) const;

    public:
      SFNode  color;
      SFNode  normal;
      SFNode  texCoord;
      MFFloat height;
      SFBool  ccw;
      SFBool  colorPerVertex;
      SFFloat creaseAngle;
      SFBool  normalPerVertex;
      SFBool  solid;
      SFInt32 xDimension;
      SFFloat xSpacing;
      SFInt32 zDimension;
      SFFloat zSpacing;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  class Extrusion : public Geometry
  {
    public:
      Extrusion();
      virtual ~Extrusion()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool      beginCap;
      SFBool      ccw;
      SFBool      convex;
      SFFloat     creaseAngle;
      MFVec2f     crossSection;
      SFBool      endCap;
      MFRotation  orientation;
      MFVec2f     scale;
      SFBool      solid;
      MFVec3f     spine;
  };

  class Fog : public Object
  {
    public:
      Fog();
      virtual ~Fog()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFColor   color;
      SFString  fogType;
      SFFloat   visibilityRange;
  };

  class FontStyle : public Object
  {
    public:
      FontStyle();
      virtual ~FontStyle()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFString  family;
      SFBool    horizontal;
      MFString  justify;
      SFString  language;
      SFBool    leftToRight;
      SFFloat   size;
      SFFloat   spacing;
      SFString  style;
      SFBool    topToBottom;
  };

  class ImageTexture : public Texture
  {
    public:
      ImageTexture();
      virtual ~ImageTexture();
      virtual const std::string & getType( void ) const;

    public:
      MFString  url;

      dp::sg::core::ParameterGroupDataSharedPtr textureData;
  };

  class IndexedFaceSet : public Geometry
  {
    public:
      IndexedFaceSet();
      virtual ~IndexedFaceSet();
      virtual const std::string & getType( void ) const;

    public:
      SFNode  color;
      SFNode  coord;
      SFNode  normal;
      SFNode  texCoord;
      SFBool  ccw;
      MFInt32 colorIndex;
      SFBool  colorPerVertex;
      SFBool  convex;
      MFInt32 coordIndex;
      SFFloat creaseAngle;
      MFInt32 normalIndex;
      SFBool  normalPerVertex;
      SFBool  solid;
      MFInt32 texCoordIndex;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
      dp::sg::core::PrimitiveSharedPtr pPolygons;
  };

  class IndexedLineSet : public Geometry
  {
    public:
      IndexedLineSet();
      virtual ~IndexedLineSet();
      virtual const std::string & getType( void ) const;

    public:
      SFNode  color;
      SFNode  coord;
      MFInt32 colorIndex;
      SFBool  colorPerVertex;
      MFInt32 coordIndex;

      dp::sg::core::PrimitiveSharedPtr  pLineStrips;
  };

  class Inline : public Object
  {
    public:
      Inline();
      virtual ~Inline();
      virtual const std::string & getType( void ) const;

    public:
      MFString  url;
      SFVec3f   bboxCenter;
      SFVec3f   bboxSize;

      dp::sg::core::NodeSharedPtr pNode;
  };

  class LOD : public Group
  {
    public:
      LOD();
      virtual ~LOD();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f center;
      MFFloat range;

      dp::sg::core::LODSharedPtr  pLOD;
  };

  class Material : public Object
  {
    public:
      Material();
      virtual ~Material();
      virtual const std::string & getType( void ) const;

    public:
      SFFloat ambientIntensity;
      SFColor diffuseColor;
      SFColor emissiveColor;
      SFFloat shininess;
      SFColor specularColor;
      SFFloat transparency;

      dp::sg::core::ParameterGroupDataSharedPtr materialParameters;
  };

  class MovieTexture : public Texture
  {
    public:
      MovieTexture();
      virtual ~MovieTexture()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool    loop;
      SFFloat   speed;
      SFTime    startTime;
      SFTime    stopTime;
      MFString  url;
  };

  class NavigationInfo : public Object
  {
    public:
      NavigationInfo();
      virtual ~NavigationInfo() {}
      virtual const std::string & getType( void ) const;

    public:
      MFFloat   avatarSize;
      SFBool    headlight;
      SFFloat   speed;
      MFString  type;
      SFFloat   visibilityLimit;
  };

  class NormalInterpolator : public Interpolator
  {
    public:
      virtual ~NormalInterpolator()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFVec3f keyValue;
  };

  class Normal : public Object
  {
    public:
      Normal();
      virtual ~Normal();
      virtual const std::string & getType( void ) const;

    public:
      MFVec3f                               vector;
      bool                                  interpreted;
      dp::util::SmartPtr<NormalInterpolator>  set_vector;
  };

  class OrientationInterpolator : public Interpolator
  {
    public:
      virtual ~OrientationInterpolator()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFRotation  keyValue;

      std::vector<dp::math::Quatf> keyValueQuatf;
  };

  class PixelTexture : public Texture
  {
    public:
      PixelTexture();
      virtual ~PixelTexture()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFImage image;
  };

  class PlaneSensor : public Sensor
  {
    public:
      PlaneSensor();
      virtual ~PlaneSensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool  autoOffset;
      SFVec2f maxPosition;
      SFVec2f minPosition;
      SFVec3f offset;
  };

  class PointLight : public Light
  {
    public:
      PointLight();
      virtual ~PointLight();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f attenuation;
      SFVec3f location;
      SFFloat radius;
  };

  class PointSet : public Geometry
  {
    public:
      PointSet();
      virtual ~PointSet();
      virtual const std::string & getType( void ) const;

    public:
      SFNode  color;
      SFNode  coord;

      dp::sg::core::PrimitiveSharedPtr  pPoints;
  };

  class PositionInterpolator : public Interpolator
  {
    public:
      virtual ~PositionInterpolator()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFVec3f keyValue;
  };

  class ProximitySensor : public Sensor
  {
    public:
      ProximitySensor();
      virtual ~ProximitySensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f center;
      SFVec3f size;
  };

  class ScalarInterpolator : public Interpolator
  {
    public:
      virtual ~ScalarInterpolator()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFFloat keyValue;
  };

  class Script : public Object
  {
    public:
      Script();
      virtual ~Script() {}
      virtual const std::string & getType( void ) const;

    public:
      MFString  url;
      SFBool    directOutput;
      SFBool    mustEvaluate;
  };

  class Shape : public Object
  {
    public:
      Shape();
      virtual ~Shape();
      virtual const std::string & getType( void ) const;

    public:
      SFNode  appearance;
      SFNode  geometry;

      dp::sg::core::NodeSharedPtr  pNode;
  };

  class Sound : public Object
  {
    public:
      Sound();
      virtual ~Sound();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f direction;
      SFFloat intensity;
      SFVec3f location;
      SFFloat maxBack;
      SFFloat maxFront;
      SFFloat minBack;
      SFFloat minFront;
      SFFloat priority;
      SFNode  source;
      SFBool  spatialize;
  };

  class Sphere : public Geometry
  {
    public:
      Sphere();
      virtual ~Sphere() {}
      virtual const std::string & getType( void ) const;

    public:
      SFFloat radius;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  class SphereSensor : public Sensor
  {
    public:
      SphereSensor();
      virtual ~SphereSensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFBool      autoOffset;
      SFRotation  offset;
  };

  class SpotLight : public Light
  {
    public:
      SpotLight();
      virtual ~SpotLight();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f attenuation;
      SFFloat beamWidth;
      SFFloat cutOffAngle;
      SFVec3f direction;
      SFVec3f location;
      SFFloat radius;
  };

  class Switch : public Group
  {
    public:
      Switch();
      virtual ~Switch();
      virtual const std::string & getType( void ) const;

    public:
      SFInt32 whichChoice;

      dp::sg::core::SwitchSharedPtr pSwitch;
  };

  class Text : public Geometry
  {
    public:
      Text();
      virtual ~Text();
      virtual const std::string & getType( void ) const;

    public:
      MFString  string;
      SFNode    fontStyle;
      MFFloat   length;
      SFFloat   maxExtent;
  };

  class TextureCoordinate : public Object
  {
    public:
      virtual ~TextureCoordinate()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFVec2f point;
  };

  class TextureTransform : public Object
  {
    public:
      TextureTransform();
      virtual ~TextureTransform()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFVec2f center;
      SFFloat rotation;
      SFVec2f scale;
      SFVec2f translation;
  };

  class TouchSensor : public Sensor
  {
    public:
      virtual ~TouchSensor()  {}
      virtual const std::string & getType( void ) const;
  };

  class Transform : public Group
  {
    public:
      Transform();
      virtual ~Transform();
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f     center;
      SFRotation  rotation;
      SFVec3f     scale;
      SFRotation  scaleOrientation;
      SFVec3f     translation;

      dp::util::SmartPtr<PositionInterpolator>    set_center;
      dp::util::SmartPtr<OrientationInterpolator> set_rotation;
      dp::util::SmartPtr<PositionInterpolator>    set_scale;
      dp::util::SmartPtr<PositionInterpolator>    set_translation;

      dp::sg::core::TransformSharedPtr  pTransform;
  };

  class Viewpoint : public Object
  {
    public:
      Viewpoint();
      virtual ~Viewpoint();
      virtual const std::string & getType( void ) const;

    public:
      SFFloat     fieldOfView;
      SFBool      jump;
      SFRotation  orientation;
      SFVec3f     position;
      SFString    description;

      dp::util::SmartPtr<OrientationInterpolator> set_orientation;
      dp::util::SmartPtr<PositionInterpolator>    set_position;
  };

  class VisibilitySensor : public Sensor
  {
    public:
      VisibilitySensor();
      virtual ~VisibilitySensor()  {}
      virtual const std::string & getType( void ) const;

    public:
      SFVec3f center;
      SFVec3f size;
  };

  class WorldInfo : public Object
  {
    public:
      virtual ~WorldInfo()  {}
      virtual const std::string & getType( void ) const;

    public:
      MFString  info;
      SFString  title;
  };


  class ROUTE
  {
    public:
      ROUTE();
      ~ROUTE();

    public:
      SFNode  fromNode;
      std::string  fromAction;
      SFNode  toNode;
      std::string  toAction;
  };


  template  <typename T>  inline  bool  is( const Object *p )
  {
    return( dynamic_cast<const T *>( p ) != NULL );
  }

#endif  //  DOXYGEN_IGNORE

} //  namespace vrml
