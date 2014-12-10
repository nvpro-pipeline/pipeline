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

  DEFINE_PTR_TYPES( Object );

  class Object
  {
    public:
      static ObjectSharedPtr create();
      virtual ~Object() {}
      virtual const std::string & getType() const;
      const std::string& getName() const { return name; } 
      void setName(const std::string& n) { name = n; }

    protected:
      Object() {}

    private:
      std::string name;
  };

  template<typename T> 
  inline bool isTypeOf( Object *p )
  {
    return( dynamic_cast<T *>( p ) != NULL );
  }

  typedef bool                  SFBool;
  typedef dp::math::Vec3f       SFColor;
  typedef std::vector<SFColor>  MFColor;
  typedef float                 SFFloat;
  typedef std::vector<SFFloat>  MFFloat;
  typedef char                  SFInt8;
  typedef int                   SFInt32;
  typedef std::vector<SFInt32>  MFInt32;
  typedef ObjectSharedPtr       SFNode;
  typedef std::vector<SFNode>   MFNode;
  typedef std::string           SFString;
  typedef std::vector<SFString> MFString;
  typedef double                SFTime;
  typedef std::vector<SFTime>   MFTime;
  typedef dp::math::Vec2f       SFVec2f;
  typedef std::vector<SFVec2f>  MFVec2f;
  typedef dp::math::Vec3f       SFVec3f;
  typedef std::vector<SFVec3f>  MFVec3f;

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


  DEFINE_PTR_TYPES( Geometry );
  class Geometry : public Object
  {
    public:
      virtual ~Geometry() {}
      virtual const std::string & getType( void ) const;

    protected:
      Geometry()  {}
  };

  DEFINE_PTR_TYPES( Group );
  class Group : public Object
  {
    public:
      static GroupSharedPtr create();
      virtual ~Group();
      virtual const std::string & getType( void ) const;

    protected:
      Group();

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

  DEFINE_PTR_TYPES( TimeSensor );
  class TimeSensor : public Sensor
  {
    public:
      static TimeSensorSharedPtr create();
      virtual ~TimeSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      TimeSensor();

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
      MFFloat                         key;
      bool                            interpreted;
      dp::util::SharedPtr<TimeSensor> set_fraction;
      std::vector<unsigned int>       steps;
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

  DEFINE_PTR_TYPES( Texture );
  class Texture : public Object
  {
    public:
      static TextureSharedPtr create();
      virtual ~Texture()  {}
      virtual const std::string & getType( void ) const;

    protected:
      Texture();

    public:
      SFBool  repeatS;
      SFBool  repeatT;
  };

  DEFINE_PTR_TYPES( Anchor );
  class Anchor : public Group
  {
    public:
      static AnchorSharedPtr create();
      virtual ~Anchor() {}
      virtual const std::string & getType( void ) const;

    protected:
      Anchor() {}

    public:
      SFString  description;
      MFString  parameter;
      MFString  url;
  };

  DEFINE_PTR_TYPES( Appearance );
  class Appearance : public Object
  {
    public:
      static AppearanceSharedPtr create();
      virtual ~Appearance();
      virtual const std::string & getType( void ) const;

    protected:
      Appearance();

    public:
      SFNode  material;
      SFNode  texture;
      SFNode  textureTransform;

      dp::sg::core::EffectDataSharedPtr materialEffect;
  };

  DEFINE_PTR_TYPES( AudioClip );
  class AudioClip : public Object
  {
    public:
      static AudioClipSharedPtr create();
      virtual ~AudioClip()  {}
      virtual const std::string & getType( void ) const;

    protected:
      AudioClip();

    public:
      SFString  description;
      SFBool    loop;
      SFFloat   pitch;
      SFTime    startTime;
      SFTime    stopTime;
      MFString  url;
  };

  DEFINE_PTR_TYPES( Background );
  class Background : public Object
  {
    public:
      static BackgroundSharedPtr create();
      virtual ~Background() {}
      virtual const std::string & getType( void ) const;

    protected:
      Background();

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

  DEFINE_PTR_TYPES( Billboard );
  class Billboard : public Group
  {
    public:
      static BillboardSharedPtr create();
      virtual ~Billboard();
      virtual const std::string & getType( void ) const;

    protected:
      Billboard();

    public:
      SFVec3f axisOfRotation;

      dp::sg::core::BillboardSharedPtr  pBillboard;
  };

  DEFINE_PTR_TYPES( Box );
  class Box : public Geometry
  {
    public:
      static BoxSharedPtr create();
      virtual ~Box();
      virtual const std::string & getType( void ) const;

    protected:
      Box();

    public:
      SFVec3f size;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  DEFINE_PTR_TYPES( Collision );
  class Collision : public Group
  {
    public:
      static CollisionSharedPtr create();
      virtual ~Collision();
      virtual const std::string & getType( void ) const;

    protected:
      Collision();

    public:
      SFBool  collide;
      SFNode  proxy;
  };

  DEFINE_PTR_TYPES( ColorInterpolator );
  class ColorInterpolator : public Interpolator
  {
    public:
      static ColorInterpolatorSharedPtr create();
      virtual ~ColorInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      ColorInterpolator() {}

    public:
      MFColor keyValue;
  };

  DEFINE_PTR_TYPES( Color );
  class Color : public Object
  {
    public:
      static ColorSharedPtr create();
      virtual ~Color() {}
      virtual const std::string & getType( void ) const;

    protected:
      Color();

    public:
      MFColor                     color;
      bool                        interpreted;
      ColorInterpolatorSharedPtr  set_color;
  };

  DEFINE_PTR_TYPES( Cone );
  class Cone : public Geometry
  {
    public:
      static ConeSharedPtr create();
      virtual ~Cone() {}
      virtual const std::string & getType( void ) const;

    protected:
      Cone();

    public:
      SFBool  bottom;
      SFFloat bottomRadius;
      SFFloat height;
      SFBool  side;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  DEFINE_PTR_TYPES( CoordinateInterpolator );
  class CoordinateInterpolator : public Interpolator
  {
    public:
      static CoordinateInterpolatorSharedPtr create();
      virtual ~CoordinateInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      CoordinateInterpolator() {}

    public:
      MFVec3f keyValue;
  };

  DEFINE_PTR_TYPES( Coordinate );
  class Coordinate : public Object
  {
    public:
      static CoordinateSharedPtr create();
      virtual ~Coordinate();
      virtual const std::string & getType( void ) const;

    protected:
      Coordinate();

    public:
      MFVec3f                         point;
      bool                            interpreted;
      CoordinateInterpolatorSharedPtr set_point;
  };

  DEFINE_PTR_TYPES( Cylinder );
  class Cylinder : public Geometry
  {
    public:
      static CylinderSharedPtr create();
      virtual ~Cylinder() {}
      virtual const std::string & getType( void ) const;

    protected:
      Cylinder();

    public:
      SFBool  bottom;
      SFFloat height;
      SFFloat radius;
      SFBool  side;
      SFBool  top;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  DEFINE_PTR_TYPES( CylinderSensor );
  class CylinderSensor : public Sensor
  {
    public:
      static CylinderSensorSharedPtr create();
      virtual ~CylinderSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      CylinderSensor();

    public:
      SFBool  autoOffset;
      SFFloat diskAngle;
      SFFloat maxAngle;
      SFFloat minAngle;
      SFFloat offset;
  };

  DEFINE_PTR_TYPES( DirectionalLight );
  class DirectionalLight : public Light
  {
    public:
      static DirectionalLightSharedPtr create();
      virtual ~DirectionalLight();
      virtual const std::string & getType( void ) const;

    protected:
      DirectionalLight();

    public:
      SFVec3f direction;
  };

  DEFINE_PTR_TYPES( ElevationGrid );
  class ElevationGrid : public Geometry
  {
    public:
      static ElevationGridSharedPtr create();
      virtual ~ElevationGrid();
      virtual const std::string & getType( void ) const;

    protected:
      ElevationGrid();

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

  DEFINE_PTR_TYPES( Extrusion );
  class Extrusion : public Geometry
  {
    public:
      static ExtrusionSharedPtr create();
      virtual ~Extrusion()  {}
      virtual const std::string & getType( void ) const;

    protected:
      Extrusion();

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

  DEFINE_PTR_TYPES( Fog );
  class Fog : public Object
  {
    public:
      static FogSharedPtr create();
      virtual ~Fog()  {}
      virtual const std::string & getType( void ) const;

    protected:
      Fog();

    public:
      SFColor   color;
      SFString  fogType;
      SFFloat   visibilityRange;
  };

  DEFINE_PTR_TYPES( FontStyle );
  class FontStyle : public Object
  {
    public:
      static FontStyleSharedPtr create();
      virtual ~FontStyle()  {}
      virtual const std::string & getType( void ) const;

    protected:
      FontStyle();

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

  DEFINE_PTR_TYPES( ImageTexture );
  class ImageTexture : public Texture
  {
    public:
      static ImageTextureSharedPtr create();
      virtual ~ImageTexture();
      virtual const std::string & getType( void ) const;

    protected:
      ImageTexture();

    public:
      MFString  url;

      dp::sg::core::ParameterGroupDataSharedPtr textureData;
  };

  DEFINE_PTR_TYPES( IndexedFaceSet );
  class IndexedFaceSet : public Geometry
  {
    public:
      static IndexedFaceSetSharedPtr create();
      virtual ~IndexedFaceSet();
      virtual const std::string & getType( void ) const;

    protected:
      IndexedFaceSet();

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

  DEFINE_PTR_TYPES( IndexedLineSet );
  class IndexedLineSet : public Geometry
  {
    public:
      static IndexedLineSetSharedPtr create();
      virtual ~IndexedLineSet();
      virtual const std::string & getType( void ) const;

    protected:
      IndexedLineSet();

    public:
      SFNode  color;
      SFNode  coord;
      MFInt32 colorIndex;
      SFBool  colorPerVertex;
      MFInt32 coordIndex;

      dp::sg::core::PrimitiveSharedPtr  pLineStrips;
  };

  DEFINE_PTR_TYPES( Inline );
  class Inline : public Object
  {
    public:
      static InlineSharedPtr create();
      virtual ~Inline();
      virtual const std::string & getType( void ) const;

    protected:
      Inline();

    public:
      MFString  url;
      SFVec3f   bboxCenter;
      SFVec3f   bboxSize;

      dp::sg::core::NodeSharedPtr pNode;
  };

  DEFINE_PTR_TYPES( LOD );
  class LOD : public Group
  {
    public:
      static LODSharedPtr create();
      virtual ~LOD();
      virtual const std::string & getType( void ) const;

    protected:
      LOD();

    public:
      SFVec3f center;
      MFFloat range;

      dp::sg::core::LODSharedPtr  pLOD;
  };

  DEFINE_PTR_TYPES( Material );
  class Material : public Object
  {
    public:
      static MaterialSharedPtr create();
      virtual ~Material();
      virtual const std::string & getType( void ) const;

    protected:
      Material();

    public:
      SFFloat ambientIntensity;
      SFColor diffuseColor;
      SFColor emissiveColor;
      SFFloat shininess;
      SFColor specularColor;
      SFFloat transparency;

      dp::sg::core::ParameterGroupDataSharedPtr materialParameters;
  };

  DEFINE_PTR_TYPES( MovieTexture );
  class MovieTexture : public Texture
  {
    public:
      static MovieTextureSharedPtr create();
      virtual ~MovieTexture()  {}
      virtual const std::string & getType( void ) const;

    protected:
      MovieTexture();

    public:
      SFBool    loop;
      SFFloat   speed;
      SFTime    startTime;
      SFTime    stopTime;
      MFString  url;
  };

  DEFINE_PTR_TYPES( NavigationInfo );
  class NavigationInfo : public Object
  {
    public:
      static NavigationInfoSharedPtr create();
      virtual ~NavigationInfo() {}
      virtual const std::string & getType( void ) const;

    protected:
      NavigationInfo();

    public:
      MFFloat   avatarSize;
      SFBool    headlight;
      SFFloat   speed;
      MFString  type;
      SFFloat   visibilityLimit;
  };

  DEFINE_PTR_TYPES( NormalInterpolator );
  class NormalInterpolator : public Interpolator
  {
    public:
      static NormalInterpolatorSharedPtr create();
      virtual ~NormalInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      NormalInterpolator() {}

    public:
      MFVec3f keyValue;
  };

  DEFINE_PTR_TYPES( Normal );
  class Normal : public Object
  {
    public:
      static NormalSharedPtr create();
      virtual ~Normal();
      virtual const std::string & getType( void ) const;

    protected:
      Normal();

    public:
      MFVec3f                     vector;
      bool                        interpreted;
      NormalInterpolatorSharedPtr set_vector;
  };

  DEFINE_PTR_TYPES( OrientationInterpolator );
  class OrientationInterpolator : public Interpolator
  {
    public:
      static OrientationInterpolatorSharedPtr create();
      virtual ~OrientationInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      OrientationInterpolator() {}

    public:
      MFRotation  keyValue;

      std::vector<dp::math::Quatf> keyValueQuatf;
  };

  DEFINE_PTR_TYPES( PixelTexture );
  class PixelTexture : public Texture
  {
    public:
      static PixelTextureSharedPtr create();
      virtual ~PixelTexture()  {}
      virtual const std::string & getType( void ) const;

    protected:
      PixelTexture();

    public:
      SFImage image;
  };

  DEFINE_PTR_TYPES( PlaneSensor );
  class PlaneSensor : public Sensor
  {
    public:
      static PlaneSensorSharedPtr create();
      virtual ~PlaneSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      PlaneSensor();

    public:
      SFBool  autoOffset;
      SFVec2f maxPosition;
      SFVec2f minPosition;
      SFVec3f offset;
  };

  DEFINE_PTR_TYPES( PointLight );
  class PointLight : public Light
  {
    public:
      static PointLightSharedPtr create();
      virtual ~PointLight();
      virtual const std::string & getType( void ) const;

    protected:
      PointLight();

    public:
      SFVec3f attenuation;
      SFVec3f location;
      SFFloat radius;
  };

  DEFINE_PTR_TYPES( PointSet );
  class PointSet : public Geometry
  {
    public:
      static PointSetSharedPtr create();
      virtual ~PointSet();
      virtual const std::string & getType( void ) const;

    protected:
      PointSet();

    public:
      SFNode  color;
      SFNode  coord;

      dp::sg::core::PrimitiveSharedPtr  pPoints;
  };

  DEFINE_PTR_TYPES( PositionInterpolator );
  class PositionInterpolator : public Interpolator
  {
    public:
      static PositionInterpolatorSharedPtr create();
      virtual ~PositionInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      PositionInterpolator() {}

    public:
      MFVec3f keyValue;
  };

  DEFINE_PTR_TYPES( ProximitySensor );
  class ProximitySensor : public Sensor
  {
    public:
      static ProximitySensorSharedPtr create();
      virtual ~ProximitySensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      ProximitySensor();

    public:
      SFVec3f center;
      SFVec3f size;
  };

  DEFINE_PTR_TYPES( ScalarInterpolator );
  class ScalarInterpolator : public Interpolator
  {
    public:
      static ScalarInterpolatorSharedPtr create();
      virtual ~ScalarInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      ScalarInterpolator()  {}

    public:
      MFFloat keyValue;
  };

  DEFINE_PTR_TYPES( Script );
  class Script : public Object
  {
    public:
      static ScriptSharedPtr create();
      virtual ~Script() {}
      virtual const std::string & getType( void ) const;

    protected:
      Script();

    public:
      MFString  url;
      SFBool    directOutput;
      SFBool    mustEvaluate;
  };

  DEFINE_PTR_TYPES( Shape );
  class Shape : public Object
  {
    public:
      static ShapeSharedPtr create();
      virtual ~Shape();
      virtual const std::string & getType( void ) const;

    protected:
      Shape();

    public:
      SFNode  appearance;
      SFNode  geometry;

      dp::sg::core::NodeSharedPtr  pNode;
  };

  DEFINE_PTR_TYPES( Sound );
  class Sound : public Object
  {
    public:
      static SoundSharedPtr create();
      virtual ~Sound();
      virtual const std::string & getType( void ) const;

    protected:
      Sound();

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

  DEFINE_PTR_TYPES( Sphere );
  class Sphere : public Geometry
  {
    public:
      static SphereSharedPtr create();
      virtual ~Sphere() {}
      virtual const std::string & getType( void ) const;

    protected:
      Sphere();

    public:
      SFFloat radius;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  DEFINE_PTR_TYPES( SphereSensor );
  class SphereSensor : public Sensor
  {
    public:
      static SphereSensorSharedPtr create();
      virtual ~SphereSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      SphereSensor();

    public:
      SFBool      autoOffset;
      SFRotation  offset;
  };

  DEFINE_PTR_TYPES( SpotLight );
  class SpotLight : public Light
  {
    public:
      static SpotLightSharedPtr create();
      virtual ~SpotLight();
      virtual const std::string & getType( void ) const;

    protected:
      SpotLight();

    public:
      SFVec3f attenuation;
      SFFloat beamWidth;
      SFFloat cutOffAngle;
      SFVec3f direction;
      SFVec3f location;
      SFFloat radius;
  };

  DEFINE_PTR_TYPES( Switch );
  class Switch : public Group
  {
    public:
      static SwitchSharedPtr create();
      virtual ~Switch();
      virtual const std::string & getType( void ) const;

    protected:
      Switch();

    public:
      SFInt32 whichChoice;

      dp::sg::core::SwitchSharedPtr pSwitch;
  };

  DEFINE_PTR_TYPES( Text );
  class Text : public Geometry
  {
    public:
      static TextSharedPtr create();
      virtual ~Text();
      virtual const std::string & getType( void ) const;

    protected:
      Text();

    public:
      MFString  string;
      SFNode    fontStyle;
      MFFloat   length;
      SFFloat   maxExtent;
  };

  DEFINE_PTR_TYPES( TextureCoordinate );
  class TextureCoordinate : public Object
  {
    public:
      static TextureCoordinateSharedPtr create();
      virtual ~TextureCoordinate()  {}
      virtual const std::string & getType( void ) const;

    protected:
      TextureCoordinate() {}

    public:
      MFVec2f point;
  };

  DEFINE_PTR_TYPES( TextureTransform );
  class TextureTransform : public Object
  {
    public:
      static TextureTransformSharedPtr create();
      virtual ~TextureTransform()  {}
      virtual const std::string & getType( void ) const;

    protected:
      TextureTransform();

    public:
      SFVec2f center;
      SFFloat rotation;
      SFVec2f scale;
      SFVec2f translation;
  };

  DEFINE_PTR_TYPES( TouchSensor );
  class TouchSensor : public Sensor
  {
    public:
      static TouchSensorSharedPtr create();
      virtual ~TouchSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      TouchSensor() {}
  };

  DEFINE_PTR_TYPES( Transform );
  class Transform : public Group
  {
    public:
      static TransformSharedPtr create();
      virtual ~Transform();
      virtual const std::string & getType( void ) const;

    protected:
      Transform();

    public:
      SFVec3f     center;
      SFRotation  rotation;
      SFVec3f     scale;
      SFRotation  scaleOrientation;
      SFVec3f     translation;

      PositionInterpolatorSharedPtr    set_center;
      OrientationInterpolatorSharedPtr set_rotation;
      PositionInterpolatorSharedPtr    set_scale;
      PositionInterpolatorSharedPtr    set_translation;

      dp::sg::core::TransformSharedPtr  pTransform;
  };

  DEFINE_PTR_TYPES( Viewpoint );
  class Viewpoint : public Object
  {
    public:
      static ViewpointSharedPtr create();
      virtual ~Viewpoint();
      virtual const std::string & getType( void ) const;

    protected:
      Viewpoint();

    public:
      SFFloat     fieldOfView;
      SFBool      jump;
      SFRotation  orientation;
      SFVec3f     position;
      SFString    description;

      OrientationInterpolatorSharedPtr set_orientation;
      PositionInterpolatorSharedPtr    set_position;
  };

  DEFINE_PTR_TYPES( VisibilitySensor );
  class VisibilitySensor : public Sensor
  {
    public:
      static VisibilitySensorSharedPtr create();
      virtual ~VisibilitySensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      VisibilitySensor();

    public:
      SFVec3f center;
      SFVec3f size;
  };

  DEFINE_PTR_TYPES( WorldInfo );
  class WorldInfo : public Object
  {
    public:
      static WorldInfoSharedPtr create();
      virtual ~WorldInfo()  {}
      virtual const std::string & getType( void ) const;

    protected:
      WorldInfo() {}

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
