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

  SHARED_TYPES( Object );

  class Object
  {
    public:
      static SharedObject create();
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
  typedef SharedObject          SFNode;
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


  SHARED_TYPES( Geometry );
  class Geometry : public Object
  {
    public:
      virtual ~Geometry() {}
      virtual const std::string & getType( void ) const;

    protected:
      Geometry()  {}
  };

  SHARED_TYPES( Group );
  class Group : public Object
  {
    public:
      static SharedGroup create();
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

  SHARED_TYPES( TimeSensor );
  class TimeSensor : public Sensor
  {
    public:
      static SharedTimeSensor create();
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

  SHARED_TYPES( Texture );
  class Texture : public Object
  {
    public:
      static SharedTexture create();
      virtual ~Texture()  {}
      virtual const std::string & getType( void ) const;

    protected:
      Texture();

    public:
      SFBool  repeatS;
      SFBool  repeatT;
  };

  SHARED_TYPES( Anchor );
  class Anchor : public Group
  {
    public:
      static SharedAnchor create();
      virtual ~Anchor() {}
      virtual const std::string & getType( void ) const;

    protected:
      Anchor() {}

    public:
      SFString  description;
      MFString  parameter;
      MFString  url;
  };

  SHARED_TYPES( Appearance );
  class Appearance : public Object
  {
    public:
      static SharedAppearance create();
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

  SHARED_TYPES( AudioClip );
  class AudioClip : public Object
  {
    public:
      static SharedAudioClip create();
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

  SHARED_TYPES( Background );
  class Background : public Object
  {
    public:
      static SharedBackground create();
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

  SHARED_TYPES( Billboard );
  class Billboard : public Group
  {
    public:
      static SharedBillboard create();
      virtual ~Billboard();
      virtual const std::string & getType( void ) const;

    protected:
      Billboard();

    public:
      SFVec3f axisOfRotation;

      dp::sg::core::BillboardSharedPtr  pBillboard;
  };

  SHARED_TYPES( Box );
  class Box : public Geometry
  {
    public:
      static SharedBox create();
      virtual ~Box();
      virtual const std::string & getType( void ) const;

    protected:
      Box();

    public:
      SFVec3f size;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  SHARED_TYPES( Collision );
  class Collision : public Group
  {
    public:
      static SharedCollision create();
      virtual ~Collision();
      virtual const std::string & getType( void ) const;

    protected:
      Collision();

    public:
      SFBool  collide;
      SFNode  proxy;
  };

  SHARED_TYPES( ColorInterpolator );
  class ColorInterpolator : public Interpolator
  {
    public:
      static SharedColorInterpolator create();
      virtual ~ColorInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      ColorInterpolator() {}

    public:
      MFColor keyValue;
  };

  SHARED_TYPES( Color );
  class Color : public Object
  {
    public:
      static SharedColor create();
      virtual ~Color() {}
      virtual const std::string & getType( void ) const;

    protected:
      Color();

    public:
      MFColor                 color;
      bool                    interpreted;
      SharedColorInterpolator set_color;
  };

  SHARED_TYPES( Cone );
  class Cone : public Geometry
  {
    public:
      static SharedCone create();
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

  SHARED_TYPES( CoordinateInterpolator );
  class CoordinateInterpolator : public Interpolator
  {
    public:
      static SharedCoordinateInterpolator create();
      virtual ~CoordinateInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      CoordinateInterpolator() {}

    public:
      MFVec3f keyValue;
  };

  SHARED_TYPES( Coordinate );
  class Coordinate : public Object
  {
    public:
      static SharedCoordinate create();
      virtual ~Coordinate();
      virtual const std::string & getType( void ) const;

    protected:
      Coordinate();

    public:
      MFVec3f                       point;
      bool                          interpreted;
      SharedCoordinateInterpolator  set_point;
  };

  SHARED_TYPES( Cylinder );
  class Cylinder : public Geometry
  {
    public:
      static SharedCylinder create();
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

  SHARED_TYPES( CylinderSensor );
  class CylinderSensor : public Sensor
  {
    public:
      static SharedCylinderSensor create();
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

  SHARED_TYPES( DirectionalLight );
  class DirectionalLight : public Light
  {
    public:
      static SharedDirectionalLight create();
      virtual ~DirectionalLight();
      virtual const std::string & getType( void ) const;

    protected:
      DirectionalLight();

    public:
      SFVec3f direction;
  };

  SHARED_TYPES( ElevationGrid );
  class ElevationGrid : public Geometry
  {
    public:
      static SharedElevationGrid create();
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

  SHARED_TYPES( Extrusion );
  class Extrusion : public Geometry
  {
    public:
      static SharedExtrusion create();
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

  SHARED_TYPES( Fog );
  class Fog : public Object
  {
    public:
      static SharedFog create();
      virtual ~Fog()  {}
      virtual const std::string & getType( void ) const;

    protected:
      Fog();

    public:
      SFColor   color;
      SFString  fogType;
      SFFloat   visibilityRange;
  };

  SHARED_TYPES( FontStyle );
  class FontStyle : public Object
  {
    public:
      static SharedFontStyle create();
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

  SHARED_TYPES( ImageTexture );
  class ImageTexture : public Texture
  {
    public:
      static SharedImageTexture create();
      virtual ~ImageTexture();
      virtual const std::string & getType( void ) const;

    protected:
      ImageTexture();

    public:
      MFString  url;

      dp::sg::core::ParameterGroupDataSharedPtr textureData;
  };

  SHARED_TYPES( IndexedFaceSet );
  class IndexedFaceSet : public Geometry
  {
    public:
      static SharedIndexedFaceSet create();
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

  SHARED_TYPES( IndexedLineSet );
  class IndexedLineSet : public Geometry
  {
    public:
      static SharedIndexedLineSet create();
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

  SHARED_TYPES( Inline );
  class Inline : public Object
  {
    public:
      static SharedInline create();
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

  SHARED_TYPES( LOD );
  class LOD : public Group
  {
    public:
      static SharedLOD create();
      virtual ~LOD();
      virtual const std::string & getType( void ) const;

    protected:
      LOD();

    public:
      SFVec3f center;
      MFFloat range;

      dp::sg::core::LODSharedPtr  pLOD;
  };

  SHARED_TYPES( Material );
  class Material : public Object
  {
    public:
      static SharedMaterial create();
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

  SHARED_TYPES( MovieTexture );
  class MovieTexture : public Texture
  {
    public:
      static SharedMovieTexture create();
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

  SHARED_TYPES( NavigationInfo );
  class NavigationInfo : public Object
  {
    public:
      static SharedNavigationInfo create();
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

  SHARED_TYPES( NormalInterpolator );
  class NormalInterpolator : public Interpolator
  {
    public:
      static SharedNormalInterpolator create();
      virtual ~NormalInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      NormalInterpolator() {}

    public:
      MFVec3f keyValue;
  };

  SHARED_TYPES( Normal );
  class Normal : public Object
  {
    public:
      static SharedNormal create();
      virtual ~Normal();
      virtual const std::string & getType( void ) const;

    protected:
      Normal();

    public:
      MFVec3f                   vector;
      bool                      interpreted;
      SharedNormalInterpolator  set_vector;
  };

  SHARED_TYPES( OrientationInterpolator );
  class OrientationInterpolator : public Interpolator
  {
    public:
      static SharedOrientationInterpolator create();
      virtual ~OrientationInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      OrientationInterpolator() {}

    public:
      MFRotation  keyValue;

      std::vector<dp::math::Quatf> keyValueQuatf;
  };

  SHARED_TYPES( PixelTexture );
  class PixelTexture : public Texture
  {
    public:
      static SharedPixelTexture create();
      virtual ~PixelTexture()  {}
      virtual const std::string & getType( void ) const;

    protected:
      PixelTexture();

    public:
      SFImage image;
  };

  SHARED_TYPES( PlaneSensor );
  class PlaneSensor : public Sensor
  {
    public:
      static SharedPlaneSensor create();
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

  SHARED_TYPES( PointLight );
  class PointLight : public Light
  {
    public:
      static SharedPointLight create();
      virtual ~PointLight();
      virtual const std::string & getType( void ) const;

    protected:
      PointLight();

    public:
      SFVec3f attenuation;
      SFVec3f location;
      SFFloat radius;
  };

  SHARED_TYPES( PointSet );
  class PointSet : public Geometry
  {
    public:
      static SharedPointSet create();
      virtual ~PointSet();
      virtual const std::string & getType( void ) const;

    protected:
      PointSet();

    public:
      SFNode  color;
      SFNode  coord;

      dp::sg::core::PrimitiveSharedPtr  pPoints;
  };

  SHARED_TYPES( PositionInterpolator );
  class PositionInterpolator : public Interpolator
  {
    public:
      static SharedPositionInterpolator create();
      virtual ~PositionInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      PositionInterpolator() {}

    public:
      MFVec3f keyValue;
  };

  SHARED_TYPES( ProximitySensor );
  class ProximitySensor : public Sensor
  {
    public:
      static SharedProximitySensor create();
      virtual ~ProximitySensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      ProximitySensor();

    public:
      SFVec3f center;
      SFVec3f size;
  };

  SHARED_TYPES( ScalarInterpolator );
  class ScalarInterpolator : public Interpolator
  {
    public:
      static SharedScalarInterpolator create();
      virtual ~ScalarInterpolator()  {}
      virtual const std::string & getType( void ) const;

    protected:
      ScalarInterpolator()  {}

    public:
      MFFloat keyValue;
  };

  SHARED_TYPES( Script );
  class Script : public Object
  {
    public:
      static SharedScript create();
      virtual ~Script() {}
      virtual const std::string & getType( void ) const;

    protected:
      Script();

    public:
      MFString  url;
      SFBool    directOutput;
      SFBool    mustEvaluate;
  };

  SHARED_TYPES( Shape );
  class Shape : public Object
  {
    public:
      static SharedShape create();
      virtual ~Shape();
      virtual const std::string & getType( void ) const;

    protected:
      Shape();

    public:
      SFNode  appearance;
      SFNode  geometry;

      dp::sg::core::NodeSharedPtr  pNode;
  };

  SHARED_TYPES( Sound );
  class Sound : public Object
  {
    public:
      static SharedSound create();
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

  SHARED_TYPES( Sphere );
  class Sphere : public Geometry
  {
    public:
      static SharedSphere create();
      virtual ~Sphere() {}
      virtual const std::string & getType( void ) const;

    protected:
      Sphere();

    public:
      SFFloat radius;

      dp::sg::core::PrimitiveSharedPtr pTriangles;
      dp::sg::core::PrimitiveSharedPtr pQuads;
  };

  SHARED_TYPES( SphereSensor );
  class SphereSensor : public Sensor
  {
    public:
      static SharedSphereSensor create();
      virtual ~SphereSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      SphereSensor();

    public:
      SFBool      autoOffset;
      SFRotation  offset;
  };

  SHARED_TYPES( SpotLight );
  class SpotLight : public Light
  {
    public:
      static SharedSpotLight create();
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

  SHARED_TYPES( Switch );
  class Switch : public Group
  {
    public:
      static SharedSwitch create();
      virtual ~Switch();
      virtual const std::string & getType( void ) const;

    protected:
      Switch();

    public:
      SFInt32 whichChoice;

      dp::sg::core::SwitchSharedPtr pSwitch;
  };

  SHARED_TYPES( Text );
  class Text : public Geometry
  {
    public:
      static SharedText create();
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

  SHARED_TYPES( TextureCoordinate );
  class TextureCoordinate : public Object
  {
    public:
      static SharedTextureCoordinate create();
      virtual ~TextureCoordinate()  {}
      virtual const std::string & getType( void ) const;

    protected:
      TextureCoordinate() {}

    public:
      MFVec2f point;
  };

  SHARED_TYPES( TextureTransform );
  class TextureTransform : public Object
  {
    public:
      static SharedTextureTransform create();
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

  SHARED_TYPES( TouchSensor );
  class TouchSensor : public Sensor
  {
    public:
      static SharedTouchSensor create();
      virtual ~TouchSensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      TouchSensor() {}
  };

  SHARED_TYPES( Transform );
  class Transform : public Group
  {
    public:
      static SharedTransform create();
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

      SharedPositionInterpolator    set_center;
      SharedOrientationInterpolator set_rotation;
      SharedPositionInterpolator    set_scale;
      SharedPositionInterpolator    set_translation;

      dp::sg::core::TransformSharedPtr  pTransform;
  };

  SHARED_TYPES( Viewpoint );
  class Viewpoint : public Object
  {
    public:
      static SharedViewpoint create();
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

      SharedOrientationInterpolator set_orientation;
      SharedPositionInterpolator    set_position;
  };

  SHARED_TYPES( VisibilitySensor );
  class VisibilitySensor : public Sensor
  {
    public:
      static SharedVisibilitySensor create();
      virtual ~VisibilitySensor()  {}
      virtual const std::string & getType( void ) const;

    protected:
      VisibilitySensor();

    public:
      SFVec3f center;
      SFVec3f size;
  };

  SHARED_TYPES( WorldInfo );
  class WorldInfo : public Object
  {
    public:
      static SharedWorldInfo create();
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
