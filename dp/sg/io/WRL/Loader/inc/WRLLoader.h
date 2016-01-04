// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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

#include  <dp/sg/io/PlugInterface.h>
#include  <dp/sg/algorithm/SmoothTraverser.h>
#include  <dp/util/FileFinder.h>
#include  "VRMLTypes.h"
#include  <fstream>
#include  <set>


//  Don't need to document the API specifier
#if ! defined( DOXYGEN_IGNORE ) && defined(_WIN32)
# ifdef WRLLOADER_EXPORTS
# define WRLLOADER_API __declspec(dllexport)
# else
# define WRLLOADER_API __declspec(dllimport)
# endif
#else
# define WRLLOADER_API
#endif

#if defined(LINUX)
void lib_init() __attribute__ ((constructor));   // will be called before dlopen() returns
#endif

extern "C"
{
//! Get the PlugIn interface for this scene loader.
/** Every PlugIn has to resolve this function. It is used to get a pointer to a PlugIn class, in this case a WRLLoader.
  * If the PlugIn ID \a piid equals \c PIID_WRL_SCENE_LOADER, a WRLLoader is created and returned in \a pi.
  * \returns  true, if the requested PlugIn could be created, otherwise false
  */
WRLLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi);

//! Query the supported types of PlugIn Interfaces.
WRLLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

/*! \brief A Scene loader for wrl files.
 *  \remarks This loader does not support the complete VRML 2.0 standard. Only a subset of the Nodes
 *  are mapped to SceniX objects. Unsupported VRML Nodes are just ignored.\n
 *  The following VRML Nodes are supported:
 *  - Appearance\n
 *    An Appearance Node is translated into an dp::sg::core::PipelineData, holding a "standardMaterialParameters" and
 *    maybe a "standardTextureParameters". If the opacityof the Appearence Node is less than one, or if the
 *    Texture holds a texture with alpha channel, the transparent hint is set on that EffectData.
 *  - Background\n
 *    The Background::skyColor is used as the back color of the dp::sg::core::Scene.
 *  - Billboard\n
 *    A Billboard Node is translated into a dp::sg::core::Billboard. If the length of
 *    Billboard::axisOfRotation is less than FLT_EPSILON, the dp::sg::core::Billboard is set to be screen
 *    aligned, otherwise it is not and the Billboard::axisOfRotation is used as the rotation axis.
 *    All the Billboard::children are added to the dp::sg::core::Billboard as children.
 *  - Box\n
 *    A Box Node first is translated into an IndexedFaceSet Node with eight faces. If the associated
 *    Appearance Node holds a Texture Node, texture coordinates are created. Then the IndexedFaceSet
 *    Node is interpreted as described below.
 *  - Color\n
 *    A Color Node is translated into the colors of a dp::sg::core::VertexAttributeSet.
 *  - ColorInterpolator\n
 *    A ROUTE with an eventOut Node as a ColorInterpolator, an eventOut name of "value_changed",
 *    an eventIn Node as a Color, and an eventIn name of "color" or "set_color" is translated
 *    into an animation fo colors, used for example in an dp::sg::core::AnimatedTriangles.
 *  - Coordinate\n
 *    A Coordinate Node is translated into the vertices of a dp::sg::core::VertexAttributeSet.
 *  - CoordinateInterpolator\n
 *    A ROUTE with an eventOut Node as a CoordinateInterpolator, an eventOut name of "value_changed",
 *    an eventIn Node as a Coordinate, and an eventIn name of "point" or "set_point" is translated
 *    into an animation of vertices, used for example in an dp::sg::core::AnimatedTriangles.
 *  - DirectionalLight\n
 *    A DirectionalLight Node is translated into an dp::sg::core::LightSource with a ParameterGroupData created with
 *    a ParameterGroupSpec named "standardDirectedLightParameters". The DirectionalLight::ambientIntensity times
 *    the DirectionalLight::color is set as the ambient color; the DirectionalLight::intensity times the
 *    DirectionalLight::color is set as the diffuse and the specular color; the DirectionalLight::direction is
 *    set as the direction.
 *  - ElevationGrid\n
 *    An ElevationGrid Node is translated into an IndexedFaceSet Node with one Quad per grid cell,
 *    using xDimension, xSpacing, height, zDimension, and zSpacing. If ElevationGrid::color is not
 *    NULL, IndexedFaceSet::color is set to ElevationGrid::color and IndexedFaceSet::colorPerVertex
 *    is set to ElevationGrid::colorPerVertex. If ElevationGrid::normal is not NULL,
 *    IndexedFaceSet::normal is set to ElevationGrid::normal and IndexedFaceSet::normalPerVertex is
 *    set to ElevationGrid::normalPerVertex. If ElevationGrid::texCoord is not NULL,
 *    IndexedFaceSet::texCoord is set to ElevationGrid::texCoord. If ElevationGrid::texCoord is NULL,
 *    IndexedFaceSet::texCoord is set by creating texture coordinates ranging from (0,0) at the
 *    first vertex to (1,1) at the last vertex. IndexedFaceSet::ccw, IndexedFaceSet::creaseAngle,
 *    and IndexedFaceSet::solid are set to ElevationGrid::ccw, ElevationGrid::creaseAngle, and
 *    ElevationGrid::solid, respectively.
 *  - Group\n
 *    A Group Node is translated into a dp::sg::core::Group. All the Group::children are added to the
 *    dp::sg::core::Group as children.
 *  - ImageTexture\n
 *    An ImageTexture Node is translated into a ParameterGroupData. If a file could be found at
 *    Node::url, and a dp::sg::core::TextureHost could be created out of that, a dp::sg::core::ParameterGroupData
 *    is created to hold that dp::sg::core::TextureHost. ImageTexture::repeatS and ImageTexture::repeatT
 *    are translated to the dp::sg::core::TextureWrapMode TextureWrapMode::REPEAT or TextureWrapMode::CLAMP for the
 *    dp::sg::core::TextureWrapCoordAxis TexWrapCoordAxis::S and TexWrapCoordAxis::T, respectively.
 *  - IndexedFaceSet\n
 *    An IndexedFaceSet Node is translated int a Primitive of type PrimitiveType::TRIANGLES, and/or one of type
 *    PrimitiveType::QUADS, and/or one of type PrimitiveType::POLYGON, for faces with 3, 4, or more vertices, respectively.
 *    If IndexedFaceSet::normal is NULL, a Normal Node holding a set of smoothed normals is created,
 *    using IndexedFaceSet::creaseAngle. If there is a ROUTE with a "set_point" eventIn on
 *    IndexedFaceSet::coord or if there is a ROUTE with a "set_vector" eventIn on
 *    IndexedFaceSet::normal, instead of a dp::sg::core::Triangles and/or a dp::sg::core::Quads a
 *    dp::sg::core::AnimatedTriangles and/or a dp::sg::core::AnimatedQuads are created, each one with a
 *    dp::sg::core::LinearInterpolatedAnimationDescription of a dp::sg::core::VNVector. If IndexedFaceSet::color is not NULL,
 *    one color per vertex is created, using IndexedFaceSet::colorIndex and
 *    IndexedFaceSet::colorPerVertex if needed, and set as the color. The IndexedFaceSet::coord is
 *    set as the vertices, using IndexedFaceSet::coordIndex if needed. If IndexedFaceSet::normal is
 *    not NULL, one normal per vertex is created, using IndexedFaceSet::normalIndex and
 *    IndexedFaceSet::normalPerVertex if needed, and set as the normals. If IndexedFaceSet::texCoord
 *    is not NULL, one two-dimensional texture coordinate per vertex is created, using
 *    IndexedFaceSet::texCoordIndex, and as the texture coordinates for texture unit 0. If
 *    IndexedFaceSet::ccw is true, the triangles and quads are inverted. IndexedFaceSet::convex and
 *    IndexedFaceSet::solid are ignored.
 *  - IndexedLineSet\n
 *    An IndexedLineSet Node is translated into a dp::sg::core::Primitive of type PRIMITIVE_LINE_STRIPS. For each index in
 *    IndexedLineSet::coordIndex, a vertex is created out of IndexedLineSet::coord. If
 *    IndexedLineSet::color is not NULL, one color per vertex is created, using
 *    IndexedLineSet::colorIndex and IndexedLineSet::colorPerVertex if needed.
 *  - Inline\n
 *    If the file specified by Inline::url can be loaded, it is loaded into it's own dp::sg::core::Scene,
 *    and the root node of that dp::sg::core::Scene is used.
 *  - LOD\n
 *    A LOD Node is translated into a dp::sg::core::LOD. All the LOD::level are added to the dp::sg::core::LOD as
 *    children. The LOD::center is set as the center in the dp::sg::core::LOD. If LOD::range is not empty,
 *    those values are used as the ranges in dp::sg::core::LOD. Otherwise some default ranges are created.
 *  - Material\n
 *    A Material Node is translated into an dp::sg::core::PipelineData holding an dp::sg::core::ParameterGroupData named
 *    "standardMaterialParameters". The Material::ambientIntensity times the Material::diffuseColor is set as the
 *    ambient color of the EffectData. The Material::diffuseColor, Material::emissiveColor, and
 *    Material::specularColor are set as the diffuse color, emissive color, and specular color, respectively, of
 *    the dp::sg::core::PipelineData. The Material::shininess is multiplied by 128 and set as the specular exponent of the
 *    dp::sg::core::PipelineData. One minus Material::transparency is set as the opacity of the dp::sg::core::PipelineData.
 *  - Normal\n
 *    A Normal Node is translated into the normals of a dp::sg::core::VertexAttributeSet.
 *  - NormalInterpolator\n
 *    A ROUTE with an eventOut Node as a NormalInterpolator, an eventOut name of "value_changed", an
 *    eventIn Node as a Normal, and an eventIn name of "vector" or "set_vector" is translated into
 *    an animation of normals, used for example in an dp::sg::core::AnimatedTriangles.
 *  - OrientationInterpolator\n
 *    A ROUTE with an eventOut Node as an OrientationInterpolator, an eventOut name of
 *    "value_changed", an eventIn Node as a Transform or a Viewpoint, and an eventIn name of
 *    "orientation" or "set_orientation" is translated into an animation of orientations, used for
 *    example in an dp::sg::core::AnimatedTransform.
 *  - PointLight\n
 *    A PointLight Node is translated into an dp::sg::core::LightSource with a ParameterGroupData created with
 *    a ParameterGroupSpec named "standardPointLightParameters". The PointLight::ambientIntensity
 *    times the PointLight::color is set as the ambient color of the dp::sg::core::LightSource. The
 *    PointLight::intensity times the PointLight::color is set as the diffuse color of the LightSource.
*     The PointLight::diffuseColor, PointLight::attenuation, and PointLight::location are set as the specular
*     color, attenuation, and position, respectively, of the LightSource.
 *  - PointSet\n
 *    A PointSet Node is translated into a dp::sg::core::Primitive of type PrimitiveType::POINTS. The PointSet::coord is set as
*     the vertices of the dp::sg::core::Primitive. If PointSet::colors is not NULL, they are set as the colors of the
 *    dp::sg::core::Primitive.
 *  - PositionInterpolator\n
 *    A ROUTE with an eventOut Node as an PositionInterpolator, an eventOut name of "value_changed",
 *    an eventIn Node as a Transform or a Viewpoint, and an eventIn name of "center", "set_center",
 *    "scale", "set_scale", "translation", or "set_translation", in case of a Transform as the
 *    eventIn Node, or "position" or "set_position", in case of a Viewpoint as the eventIn Node, is
 *    translated into an animation of dp::math::Vec3f, used for example in an AnimatedTransform.
 *  - Shape\n
 *    A Shape Node is translated into a number of dp::sg::core::GeoNodes. If the Shape::appearance is not NULL, it is
 *    translated into a dp::sg::core::StateSet, and if Shape::geometry is not NULL, it is translated into a
 *    number of dp::sg::core::Primitive, each one added to a the dp::sg::core::GeoNode, using that same dp::sg::core::StateSet.
 *  - SpotLight\n
 *    A SpotLight Node is translated into an dp::sg::core::LightSource with a ParameterGroupData created with
 *    a ParameterGroupSpec named "standardSpotLightParameters". The SpotLight::ambientIntensity times
 *    the SpotLight::color is set as the ambient color of the dp::sg::core::LightSource. The
 *    SpotLight::intensity times the SpotLight::color is set as the diffuse color of the
 *    dp::sg::core::LightSource. The SpotLight::diffuseColor, SpotLight::attenuation, SpotLight::direction,
 *    SpotLight::location, and half of the SpotLight::cutOffAngle are set as the specular color,
 *    attenuation, direction, and cut off angle, respectively, of the dp::sg::core::LightSource. If the
 *    SpotLight::beamWidth is less than the SpotLight::cutOffAngle, the SpotLight::cutOffAngle
 *    divided by the SpotLight::beamWidth is set as the falloff exponent of the dp::sg::core::LightSource.
 *  - Switch\n
 *    A Switch Node is translated into a dp::sg::core::Switch. All the Switch::choice are set as the
 *    children of the dp::sg::core::Switch. If Switch::whichChoice is between 0 and the number of children,
 *    that child is set as active, otherwise no child is set as active.
 *  - TextureCoordinate\n
 *    A TextureCoordinate Node is translated into two-dimensional texture coordinates for the first
 *    texture unit in a dp::sg::core::VertexAttributeSet.
 *  - TextureTransform\n
 *    A TextureTransform Node is translated into the orientation (rotation around the z-axis),
 *    scaling (in x and y), and translation (in x and y) of a dp::sg::core::ParameterGroupData. The
 *    TextureTransform::center is ignored.
 *  - TimeSensor\n
 *    A ROUTE with an eventOut Node as an TimeSensor, an eventOut name of "fraction_changed", an
 *    eventIn Node as an Interpolator, and an eventIn name of "fraction" or "set_fraction" is
 *    translated into a TimeSensor for that Interpolator. Currently, only the cycleInterval is
 *    used to determine the duration of the animation.
 *  - Transform\n
 *    A Transform Node is translated into a dp::sg::core::Transform. If there is a ROUTE with a Transform
 *    as the eventIn Node and "center", "set_center", "rotation", "set_rotation", "scale",
 *    "set_scale", "translation" or "set_translation" as the eventIn name, an
 *    dp::sg::core::AnimatedTransform is created instead. Transform::center, Transform::rotation,
 *    Transform::scale, Transform::scaleOrientation, and Transform::translation are used to form the
 *    dp::math::Trafo of the dp::sg::core::Transform. All the Transform::children are added to the
 *    dp::sg::core::Transform as children.
 *  - Viewpoint\n
 *    A Viewpoint Node is translated into a dp::sg::core::PerspectiveCamera. Viewpoint::fieldOfView,
 *    Viewpoint::orientation, and Viewpoint::position are set as field of view, orientation and
 *    position, respectively of the dp::sg::core::PerspectiveCamera. If there is a ROUTE with an eventOut
 *    Node as an OrientationInterpolator, an eventOut name of "value_changed", an eventIn Node as a
 *    Viewpoint, and an eventIn name of "orientation" or "set_orientation", or if there is a ROUTE
 *    with an eventOut Node as a PositionInterpolator, an eventOut name of "value_changed", an
 *    eventIn Node as a Viewpoint, and an eventIn name of "position" or "set_position", a camera
 *    animation of type dp::sg::core::LinearInterpolatedAnimationDescription of dp::math::Trafo is created in addition.
 *    The Viewpoint::jump and Viewpoint::description are ignored.
 *
 *  That is, the followng VRML Nodes are not supported: Anchor, AudioClip, Collision, Cone,
 *  Cylinder, CylinderSensor, Extrusion, Fog, FontStyle, MovieTexture, NavigationInfo,
 *  PixelTexture, PlaneSensor, ProximitySensor, ScalarInterpolator, Script, Sound, Sphere,
 *  SphereSensor, Text, TouchSensor, VisibilitySensor, WorldInfo.
 *  Additionally, as described in detail above, only a very limited number of events, and no sensors
 *  are supported. No PROTO or EXTERNPROTO is supported.
 *  \sa SceneLoader */
 DEFINE_PTR_TYPES( WRLLoader );

class WRLLoader : public dp::sg::io::SceneLoader
{
  public :
    static WRLLoaderSharedPtr create();
    ~WRLLoader();

  public :
    /*! \brief Get the number of steps per time unit.
     *  \return The number of steps per time unit used in animations.
     *  \remarks The animations in a WRL file are specified by key values, usually in the range
     *  [0...1]. This is one time unit and this WRLLoader uses \c stepsPerUnit intervals for
     *  remapping them into the frame based animation approach of SceniX.
     *  \sa setStepsPerUnit */
    unsigned int getStepsPerUnit() const;

    /*! \brief Get the number of steps per time unit.
     *  \param stepsPerUnit The number of steps per time unit used in animations.
     *  \remarks The animations in a WRL file are specified by key values, usually in the range
     *  [0...1]. This is one time unit and this WRLLoader uses \c stepsPerUnit intervals for
     *  remapping them into the frame based animation approach of SceniX.
     *  \sa getStepsPerUnit */
    void setStepsPerUnit( unsigned int stepsPerUnit );

    //! Realization of the pure virtual interface function of a SceneLoader.
    /** Loads a VRML file given by \a filename. It looks for this file and
      * possibly referenced other files like textures or effects at the given
      * path first, then at the current location and finally it searches
      * through the \a searchPaths.
      * \returns  A pointer to the loaded scene. */
    dp::sg::core::SceneSharedPtr load( std::string const& filename                  //!<  file to load
                                     , dp::util::FileFinder const& fileFinder       //!<  file finder to use for searching
                                     , dp::sg::ui::ViewStateSharedPtr & viewState   /*!< If the function succeeded, this points to the optional
                                                                                         ViewState stored with the scene. */
                                     );

  protected:
    WRLLoader();

  private :
    void                                        createBox( vrml::IndexedFaceSetSharedPtr & pIndexedFaceSet, const vrml::SFVec3f& size, bool textured );
    void                                        createCone( vrml::IndexedFaceSetSharedPtr & pIndexedFaceSet, float radius, float height, bool bottom, bool side, bool textured );
    void                                        createCylinder( vrml::IndexedFaceSetSharedPtr & indexedFaceSet, float radius, float height, bool bottom, bool side, bool top, bool textured );
    void                                        createSphere( vrml::IndexedFaceSetSharedPtr & indexedFaceSet, float radius, bool textured );
    void                                        determineTexGen( vrml::IndexedFaceSetSharedPtr const& pIndexedFaceSet, const dp::sg::core::ParameterGroupDataSharedPtr & parameterGroupData );
    template<typename VType> void               eraseIndex( unsigned int f, unsigned int i, unsigned int count, bool perVertex, vrml::MFInt32 &index, VType &vec );
    vrml::SFNode                                findNode( const vrml::SFNode currentNode, std::string name );
    std::vector<unsigned int>                   getCombinedKeys( vrml::PositionInterpolatorSharedPtr const& center, vrml::OrientationInterpolatorSharedPtr const& rotation
                                                               , vrml::PositionInterpolatorSharedPtr const& scale, vrml::PositionInterpolatorSharedPtr const& translation );
    bool                                        getNextLine( void );
    std::string                               & getNextToken( void );
    vrml::SFNode                                getNode( const std::string &nodeName, std::string &token );
    void                                        ignoreBlock( const std::string &open, const std::string &close, std::string &token );
    dp::sg::core::SceneSharedPtr                import( const std::string &filename );
    dp::sg::core::PipelineDataSharedPtr         interpretAppearance( vrml::AppearanceSharedPtr const& pAppearance );
    void                                        interpretChildren( vrml::MFNode &children, dp::sg::core::GroupSharedPtr const& pGroup );
    void                                        interpretBackground( vrml::BackgroundSharedPtr const& pBackground );
    dp::sg::core::BillboardSharedPtr            interpretBillboard( vrml::BillboardSharedPtr const& pVRMLBillboard );
    void                                        interpretBox( vrml::BoxSharedPtr const& pBox, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    void                                        interpretColor( vrml::ColorSharedPtr const& pColor );
    void                                        interpretColorInterpolator( vrml::ColorInterpolatorSharedPtr const& pColorInterpolator, unsigned int colorCount );
    void                                        interpretCone( vrml::ConeSharedPtr const& pCone, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    void                                        interpretCoordinate( vrml::CoordinateSharedPtr const& pCoordinate );
    void                                        interpretCoordinateInterpolator( vrml::CoordinateInterpolatorSharedPtr const& pCoordinateInterpolator, unsigned int pointCount );
    void                                        interpretCylinder( vrml::CylinderSharedPtr const& pCylinder, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    dp::sg::core::LightSourceSharedPtr          interpretDirectionalLight( vrml::DirectionalLightSharedPtr const& pDirectionalLight );
    void                                        interpretElevationGrid( vrml::ElevationGridSharedPtr const& pElevationGrid, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    void                                        interpretGeometry( vrml::GeometrySharedPtr const& pGeometry, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    dp::sg::core::GroupSharedPtr                interpretGroup( vrml::GroupSharedPtr const& pGroup );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretImageTexture( vrml::ImageTextureSharedPtr const& pImageTexture );
    void                                        interpretIndexedFaceSet( vrml::IndexedFaceSetSharedPtr const& pIndexedFaceSet, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    void                                        interpretIndexedLineSet( vrml::IndexedLineSetSharedPtr const& pIndexedLineSet, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    dp::sg::core::NodeSharedPtr                 interpretInline( vrml::InlineSharedPtr const& pInline );
    dp::sg::core::LODSharedPtr                  interpretLOD( vrml::LODSharedPtr const& pVRMLLOD );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretMaterial( vrml::MaterialSharedPtr const& pVRMLMaterial );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretMovieTexture( vrml::MovieTextureSharedPtr const& pMovieTexture );
    void                                        interpretNormal( vrml::NormalSharedPtr const& pNormal );
    void                                        interpretNormalInterpolator( vrml::NormalInterpolatorSharedPtr const& pNormalInterpolator, unsigned int vectorCount );
    void                                        interpretOrientationInterpolator( vrml::OrientationInterpolatorSharedPtr const& pOI );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretPixelTexture( vrml::PixelTextureSharedPtr const& pPixelTexture );
    dp::sg::core::LightSourceSharedPtr          interpretPointLight( vrml::PointLightSharedPtr const& pPointLight );
    void                                        interpretPointSet( vrml::PointSetSharedPtr const& pPointSet, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    void                                        interpretPositionInterpolator( vrml::PositionInterpolatorSharedPtr const& pPositionInterpolator );
    dp::math::Vec3f                             interpretSFColor( const vrml::SFColor &c );
    dp::sg::core::ObjectSharedPtr               interpretSFNode( const vrml::SFNode n );
    dp::math::Quatf                             interpretSFRotation( const vrml::SFRotation &r );
    dp::sg::core::NodeSharedPtr                 interpretShape( vrml::ShapeSharedPtr const& pShape );
    void                                        interpretSphere( vrml::SphereSharedPtr const& pSphere, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    dp::sg::core::LightSourceSharedPtr          interpretSpotLight( vrml::SpotLightSharedPtr const& pVRMLSpotLight );
    dp::sg::core::SwitchSharedPtr               interpretSwitch( vrml::SwitchSharedPtr const& pVRMLSwitch );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretTexture( vrml::TextureSharedPtr const& pTexture );
    void                                        interpretTextureTransform( vrml::TextureTransformSharedPtr const& pTextureTransform, const dp::sg::core::ParameterGroupDataSharedPtr & textureData );
    dp::sg::core::TransformSharedPtr            interpretTransform( vrml::TransformSharedPtr const& pVRMLTransform );
    bool                                        interpretURL( const vrml::MFString &url, std::string &fileName );
    dp::sg::core::VertexAttributeSetSharedPtr   interpretVertexAttributeSet( vrml::IndexedFaceSetSharedPtr const& pIndexedFaceSet, unsigned int numberOfVertices, const std::vector<unsigned int> & startIndices
                                                                           , const std::vector<unsigned int> & faceIndices );
    dp::sg::core::ObjectSharedPtr               interpretViewpoint( vrml::ViewpointSharedPtr const& pViewpoint );
    void                                        interpretVRMLTree( void );
    bool                                        isValidScaling( vrml::PositionInterpolatorSharedPtr const& pPositionInterpolator ) const;
    bool                                        isValidScaling( const vrml::SFVec3f &sfVec3f ) const;
    bool                                        onIncompatibleValues( int value0, int value1, const std::string &node, const std::string &field0, const std::string &field1 ) const;
    template<typename T> bool                   onInvalidValue( T value, const std::string &node, const std::string &field ) const;
    bool                                        onEmptyToken( const std::string &tokenType, const std::string &token ) const;
    bool                                        onFileNotFound( const vrml::SFString &url ) const;
    bool                                        onFilesNotFound( bool found, const vrml::MFString &url ) const;
    void                                        onUnexpectedEndOfFile( bool error ) const;
    void                                        onUnexpectedToken( const std::string &expected, const std::string &token ) const;
    void                                        onUnknownToken( const std::string &tokenType, const std::string &token ) const;
    bool                                        onUndefinedToken( const std::string &tokenType, const std::string &token ) const;
    bool                                        onUnsupportedToken( const std::string &tokenType, const std::string &token ) const;
    vrml::AnchorSharedPtr                       readAnchor( const std::string &nodeName );
    vrml::AppearanceSharedPtr                   readAppearance( const std::string &nodeName );
    vrml::AudioClipSharedPtr                    readAudioClip( const std::string &nodeName );
    vrml::BackgroundSharedPtr                   readBackground( const std::string &nodeName );
    vrml::BillboardSharedPtr                    readBillboard( const std::string &nodeName );
    vrml::BoxSharedPtr                          readBox( const std::string &nodeName );
    vrml::CollisionSharedPtr                    readCollision( const std::string &nodeName );
    vrml::ColorSharedPtr                        readColor( const std::string &nodeName );
    vrml::ColorInterpolatorSharedPtr            readColorInterpolator( const std::string &nodeName );
    vrml::ConeSharedPtr                         readCone( const std::string &nodeName );
    vrml::CoordinateSharedPtr                   readCoordinate( const std::string &nodeName );
    vrml::CoordinateInterpolatorSharedPtr       readCoordinateInterpolator( const std::string &nodeName );
    vrml::CylinderSharedPtr                     readCylinder( const std::string &nodeName );
    vrml::CylinderSensorSharedPtr               readCylinderSensor( const std::string &nodeName );
    vrml::DirectionalLightSharedPtr             readDirectionalLight( const std::string &nodeName );
    vrml::ElevationGridSharedPtr                readElevationGrid( const std::string &nodeName );
    void                                        readEXTERNPROTO( void );
    vrml::ExtrusionSharedPtr                    readExtrusion( const std::string &nodeName );
    vrml::FogSharedPtr                          readFog( const std::string &nodeName );
    vrml::FontStyleSharedPtr                    readFontStyle( const std::string &nodeName );
    vrml::GroupSharedPtr                        readGroup( const std::string &nodeName );
    vrml::ImageTextureSharedPtr                 readImageTexture( const std::string &nodeName );
    void                                        readIndex( std::vector<vrml::SFInt32> &mf );
    vrml::IndexedFaceSetSharedPtr               readIndexedFaceSet( const std::string &nodeName );
    vrml::IndexedLineSetSharedPtr               readIndexedLineSet( const std::string &nodeName );
    vrml::InlineSharedPtr                       readInline( const std::string &nodeName );
    vrml::LODSharedPtr                          readLOD( const std::string &nodeName );
    vrml::MaterialSharedPtr                     readMaterial( const std::string &nodeName );
    void                                        readMFNode( vrml::GroupSharedPtr const& fatherNode );
    template<typename SFType> void              readMFType( std::vector<SFType> &mf, void (WRLLoader::*readSFType)( SFType &sf, std::string &token ) );
    vrml::MovieTextureSharedPtr                 readMovieTexture( const std::string &nodeName );
    vrml::NavigationInfoSharedPtr               readNavigationInfo( const std::string &nodeName );
    vrml::NormalSharedPtr                       readNormal( const std::string &nodeName );
    vrml::NormalInterpolatorSharedPtr           readNormalInterpolator( const std::string &nodeName );
    vrml::OrientationInterpolatorSharedPtr      readOrientationInterpolator( const std::string &nodeName );
    vrml::PixelTextureSharedPtr                 readPixelTexture( const std::string &nodeName );
    vrml::PlaneSensorSharedPtr                  readPlaneSensor( const std::string &nodeName );
    vrml::PointLightSharedPtr                   readPointLight( const std::string &nodeName );
    vrml::PointSetSharedPtr                     readPointSet( const std::string &nodeName );
    vrml::PositionInterpolatorSharedPtr         readPositionInterpolator( const std::string &nodeName );
    void                                        readPROTO( void );
    vrml::ProximitySensorSharedPtr              readProximitySensor( const std::string &nodeName );
    void                                        readROUTE( const vrml::SFNode currentNode );
    vrml::ScalarInterpolatorSharedPtr           readScalarInterpolator( const std::string &nodeName );
    vrml::ScriptSharedPtr                       readScript( const std::string &nodeName );
    void                                        readSFBool( vrml::SFBool &b );
    void                                        readSFColor( vrml::SFColor &c, std::string &token );
    void                                        readSFFloat( vrml::SFFloat &f, std::string &token );
    void                                        readSFImage( vrml::SFImage &i );
    void                                        readSFInt8( vrml::SFInt8 &i );
    void                                        readSFInt32( vrml::SFInt32 &i, std::string &token );
    void                                        readSFNode( const vrml::SFNode fatherNode, vrml::SFNode &n, std::string &token );
    void                                        readSFRotation( vrml::SFRotation &r, std::string &token );
    void                                        readSFString( vrml::SFString &s, std::string &token );
    void                                        readSFTime( vrml::SFTime &t );
    void                                        readSFVec2f( vrml::SFVec2f &v, std::string &token );
    void                                        readSFVec3f( vrml::SFVec3f &v, std::string &token );
    vrml::ShapeSharedPtr                        readShape( const std::string &nodeName );
    vrml::SoundSharedPtr                        readSound( const std::string &nodeName );
    vrml::SphereSharedPtr                       readSphere( const std::string &nodeName );
    vrml::SphereSensorSharedPtr                 readSphereSensor( const std::string &nodeName );
    vrml::SpotLightSharedPtr                    readSpotLight( const std::string &nodeName );
    void                                        readStatements( void );
    vrml::SwitchSharedPtr                       readSwitch( const std::string &nodeName );
    vrml::TextSharedPtr                         readText( const std::string &nodeName );
    vrml::TextureCoordinateSharedPtr            readTextureCoordinate( const std::string &nodeName );
    vrml::TextureTransformSharedPtr             readTextureTransform( const std::string &nodeName );
    vrml::TimeSensorSharedPtr                   readTimeSensor( const std::string &nodeName );
    vrml::TouchSensorSharedPtr                  readTouchSensor( const std::string &nodeName );
    vrml::TransformSharedPtr                    readTransform( const std::string &nodeName );
    vrml::ViewpointSharedPtr                    readViewpoint( const std::string &nodeName );
    vrml::VisibilitySensorSharedPtr             readVisibilitySensor( const std::string &nodeName );
    vrml::WorldInfoSharedPtr                    readWorldInfo( const std::string &nodeName );
    template<typename T> void                   resampleKeyValues( vrml::MFFloat & keys, std::vector<T> & values, unsigned int valuesPerKey, std::vector<unsigned int> & steps, vrml::SFTime cycleInterval );
    void                                        setNextToken( void );
    bool                                        testWRLVersion( const std::string &filename );

  private:
    struct Subdivisions
    {
      int sphereMin;
      int sphereDefault;
      int sphereMax;
      int boxMin;
      int boxDefault;
      int boxMax;
    };

  private :
    std::string                                               m_currentString;
    std::string                                               m_currentToken;
    std::map<std::string,vrml::SFNode>                        m_defNodes;
    bool                                                      m_eof;
    FILE                                                    * m_fh;
    dp::util::FileFinder                                      m_fileFinder;
    std::map<vrml::MFString,vrml::InlineSharedPtr>            m_inlines;
    char                                                    * m_line;
    unsigned int                                              m_lineLength;
    unsigned int                                              m_lineNumber;
    std::string::size_type                                    m_nextTokenEnd;
    std::string::size_type                                    m_nextTokenStart;
    std::vector<vrml::SFNode>                                 m_openNodes;
    std::set<std::string>                                     m_PROTONames;
    dp::sg::core::GroupSharedPtr                              m_rootNode;
    dp::sg::core::SceneSharedPtr                              m_scene;
    std::unique_ptr<dp::sg::algorithm::SmoothTraverser>       m_smoothTraverser;
    unsigned int                                              m_stepsPerUnit;
    vrml::GroupSharedPtr                                      m_topLevelGroup;
    bool                                                      m_strict;
    std::map<std::string,dp::sg::core::TextureHostWeakPtr>    m_textureFiles;
    std::string                                               m_ungetToken;
    // Circular and rectangular subdivision limits. The creation functions attempt to subdivide with square sized quads.
    // Minimum, standard at radius 1.0, and maximum subdivisions for a full circle of a sphere, cylinder, cone tessellation depending on their radii and heights.
    // Minimum, standard at size 1.0, and maximum subdivisions for a Box depending on its size.
    // These six values can be defined by the user via the environment variable DP_WRL_SUBDIVISIONS.
    // Defaults are DP_WRL_SUBDIVISIONS = 12 36 90  2 4 8
    Subdivisions                                              m_subdivisions;
};

inline unsigned int WRLLoader::getStepsPerUnit() const
{
  return( m_stepsPerUnit );
}

inline void WRLLoader::setStepsPerUnit( unsigned int stepsPerUnit )
{
  m_stepsPerUnit = stepsPerUnit;
}

inline  dp::math::Vec3f WRLLoader::interpretSFColor( const vrml::SFColor &c )
{
  return( * (dp::math::Vec3f *) &c );
}
