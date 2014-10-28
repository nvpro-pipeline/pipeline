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

#include  <fstream>
#include  <set>
#include  <dp/sg/core/nvsg.h>
#include  <dp/sg/io/PlugInterface.h>
#include  <dp/sg/algorithm/SmoothTraverser.h>
#include  "VRMLTypes.h"


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
WRLLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPlugIn & pi);

//! Query the supported types of PlugIn Interfaces.
WRLLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

/*! \brief A Scene loader for wrl files.
 *  \remarks This loader does not support the complete VRML 2.0 standard. Only a subset of the Nodes
 *  are mapped to SceniX objects. Unsupported VRML Nodes are just ignored.\n
 *  The following VRML Nodes are supported:
 *  - Appearance\n
 *    An Appearance Node is translated into an dp::sg::core::EffectData, holding a "standardMaterialParameters" and
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
 *    are translated to the dp::sg::core::TextureWrapMode TWM_REPEAT or TWM_CLAMP for the
 *    dp::sg::core::TextureWrapCoordAxis TWCA_S and TWCA_T, respectively.
 *  - IndexedFaceSet\n
 *    An IndexedFaceSet Node is translated int a Primitive of type PRIMITIVE_TRIANGLES, and/or one of type
 *    PRIMITIVE_QUADS, and/or one of type PRIMITIVE_POLYGON, for faces with 3, 4, or more vertices, respectively.
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
 *    A Material Node is translated into an dp::sg::core::EffectData holding an dp::sg::core::ParameterGroupData named
 *    "standardMaterialParameters". The Material::ambientIntensity times the Material::diffuseColor is set as the
 *    ambient color of the EffectData. The Material::diffuseColor, Material::emissiveColor, and
 *    Material::specularColor are set as the diffuse color, emissive color, and specular color, respectively, of
 *    the dp::sg::core::EffectData. The Material::shininess is multiplied by 128 and set as the specular exponent of the
 *    dp::sg::core::EffectData. One minus Material::transparency is set as the opacity of the dp::sg::core::EffectData.
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
 *    A PointSet Node is translated into a dp::sg::core::Primitive of type PRIMITIVE_POINTS. The PointSet::coord is set as
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
 SMART_TYPES( WRLLoader );

class WRLLoader : public dp::sg::io::SceneLoader
{
  public :
    static SmartWRLLoader create();
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
                                     , std::vector<std::string> const& searchPaths  //!<  paths to search through
                                     , dp::sg::ui::ViewStateSharedPtr & viewState   /*!< If the function succeeded, this points to the optional
                                                                                         ViewState stored with the scene. */
                                     );

  protected:
    WRLLoader();

  private :
    void                                        createBox( vrml::SharedIndexedFaceSet & pIndexedFaceSet, const vrml::SFVec3f& size, bool textured );
    void                                        createCone( vrml::SharedIndexedFaceSet & pIndexedFaceSet, float radius, float height, bool bottom, bool side, bool textured );
    void                                        createCylinder( vrml::SharedIndexedFaceSet & indexedFaceSet, float radius, float height, bool bottom, bool side, bool top, bool textured );
    void                                        createSphere( vrml::SharedIndexedFaceSet & indexedFaceSet, float radius, bool textured );
    void                                        determineTexGen( vrml::SharedIndexedFaceSet const& pIndexedFaceSet, const dp::sg::core::ParameterGroupDataSharedPtr & parameterGroupData );
    template<typename VType> void               eraseIndex( unsigned int f, unsigned int i, unsigned int count, bool perVertex, vrml::MFInt32 &index, VType &vec );
    vrml::SFNode                                findNode( const vrml::SFNode currentNode, std::string name );
    std::vector<unsigned int>                   getCombinedKeys( vrml::SharedPositionInterpolator const& center, vrml::SharedOrientationInterpolator const& rotation
                                                               , vrml::SharedPositionInterpolator const& scale, vrml::SharedPositionInterpolator const& translation );
    bool                                        getNextLine( void );
    std::string                               & getNextToken( void );
    vrml::SFNode                                getNode( const std::string &nodeName, std::string &token );
    void                                        ignoreBlock( const std::string &open, const std::string &close, std::string &token );
    dp::sg::core::SceneSharedPtr                import( const std::string &filename );
    dp::sg::core::EffectDataSharedPtr           interpretAppearance( vrml::SharedAppearance const& pAppearance );
    void                                        interpretChildren( vrml::MFNode &children, dp::sg::core::GroupSharedPtr const& pNVSGGroup );
    void                                        interpretBackground( vrml::SharedBackground const& pBackground );
    dp::sg::core::BillboardSharedPtr            interpretBillboard( vrml::SharedBillboard const& pVRMLBillboard );
    void                                        interpretBox( vrml::SharedBox const& pBox, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    void                                        interpretColor( vrml::SharedColor const& pColor );
    void                                        interpretColorInterpolator( vrml::SharedColorInterpolator const& pColorInterpolator, unsigned int colorCount );
    void                                        interpretCone( vrml::SharedCone const& pCone, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    void                                        interpretCoordinate( vrml::SharedCoordinate const& pCoordinate );
    void                                        interpretCoordinateInterpolator( vrml::SharedCoordinateInterpolator const& pCoordinateInterpolator, unsigned int pointCount );
    void                                        interpretCylinder( vrml::SharedCylinder const& pCylinder, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    dp::sg::core::LightSourceSharedPtr          interpretDirectionalLight( vrml::SharedDirectionalLight const& pDirectionalLight );
    void                                        interpretElevationGrid( vrml::SharedElevationGrid const& pElevationGrid, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    void                                        interpretGeometry( vrml::SharedGeometry const& pGeometry, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    dp::sg::core::GroupSharedPtr                interpretGroup( vrml::SharedGroup const& pGroup );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretImageTexture( vrml::SharedImageTexture const& pImageTexture );
    void                                        interpretIndexedFaceSet( vrml::SharedIndexedFaceSet const& pIndexedFaceSet, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    void                                        interpretIndexedLineSet( vrml::SharedIndexedLineSet const& pIndexedLineSet, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    dp::sg::core::NodeSharedPtr                 interpretInline( vrml::SharedInline const& pInline );
    dp::sg::core::LODSharedPtr                  interpretLOD( vrml::SharedLOD const& pVRMLLOD );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretMaterial( vrml::SharedMaterial const& pVRMLMaterial );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretMovieTexture( vrml::SharedMovieTexture const& pMovieTexture );
    void                                        interpretNormal( vrml::SharedNormal const& pNormal );
    void                                        interpretNormalInterpolator( vrml::SharedNormalInterpolator const& pNormalInterpolator, unsigned int vectorCount );
    void                                        interpretOrientationInterpolator( vrml::SharedOrientationInterpolator const& pOI );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretPixelTexture( vrml::SharedPixelTexture const& pPixelTexture );
    dp::sg::core::LightSourceSharedPtr          interpretPointLight( vrml::SharedPointLight const& pPointLight );
    void                                        interpretPointSet( vrml::SharedPointSet const& pPointSet, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives );
    void                                        interpretPositionInterpolator( vrml::SharedPositionInterpolator const& pPositionInterpolator );
    dp::math::Vec3f                             interpretSFColor( const vrml::SFColor &c );
    dp::sg::core::ObjectSharedPtr               interpretSFNode( const vrml::SFNode n );
    dp::math::Quatf                             interpretSFRotation( const vrml::SFRotation &r );
    dp::sg::core::NodeSharedPtr                 interpretShape( vrml::SharedShape const& pShape );
    void                                        interpretSphere( vrml::SharedSphere const& pSphere, std::vector<dp::sg::core::PrimitiveSharedPtr> &primitives, bool textured = false );
    dp::sg::core::LightSourceSharedPtr          interpretSpotLight( vrml::SharedSpotLight const& pVRMLSpotLight );
    dp::sg::core::SwitchSharedPtr               interpretSwitch( vrml::SharedSwitch const& pVRMLSwitch );
    dp::sg::core::ParameterGroupDataSharedPtr   interpretTexture( vrml::SharedTexture const& pTexture );
    void                                        interpretTextureTransform( vrml::SharedTextureTransform const& pTextureTransform, const dp::sg::core::ParameterGroupDataSharedPtr & textureData );
    dp::sg::core::TransformSharedPtr            interpretTransform( vrml::SharedTransform const& pVRMLTransform );
    bool                                        interpretURL( const vrml::MFString &url, std::string &fileName );
    dp::sg::core::VertexAttributeSetSharedPtr   interpretVertexAttributeSet( vrml::SharedIndexedFaceSet const& pIndexedFaceSet, unsigned int numberOfVertices, const std::vector<unsigned int> & startIndices
                                                                           , const std::vector<unsigned int> & faceIndices );
    dp::sg::core::ObjectSharedPtr               interpretViewpoint( vrml::SharedViewpoint const& pViewpoint );
    void                                        interpretVRMLTree( void );
    bool                                        isValidScaling( vrml::SharedPositionInterpolator const& pPositionInterpolator ) const;
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
    vrml::SharedAnchor                          readAnchor( const std::string &nodeName );
    vrml::SharedAppearance                      readAppearance( const std::string &nodeName );
    vrml::SharedAudioClip                       readAudioClip( const std::string &nodeName );
    vrml::SharedBackground                      readBackground( const std::string &nodeName );
    vrml::SharedBillboard                       readBillboard( const std::string &nodeName );
    vrml::SharedBox                             readBox( const std::string &nodeName );
    vrml::SharedCollision                       readCollision( const std::string &nodeName );
    vrml::SharedColor                           readColor( const std::string &nodeName );
    vrml::SharedColorInterpolator               readColorInterpolator( const std::string &nodeName );
    vrml::SharedCone                            readCone( const std::string &nodeName );
    vrml::SharedCoordinate                      readCoordinate( const std::string &nodeName );
    vrml::SharedCoordinateInterpolator          readCoordinateInterpolator( const std::string &nodeName );
    vrml::SharedCylinder                        readCylinder( const std::string &nodeName );
    vrml::SharedCylinderSensor                  readCylinderSensor( const std::string &nodeName );
    vrml::SharedDirectionalLight                readDirectionalLight( const std::string &nodeName );
    vrml::SharedElevationGrid                   readElevationGrid( const std::string &nodeName );
    void                                        readEXTERNPROTO( void );
    vrml::SharedExtrusion                       readExtrusion( const std::string &nodeName );
    vrml::SharedFog                             readFog( const std::string &nodeName );
    vrml::SharedFontStyle                       readFontStyle( const std::string &nodeName );
    vrml::SharedGroup                           readGroup( const std::string &nodeName );
    vrml::SharedImageTexture                    readImageTexture( const std::string &nodeName );
    void                                        readIndex( std::vector<vrml::SFInt32> &mf );
    vrml::SharedIndexedFaceSet                  readIndexedFaceSet( const std::string &nodeName );
    vrml::SharedIndexedLineSet                  readIndexedLineSet( const std::string &nodeName );
    vrml::SharedInline                          readInline( const std::string &nodeName );
    vrml::SharedLOD                             readLOD( const std::string &nodeName );
    vrml::SharedMaterial                        readMaterial( const std::string &nodeName );
    void                                        readMFNode( vrml::SharedGroup const& fatherNode );
    template<typename SFType> void              readMFType( std::vector<SFType> &mf, void (WRLLoader::*readSFType)( SFType &sf, std::string &token ) );
    vrml::SharedMovieTexture                    readMovieTexture( const std::string &nodeName );
    vrml::SharedNavigationInfo                  readNavigationInfo( const std::string &nodeName );
    vrml::SharedNormal                          readNormal( const std::string &nodeName );
    vrml::SharedNormalInterpolator              readNormalInterpolator( const std::string &nodeName );
    vrml::SharedOrientationInterpolator         readOrientationInterpolator( const std::string &nodeName );
    vrml::SharedPixelTexture                    readPixelTexture( const std::string &nodeName );
    vrml::SharedPlaneSensor                     readPlaneSensor( const std::string &nodeName );
    vrml::SharedPointLight                      readPointLight( const std::string &nodeName );
    vrml::SharedPointSet                        readPointSet( const std::string &nodeName );
    vrml::SharedPositionInterpolator            readPositionInterpolator( const std::string &nodeName );
    void                                        readPROTO( void );
    vrml::SharedProximitySensor                 readProximitySensor( const std::string &nodeName );
    void                                        readROUTE( const vrml::SFNode currentNode );
    vrml::SharedScalarInterpolator              readScalarInterpolator( const std::string &nodeName );
    vrml::SharedScript                          readScript( const std::string &nodeName );
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
    vrml::SharedShape                           readShape( const std::string &nodeName );
    vrml::SharedSound                           readSound( const std::string &nodeName );
    vrml::SharedSphere                          readSphere( const std::string &nodeName );
    vrml::SharedSphereSensor                    readSphereSensor( const std::string &nodeName );
    vrml::SharedSpotLight                       readSpotLight( const std::string &nodeName );
    void                                        readStatements( void );
    vrml::SharedSwitch                          readSwitch( const std::string &nodeName );
    vrml::SharedText                            readText( const std::string &nodeName );
    vrml::SharedTextureCoordinate               readTextureCoordinate( const std::string &nodeName );
    vrml::SharedTextureTransform                readTextureTransform( const std::string &nodeName );
    vrml::SharedTimeSensor                      readTimeSensor( const std::string &nodeName );
    vrml::SharedTouchSensor                     readTouchSensor( const std::string &nodeName );
    vrml::SharedTransform                       readTransform( const std::string &nodeName );
    vrml::SharedViewpoint                       readViewpoint( const std::string &nodeName );
    vrml::SharedVisibilitySensor                readVisibilitySensor( const std::string &nodeName );
    vrml::SharedWorldInfo                       readWorldInfo( const std::string &nodeName );
    template<typename T> void                   resampleKeyValues( vrml::MFFloat & keys, std::vector<T> & values, unsigned int valuesPerKey, std::vector<unsigned int> & steps, vrml::SFTime cycleInterval );
    void                                        setNextToken( void );
    bool                                        testWRLVersion( const std::string &filename );

  private :
    std::string                                               m_currentString;
    std::string                                               m_currentToken;
    std::map<std::string,vrml::SFNode>                        m_defNodes;
    bool                                                      m_eof;
    FILE                                                    * m_fh;
    std::map<vrml::MFString,vrml::SharedInline>               m_inlines;
    char                                                    * m_line;
    unsigned int                                              m_lineLength;
    unsigned int                                              m_lineNumber;
    std::string::size_type                                    m_nextTokenEnd;
    std::string::size_type                                    m_nextTokenStart;
    std::vector<vrml::SFNode>                                 m_openNodes;
    std::set<std::string>                                     m_PROTONames;
    dp::sg::core::GroupSharedPtr                              m_rootNode;
    dp::sg::core::SceneSharedPtr                              m_scene;
    std::vector<std::string>                                  m_searchPaths;
    std::unique_ptr<dp::sg::algorithm::SmoothTraverser>       m_smoothTraverser;
    unsigned int                                              m_stepsPerUnit;
    vrml::SharedGroup                                         m_topLevelGroup;
    bool                                                      m_strict;
    std::map<std::string,dp::sg::core::TextureHostWeakPtr>    m_textureFiles;
    std::string                                               m_ungetToken;
    // Circular and rectangular subdivision limits. The creation functions attempt to subdivide with square sized quads.
    // Minimum, standard at radius 1.0, and maximum subdivisions for a full circle of a sphere, cylinder, cone tessellation depending on their radii and heights.
    // Minimum, standard at size 1.0, and maximum subdivisions for a Box depending on its size.
    // These six values can be defined by the user via the environment variable NVSG_WRL_SUBDIVISIONS. 
    // Defaults are NVSG_WRL_SUBDIVISIONS = 12 36 90  2 4 8
    int                                                       m_subdivisions[6];
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
