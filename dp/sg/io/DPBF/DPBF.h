// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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
/** @file */

#include <dp/sg/core/Config.h>
#include <dp/util/BitMask.h>

// DPBF is independent of packing. However, this forces
// the compile-time asserts below to fire on inconsistencies
#pragma pack(push, 1)

// convenient aliases for used built-in types
typedef char                    byte_t;   //!< Specifies an 8-bit signed type.
typedef unsigned char           ubyte_t;  //!< Specifies an 8-bit unsigned type.
typedef short                   short_t;  //!< Specifies a 16-bit signed integer type
typedef unsigned short          ushort_t; //!< Specifies a 16-bit unsigned integer type
typedef int                     int_t;    //!< Specifies a 32-bit signed integer type.
typedef unsigned int            uint_t;   //!< Specifies a 32-bit unsigned integer type.
#ifdef LINUX
# if __WORDSIZE == 64           // defined indirectly through stdint.h
// avoid a conflict with GNU stdint.h on Linux64
// note: long is a 64-bit type on Linux64, while it is 32bit on Win64
// Linux64
typedef unsigned long           uint64_t; //!< Specifies a 64-bit unsigned integer type.
# else
// Linux32
typedef unsigned long long      uint64_t; //!< Specifies a 64-bit unsigned integer type.
# endif
#else
// Win32 and Win64
typedef unsigned long long      uint64_t; //!< Specifies a 64-bit unsigned integer type.
#endif
typedef float                   float2_t[2]; //!< Specifies a 2-component float vector.
typedef float                   float3_t[3]; //!< Specifies a 3-component float vector.
typedef float                   float4_t[4]; //!< Specifies a 4-component float vector.
typedef float                   float44_t[4][4];  //!< Specifies a 4x4-component float matrix.

#define PADDING(n) PADDING_i(n,__LINE__)        //!< Convenient macro to add padding bits, part one of three
#define PADDING_i(n,l) PADDING_ii(n,l)          //!< Convenient macro to add padding bits, part two of three
#define PADDING_ii(n,l) ubyte_t padding##l[n]   //!< Convenient macro to add padding bits, part three of three

// DPBF version. DPBF uses the same version numbers as NBF up to version 0x56.00
const ubyte_t DPBF_VER_MAJOR  =  0x56; //!< DPBF major version number
const ubyte_t DPBF_VER_MINOR  =  0x00; //!< DPBF version compatibility level
const ubyte_t DPBF_VER_BUGFIX =  0x00; //!< DPBF version bugfix level
  
// constants specifying a certain byte order
const ubyte_t DPBF_LITTLE_ENDIAN = 0x00; //!< Specifies little endian byte order
const ubyte_t DPBF_BIG_ENDIAN    = 0x01; //!< Specifies big endian byte order

// convenient helper types

//! The byteArray_t structure specifies how an array of bytes is stored in a .DPBF file.
struct byteArray_t
{
  uint_t  numBytes;               //!< Specifies the number of bytes in the array.
  uint_t  bytes;                  //!< Specifies the file offset to the byte data.
};

//! The str_t structure specifies how a string is stored in a .DPBF file.
struct str_t 
{ 
  uint_t      numChars;           //!< Specifies the number of characters in the actual string,
                                  //!< not including the terminating null character.
  uint_t      chars;              //!< Specifies the file offset to the string characters.
};
DP_STATIC_ASSERT( ( sizeof(str_t) % 4 ) == 0 );   //!< Compile-time assert on size of structure

//! The sstr_t structure specifies how a small string is stored in a .DPBF file.
/** \note A small string is limited to 65535 characters, including the terminating 0 */
struct sstr_t
{
  ushort_t    numChars;           //!< Specifies the number of characters in the actual small string,
                                  //!< not including the terminating null character.
  uint_t      chars;              //!< Specifies the file offset to the string characters.
};

//! The vertexAttrib_t structure specifies how vertex attributes are stored in a .DPBF file
/** Objects of type vertexAttrib_t alway are considered in conjunction with NBFVertexAttributeSet objects. */
struct vertexAttrib_t
{
  uint_t      size;               //!< Specifies the number of coordinates per vertex.
  uint_t      type;               //!< Symbolic constant indicating the data type of each coordinate.
  uint_t      numVData;           //!< Specifies the number of vertex data stored at offset \a vdata.
  uint_t      vdata;              //!< Specifies the file offset to the raw vertex data
};
DP_STATIC_ASSERT( ( sizeof(vertexAttrib_t) % 4 ) == 0 );    //!< Compile-time assert on size of structure

//! The texCoordSet_t structure specifies how a texture coordinate set is stored in a .DPBF file.
/** Texture coordinate sets, in this context, need to be considered in conjunction with NBFVertexAttributeSet objects. */
struct texCoordSet_t 
{
  uint_t      numTexCoords;       //!< Specifies the number of texture coordinates contained in the actual set.
  uint_t      coordDim;           //!< Specifies the dimension, in terms of float, of the contained texture coordinates.
                                  //!< Texture coordinates can be either one, two, three, or four dimensional.
  uint_t      texCoords;          //!< Specifies the file offset to the contained texture coordinates. 
};
DP_STATIC_ASSERT( ( sizeof(texCoordSet_t) % 4 ) == 0 );   //!< Compile-time assert on size of structure

//! The indexList_t structure specifies how an index set is stored in a .DPBF file.
/** Index sets, in this context, need to be considered in conjunction with NBFStrippedPrimitiveSet objects. */
struct indexList_t
{
  uint_t      numIndices;         //!< Specifies the number of indices in the actual index set.
  uint_t      indices;            //!< Specifies the file offset to the indices. As specified by the DPBF format,
                                  //!< a index is a 32-bit unsigned integer value.
};
DP_STATIC_ASSERT( ( sizeof(indexList_t) % 4 ) == 0 );    //!< Compile-time assert on size of structure

//! The meshSet_t structure specifies how an mesh set is stored in a .DPBF file.
/** Mesh sets, in this context, need to be considered in conjunction with NBFMeshedPrimitiveSet objects. */
struct meshSet_t
{
  uint_t      width;              //!< Specifies the width of the mesh
  uint_t      height;             //!< Specifies the height of the mesh
  uint_t      indices;            //!< Specifies the file offset to the indices. As specified by the NBF format,
  //!< a index is a 32-bit unsigned integer value.
};
DP_STATIC_ASSERT( ( sizeof(meshSet_t) % 4 ) == 0 );   //!< Compile-time assert on size of structure

//! The texImage_t structure specifies how a texture image is stored in a .DPBF file.
/** Texture images are considered in conjunction with NBFParameterGroupData objects. */
struct texImage_t
{
  uint_t      flags;              //!< Creation flags.
  str_t       file;               //!< Specifies the filename of the image file in case the image is from a file.
  // the following are only relevant in case the image is not from a file but from a image data lump.
  uint_t      width;              //!< Specifies the width of the texture in pixels.
  uint_t      height;             //!< Specifies the height of the texture in pixels.
  uint_t      depth;              //!< Specifies the depth of the texture in pixels.
  uint_t      target;             //!< texture target.
  PADDING(8);                     //!< Padding bits to ensure offset of scene is on a 4-byte boundary, regardless of packing.
  uint_t      pixelFormat;        //!< Specifies the format of the pixel data. 
  uint_t      dataType;           //!< Specifies the type of the pixel data.
  uint_t      pixels;             //!< Specifies the file offset to the raw pixel data.
};
DP_STATIC_ASSERT( ( sizeof(texImage_t) % 4 ) == 0 );    //!< Compile-time assert on size of structure

//! The trafo_t structure specifies how a transformation is stored in a .DPBF file.
struct trafo_t
{
  float4_t    orientation;        //!< Specifies the orientational part of the transformation.
  float3_t    scaling;            //!< Specifies the scaling part of the transformation.
  float3_t    translation;        //!< Specifies the translational part of the transformation.
  float3_t    center;             //!< Specifies the center of rotation of the transformation.
  float4_t    scaleOrientation;   //!< Specifies the scale orientational part of the transformation.
};
DP_STATIC_ASSERT( ( sizeof(trafo_t) % 4 ) == 0 );   //!< Compile-time assert on size of structure

//! The plane_t structure specifies how a clipping plane is stored in an .DPBF file.
struct plane_t
{
  uint_t      active;             //!< Specifies if this plane is active
  float3_t    normal;             //!< Specifies the normal of the plane
  float       offset;             //!< Specifies the offset of the plane
};
DP_STATIC_ASSERT( ( sizeof(plane_t) % 4 ) == 0 );   //!< Compile-time assert on size of structure

//! The switchMask_t structure specifies how a SwitchMask is stored in a .DPBF file.
struct switchMask_t
{
  uint_t        maskKey;          //!< Specifies the key to identify this mask
  uint_t        numChildren;      //!< Specifies the number of active children stored with this mask
  uint_t        children;         //!< Specifies the file offset to the zero-based indexes referencing the active children
};
DP_STATIC_ASSERT( ( sizeof(switchMask_t) % 4 ) == 0 );    //!< Compile-time assert on size of structure

//! Unique DPBF Object Codes
/** Each concrete NBFObject type is assigned to a unique DPBF object code. 
  * This code is a 32-bit unsigned integer value, stored at offset 0, of each concrete NBFObject.
  * The purpose of the unique 'per-object' code is to provide a Load-Time Type Information (LTTI) 
  * to resolve concrete NBFObjects while loading DPBF files. */   
enum
{
  // scene object 
  NBF_UNKNOWN                     = 0x00000000  //!< Unknown, and hence, invalid object code.
  // animation objects (0x100 - 0x1FF)
, NBF_TRAFO_ANIMATION           = 0x00000100  //!< Obsolete
, NBF_VNVECTOR_ANIMATION                      //!< Obsolete
, NBF_VERTEX_ATTRIBUTE_ANIMATION              //!< Obsolete
, NBF_INDEX_ANIMATION                         //!< Obsolete
  // framed animation descriptions
, NBF_FRAMED_ANIMATION          = 0x00000120
, NBF_FRAMED_TRAFO_ANIMATION_DESCRIPTION      //!< Obsolete
, NBF_FRAMED_VNVECTOR_ANIMATION               //!< Obsolete
, NBF_FRAMED_VERTEX_ATTRIBUTE_ANIMATION_DESCRIPTION //!< Obsolete
, NBF_FRAMED_INDEX_ANIMATION_DESCRIPTION      //!< Obsolete
  // interpolated animation descriptions
, NBF_INTERPOLATED_ANIMATION    = 0x00000140
, NBF_LINEAR_INTERPOLATED_TRAFO_ANIMATION_DESCRIPTION             //!< Obsolete
, NBF_LINEAR_INTERPOLATED_VNVECTOR_ANIMATION                      //!< Obsolete
, NBF_LINEAR_INTERPOLATED_VERTEX_ATTRIBUTE_ANIMATION_DESCRIPTION  //!< Obsolete
  // camera objects (0x200 - 0x2FF)
, NBF_CAMERA                    = 0x00000200
, NBF_MONO_CAMERA
, NBF_JITTER_CAMERA
, NBF_SIMPLE_CAMERA
, NBF_PARALLEL_CAMERA                         //!< Identifies an NBFParallelCamera object.
, NBF_PERSPECTIVE_CAMERA                      //!< Identifies an NBFPerspectiveCamera object.
, NBF_STEREO_CAMERA  
, NBF_MATRIX_CAMERA                           //!< Identifies an NBFMatrixCamera object.
  // drawable objects (0x300 - 0x3FF) 
, NBF_DRAWABLE                  = 0x00000300
, NBF_VERTEX_ATTRIBUTE_SET                    //!< Identifies an NBFVertexAttributeSet object.
, NBF_TRIANGLES                               //!< Obsolete
, NBF_ANIMATED_TRIANGLES                      //!< Obsolete
, NBF_SKINNED_TRIANGLES                       //!< Obsolete
, NBF_TRISTRIPS                               //!< Obsolete
, NBF_QUADS                                   //!< Obsolete
, NBF_ANIMATED_QUADS                          //!< Obsolete
, NBF_QUADSTRIPS                              //!< Obsolete
, NBF_LINES                                   //!< Obsolete
, NBF_LINESTRIPS                              //!< Obsolete
, NBF_TRIFANS                                 //!< Obsolete
, NBF_POINTS                                  //!< Obsolete
, NBF_QUADMESHES                              //!< Obsolete
, NBF_ANIMATED_VERTEX_ATTRIBUTE_SET           //!< Obsolete
, NBF_SKIN                                    //!< Identifies an NBFSkin
, NBF_PATCHES                                 //!< Obsolete
, NBF_QUAD_PATCHES                            //!< Obsolete
, NBF_QUAD_PATCHES_4X4                        //!< Obsolete
, NBF_RECT_PATCHES                            //!< Obsolete
, NBF_TRI_PATCHES                             //!< Obsolete
, NBF_TRI_PATCHES_4                           //!< Obsolete
, NBF_PRIMITIVE                               //!< Identifies an NBFPrimitive
, NBF_INDEX_SET                               //!< Identifies an NBFIndexSet object.
  // node objects (0x400 - 0x4FF)
, NBF_NODE                      = 0x00000400
, NBF_GEO_NODE                                //!< Identifies a NBFGeoNode object.
, NBF_GROUP                                   //!< Identifies a NBFGroup object.
, NBF_LOD                                     //!< Identifies a NBFLOD object.
, NBF_SWITCH                                  //!< Identifies a NBFSwitch object.
, NBF_TRANSFORM                               //!< Identifies a NBFTransform object.
, NBF_ANIMATED_TRANSFORM                      //!< Obsolete
, NBF_LIGHT_SOURCE                            //!< Identifies an NBFLightSource object.
, NBF_DIRECTED_LIGHT                          //!< Identifies an NBFDirectedLight object.
, NBF_POINT_LIGHT                             //!< Identifies an NBFPointLight object.
, NBF_SPOT_LIGHT                              //!< Identifies an NBFSpotLight object.
, NBF_BILLBOARD                               //!< Identifies an NBFBillboard object.
, NBF_VOLUME_NODE                             //!< Identifies an NBFVolumeNode object.
, NBF_FLIPBOOK_ANIMATION                      //!< Obsolete
  // state set objects (0x500 - 0x5FF)
, NBF_STATE_SET                 = 0x00000500  //!< Identifies an NBFStateSet object.
, NBF_STATE_VARIANT                           //!< Obsolete
, NBF_STATE_PASS                              //!< Obsolete
  // state attribute objects (0x600 - 0x6FF)
, NBF_STATE_ATTRIBUTE           = 0x00000600
, NBF_CGFX                                    //!< Obsolete
, NBF_MATERIAL                                //!< Obsolete
, NBF_FACE_ATTRIBUTE                          //!< Obsolete
, NBF_TEXTURE_ATTRIBUTE                       //!< Obsolete
, NBF_TEXTURE_ATTRIBUTE_ITEM                  //!< Obsolete
, NBF_LINE_ATTRIBUTE                          //!< Obsolete
, NBF_POINT_ATTRIBUTE                         //!< Obsolete
, NBF_BLEND_ATTRIBUTE                         //!< Obsolete
, NBF_DEPTH_ATTRIBUTE                         //!< Obsolete
, NBF_ALPHA_TEST_ATTRIBUTE                    //!< Obsolete
, NBF_LIGHTING_ATTRIBUTE                      //!< Obsolete
, NBF_UNLIT_COLOR_ATTRIBUTE                   //!< Obsolete
, NBF_STENCIL_ATTRIBUTE                       //!< Obsolete
, NBF_RTFX                                    //!< Obsolete
, NBF_RTBUFFER_ATTRIBUTE                      //!< Obsolete
, NBF_RTFX_SCENE_ATTRIBUTE                    //!< Obsolete
, NBF_RTFX_PROGRAM                            //!< Obsolete
, NBF_PIPELINE_DATA                           //!< Identifies an NBFPipelineData object.
, NBF_PARAMETER_GROUP_DATA                    //!< Identifies an NBFParameterGroup object.
, NBF_SAMPLER                                 //!< Identifies an NBFSampler object.
, NBF_SAMPLER_STATE                           //!< Identifies an NBFSamplerState object.
  // custom objects (>=0x700)
, NBF_CUSTOM_OBJECT             = 0x00000700  //!< Custom objects must not have codes lower than this.
};

//! The NBFHeader structure represents the NBF header format.
/** The NBFHeader structure is the primary location where NBF specifics are stored.\n
  * For a valid NBF file, the NBFHeader structure is stored at file offset 0. Note that,
  * except for the NBFHeader object, a file offset of 0 indicates an invalid file offset!\n
  * This structure mainly serves as validation and compatibility checks for verification 
  * purposes. It also maintains the file offset to the contained NBFScene object, which  
  * represents a scene in the context of computer graphics. */ 
struct NBFHeader
{
  // signature
  byte_t      signature[4];       //!< A 4-byte signature identifying the file as a valid NBF file. The bytes are "#NBF".
  // NBF version
  ubyte_t     nbfMajorVersion;    //!< Specifies the major part of the NBF version used to save the file.
  ubyte_t     nbfMinorVersion;    //!< Specifies the minor part (compatibility level) of the NBF version used to save the file.
  ubyte_t     nbfBugfixLevel;     //!< Specifies the bugfix level of the NBF version used to save the file. This is optional
                                  //!< information, as a bugfix level does not influence compatibility issues, and hence 
                                  //!< must not be taken into account for compatibility checks.
  // SceniX version
  ubyte_t     dpMajorVersion;   //!< Specifies the major part of the pipeline version the content of this file is compatible to.
  ubyte_t     dpMinorVersion;   //!< Specifies the minor part of the pipeline version the content of this file is compatible to.
  ubyte_t     dpBugfixLevel;    //!< Specifies the bugfix level of the pipeline version. This is optional information, as a 
                                  //!< bugfix level does not influence compatibility issues, and hence must not be taken 
                                  //!< into account for compatibility checks.
  // Reserved bytes 
  ubyte_t     reserved[16];       //!< Reserved bytes for future extensions.
  // Date
  ubyte_t     dayLastModified;    //!< Specifies the day (1-31) of last modification.
  ubyte_t     monthLastModified;  //!< Specifies the month (1-12) of last modification.
  ubyte_t     yearLastModified[2]; //!< Specifies the year of last modification.
  // Time stamp
  ubyte_t     secondLastModified; //!< Specifies the second (0-59) of last modification. 
  ubyte_t     minuteLastModified; //!< Specifies the minute (0-59) of last modification.
  ubyte_t     hourLastModified;   //!< Specifies the hour (0-23) of last modification.
  // endianess
  ubyte_t     byteOrder;          //!< Specifies the byte order used to save the contained data.
                                  //!< A value of 0 implies little-endian byte order, a value of 1 implies big-endian byte order.
                                  //!< It is more convenient to use the symbolic constants NBF_LITTLE_ENDIAN and NBF_BIG_ENDIAN here.
  PADDING(2);                     //!< Padding bits to ensure offset of scene is on a 4-byte boundary, regardless of packing.
  // scene object
  uint_t      scene;              //!< Specifies the file offset to the contained NBFScene object.
  // optional view state
  uint_t      viewState;          //!< Specifies the file offset to an optional NBFViewState object. 
                                  //!< An offset of 0 indicates that no NBFViewState object is available in this file. 
};
DP_STATIC_ASSERT( ( sizeof(NBFHeader) % 4 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFScene structure represents a scene in the context of computer graphics.
/** A valid NBF file always contains one - and only one - NBFScene object. 
  * The file offset to this NBFScene object is specified within the NBFHeader structure. */
struct NBFScene
{
  float3_t    ambientColor;             //!< Specifies the global ambient color to be used for rendering.
  float4_t    backColor;                //!< Specifies the scene's RGBA background color used with rendering.
  uint_t      backImg;                  //!< Specifies the file offset to the back image object
  uint_t      numCameras;               //!< Specifies the number of the scene's NBFCamera objects.
  uint_t      cameras;                  //!< Specifies the file offset to the offsets of the scene's NBFCamera objects.
  uint_t      numCameraAnimations;      //!< Obsolete
  uint_t      cameraAnimations;         //!< Obsolete
  uint_t      numberOfAnimationFrames;  //!< Obsolete
  uint_t      root;                     //!< Specifies the file offset to the scene's root node, which always is of a NBFNode derived type.
  uint_t      numObjectLinks;           //!< Specifies the number of objects links in the scene
  uint_t      objectLinks;              //!< Specifies the file offset to the scenes's object links
  uint_t      numAttributes;            //!< Specifies the number of attributes in the scene
  uint_t      attributes;               //!< Specifies the file offset to the scenes's attributes
};
DP_STATIC_ASSERT( ( sizeof(NBFScene) % 4 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFViewState represents an optional view state used to render the scene.
/** The file offset to an optional NBFViewState object is specified within the NBFHeader structure. */
struct NBFViewState
{
  uint_t      objectCode;           //!< Specifies the object code of the actual object. The object code is unique per object type! 
  uint_t      camera;               //!< Specifies the offset to the NBFCamera object to be used for viewing.
  ubyte_t     isStereo;             //!< Indicates whether the view is in stereo mode.
  ubyte_t     isStereoAutomatic;    //!< Indicates whether eye distance is automatically adjusted in stereo mode.
  ubyte_t     isAutoClipPlanes;     //!< Indicates if automatic generation of clipping planes is enabled.
  PADDING(1);                       //!< Padding bits to ensure offset of next member is on a 4-byte boundary, regardless of packing.
  PADDING(20); //!< discontinuing support for jitter
  float       stereoAutomaticFactor;//!< Specifies the automatic eye distance adjustment factor in stereo mode.
  float       stereoEyeDistance;    //!< Specifies the stereo eye distance used if the view is in stereo mode.
  float       targetDistance;       //!< Specifies the target distance to the projection plane.
};
DP_STATIC_ASSERT( ( sizeof(NBFViewState) % 4 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFObject structure represents general object data. 
struct NBFObject
{
  uint_t      objectCode;         //!< Specifies the object code of the actual object. The object code is unique per object type! 
  ubyte_t     isShared;           //!< Indicates whether the data of the actual object is shared among different objects.
                                  //!< A value of 1 indicates that this object's data is shared, whereas a value of 0 indicates
                                  //!< that this object's data is not shared.

  PADDING(3);                     //!< Padding bits to keep compatibility to earlier versions.

  uint64_t    objectDataID;       //!< A unique 64-bit value to identify shared object data while loading.
  uint_t      sourceObject;       //!< Specifies the file offset to the source object in case of data sharing. 
                                  //!< A file offset of 0 always indicates that no source object is available for the actual object.
  uint_t       objectName;        //!< Specifies the offset to an optional name. A 0-offset implies no name.
                                  //!< The name is stored as a str_t object.
  uint_t       objectAnno;        //!< Specifies the offset to an optional annotation that can be specified for an object.
                                  //!< A 0-offset implies no annotation. An annotation is stored as a str_t object.
  uint_t       hints;             //!< Hints vars for node, user, object
};
// NOTE: Because of the uint64_t member objectDataID above, which is 8-byte aligned, 
// we need to ensure the size of NBFObject is fixed - that is, independent of whatever
// the compilers actual packing value might be! We achieve this by making the size of 
// NBFObject a multiple of 8 bytes (see compile time assert below).  
DP_STATIC_ASSERT( ( sizeof(NBFObject) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFCamera represents a camera.
/** A NBFCamera serves as base class only.\n
  * Concrete object codes valid for a NBFCamera are NBF_PARALLEL_CAMERA
  * and NBF_PERSPECTIVE_CAMERA. Further object codes valid for a NBFCamera
  * are subject to future extensions for the NBF format. */
struct NBFCamera : public NBFObject
{
  uint_t      numHeadLights;    //!< Specifies the number of headlights attached.
  uint_t      headLights;       //!< Specifies the file offset to the offsets to the attached headlight objects.
                                //!< Headlights are of type NBFLightSource. 
  float3_t    upVector;         //!< Specifies the camera's normalized up vector.
  float3_t    position;         //!< Specifies the actual position of camera in world space.
  float3_t    direction;        //!< Specifies the normalized direction for the camera to look along.
  float       focusDist;        //!< Specifies the distance to the projection plane.
};
DP_STATIC_ASSERT( ( sizeof(NBFCamera) % 8 ) == 0 );   //!< Compile-time assert on size of structure

/*! \brief The NBFFrustumCamera structure is the base of the NBFParalleleCamera and the NBFPerspectiveCamera. */
struct NBFFrustumCamera : public NBFCamera
{
  float       farDist;          //!< Specifies the distance from the actual camera position to the far clipping plane.
  PADDING(4);      //!< Padding bits to ensure offset of windowSize is on a 4-byte boundary, regardless of packing
  float       nearDist;         //!< Specifies the distance from the actual camera position to the near clipping plane.
  float2_t    windowOffset;     //!< Specifies the world-relative offset from the viewing reference point to the center 
                                //!< of the viewing window.
  float2_t    windowSize;       //!< Specifies the world-relative size of the viewing window. Whereas the x-component of 
                                //!< of the vector specifies the width, and the y-component of the vector specifies the height.
  PADDING(4);        //!< Padding bits to ensure the size of NBFCamera is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFFrustumCamera) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFParallelCamera represents a parallel camera.
/** A NBFParallelCamera is a concrete camera type. 
  * The object code for a NBFParallelCamera is NBF_PARALLEL_CAMERA. */
struct NBFParallelCamera : public NBFFrustumCamera
{
};
DP_STATIC_ASSERT( ( sizeof(NBFParallelCamera) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFPerspectiveCamera represents a perspective camera.
/** A NBFPerspectiveCamera is a concrete camera type. 
  * The object code for a NBFPerspectiveCamera is NBF_PERSPECTIVE_CAMERA. */
struct NBFPerspectiveCamera : public NBFFrustumCamera
{
};
DP_STATIC_ASSERT( ( sizeof(NBFPerspectiveCamera) % 8 ) == 0 );    //!< Compile-time assert on size of structure

/*! \brief The NBFMatrixCamera structure represents a general matrix camera.
 *  \remarks The object code for a NBFMatrixCamera is NBF_MATRIX_CAMERA. */
struct NBFMatrixCamera : public NBFCamera
{
  float44_t   projection;         //!< Specifies the projection matrix
  float44_t   inverseProjection;  //!< Specifies the inverse projection matrix
};
DP_STATIC_ASSERT( ( sizeof(NBFMatrixCamera) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFVertexAttributeSet structure represents a set of vertex attributes.
/** A NBFVertexAttributeSet maintains a full set of geometry
  * data (vertex attributes) that is used by all topology classes. */
struct NBFVertexAttributeSet : public NBFObject
{
  uint_t      enableFlags;            //!< Specifies which of the 16 vertex attributes is enabled.
  uint_t      normalizeEnableFlags;   //!< Specifies for which of the 16 vertex attributes normalization is enabled.
  vertexAttrib_t  vattribs[16];       //!< Specifies the 16 vertex attributes.
};
DP_STATIC_ASSERT( ( sizeof(NBFVertexAttributeSet) % 8 ) == 0 );   //!< Compile-time assert on size of structure

/*! \brief The NBFSkin structure represents a set of skin information for a vertex attribute.
 *  \remarks An NBFSkin derives from NBFObject, and holds the bindShapeMatrix, the number
 *  of influences (numCounts), the number of indices/weights per influence (counts), the indices
 *  into the array of joints and the corresponding weights, the skinning types for each vertex
 *  attribute, the number of joints, and the offsets to the inverseBindMatrices and the joints. */
struct NBFSkin : public NBFObject
{
  float44_t   bindShapeMatrix;    //!< Specifies the bindShapeMatrix
  uint_t      numCounts;          //!< Specifies the number of influences (counts).
  uint_t      counts;             //!< Specifies the file offset to the indices/weights per vertex.
  uint_t      indices;            //!< Specifies the file offset to the indices.
  uint_t      weights;            //!< Specifies the file offset to the weights.
  ubyte_t     skinningType[16];   //!< Specifies the skinning type for each vertex attribute.
  uint_t      numJoints;          //!< Specifies the number of joints.
  uint_t      inverseBindMatrices;  //!< Specifies the file offset to the inverseBindMatrices.
  uint_t      joints;             //!< Specifies the file offset to the joints.
  uint_t      bindPose;           //!< Specifies the file offset to the bind pose vertex attribute set.
};
DP_STATIC_ASSERT( ( sizeof(NBFSkin) % 8 ) == 0 );

//! The NBFPrimitiveSet structure represents a geometry with an NBFVertexAttributeSet.
/** A NBFPrimitiveSet holds the offset to an NBFVertexAttributeSet. */
struct NBFPrimitiveSet : public NBFObject
{
  uint_t      vertexAttributeSet; //!< Specifies the file offset to the vertex attribute set.
  uint_t      skin;               //!< Specifies the file offset to the skin (from version 0x3e on!)
};
DP_STATIC_ASSERT( ( sizeof(NBFPrimitiveSet) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFPrimitive structure represents a geometry with an NBFVertexAttributeSet, and possibly an index set
/** A NBFPrimitive holds the offset to an NBFVertexAttributeSet, and possibly an index set */
struct NBFPrimitive : public NBFObject
{
  uint_t      primitiveType;      //!< Specified the primitive type (unit because it may be a user enum)
  uint_t      elementOffset;      //!< Specifies the element offset
  uint_t      elementCount;       //!< Specifies the element count
  uint_t      instanceCount;      //!< Specified the instance count
  PADDING(4);                     //!< Padding bits to ensure the size of NBFPrimitive is a multiple of 8, regardless of packing.    
  uint_t      renderFlags;        //!< Specified the rendering flags
  uint_t      vertexAttributeSet; //!< Specifies the file offset to the vertex attribute set.
  uint_t      indexSet;           //!< Specifies the file offset to the index set
  ubyte_t     patchesMode;        //!< Specifies the patches mode, if this Primitive is a patch
  ubyte_t     patchesOrdering;    //!< Specifies the patches ordering, if this Primitive is a patch
  ubyte_t     patchesSpacing;     //!< Specifies the patches spacing, if this Primitive is a patch
  ubyte_t     patchesType;        //!< Specifies the patches type, if this Primitive is a patch
  PADDING(4);                     //!< Padding bits to ensure the size of NBFPrimitive is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFPrimitive) % 8 ) == 0 );   // Compile-time assert on size of structure

//! The NBFIndependentPrimitiveSet structure represents a set of independent primitives.
/** A NBFIndependentPrimitiveSet is an abstract topology class derived from NBFPrimitiveSet.
  * It is used with NBF_LINES, NBF_POINTS, NBF_QUADS, and NBF_TRIANGLES. */
struct NBFIndependentPrimitiveSet : public NBFPrimitiveSet
{
  uint_t      numIndices;         //!< Specifies the number of contained indices
  uint_t      indices;            //!< Specifies the file offset to the Independent objects.
};
DP_STATIC_ASSERT( ( sizeof(NBFIndependentPrimitiveSet) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFMeshedPrimitiveSet structure represents a mesh set.
/** A NBFMeshedPrimitiveSet is an abstract topology class derived from NBFPrimitiveSet. 
* Valid object codes for a NBFMeshedPrimitiveSet are NBF_QUADMESHES. */
struct NBFMeshedPrimitiveSet : public NBFPrimitiveSet
{
  uint_t      numMeshes;          //!< Specifies the number of meshes
  uint_t      meshes;             //!< Specifies the file offset to the meshes. 
  //!< Strips are stored as indexList_t objects.
};
DP_STATIC_ASSERT( ( sizeof(NBFMeshedPrimitiveSet) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFStrippedPrimitiveSet structure represents a strip set.
/** A NBFStrippedPrimitiveSet is an abstract topology class derived from NBFPrimitiveSet. 
  * Valid object codes for a NBFStrippedPrimitiveSet are NBF_TRIFANS, NBF_TRISTRIPS, and NBF_QUADSTRIPS. */
struct NBFStrippedPrimitiveSet : public NBFPrimitiveSet
{
  uint_t      numStrips;          //!< Specifies the number of strips
  uint_t      strips;             //!< Specifies the file offset to the strips. 
                                  //!< Strips are stored as indexList_t objects.
};
DP_STATIC_ASSERT( ( sizeof(NBFStrippedPrimitiveSet) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFNode structure represents a general node.
/** A NBFNode serves as base class only. Concrete object codes valid for a NBFNode are
  * NBF_GEO_NODE, NBF_LOD, NBF_SWITCH, NBF_TRANSFORM, NBF_ANIMATED_TRANSFORM,
  * NBF_DIRECTED_LIGHT, NBF_POINT_LIGHT, NBF_SPOT_LIGHT, and NBF_VOLUME_NODE. Further concrete
  * object codes valid for a NBFNode are subject to future extensions of the NBF format. */
struct NBFNode : public NBFObject
{
  str_t       annotation;         //!< Specifies an optional annotation string. Unused since v61.2!
  PADDING(6);                     //!< Padding bits to ensure the size of NBFStateAttribute is a multiple of 4, regardless of packing.
  PADDING(2);                     //!< Two more padding for backwards compat
};
DP_STATIC_ASSERT( ( sizeof(NBFNode) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFGeoNode structure represents a geometry node.
/** The object code for a NBFGeoNode is NBF_GEO_NODE. */
struct NBFGeoNode : public NBFNode
{
  uint_t      materialPipeline;   //!< Specifies the file offset to the corresponding NBFPipelineData object
  uint_t      primitive;          //!< Specifies the file offset to the corresponding NBFPrimitive object. 
  uint_t      stateSet;           //!< Obsolete
  PADDING(4);        //!< Padding bits to ensure the size of NBFGeoNode is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFGeoNode) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFGroup structure represents a group node.
/** A NBFGroup serves as base class only. Concrete object codes valid for a NBFGroup are
  * NBF_LOD, NBF_SWITCH, NBF_TRANSFORM, and NBF_ANIMATED_TRANSFORM. Further concrete object
  * codes valid for a NBFGroup are subject to future extensions of the NBF format. */
struct NBFGroup : public NBFNode
{
  uint_t      numChildren;        //!< Specifies the number of maintained children.
  uint_t      children;           //!< Specifies the file offset to the offsets to the maintained children.
                                  //!< NBFGroup's children always are of NBFNode-derived types.
  uint_t      numClipPlanes;      //!< Specifies the number of clipping planes.
  uint_t      clipPlanes;         //!< Specifies the file offset to the clipping planes
  uint_t      numLightSource;     //!< Specifies the number of light sources.
  uint_t      lightSources;       //!< Specifies the fie offset to the offsets to the light sources.
};
DP_STATIC_ASSERT( ( sizeof(NBFGroup) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFLOD structure represents a 'Level Of Detail' group node.
/** The object code for a NBFLOD is NBF_LOD. */
struct NBFLOD : public NBFGroup
{
  float3_t    center;             //!< Specifies the center point used for distance calculations.
  uint_t      numRanges;          //!< Specifies the number of contained ranges.
  uint_t      ranges;             //!< Specifies the file offset to the ranges. 
                                  //!< Ranges are stored as 32-bit floating point numbers.
  PADDING(4);        //!< Padding bits to ensure the size of NBFLOD is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFLOD) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFSwitch structure represents a switch group node.
/** The object code for a NBFSwitch is NBF_SWITCH. */
struct NBFSwitch : public NBFGroup
{
  uint_t      activeMaskKey;      //!< Specifies the key of the active mask
  uint_t      numMasks;           //!< Specifies the number of masks stored at offset masks
  uint_t      masks;              //!< Specifies the file offset to the masks stored as switchMask_t objects
  PADDING(4);        //!< Padding bits to ensure the size of NBFSwitch is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFSwitch) % 8 ) == 0 );   //!< Compile-time assert on size of structure

//! The NBFBillboard structure represents a billboard group node.
/** The object code for a NBFBillboard is NBF_BILLBOARD. */
struct NBFBillboard: public NBFGroup
{
  float3_t  rotationAxis;         //!< Specifies the axis to rotate the Billboard around
  ubyte_t   alignment;            //!< Specifies the alignment (axis, viewer, or screen aligned)
  PADDING(3);        //!< Padding bits to ensure the size of NBFBillboard is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFBillboard) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFTransform structure represents a transform group node.
/** The object code for a NBFTransform is NBF_TRANSFORM. */
struct NBFTransform : public NBFGroup
{
  trafo_t             trafo;      //!< Specifies the transformation of the NBFTransform.
  PADDING(4);        //!< Padding bits to ensure the size of NBFTransform is a multiple of 8, regardless of packing.    
};
DP_STATIC_ASSERT( ( sizeof(NBFTransform) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFLightSource structure represents a light source node.
/** A NBFLightSource serves as base class only. Concrete object codes valid for 
  * a NBFLightSource are NBF_DIRECTED_LIGHT, NBF_POINT_LIGHT, and NBF_SPOT_LIGHT. */
struct NBFLightSource : public NBFNode
{
  uint_t      animation;          //!< Obsolete
  ubyte_t     castShadow;         //!< flag that determines if this light source creates shadows.
  ubyte_t     enabled;            //!< flag to indicate enabled state.
  PADDING(2);        //!< Padding bits to ensure the offset of the next struct member is on a 4-byte boundary.
  uint_t      lightEffect;        //!< Specifies the file offset to an optional NBFPipelineData
  PADDING(4);        //!< Padding bits to ensure the offset of the next struct member is on a 4-byte boundary.
};
DP_STATIC_ASSERT( ( sizeof(NBFLightSource) % 8 ) == 0 );    //!< Compile-time assert on size of structure

//! The NBFPipelineData structure represents a set of ParameterGroupDatas
/** The object code for an NBFPipelineData is NBF_PIPELINE_DATA. */
struct NBFPipelineData : public NBFObject
{
  str_t       effectFileName;       //!< Specifies the (potentially relative) file with the EffectSpec
  str_t       effectSpecName;       //!< Specifies the name of the corresponding EffectSpec
  uint_t      parameterGroupData;   //!< Specifies the file offset to the offsets to the NBFParameterGroupData objects
  ubyte_t     transparent;          //!< Specifies if this EffectData is to be handled as transparent
  PADDING(3);        //!< Padding bits to ensure the offset of the next struct member is on a 4-byte boundary.
};
DP_STATIC_ASSERT( ( sizeof(NBFPipelineData) % 8 ) == 0 );   //!< Compile-time assert on size of structure

struct NBFParameterGroupData : public NBFObject
{
  str_t       parameterGroupSpecName; //!< Specifies the name of the corresponding ParameterGroupSpec
  uint_t      numData;
  uint_t      data;                 //!< Specifies the file offset to the data
};
DP_STATIC_ASSERT( ( sizeof(NBFParameterGroupData) % 8 ) == 0 );   //!< Compile-time assert on size of structure

/*! \brief The NBFLink structure represents a link between two objects using a callback. */
struct NBFLink
{
  uint_t  linkID;                   //!< Specifies the class id of the callback
  uint_t  subject;                  //!< Specifies the offset of the subject
  uint_t  observer;                 //!< Specifies the offset of the observer
};
DP_STATIC_ASSERT( ( sizeof(NBFLink) % 4 ) == 0 );               //!< Compile-time assert on size of structure

/*! The NBFIndexSet structure specifies how indices are stored in a .DPBF file
 *  \remarks The object code for a NBFIndexSet is NBF_INDEX_SET. */
struct NBFIndexSet : public NBFObject
{
  uint_t      dataType;               //!< Data type
  uint_t      primitiveRestartIndex;  //!< Primitive Restart Index
  uint_t      numberOfIndices;        //!< Number of indices in buffer
  uint_t      idata;                  //!< the index data
};
DP_STATIC_ASSERT( ( sizeof(NBFIndexSet) % 8 ) == 0 );

struct NBFSampler : public NBFObject
{
  uint_t      texture;              //!< Specifies the offset of the Texture
  float4_t    borderColor;          //!< Speicifies the texture border RGBA color
  uint_t      magFilter;            //!< Specifies the filter used with magnifying.
                                    //!< Valid values are TFM_MAG_NEAREST, and TFM_MAG_LINEAR.
  uint_t      minFilter;            //!< Specifies the filter used with minimizing. 
                                    //!< //!< Valid values are TFM_MIN_NEAREST, TFM_MIN_LINEAR, TFM_MIN_LINEAR_MIPMAP_LINEAR,
                                    //!< TFM_MIN_NEAREST_MIPMAP_NEAREST, TFM_MIN_NEAREST_MIPMAP_LINEAR, TFM_MIN_LINEAR_MIPMAP_NEAREST.
  uint_t      texWrapS;             //!< Specifies the wrap parameter for texture coordinate s. 
  uint_t      texWrapT;             //!< Specifies the wrap parameter for texture coordinate t.
  uint_t      texWrapR;             //!< Specifies the wrap parameter for texture coordinate r.
  uint_t      compareMode;          //!> Specifies the compare mode parameter for a texture. Valid values are TCM_NONE and TCM_R_TO_TEXTURE.
  PADDING(4);                       //!< Padding bits ensure offset of next elements is on a 4-byte boundary, regardless of packing
};
DP_STATIC_ASSERT( ( sizeof(NBFSampler) % 8 ) == 0 );

#pragma pack(pop)

#undef __DP_STATIC_ASSERT_PREFIX
