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


//
// FLTLoader.cpp
//

#include <dp/Exception.h>
#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/PlugInterface.h>
#include <flt.h>
#include "FLTLoader.h"

// NVSG types used
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>

// optimizers
#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>
#include <dp/sg/algorithm/StrippingTraverser.h>
#include <dp/sg/algorithm/NormalizeTraverser.h>

#include "SimpleTesselator.h"
#include "StackedAtlas.h"

// stl headers
#include <algorithm>
#include <sstream>

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;

using std::make_pair;
using std::multimap;
using std::pair;
using std::string;

FLTLoader::FLTLoader()
  : m_numTextures(0), m_numMaterials(0), m_rgbColorMode(false), m_fileIDs(0), 
    m_buildMipmapsAtLoadTime( true ), m_taiManager(0),
    m_stackedAtlasManager(0), m_useStackedAtlasManager(false)
{
}

FLTLoader::~FLTLoader()
{

  // unreference all gathered nodes
  m_instanceNodes.clear();

  // delete tai manager if we had one
  if( getTaiManager() )
  {
    delete getTaiManager();
  }

  if( getStackedAtlasManager() )
  {
    delete getStackedAtlasManager();
  }
}

void
FLTLoader::deleteThis( void )
{
  // was instantiated using 'new'. hence kill it with 'delete'
  delete this;
}

bool 
FLTLoader::createInstance( FltFile * flt, FltNode * root )
{
  FltInstanceDefinition * finst = (FltInstanceDefinition *)root;

  //
  // Create a group node to store for later
  //
  GroupSharedPtr gh = Group::create();
  gh->setName( fltSafeNodeName( root ) );

  collectGeometry( gh, root, flt );
  for (unsigned int i=0;i<root->numChildren;i++)
  {
    if( !collect( root->child[i] ) )
    {
      buildScene( gh, root->child[i], flt, 0 /*level*/ );
    }
  }

  // add the instance to the list
  m_instanceNodes[ cvtToMapEntry( flt, finst->instance ) ] = gh;

  return true;
}

const GroupSharedPtr &
FLTLoader::lookupInstance( FltFile * flt, uint16 inst )
{
  std::map<mapEntryType,GroupSharedPtr>::iterator iter;

  iter = m_instanceNodes.find( cvtToMapEntry( flt, inst ) );

  //
  // re-use the parent group
  //
  if( iter != m_instanceNodes.end() )
  {
    return iter->second;
  }

  // log error
#if 0
  NVSG_TRACE_OUT_F(("Instance Reference: %d not found!!\n", inst ));
#endif
  static GroupSharedPtr nullGroup;
  return nullGroup;
}

//
// Get face color.  Returns true if face color was actually specified,
// false otherwise.  Sets result to white if no color specified.
//
FLTLoader::ColorLookupEnum
FLTLoader::getFaceColor( FltFile * flt, FltFace * face, Vec4f & result )
{
  if( !(face->miscFlags & FLTFACEMF_NOCOLOR) )
  {
    // apparently the standard is to assume packed colors are available
    // throughout if the header flag "RGB color mode" is selected.
    if( ((face->miscFlags & FLTFACEMF_PACKEDCOLOR) || m_rgbColorMode) &&
         // i'm not sure about this check - but some files claim to have
         // packed colors available and yet this value is zero, which
         // could be black, but black with full transparency does not
         // seem like a likely choice for a color.  Perhaps we will
         // need to add a workaround for this as well, but this seems to
         // work on the test files.
         face->packedColorPrimary != 0 )
    {
      // color is in ABGR format, with only BGR used at the face level
      setVec( result, FLTPACKED_COLOR_R(face->packedColorPrimary) / 255.0f,
                      FLTPACKED_COLOR_G(face->packedColorPrimary) / 255.0f,
                      FLTPACKED_COLOR_B(face->packedColorPrimary) / 255.0f,
                      1.0f );

      return COLOR_OK;
    }
    // check for paletted color
    else if( face->primaryColorIndex != 0xffffffff )
    {
      real32 r, g, b, a;

      // NOTE: colorNameIndex check removed because the fltlib
      //       now tries to transparently handle this based on
      //       file version.

      if (fltLookupColor( flt, face->primaryColorIndex, &r, &g, &b, &a ))
      {
        // alpha not yet supported
        setVec(result, r, g, b, 1.0f);

        return COLOR_OK;
      }
      else
      {
        setVec( result, 1.f, 1.f, 1.f, 1.f );

        return COLOR_NOTAVAILABLE;
      }
    }
    else
    {
      setVec( result, 1.f, 1.f, 1.f, 1.f );

      return COLOR_NOTAVAILABLE;
    }
  }
  else
  {
    // make it white
    setVec(result, 1.0f, 1.0f, 1.0f, 1.0f);

    return COLOR_NOCOLOR;
  }
}

//
// Get vertex color.  Returns true if vertex color was actually specified,
// false otherwise.  Sets result to white if no color specified.
//
FLTLoader::ColorLookupEnum
FLTLoader::getVertexColor( FltFile * flt, FltVertex * vert, Vec4f & result )
{
  if( !(vert->flags & FVNO_COLOR ) )
  {
    // why do we have to check m_rgbColor for face, but not for vertex??
    // i suppose we should check for packedColor = 0 here as well, but I
    // have not run into a file where this was important yet.  the spec
    // is silent on this too, of course.
    if( ((vert->flags & FVPACKED_COLOR) /*|| m_rgbColorMode*/ ) /*&&
          vert->packedColor != 0*/ )
    {
      uint32 pcolor = vert->packedColor;

      // color is in ABGR format
      // alpha is supported per-vertex
      setVec( result, FLTPACKED_COLOR_R( pcolor ) / 255.0f,
                      FLTPACKED_COLOR_G( pcolor ) / 255.0f,
                      FLTPACKED_COLOR_B( pcolor ) / 255.0f,
                      FLTPACKED_COLOR_A( pcolor ) / 255.0f );

      return COLOR_OK;
    }
    else if( vert->colorIndex != 0xffffffff )
    {
      float r, g, b, a;

      if ( fltLookupColor( flt, vert->colorIndex, &r, &g, &b, &a ) )
      {
        // alpha is supported for vertex colors, but not in the
        // palette.  You have to read it from the packed color
        // attribute.
        //a = FLTPACKED_COLOR_A( vlist->list[i]->packedColor ) / 
         //     255.0f;
        // the above is apparently not the case for all files!!

        setVec( result, r, g, b, 1.0f );

        return COLOR_OK;
      }
      else
      {
        setVec( result, 1.f, 1.f, 1.f, 1.f );

        return COLOR_NOTAVAILABLE;
      }
    }
    else
    {
      setVec( result, 1.f, 1.f, 1.f, 1.f );

      return COLOR_NOTAVAILABLE;
    }
  }
  else
  {
    setVec( result, 1.f, 1.f, 1.f, 1.f );

    return COLOR_NOCOLOR;
  }
}

bool isTransparent( const EffectDataSharedPtr & material )
{
  bool transparent = false;
  const ParameterGroupDataSharedPtr & parameterGroupData = material->findParameterGroupData( string( "standardMaterialParameters" ) );
  if ( parameterGroupData )
  {
    const dp::fx::SmartParameterGroupSpec & pgs = parameterGroupData->getParameterGroupSpec();
    transparent =   ( parameterGroupData->getParameter<float>( pgs->findParameterSpec( "frontOpacity" ) ) < 1.0f )
                ||  ( parameterGroupData->getParameter<float>( pgs->findParameterSpec( "backOpacity" ) ) < 1.0f );
  }
  return( transparent );
}

ParameterGroupDataSharedPtr getParameterGroupData( EffectDataSharedPtr & effectData, const string & name )
{
  if ( ! effectData )
  {
    effectData = EffectData::create( getStandardMaterialSpec() );
  }
  ParameterGroupDataSharedPtr pgd;
  {
    pgd = effectData->findParameterGroupData( name );
    if ( !pgd )
    {
      const dp::fx::SmartEffectSpec es = effectData->getEffectSpec();
      dp::fx::EffectSpec::iterator it = es->findParameterGroupSpec( name );
      DP_ASSERT( it != es->endParameterGroupSpecs() );
      pgd = ParameterGroupData::create( *it );
      effectData->setParameterGroupData( it, pgd );
    }
  }
  return( pgd );
}

void
FLTLoader::buildFace( GroupSharedPtr const& group,
                       FltNode * node, 
                       FltFile * flt,
                       bool isSubFace )
{
  if (node->type == FLTRECORD_FACE)
  {
    FltFace * face = (FltFace *)node;
    const dp::fx::SmartEffectSpec & es = getStandardMaterialSpec();
    EffectDataSharedPtr material = EffectData::create( es );

    bool hasTransparency = false;
    bool hasMaterialTransparency = false;
    unsigned int i;

    // skip hidden faces
    if( face->miscFlags & FLTFACEMF_HIDDEN )
    {
       return;
    }

    enum 
    {
      UNLIT_FLAT,
      UNLIT_GOURAUD,
      LIT,
      LIT_GOURAUD
    } drawMode = UNLIT_FLAT;

    // first, check face flags to see how this node should be lit
    switch( face->lightMode )
    {
      // FLAT shaded
      case FLTFACELM_FCNOTILLUM:    // only use face color, not lit
        drawMode = UNLIT_FLAT;
        break;

      // Gouraud shaded
      case FLTFACELM_VCNOTILLUM:    // only use vertex colors, not lit
        drawMode = UNLIT_GOURAUD;
        break;

      // "LIT" using face color * material
      case FLTFACELM_FCVN:          // use face colors and vertex normals
        drawMode = LIT;
        break;

      case FLTFACELM_VCVN:          // use vertex colors and vertex normals
        drawMode = LIT_GOURAUD;
        break;
    }

    //
    // check for lighting workaround
    //
    std::string walMode;
    if( getWorkaround( "LIGHTING_MODE", &walMode ) )
    {
      if( walMode == "FLAT" )
      {
        drawMode = UNLIT_FLAT;
      }
      else if( walMode == "GOURAUD" )
      {
        drawMode = UNLIT_GOURAUD;
      }
      else if( walMode == "LIT" )
      {
        drawMode = LIT;
      }
      else if( walMode == "LIT_GOURAUD" )
      {
        drawMode = LIT_GOURAUD;
      }
    }

    group->setName( fltSafeNodeName( node ) );

    Vec4f * fcolor = 0;
    Vec4f theColor( 1.f, 1.f, 1.f, 1.f );

    // 
    // Primary Face Color
    //
    if ( (drawMode == UNLIT_FLAT || drawMode == LIT ) )
    {
      // getFaceColor will return white if no color was available
      // and that is probably the right default
      fcolor = &theColor;
      getFaceColor( flt, face, theColor );

      // set the alpha, from the guide.
      // i guess this should be independent from whether the face
      // actually contains a color or not, since we default it to
      // white anyway
      theColor[3] = 1.0f - ((float)face->transparency / 65535.0f);

      if( face->transparency > 0 )
      {
        hasMaterialTransparency = true; 
      }
    }

    // set face color to white
    bool whiteColors = false;
    if( face->textureWhite && face->texturePatternIndex != -1 )
    {
      fcolor = &theColor;

      setVec(*fcolor, 1.0f, 1.0f, 1.0f, 1.0f);

      whiteColors = true;
    }

    // 
    // Materials
    // 
    if( drawMode == LIT )
    {
      if( face->materialIndex != -1 )
      {
        Vec3f fc( 1.0f, 1.0f, 1.0f );

        if( fcolor )
        {
          fc[0] = (*fcolor)[0];
          fc[1] = (*fcolor)[1];
          fc[2] = (*fcolor)[2];
        }

        // may fail, which is OK
        material = lookupMaterial( flt, face->materialIndex, fc, face->transparency );
      }
    }
    else if( drawMode == LIT_GOURAUD )
    {
      //
      // must lookup material but we don't actually add it to the
      // scene.  later, we pre-multiply the vertex colors by the material
      // and let NVSG turn on COLORMATERIAL mode.
      //
      fcolor = &theColor;
      setVec(*fcolor, 1.f, 1.f, 1.f, 1.f );

      if( face->materialIndex != -1 )
      {
        FltMaterial * fmat = fltLookupMaterial( flt, face->materialIndex );

        // set this as the "face color" and we can use it later

        if( fmat )
        {
          setVec(theColor, fmat->diffuseRed, 
                           fmat->diffuseGreen, 
                           fmat->diffuseBlue,
                           1.0f );
        }
      }
    }

    //
    // check to see if there is material transparency
    //
    hasTransparency = material && isTransparent( material );

    // 
    // Billboard Flags
    //
    if(  face->billboardFlags == FLTFACEBB_AXIALROTATE
         || face->billboardFlags == FLTFACEBB_POINTROTATE
         || face->billboardFlags == FLTFACEBB_FIXEDALPHA )

    {
      hasTransparency = true;
    }

    for (uint32 j=0;j<node->numChildren;j++)
    {
      if( node->child[j]->type == FLTRECORD_FACE )
      {
        // if it is a subface, send it back through
        buildFace( group, node->child[j], flt, true );

        continue;
      }
      else if( node->child[j]->type != FLTRECORD_VERTEXLIST )
      {
        char buf[256];
        sprintf(buf, "Child of face: %s", fltSafeNodeName(node) );

        INVOKE_CALLBACK(onUnsupportedToken(0, buf, "not vertexlist or face!"));

        continue;
      }

      FltVertexList * vlist = (FltVertexList *)node->child[j];

      // check for two-sided prims
      if ( face->drawType == FLTFACEDT_DRAWSOLIDNOBACKFACE )
      {
        ParameterGroupDataSharedPtr materialData = getParameterGroupData( material, "standardMaterialParameters" );
        DP_VERIFY( materialData->setParameter( "twoSidedLighting", true ) );
      }

      PrimitiveSharedPtr hGeom;

      // check for lines
      bool isLine = false;
      bool closeLine = false;

      if( face->drawType == FLTFACEDT_DRAWWIREFRAME || 
         face->drawType == FLTFACEDT_DRAWWIREFRAMECLOSE )
      {
        hGeom = Primitive::create( PRIMITIVE_LINE_STRIP );

        isLine = true;

        if( face->drawType == FLTFACEDT_DRAWWIREFRAMECLOSE )
        {
          closeLine = true;
        }
      }
      else if( vlist->numVerts < 3 )
      {
       // skip points and degenerates for the moment
       continue;
      }
      else
      {
        // is a polygon
        hGeom = Primitive::create( PRIMITIVE_TRIANGLES );
      }

      // set this in case we find a bump map and have to generate tangents
      int bumpEffectLayer = -1;
      int bumpTextureLayer = -1;

      VertexAttributeSetSharedPtr hVas = VertexAttributeSet::create();
      //
      // Coords
      //
      std::vector<Vec3f> vertex( vlist->numVerts );
      std::vector<Vec3f> normal( vlist->numVerts );
      std::vector<Vec2f> texcoord( vlist->numVerts );
      std::vector<Vec4f> colors( vlist->numVerts );

      Vec3f fNormal;
      bool fnComputed = false;
        
      for(i=0;i<vlist->numVerts;i++)
      {
        setVec(vertex[i], (float) vlist->list[i]->x,
                          (float) vlist->list[i]->y,
                          (float) vlist->list[i]->z);

        // if lit, extract normals
        if( drawMode == LIT || drawMode == LIT_GOURAUD )
        {
          if( vlist->list[i]->localFlags & FVHAS_NORMAL )
          {
            setVec(normal[i], (float) vlist->list[i]->i,
                              (float) vlist->list[i]->j,
                              (float) vlist->list[i]->k);
          }
          else
          {
            if( fnComputed == false )
            {
              // this may return false, in which case we will use
              // (0,0,1) default - just like Creator.
              if( faceNormal( fNormal, vlist->list, vlist->numVerts ) == false )
              {
                setVec( fNormal, 0.f, 0.f, 1.f );
              }

              fnComputed = true;
            }

            normal[i] = fNormal;
          }
        }

        // these may not be used
        setVec( texcoord[i], vlist->list[i]->u, vlist->list[i]->v );

        // extract vertex colors
        if( drawMode == UNLIT_GOURAUD || drawMode == LIT_GOURAUD )
        {
          if( whiteColors )
          {
            setVec( colors[i], 1.f, 1.f, 1.f, 1.f );
          }
          else
          {
            ColorLookupEnum result = getVertexColor( flt, vlist->list[i],
                                                          colors[i] );

            switch( result )
            {
              case COLOR_OK:
                break;
              case COLOR_NOCOLOR:
                // if no vertex color, fall back to face color, if available
                getFaceColor( flt, face, colors[i] );
                break;
              case COLOR_NOTAVAILABLE:
                setVec( colors[i], 1.f, 1.f, 1.f, 1.f );
                break;
            }
          }

          //
          // If we are lit_gouraud then we need to multiply each 
          // vertex color by the material color (if available) to 
          // achive the correct runtime color.
          //
          if( drawMode == LIT_GOURAUD && fcolor )
          {
            setVec( colors[i], colors[i][0] * (*fcolor)[0],
                                colors[i][1] * (*fcolor)[1],
                                colors[i][2] * (*fcolor)[2],
                                colors[i][3] * (*fcolor)[3] );
          }
        }
        // This may be sub-optimal since we use a vertex color as a face
        // color
        else if( (drawMode == UNLIT_FLAT || 
                  (drawMode == LIT && material == 0 )) && fcolor &&
                  !whiteColors )
        {
          setVec( colors[i], (*fcolor)[0], (*fcolor)[1], (*fcolor)[2],
                              (*fcolor)[3] );
        }
        else
        {
          // fallback of white
          setVec( colors[i], 1.0f, 1.0f, 1.0f, 1.0f );
        }
      }

      if( drawMode == LIT || drawMode == LIT_GOURAUD )
      {
        hVas->setNormals(&normal[0], vlist->numVerts );
      }

      if( drawMode == UNLIT_GOURAUD || drawMode == LIT_GOURAUD )
      {
        hVas->setColors(&colors[0], vlist->numVerts );
      }
        
      if( drawMode == UNLIT_FLAT || (drawMode == LIT && material == 0) )
      {
        // this is the only way to do it in NVSG - there is no face color.
        hVas->setColors(&colors[0], vlist->numVerts );
      }

      // set the verts
      hVas->setVertices(&vertex[0], vlist->numVerts);

      std::vector<unsigned int> indexList;
      for(unsigned int v=0;v<vertex.size();v++)
      {
        indexList.push_back( v );
      }

      //
      // Create face list - triangulate if necessary
      //
      if( !isLine )
      {
        // remove redundant verts and ignore degenerate tris
        if( removeDegenerates( indexList, vertex ) )
        {
          INVOKE_CALLBACK(onDegenerateGeometry(0, fltSafeNodeName( node ) ));
          continue;
        }

        // assume quads can be triangulated
        if ( 4 < indexList.size() )
        {
          SimpleTesselator st;

          if( ! st.tesselate( vertex, indexList ) )
          {
            INVOKE_CALLBACK(onDegenerateGeometry(1, std::string( "TESS: " )
                                  + std::string( fltSafeNodeName( node ) ) ));
            continue;
          }
        }
        else if ( 4 == indexList.size() )
        {
          unsigned int i3 = indexList[3];
          indexList[3] = indexList[0];
          indexList.push_back( indexList[2] );
          indexList.push_back( i3 );
        }
        IndexSetSharedPtr iset( IndexSet::create() );
        iset->setData( &indexList[0], checked_cast<unsigned int>(indexList.size()) );

        hGeom->setIndexSet( iset );
        hGeom->setVertexAttributeSet( hVas );
      }
      else
      {
        if( closeLine )
        {
          indexList.push_back( indexList[0] );
        }
        IndexSetSharedPtr iset( IndexSet::create() );
        iset->setData( &indexList[0], checked_cast<unsigned int>(indexList.size()) );

        hGeom->setVertexAttributeSet( hVas );
        hGeom->setIndexSet( iset );
      }

      if ( face->texturePatternIndex != -1 )
      {
        TextureEntry tent;
        int texUnit = 0;
        dp::fx::EffectSpec::iterator pgsit = es->findParameterGroupSpec( string( "standardTextureParameters" ) );
        DP_ASSERT( pgsit != es->endParameterGroupSpecs() );

        FltMultiTexture * mtex = (FltMultiTexture *)fltFindFirstAttrNode( node, FLTRECORD_MULTITEXTURE );
        //
        // First, check to see if we have a bump map in one of the 
        // multitexture
        // layers.  If so, we need to reorder the textures to add a 
        // normalizationcubemap as the first unit, then the bump map so
        // that the math will work out to be:
        //
        // NormalizationCube DOT3 BumpMap MODULATE (decal...) 
        //
        // Since the NormalizationCubeMap is added using TEM_REPLACE,
        // this has the unfortunate side effect of ignoring the polygon
        // color, which is sometimes used in OpenFlight files to modulate
        // the decal texture color.  We could work around this by adding
        // more support for texenv_combine extension, if necessary.
        //
        // In addition, if there is a shader, we do not add the 
        // normalization cube map because the shader is not expecting it
        // to be there.
        //
        if( mtex && (face->shaderIndex == -1) )
        {
          for(unsigned int l=0; l<6; l++)
          {
            // layers start at '1'
            if( mtex->mask & FLTMT_HASLAYER(l+1) )
            {
              // array starts at zero
              if( mtex->layer[l].effect == FLTMTEFFECT_BUMP )
              {
                if( lookupTexture( tent, flt, mtex->layer[l].index, -1, LOOKUP_NORMALMAP ))
                {
                  //
                  // add norm.cubemap as first layer
                  //
                  std::vector<Vec3f> texcoordCube( vlist->numVerts );

                  material->setParameterGroupData( pgsit, m_normalizationCubeMap );

                  for( unsigned int t=0; t<vlist->numVerts;t++)
                  {
                    // this should be TBN*LightVec but use this for
                    // now
                    setVec( texcoordCube[t], 0.0f, 0.0f, 1.0f );
                  }

                  hVas->setTexCoords(texUnit, &texcoordCube[0], vlist->numVerts);
                  texUnit++;

                  //
                  // now add bump map
                  // 
                  material->setParameterGroupData( pgsit, tent.textureParameters );

                  // create and offset texcoords if necessary
                  for( unsigned int t=0; t<vlist->numVerts;t++)
                  {
                    setVec( texcoord[t], vlist->list[t]->mtU[l] * tent.uscale + tent.utrans
                                        , vlist->list[t]->mtV[l] * tent.vscale + tent.vtrans );
                  }

                  hVas->setTexCoords( texUnit, &texcoord[0], vlist->numVerts );
                  texUnit++;

                  // only allow 1 bump map
                  break;
                }
              }
            }
            else
            {
              // quit looking as soon as a layer is missing
              break;
            }
          }
        }

        // now, start with the base texture
        if( lookupTexture( tent, flt, face->texturePatternIndex ) )
        {
          material->setParameterGroupData( pgsit, tent.textureParameters );

          // set texcoords in geometry
          if( tent.layer != -1 )
          {
            std::vector< Vec3f > newTC( vlist->numVerts );

            //
            // Allow a combination of atlas and stacked atlas
            //
            for( unsigned int t=0; t<vlist->numVerts;t++)
            {
              setVec( newTC[t], texcoord[t][0] * tent.uscale + tent.utrans,
                                texcoord[t][1] * tent.vscale + tent.vtrans,
                                (float)tent.layer );
            }

            hVas->setTexCoords( texUnit, &newTC[0], vlist->numVerts);
          }
          else if( tent.uscale != 1.f || tent.vscale != 1.f ||
                    tent.utrans != 0.f || tent.vtrans != 0.f )
          {
            for( unsigned int t=0; t<vlist->numVerts;t++)
            {
              setVec( texcoord[t], texcoord[t][0] * tent.uscale+tent.utrans,
                                    texcoord[t][1] * tent.vscale + tent.vtrans );
            }

            hVas->setTexCoords(texUnit, &texcoord[0], vlist->numVerts);
          }
          else
          {
            hVas->setTexCoords(texUnit, &texcoord[0], vlist->numVerts);
          }

          // try workaround for textures that contain transparency
          if( getWorkaround("TRANSPARENCY_FROM_TEXTUREALPHA") )
          {
            const dp::fx::SmartParameterGroupSpec pgs( tent.textureParameters->getParameterGroupSpec() );
            const SamplerSharedPtr & sampler = tent.textureParameters->getParameter<SamplerSharedPtr>( pgs->findParameterSpec( "sampler" ) );
            DP_ASSERT( sampler && sampler->getTexture() );
            const TextureSharedPtr & texture = sampler->getTexture();
            if ( texture.isPtrTo<TextureHost>() )
            {
              switch( texture.staticCast<TextureHost>()->getFormat() )
              {
                case Image::IMG_RGBA:
                case Image::IMG_BGRA:
                case Image::IMG_LUMINANCE_ALPHA:
                  hasTransparency = true;
                  break;
              }
            }
          }

          texUnit++;
        }

        // see if we have a detail texture
        if ( face->detailTexturePatternIndex != -1 )
        {
          if ( lookupTexture( tent, flt, face->detailTexturePatternIndex, -1, LOOKUP_DETAIL ) )
          {
            material->setParameterGroupData( pgsit, tent.textureParameters );

            // u and v scale are read from the attr file

            // make new texcoords.  detail texture coords are based on
            // decal texture coords.
            for( unsigned int t=0; t<vlist->numVerts;t++)
            {
              // we don't add an offset for detail textures, as they 
              // can not currently appear in the atlas
              setVec( texcoord[t],texcoord[t][0] * tent.uscale,
                                  texcoord[t][1] * tent.vscale );
            }

            // set texcoords in geometry
            hVas->setTexCoords(texUnit, &texcoord[0], vlist->numVerts);
            texUnit++;
          }
        }

        //
        // Check all multitexture layers
        //
        if( mtex )
        {
          for(unsigned int l=0; l<6; l++)
          {
            bool result = false;

            // layers start at '1'
            if( mtex->mask & FLTMT_HASLAYER(l+1) )
            {
              // array starts at zero
              if( mtex->layer[l].effect == FLTMTEFFECT_BUMP )
              {
                if( face->shaderIndex != -1 )
                {
                  // we add it to the proper unit if there was a shader
                  result = lookupTexture( tent, flt, mtex->layer[l].index, -1, LOOKUP_NORMALMAP );

                  bumpEffectLayer = mtex->layer[l].data;
                  bumpTextureLayer = texUnit;

                  if( bumpEffectLayer == 0 )
                  {
                    bumpEffectLayer = 7;
                  }
                }
                else
                {
                  // continue, as we already added the bump map above
                  continue;
                }
              }
              else
              {
                result = lookupTexture( tent, flt, mtex->layer[l].index, mtex->layer[l].mapping );
              }

              if( result )
              {
                material->setParameterGroupData( pgsit, tent.textureParameters );

                // take into account texcoord offsets if necessary
                for( unsigned int t=0; t<vlist->numVerts;t++)
                {
                  setVec( texcoord[t], vlist->list[t]->mtU[l] * tent.uscale + tent.utrans
                                      , vlist->list[t]->mtV[l] * tent.vscale + tent.vtrans );
                }

                hVas->setTexCoords(texUnit, &texcoord[0], vlist->numVerts);
                texUnit++;
              }
            }
            else
            {
              // quit looking as soon as a layer is missing
              // perhaps with shaders we should not do this?
              break;
            }
          }
        }
      }

      if ( isLine && ( face->lineStyleIndex != 0xff ) )
      {
        FltLineStyle * ls = fltLookupLineStyle( flt, face->lineStyleIndex );
        if ( ls )
        {
          {
            const dp::fx::SmartEffectSpec & es = material->getEffectSpec();
            dp::fx::EffectSpec::iterator pgsit = es->findParameterGroupSpec( string( "standardGeometryParameters" ) );
            DP_ASSERT( pgsit != es->endParameterGroupSpecs() );
            ParameterGroupDataSharedPtr geometryParameters = ParameterGroupData::create( *pgsit );
            DP_VERIFY( geometryParameters->setParameter( "lineWidth", (float)ls->lineWidth ) );
          }
          DP_ASSERT( hGeom && ( hGeom->getPrimitiveType() == PRIMITIVE_LINE_STRIP ) );

          ParameterGroupDataSharedPtr materialData = getParameterGroupData( material, "standardMaterialParameters" );
          DP_VERIFY( materialData->setParameter( "lineStipplePattern", ls->patternMask ) );
        }
      }

      bool transparent = hasMaterialTransparency || getWorkaround( "BLENDED_TRANSPARENCY" );
      material->setTransparent( transparent );

      if ( hasTransparency && !transparent )
      {
        ParameterGroupDataSharedPtr materialData = getParameterGroupData( material, "standardMaterialParameters" );
        {
          const dp::fx::SmartParameterGroupSpec & pgs = materialData->getParameterGroupSpec();
          dp::fx::ParameterGroupSpec::iterator psit = pgs->findParameterSpec( "alphaFunction" );
          DP_ASSERT( psit != pgs->endParameterSpecs() );
          DP_ASSERT( psit->first.getEnumSpec() );
          materialData->setParameter( psit, psit->first.getEnumSpec()->getValue( "AF_GREATER" ) );
          DP_VERIFY( materialData->setParameter( "alphaThreshold", 0.8f ) );
        }
      }

      // finally, check the lighting mode
      if( drawMode == UNLIT_FLAT || drawMode == UNLIT_GOURAUD )
      {
        // if it is set to unlit - force lighting off
        ParameterGroupDataSharedPtr materialData = getParameterGroupData( material, "standardMaterialParameters" );
        DP_VERIFY( materialData->setParameter( "lightingEnabled", false ) );
      }

      // now, search through current materials
      typedef multimap<HashKey,EffectDataSharedPtr>::const_iterator I;
      I it;
      HashKey hashKey;
      bool found = false;
      {
        hashKey = material->getHashKey();

        pair<I,I> itp = m_materialCache.equal_range( hashKey );
        for ( it = itp.first ; it != itp.second ; ++it )
        {
          if ( material->isEquivalent( it->second ) )
          {
            found = true;
            break;
          }
        }
      }
      if ( found )
      {
        material = it->second;
      }
      else
      {
        m_materialCache.insert( make_pair( hashKey, material ) );
      }

      GeoNodeSharedPtr geoNode = GeoNode::create();
      geoNode->setMaterialEffect( material );
      geoNode->setPrimitive( hGeom );

      if( bumpEffectLayer != -1 )
      {
        // we really don't want binormals, but oh well
        geoNode->generateTangentSpace( bumpTextureLayer  + VertexAttributeSet::NVSG_TEXCOORD0
                                , bumpEffectLayer   + VertexAttributeSet::NVSG_TEXCOORD0
                                , bumpEffectLayer-1 + VertexAttributeSet::NVSG_TEXCOORD0 );
      }
      group->addChild( geoNode );
    }
  }
}

void
FLTLoader::buildObject( GroupSharedPtr const& parent,
                        FltNode * node, 
                        FltFile * flt,
                        int level )
{
  DP_ASSERT( parent != 0 );

  //
  // Objects can only have faces as their children.  So we just need
  // to know if they have transforms or are billboards and go from there.
  //

  unsigned int numStateSets = 0;
  bool childrenEliminated = false;
  GroupSharedPtr group = Group::create();
  group->setName( std::string("GeometryIn: ") + std::string(fltSafeNodeName( node )) );

  // add the annotation to the Gruop if it is present
  addAnnotation( group, node );

  // check for preserve at runtime flag.  if set, mark as dynamic so it
  // is not optimized away
  if( ((FltObject*)node)->flags & FLTOBJECT_PRESERVE )
  {
    group->addHints( Object::DP_SG_HINT_DYNAMIC );
  }

  for(unsigned int i=0;i<node->numChildren;i++)
  {
    if( collect( node->child[i] ) )
    {
      buildFace( group, node->child[i], flt );
    }
    else
    {
      // had a transform or was a billboard
      buildScene( parent, node->child[i], flt, level );

      // set the flag to indicate that some children have been eliminated
      // from the GeoNode.  This is important later to know whether to
      // keep the GeoNode or eliminate it.
      childrenEliminated = true;
    }
  }

  numStateSets = group->getNumberOfChildren();

  // optimize it
  if( numStateSets )
  {
    parent->addChild( group );
    optimizeGeometry( group );
  }
  else
  {
    //
    // If they have the 'preserve' flag set we need to save this empty
    // object.  OR if it is possibly an empty child of a switch or an 
    // animation we need to save it as well.  But if the Object did contain
    // children, AND they were all eliminated because they were added
    // to the parent node (in the case of billboards, for instance) then
    // it is safe to eliminate this node because the children will take
    // its place.  This could be wrong later if we have an object that
    // contains a bunch of billboard nodes that is under an animation,
    // but hopefully we do not encounter this.
    //

    FltObject * object = (FltObject *)node;
    bool add = false;

    if( object->flags & FLTOBJECT_PRESERVE )
    {
      add = true;
    }
    else if( node->parent && !childrenEliminated )
    {
      // must parse if parent was switch
      if( node->parent->type == FLTRECORD_SWITCH )
      {
        add = true;
      }
      // must parse if parent was an animated group
      else if( node->parent->type == FLTRECORD_GROUP )
      {
        FltGroup * pgroup = (FltGroup *)node->parent;
        if (pgroup->flags & ( FLTGROUP_FORWARD_ANIM | FLTGROUP_BACKWARD_ANIM ))
        {
          add = true;
        }
      }
    }

    if( add )
    {
      parent->addChild( group );
    }
  }
}

inline bool
FLTLoader::collect( FltNode * node )
{
  DP_ASSERT( node != 0 );

  //
  // We must skip faces and groups that have transformations because
  // this will hoze things up later.  By definition, billboards are supposed
  // to have a transform attached to them, so this should catch billboards
  // as well.  But we will check for them explicitly, as some modelers
  // apparently do not do this.
  //
  if( node->type == FLTRECORD_FACE )
  {
    FltFace * face = (FltFace *) node;

    // can't have a matrix, or be a billboard
    if( fltFindFirstAttrNode( node, FLTRECORD_MATRIX ) ||
        face->billboardFlags == FLTFACEBB_AXIALROTATE ||
        face->billboardFlags == FLTFACEBB_POINTROTATE )
    {
      return false;
    }
    else
    {
      return true;
    }
  }
  else if( node->type == FLTRECORD_OBJECT &&
        (!fltFindFirstAttrNode( node, FLTRECORD_MATRIX ) &&
         // if they want to preserve this object, we have to not collect it
         !(((FltObject*)node)->flags & FLTOBJECT_PRESERVE) ) )
  {
    // now we must check to see if any of the children are billboards
    // if so, we need to return false, or they won't get
    // added to the scene.
    for(uint32 i=0;i<node->numChildren;i++)
    {
      FltFace * face = (FltFace *) node->child[i];

      // can't have a matrix, or be a billboard
      if( fltFindFirstAttrNode( (FltNode*)face, FLTRECORD_MATRIX ) ||
          face->billboardFlags == FLTFACEBB_AXIALROTATE ||
          face->billboardFlags == FLTFACEBB_POINTROTATE )
      {
        return false;
      }
    }

    // ok, none found
    return true;
  }
  else
  {
    return false;
  }
}

void
FLTLoader::collectDescend( GroupSharedPtr const& parent, 
                            FltNode * node, 
                            FltFile * flt )
{
  if( collect( node ) )
  {
    if( node->type == FLTRECORD_FACE )
    {
      buildFace( parent, node, flt );
    }
    else
    {
      for(unsigned int i=0;i<node->numChildren;i++)
      {
        collectDescend( parent, node->child[i], flt );
      }
    }
  }
}

void
FLTLoader::collectGeometry( GroupSharedPtr const& parent,
                            FltNode * node, 
                            FltFile * flt )
{
  GroupSharedPtr group = Group::create();
  unsigned int numStateSets = 0;

  DP_ASSERT( parent != 0 );

  group->setName( std::string("GeometryIn: ") + std::string(fltSafeNodeName( node )) );

  for(unsigned int i=0;i<node->numChildren;i++)
  {
    collectDescend( group, node->child[i], flt );
  }

  numStateSets = group->getNumberOfChildren();

  // optimize it
  if( numStateSets )
  {
    parent->addChild( group );
    optimizeGeometry( group );
  }
}

void
FLTLoader::optimizeGeometry( const GroupSharedPtr & group )
{
//  SmartPtr< NormalizeTraverser > nt = new NormalizeTraverser;
//  nt->apply( NodeSharedPtr(group)) );

  SmartPtr< dp::sg::algorithm::CombineTraverser > ct = new dp::sg::algorithm::CombineTraverser;
  ct->setCombineTargets( dp::sg::algorithm::CombineTraverser::CT_GEONODE );
  ct->apply( NodeSharedPtr(group) );

  SmartPtr< dp::sg::algorithm::UnifyTraverser > psut = new dp::sg::algorithm::UnifyTraverser;
  psut->setIgnoreNames( true );
  psut->setUnifyTargets( dp::sg::algorithm::UnifyTraverser::UT_PRIMITIVE | 
                         dp::sg::algorithm::UnifyTraverser::UT_VERTEX_ATTRIBUTE_SET |
                         dp::sg::algorithm::UnifyTraverser::UT_VERTICES );
  psut->setEpsilon( 0.00001f );
  psut->apply( NodeSharedPtr(group) );

//  SmartPtr< StrippingTraverser > st = new StrippingTraverser;
//  st->apply( NodeSharedPtr(group) );
}

// Ascending sorting function
struct LODRangeSort
{
  // linux required this version for some reason
  bool operator()(FltLOD* const& lhs, FltLOD* const& rhs) const
  {
    return lhs->switchOutDistance < rhs->switchOutDistance;
  }
};

static void
eulerToQuat( double p, double r, double y, Quatf * result )
{
  *result = Quatf( Vec3f( 0.0f, 0.0f, 1.0f ), (float) degToRad(y) )    // yaw
          * Quatf( Vec3f( 0.0f, 1.0f, 0.0f ), (float) degToRad(r) )    // roll
          * Quatf( Vec3f( 1.0f, 0.0f, 0.0f ), (float) degToRad(p) );   // pitch
}

void 
FLTLoader::buildScene( GroupSharedPtr const& parent,
                       FltNode * node, 
                       FltFile * flt,
                       int level,
                       bool parseAttr )
{

  if (node == NULL)
  {
    return;
  }

  DP_ASSERT( parent != 0 );

#if 0
  FltRecord * rec;
  rec = fltRecordGetDefinition( node->type );
  NVSG_TRACE_OUT_F(( "Parent: %s -> [%s(%d)]:%d : %s\n", 
          (parent->getName().empty())?"Unnamed!!":parent->getName().c_str(), 
                            (rec)?rec->name:"Unknown Node Type",
                            node->type,
                            level, fltSafeNodeName( node ) ));
#endif

  // first check for transform mats
  FltMatrix * matrix = 0;
  FltReplicate * replicate = 0;

  if( parseAttr )
  {
    matrix = (FltMatrix *)fltFindFirstAttrNode( node, FLTRECORD_MATRIX );

    // unfortunate
    if( node->type == FLTRECORD_LIGHTSOURCE )
    {
      // pretty sure we have to ignore mats in light sources because
      // of the way NVSG works.
      matrix = 0;
    }

    replicate = (FltReplicate *)
                        fltFindFirstAttrNode( node, FLTRECORD_REPLICATE );
  }

  if( matrix )
  {
    if( replicate )
    {
      // build the matrix
      FltMatrix * m = matrix;
      Mat44f nvmat( makeArray( m->matrix[0],  m->matrix[1],  m->matrix[2],  m->matrix[3],
                               m->matrix[4],  m->matrix[5],  m->matrix[6],  m->matrix[7],
                               m->matrix[8],  m->matrix[9],  m->matrix[10], m->matrix[11],
                               m->matrix[12], m->matrix[13], m->matrix[14], m->matrix[15] ) );

      //
      // create template node
      // 
      TransformSharedPtr hTop = Transform::create();
      hTop->setName( std::string("Replicate for: ") + 
                        std::string(fltSafeNodeName(node)) );

      // run through scene once
      buildScene( hTop, node, flt, level+1, false );

      // create the clones
      std::vector<TransformSharedPtr> clones( replicate->replications );
      for( int r=0;r<replicate->replications;r++ )
      {
        clones[r] = hTop.clone();

        Mat44f concat( nvmat );

        for( int t=0; t<r; t++ )
        {
          concat *= nvmat;
        }

        Trafo tr;
        tr.setMatrix( concat );

        clones[r]->setTrafo( tr );
      }
     
      //
      // add everyone to toplevel as siblings
      //
      parent->addChild( hTop );
      for( int r=0;r<replicate->replications;r++ )
      {
        parent->addChild( clones[r] );
      }

      return;
    }

    //
    // Not replicate - normal matrix processing
    //
    TransformSharedPtr hTrans = Transform::create();
    hTrans->setName( std::string("Transform for: ") + 
                      std::string(fltSafeNodeName(node)) );

    // add same annotation to this node as to root node
    addAnnotation( hTrans, node );

    //
    // If there was a transform, send this node back through 
    // with a newly created transform as the parent.  This feels
    // like a hack, but was the easiest way to accomplish adding
    // a transform node above this node given the begin/endEdit
    // constructs that need to be present.
    //
    buildScene( hTrans, node, flt, level+1, false );

    //
    // Check to see if there are any children.
    //
    if( hTrans->getNumberOfChildren() == 0 )
    {
#if 0
      NVSG_TRACE_OUT_F(("Hmm: %s is empty, not rooting it.\n", 
                              hTrans->getName().c_str() ));
#endif
      return;
    }

    //
    // now decompose the matrix and send it to the transform node
    // build the matrix transposed.  OpenFlight matrices are row-major.
    //
      
    FltMatrix * m = matrix;
    Mat44f nvmat( makeArray( m->matrix[0], m->matrix[1], m->matrix[2], m->matrix[3],
                              m->matrix[4], m->matrix[5], m->matrix[6], m->matrix[7],
                              m->matrix[8], m->matrix[9], m->matrix[10], m->matrix[11],
                              m->matrix[12], m->matrix[13], m->matrix[14], m->matrix[15] ) );

    Trafo t;
    t.setMatrix( nvmat );

    hTrans->setTrafo( t );

    if( parent )
    {
      parent->addChild( hTrans );
    }

    return;
  }

  //
  // BEGIN SPECIFIC NODE PARSING
  //

  if( node->type == FLTRECORD_EXTERNALREFERENCE )
  {
    FltExternalReference * fext = (FltExternalReference *)node;
    char fname[MAX_PATHLEN];

    //
    // Lookup file
    //
    if( fltFindFile( flt, fext->path, fname ) )
    {
      FltFile * extRef = 0;

      // check if it is in the cache
      GroupSharedPtr gh = lookupExtRef( fname );

      if( gh )
      {
        parent->addChild( gh.clone() );
        return;
      }

      // open file
      extRef = fltOpen( fname );

      if( extRef )
      {
        // set fileID first
        fltSetFileID( extRef, nextFileID() );

        // copy search paths first, so that the file's dir will show up first
        // in the list
        fltCopySearchPaths( extRef, flt );

        // read, parse
        fltParse( extRef, 0 );
        fltClose( extRef );

        GroupSharedPtr hDummyGroup = Group::create();
        hDummyGroup->setName("ExtRefGrp");

        //
        // add everything to the dummy group
        //
        buildScene( hDummyGroup, (FltNode*)extRef->header, extRef, level+1 );

        //
        // If only 1 child, remove the dummy group as it is
        // not needed.
        //
        GroupSharedPtr const& gh = hDummyGroup->beginChildren()->staticCast<Group>();

        if( (hDummyGroup->getNumberOfChildren() == 1) && gh )
        {
          // create the ext ref, so it will be cached for next time
          createExtRef( fname, gh );

          parent->addChild( gh );

          // remove it and the rest will be freed
          hDummyGroup->removeChild( gh );
        }
        else
        {
          createExtRef( fname, hDummyGroup );

          parent->addChild( hDummyGroup );
        }

        // free flt file
        fltFileFree( extRef );
      }
      else
      {
        // report error?
        //NVSG_TRACE_OUT_F(("Unable to find external reference: %s\n", fname ));

        INVOKE_CALLBACK(onFileNotFound( fname ));
      }
    }
    else
    {
      INVOKE_CALLBACK(onFileNotFound( fext->path ));
    }

    return;
  }

  else if (node->type == FLTRECORD_INSTANCEDEFINITION)
  {
    FltInstanceDefinition * idef = (FltInstanceDefinition *)node;

    // create the instance
#if 0
    NVSG_TRACE_OUT_F(("Creating instance: %s (%04x)\n", fltSafeNodeName( node ),
                                              idef->instance ));
#endif
    createInstance( flt, node );

    return;
  }

  else if (node->type == FLTRECORD_INSTANCEREFERENCE)
  {
    FltInstanceReference * iref = (FltInstanceReference *)node;
    GroupSharedPtr hGroup = lookupInstance( flt, iref->instance );

    if( hGroup )
    {
#if 0
      NVSG_TRACE_OUT_F(("Using instance: %04x\n", iref->instance ));
#endif
      parent->addChild( hGroup );
    }
    else
    {
      // log error
#if 0
      NVSG_TRACE_OUT_F(("Unable to find instance reference: %04x\n", 
                                                          iref->instance ));
#endif
    }

    return;
  }

  else if ( node->type == FLTRECORD_LIGHTSOURCE )
  {
    FltLightSource * light = (FltLightSource *)node;
    FltLightSourcePaletteEntry * lpe = 
             fltLookupLightSource( flt, light->paletteIndex );

    if( (light->flags & (FLTRECORD_LSENABLED | FLTRECORD_LSEXPORT))
        != (FLTRECORD_LSENABLED | FLTRECORD_LSEXPORT ) )
    {
      // if not enabled and/or not exported, don't add it
      return;
    }

    bool directionalLight = false;

    if( lpe )
    {
      // may not be needed if not positional, but thats OK
      Vec3f position( (float)light->position[0], 
                      (float)light->position[1],
                      (float)light->position[2] );

      Mat33f lightMatIT;

      if( FltMatrix * lmat = 
        (FltMatrix *)fltFindFirstAttrNode( node, FLTRECORD_MATRIX ) )
      {
        Mat44f nvmat( makeArray( lmat->matrix[0],  lmat->matrix[1],  lmat->matrix[2],  lmat->matrix[3],
                                 lmat->matrix[4],  lmat->matrix[5],  lmat->matrix[6],  lmat->matrix[7],
                                 lmat->matrix[8],  lmat->matrix[9],  lmat->matrix[10], lmat->matrix[11],
                                 lmat->matrix[12], lmat->matrix[13], lmat->matrix[14], lmat->matrix[15] ) );

        // transform position
        Vec4f v4f( position, 1.0f );
        v4f = v4f * nvmat;

        setVec( position, v4f[0], v4f[1], v4f[2] );

        //position = position * nvmat;

        Mat33f nvrot( makeArray( lmat->matrix[0], lmat->matrix[1], lmat->matrix[2],
                                 lmat->matrix[4], lmat->matrix[5], lmat->matrix[6],
                                 lmat->matrix[8], lmat->matrix[9], lmat->matrix[10] ) );

        nvrot.invert();
        lightMatIT = ~nvrot;
      }
      else
      {
        setIdentity( lightMatIT );
      }

      LightSourceSharedPtr ls;

      switch( lpe->type )
      {
        case FLTLIGHT_INFINITE:
        {
          // no corona for directional lights
          directionalLight = true;

          // set direction from yaw/pitch
          Vec3f direction;
          float dtor = PI/180.0f;

          direction[0] = - sin(dtor * light->yaw) * cos( dtor * light->pitch );
          direction[1] =   cos(dtor * light->yaw) * cos( dtor * light->pitch );
          direction[2] =   sin(dtor * light->pitch);

          direction = lightMatIT * direction;

          ls = createStandardDirectedLight( direction
                                          , Vec3f( lpe->ambient[0], lpe->ambient[1], lpe->ambient[2] )
                                          , Vec3f( lpe->diffuse[0], lpe->diffuse[1], lpe->diffuse[2] )
                                          , Vec3f( lpe->specular[0], lpe->specular[1], lpe->specular[2] ) );
        }
        break;

        case FLTLIGHT_SPOT:
        {
          // set direction from yaw/pitch
          Vec3f direction;
          float dtor = PI/180.0f;

          direction[0] = - sin(dtor * light->yaw) * cos( dtor * light->pitch );
          direction[1] =   cos(dtor * light->yaw) * cos( dtor * light->pitch );
          direction[2] =   sin(dtor * light->pitch);

          direction = lightMatIT * direction;

          ls = createStandardSpotLight( position, direction
                                      , Vec3f( lpe->ambient[0], lpe->ambient[1], lpe->ambient[2] )
                                      , Vec3f( lpe->diffuse[0], lpe->diffuse[1], lpe->diffuse[2] )
                                      , Vec3f( lpe->specular[0], lpe->specular[1], lpe->specular[2] )
                                      , makeArray( lpe->attenC, lpe->attenL, lpe->attenQ )
                                      , lpe->spotExponent, lpe->spotCutoff );
        }
        break;

        case FLTLIGHT_LOCAL:
        {
          ls = createStandardPointLight( position
                                       , Vec3f( lpe->ambient[0], lpe->ambient[1], lpe->ambient[2] )
                                       , Vec3f( lpe->diffuse[0], lpe->diffuse[1], lpe->diffuse[2] )
                                       , Vec3f( lpe->specular[0], lpe->specular[1], lpe->specular[2] )
                                       , makeArray( lpe->attenC, lpe->attenL, lpe->attenQ ) );
        }
        break;
      }

      if( ls )
      {
        ls->setName( fltSafeNodeName( node ) );

        parent->addChild( ls );
        // disregard the FLTRECORD_LSGLOBAL flag, lights are global
      }
    }
    else
    {
#if 0
       "Unable to find light source: " <<
                               fltSafeNodeName(node) <<
                               " (" << light->paletteIndex << ") " << 
                               " in palette!" << std::endl;
#endif
    }

    return;
  }

  else if ( node->type == FLTRECORD_LIGHTPOINTSYSTEM )
  {
    FltLightPointSystem * flps = (FltLightPointSystem *)node;

    // check to see whether this is enabled or not.  If not,
    // do not bother building children
    if( flps->flags & FLTLPS_FLAGS_ENABLED )
    {
      // children should only be lightpoints
      for (unsigned int i=0;i<node->numChildren;i++)
      {
        buildScene(parent, node->child[i], flt, level+1);
      }
    }

    return;
  }

  else if ( node->type == FLTRECORD_LIGHTPOINT || 
            node->type == FLTRECORD_INDEXEDLIGHTPOINT )
  {
    FltLightPoint * lp = (FltLightPoint*)node;

    std::vector<Vec4f> colors;
    std::vector<Vec3f> positions;
    std::vector<Vec3f> normals;

    unsigned int indexCounter = 0;

    // only verts can be children of light points
    for( unsigned int i=0;i<node->numChildren;i++)
    {
      if( node->child[i]->type != FLTRECORD_VERTEXLIST )
      {
        char buf[256];
        sprintf(buf, "Child of lightPoint: %s", fltSafeNodeName(node) );

        INVOKE_CALLBACK(onUnsupportedToken(0, buf, "not vertexlist!"));
        continue;
      }

      FltVertexList * vlist = (FltVertexList *)node->child[i];

      //
      // Extract their colors
      //
      for(unsigned int j=0;j<vlist->numVerts;j++)
      {
        Vec4f color;
        Vec3f position;
        Vec3f normal;

        getVertexColor( flt, vlist->list[j], color );

        setVec( position, (float)vlist->list[j]->x, 
                          (float)vlist->list[j]->y,
                          (float)vlist->list[j]->z );

        if( vlist->list[j]->localFlags & FVHAS_NORMAL )
        {
          setVec( normal, (float)vlist->list[j]->i, 
                          (float)vlist->list[j]->j,
                          (float)vlist->list[j]->k );
        }
        else
        {
          setVec( normal, 0.0f, 0.0f, 1.0f );
        }

        // add these to the list
        colors.push_back( color );
        positions.push_back( position );
        normals.push_back( normal );
      }
    }

    if( colors.size() )
    {
      if ( ! m_pointGeometryEffect )
      {
        m_pointGeometryEffect = createStandardMaterialData();
        {
          const ParameterGroupDataSharedPtr & pgd = m_pointGeometryEffect->findParameterGroupData( string( "standardGeometryParameters" ) );
          DP_ASSERT( pgd );
          DP_VERIFY( pgd->setParameter( "pointSize", 5.0f ) );
        }
      }

      //
      // Write point and vert data
      //
      VertexAttributeSetSharedPtr hVas = VertexAttributeSet::create();
      hVas->setColors(&colors[0], checked_cast<unsigned int>(colors.size()));
      hVas->setVertices(&positions[0], checked_cast<unsigned int>(positions.size()));
      hVas->setNormals(&normals[0], checked_cast<unsigned int>(normals.size()));

      PrimitiveSharedPtr hPoints = Primitive::create( PRIMITIVE_POINTS );
      hPoints->setVertexAttributeSet( hVas );

      //
      // Finally, write it all out to the geode
      //
      GeoNodeSharedPtr hGeode = GeoNode::create();
      hGeode->setName( fltSafeNodeName( node ) );
      hGeode->setPrimitive( hPoints );
      hGeode->setMaterialEffect( m_pointGeometryEffect );

      // add it to parent
      parent->addChild( hGeode );
    }

    return;
  }

  else if (node->type == FLTRECORD_MESH)
  {
    INVOKE_CALLBACK(onInvalidValue(0, fltSafeNodeName(node), 
                    "MESH Records not yet supported.",0.0f));
    return;
  }

  else if (node->type == FLTRECORD_FACE)
  {
    FltFace * face = (FltFace *)node;

    GroupSharedPtr group = Group::create();
    group->setName( fltSafeNodeName( node ) );
    buildFace( group, node, flt );

    // set up billboard if necessary
    if( face->billboardFlags == FLTFACEBB_AXIALROTATE ||
        face->billboardFlags == FLTFACEBB_POINTROTATE )
    {
      BillboardSharedPtr hBB = Billboard::create();
      hBB->setName( fltSafeNodeName( node ) );

      if( face->billboardFlags == FLTFACEBB_AXIALROTATE )
      {
        // openflight has Z 'up'
        hBB->setAlignment( Billboard::BA_AXIS );
        hBB->setRotationAxis( Vec3f( 0.0f, 0.0f, 1.0f ) );
      }
      else if( face->billboardFlags == FLTFACEBB_POINTROTATE )
      {
        hBB->setAlignment( Billboard::BA_VIEWER );
      }

      hBB->addChild( group );

      addAnnotation( hBB, node );

      parent->addChild( hBB );
    }
    else
      parent->addChild( group );

    return;
  }

  else if (node->type == FLTRECORD_LOD)
  {
    FltLOD * fltLOD = (FltLOD *)node;
    Vec3f v3f;

    // if this flag is set, then we have already combined 
    // this node with the sibling node to the left in the hierarchy.
    if( fltLOD->flags & FLTLOD_USEPREVSLANT )
    {
      // only bail out if this is not the leftmost node,
      // otherwise we haven't seen it yet
      if( node->prev && node->prev->type == FLTRECORD_LOD )
        return;
    }

    // start with some preprocessing of hierarchy to see if we need
    // to combine siblings
    std::vector<FltLOD *> lods;

    // add current
    lods.push_back( fltLOD );

    FltNode * n = node->next;
    FltNode * nLast = node;
    const float LOD_EPSILON = 0.5f;
    while( n && (n->type == FLTRECORD_LOD) )
    {
      bool quit = false;

      if( ((FltLOD*)n)->flags & FLTLOD_USEPREVSLANT )
      {
        lods.push_back( (FltLOD*)n );
      }
      else
      {
        FltLOD * lod = (FltLOD*)n;
        FltLOD * lodLast = (FltLOD*)nLast;
        
        //
        // check to see if the LOD centers are "about" equal, AND
        // that the switch in distance is similar to the switch out
        // distance of the previous LOD.  If these are true, we can
        // combine them together.
        //
        if( ( lodLast->switchOutDistance == lod->switchInDistance ||
              lodLast->switchInDistance  == lod->switchOutDistance ) &&
            ( ( (lodLast->centerX - lod->centerX) < LOD_EPSILON ) &&
              ( (lodLast->centerY - lod->centerY) < LOD_EPSILON ) &&
              ( (lodLast->centerZ - lod->centerZ) < LOD_EPSILON ) ) )
        {
          // set this flag so we don't process this node again
          lod->flags |= FLTLOD_USEPREVSLANT;

          lods.push_back( (FltLOD*)n );
        }
        else
        {
          quit = true;
        }
      }

      if( quit )
      {
        break;
      }

      nLast = n;
      n = n->next;
    }

    // sort ranges
    std::sort( lods.begin(), lods.end(), LODRangeSort() );
    std::vector<float> ranges( lods.size() );

    for(unsigned int i=0;i<lods.size();i++)
    {
      ranges[i] = (float)lods[i]->switchInDistance;
    }

    // reset the last one to be outside distance
    //ranges[ lods.size() -1 ] = (float) lods[ lods.size() -1 ]->switchInDistance;

    LODSharedPtr hLod = LOD::create();
    hLod->setName( "ParentLODNode" );

    //
    // We have to insert "dummy" groups because of differences in FLT LODs 
    // vs NVSG LODs.
    //
    GroupSharedPtr hDummyGroup = Group::create();
    hDummyGroup->setName("EmptyLODGroup");

    // we need an initial dummy group if we don't switch out at zero
    if( lods[0]->switchOutDistance != 0 )
    {
      hLod->addChild( hDummyGroup );
      ranges.insert( ranges.begin(), (float)lods[0]->switchOutDistance );
    }

    // traverse children
    for( unsigned int i=0;i<lods.size();i++)
    {
      FltNode * lnode = (FltNode*) lods[i];

      GroupSharedPtr hGroup = Group::create();
      hGroup->setName( fltSafeNodeName(lnode) );
      addAnnotation( hGroup, lnode );

      collectGeometry( hGroup, lnode, flt );

      for (unsigned int i=0;i<lnode->numChildren;i++)
      {
        if( !collect( lnode->child[i] ) )
        {
          buildScene( hGroup, lnode->child[i], flt, level+1);
        }
      }

      // add the actual geometry
      hLod->addChild( hGroup );
    }

    // add a dummy group on the end
    hLod->addChild( hDummyGroup );

    //
    // Take center value from initial lod
    //
    if( fltFinite64( fltLOD->centerX ) &&
          fltFinite64( fltLOD->centerY ) &&
          fltFinite64( fltLOD->centerZ ) )
    {
      setVec( v3f, (float)fltLOD->centerX, 
                    (float)fltLOD->centerY, 
                    (float)fltLOD->centerZ );
    }
    else 
    {
      // warn that center is messed in flt file
      // compute new bound and set it correctly
      char buf[256];

      sprintf( buf, "LOD \"%s\"", fltSafeNodeName(node) );
      INVOKE_CALLBACK(onInvalidValue(0, buf, "Center is Infinite.",0.0f));

      Sphere3f bound = hLod->getBoundingSphere();

      v3f = bound.getCenter();
    }

    hLod->setCenter(v3f);

    hLod->setRanges( (float*)&ranges[0], checked_cast<unsigned int>(ranges.size()) );

    if (parent)
    {
#if 0
      NVSG_TRACE_OUT_F(( "LOD %s (%f %f %f) %f->%f added to %s\n",
                                          fltSafeNodeName( node ), 
                                          v3f[0], v3f[1], v3f[2],
                                          fltLOD->switchOutDistance,
                                          fltLOD->switchInDistance,
                                          parent->getName().c_str() ));
#endif

      parent->addChild( hLod );
    }

    return;
  }

  else if (node->type == FLTRECORD_SWITCH)
  {
    FltSwitch * fltSwitch = (FltSwitch *)node;
    SwitchSharedPtr hSwitch = Switch::create();

    // XXX: OpenFlight uses masks to select which children are active.

    // set switch info
    hSwitch->setName( fltSafeNodeName( node ) );
    addAnnotation( hSwitch, node );

    // tell optimizers to retain this switch
    hSwitch->addHints( Object::DP_SG_HINT_DYNAMIC );

    // build scene for children
    for (unsigned int i=0;i<node->numChildren;i++)
    {
      buildScene( hSwitch, node->child[i], flt, level+1 );
    }

    if ( !hSwitch->getNumberOfChildren() )
    {
      // there actually was not a single child added to the switch
      return;
      // things that happen on return:
      // (1) switch's write-lock will be released as SwitchLock runs out of scope
      // (2) switch object will be auto-deleted when the SmartPtr runs out of scope
      // (3) nothing will be added to the current parent
    }

    // now create masks
    for(unsigned int i=0;i<fltSwitch->numMasks;i++)
    {
      uint32 swMask = fltSwitch->masks[ i ];
      Switch::SwitchMask mask;

      // does not take into account masks > 31 bits
      for(unsigned int j=0;j<31;j++)
      {
        if( j >= node->numChildren )
        {
          break;
        }

        if( swMask & (1<<j) )
        {
          mask.insert( j );
        }
      }

      hSwitch->addMask( i, mask );
    }

    hSwitch->setActiveMaskKey( fltSwitch->currentMask );

    if (parent)
      parent->addChild( hSwitch );

    return;
  }

  else if (node->type == FLTRECORD_DOF)
  {
    FltDOF * fltDOF = (FltDOF *)node;

    // From OpenFlight16.4.pdf:
    //
    // Degree of Freedom Record
    // The degree of freedom (DOF) record is the primary record of the DOF node. The DOF node specifies a local
    // coordinate system and the range allowed for translation, rotation, and scale with respect to that
    // coordinate system.
    // The DOF record can be viewed as a series of applied transformations consisting of the following elements:
    //    [PTTTRRRSSSP]
    // where P denotes put, T denotes translate, R denotes rotate, and S denotes scale.
    // It is important to understand the order in which these transformations are applied to the geometry. A
    // pre-multiplication is assumed, so the sequence of transformations must be read from right to left, in order
    // to describe its effect on the geometry contained below the DOF. In this manner, a DOF is interpreted as a
    // Put followed by three Scales, three Rotates, three Translates, and a Put.
    // Taking the transformations in right to left order, they represent:
    // 1.A Put (3 point to 3 point transformation). This matrix brings the DOF coordinate system to the world
    //   origin, with its x-axis aligned along the world x-axis and its y-axis in the world x-y plane. Testing
    //   against the DOF's constraints is performed in this standard position. This matrix is therefore the
    //   inverse of the last (See Step 11 below).
    // 2.Scale in x.
    // 3.Scale in y.
    // 4.Scale in z.
    // 5.Rotation about z (yaw).
    // 6.Rotation about y (roll).
    // 7.Rotation about x (pitch).
    // 8.Translation in x.
    // 9.Translation in y.
    // 10.Translation in z.
    // 11.A final Put. This matrix moves the DOF coordinate system back to its original position in the scene.
    // The DOF record specifies the minimum, maximum, and current values for each transformation. Only the current
    // value affects the actual transformation applied to the geometry. The increment value specifies discrete
    // allowable values within the range of legal values represented by the DOF.

    Vec3f origin( (float) fltDOF->localOriginX, (float) fltDOF->localOriginY, (float) fltDOF->localOriginZ );
    Vec3f xAxisPoint( (float) fltDOF->localPointX, (float) fltDOF->localPointY, (float) fltDOF->localPointZ );
    Vec3f xyPlanePoint( (float) fltDOF->localPlanePointX, (float) fltDOF->localPlanePointY, (float) fltDOF->localPlanePointZ );

    Vec3f xAxis = xAxisPoint - origin;
    if ( isNull(xAxis) )
    {
      xAxis = Vec3f(1.0f, 0.0f, 0.0f);
      INVOKE_CALLBACK(onInvalidValue(0, fltSafeNodeName(node), "DOF contains invalid local origin or x-axis point.", 0.0f));
    }
    xAxis.normalize();

    Vec3f yAxis = xyPlanePoint - origin;
    if ( isNull(yAxis) )
    {
      yAxis = Vec3f(0.0f, 1.0f, 0.0f);
      INVOKE_CALLBACK(onInvalidValue(0, fltSafeNodeName(node), "DOF contains invalid local origin or xy-plane point.", 0.0f));
    }
    yAxis.normalize();

    Vec3f zAxis = xAxis ^ yAxis;
    if ( isNull(zAxis) ) // Could happen if xAxis and yAxis are collinear which is not covered by the above checks.
    {
      zAxis = Vec3f(0.0f, 0.0f, 1.0f);
      INVOKE_CALLBACK(onInvalidValue(0, fltSafeNodeName(node), "DOF contains invalid local origin, x-axis, or xy-plane point.", 0.0f));
    }
    zAxis.normalize();
    yAxis = zAxis ^ xAxis;
    DP_ASSERT( isNormalized( yAxis ) );

    // This is a right-handed ortho-normal basis transformation
    // which can be expressed as simple rotation (to avoid matrix 
    // decomposition of the Trafo and to simplify inversion).
    Mat33f m33BasisTrafo( dp::util::makeArray( Vec3f(xAxis[0], yAxis[0], zAxis[0]),
                                               Vec3f(xAxis[1], yAxis[1], zAxis[1]),
                                               Vec3f(xAxis[2], yAxis[2], zAxis[2]) ) );
    Quatf quatBasisTrafo( m33BasisTrafo );


    // The order of transformations of the following node hierarchy is:
    // v' = v * Put * Dof * PutInv;
    TransformSharedPtr hDofPutInv = Transform::create();
    TransformSharedPtr hDof       = Transform::create();
    TransformSharedPtr hDofPut    = Transform::create();

    // 1.) The PutInv transform node:
    // This is explicitly NOT set to DP_SG_HINT_DYNAMIC because
    // if the basis transformation is identity it can be optimized away.
    hDofPutInv->setName( std::string( fltSafeNodeName( node ) ) + std::string( " PutInv" ) );

    {
      Trafo t;
      t.setOrientation( -quatBasisTrafo ); // Inverse basis transformation (negative rotation) in PutInv.
      hDofPutInv->setTrafo( t );

      hDofPutInv->addChild( hDof );
    }

    {
      // 2.) The DOF transform node is the one which is normally manipulated.
      hDof->addHints( Object::DP_SG_HINT_DYNAMIC ); // Tell the optimizers to preserve this node!
      hDof->setName( fltSafeNodeName( node ) );
      addAnnotation( hDof, node );

      Vec3f scale( (float) fltDOF->localCurScaleX, (float) fltDOF->localCurScaleY, (float) fltDOF->localCurScaleZ );
      Quatf rotation;
      eulerToQuat( fltDOF->localCurPitch, fltDOF->localCurRoll, fltDOF->localCurYaw, &rotation );
      Vec3f translation( (float) fltDOF->localCurX, (float) fltDOF->localCurY, (float) fltDOF->localCurZ );

      Trafo t;
      // The transformation order of Trafo::getMatrix() in SceniX is: M = -C * SO^-1 * S * SO * R * C * T
      // so basically scale * rotate * translate which is exactly the desired order.
      t.setScaling( scale );
      t.setOrientation( rotation );
      t.setTranslation( translation );
      // Instead of using origin as translation inside the Put matrices, set it here as center of rotation
      // which allows to optimize the Put matrices away if the basis transformation is identity!
      // We are in local coordinates here, due to the Put matrix below applied first, 
      // transform the world-space origin to local space as well.
      t.setCenter( origin * m33BasisTrafo );
      hDof->setTrafo( t );

      hDof->addChild( hDofPut );
    }

    {
      // 3.) The Put transform node:
      // This is explicitly NOT set to DP_SG_HINT_DYNAMIC because
      // if the basis transformation is identity it can be optimized away.
      hDofPut->setName( std::string( fltSafeNodeName( node ) ) + std::string( " Put" ) );

      Trafo t;
      t.setOrientation( quatBasisTrafo ); // Basis transformation (rotation) in Put.
      hDofPut->setTrafo( t );


      // Descend children.
      collectGeometry( hDofPut, node, flt );
      for (unsigned int i = 0; i < node->numChildren; i++)
      {
        if( !collect( node->child[i] ) )
        {
          buildScene( hDofPut, node->child[i], flt, level + 1 );
        }
      }
    }

    if (parent)
    {
      parent->addChild( hDofPutInv );
    }

    return;
  }

  else if (node->type == FLTRECORD_OBJECT)
  {
    buildObject( parent, node, flt, level+1 );

    return;
  }

  else if (node->type == FLTRECORD_GROUP)
  {
    FltGroup * fgroup = (FltGroup *)node;
    bool isAnim = false;
    bool alwaysParse = false;

    isAnim = !!(fgroup->flags & ( FLTGROUP_FORWARD_ANIM | 
                                  FLTGROUP_BACKWARD_ANIM ) );

    // do some other optimizations, but only where we can
    alwaysParse = (fgroup->flags & FLTGROUP_PRESERVE) || isAnim;

    if( node->parent )
    {
      // must parse if parent was switch
      if( node->parent->type == FLTRECORD_SWITCH )
      {
        alwaysParse = true;
      }
      // must parse if parent was an animated group
      else if( node->parent->type == FLTRECORD_GROUP )
      {
        FltGroup * pgroup = (FltGroup *)node->parent;
        if (pgroup->flags & ( FLTGROUP_FORWARD_ANIM | FLTGROUP_BACKWARD_ANIM ) )
        {
          alwaysParse = true;
        }
      }
    }

    // check if this group has comments; if so, we want to preserve it
    bool commented = hasComment( node );

    if( alwaysParse || node->numChildren || commented)
    {
      //
      // parse group, taking animations into account when available
      //
      if( alwaysParse || node->numChildren > 1 || commented)
      {
        GroupSharedPtr hGroup;

#if defined(KEEP_ANIMATION)
        if( isAnim )
        {
          hGroup = FlipbookAnimation::create();
        }
        else
#endif
        {
          hGroup = Group::create();
        }

        // build the group
        hGroup->setName( fltSafeNodeName( node ) );
        if(commented)
        {
          addAnnotation( hGroup, node );
        }

        // if they want to preserve this node, mark it as dynamic
        if( (fgroup->flags & FLTGROUP_PRESERVE) || commented)
        {
          hGroup->addHints( Object::DP_SG_HINT_DYNAMIC );
        }

#if defined(KEEP_ANIMATION)
        if( isAnim )
        {
          // special handling required since all children are expected
          // to be part of the flipbook anim.
            
          for (unsigned int i=0;i<node->numChildren;i++)
          {
            buildScene(hGroup, node->child[i], flt, level+1);
          }
        }
        else
#endif
        {
          // no special handling required, so optimize
          collectGeometry( hGroup, node, flt );
          for (unsigned int i=0;i<node->numChildren;i++)
          {
            if( !collect( node->child[i] ) )
            {
              buildScene( hGroup, node->child[i], flt, level+1 );
            }
          }
        }

        //
        // If it is an animation, we also add the animation sequence
        // to it.
        //
#if defined(KEEP_ANIMATION)
        if( isAnim )
        {
          FlipbookAnimationLock flip( hGroup.staticCast<FlipbookAnimation>() );
          FramedIndexAnimationDescriptionSharedPtr hIdx = FramedIndexAnimationDescription::create();
          {
            FramedIndexAnimationDescriptionLock idx( hIdx );
            unsigned int numFrames = checked_cast<unsigned int>(flip->getNumberOfChildren());
            unsigned int animFramesPerFrame = 1;
            // assume 60 fps for now
            unsigned int loopFrames = static_cast<unsigned int>(fgroup->loopDuration * 60.0f);

            //
            // modify frames depending on the length of the anim
            //
            if( fgroup->loopDuration != 0.f )
            {
              animFramesPerFrame = loopFrames / numFrames;

              // error check!
              if( animFramesPerFrame == 0 )
              {
                animFramesPerFrame = 1;
              }
            }
            else
            {
              // set this first, just in case
              loopFrames = numFrames;

              // ok, just this once. :) check the comment field and parse
              // out the same tags that the Quantum3D compiler understands
              // leave this as undocumented?
              if(! flip->getAnnotation().empty() )
              {
                const char * cstr = flip->getAnnotation().c_str();
                const char * equal = 0;

                if( strstr( cstr, "vt_loop_duration" ) && 
                    (equal = strchr( cstr, '=' )) )
                {
                  equal++;

                  while( *equal != 0 && (*equal == ' ' || *equal == '\t') ) 
                  {
                    ++equal;
                  }

                  if( *equal )
                  {
                    float duration = (float) atof( equal );

                    if( duration > 0.f )
                    {
                      loopFrames = static_cast<unsigned int>(duration * 60.0f);
                      animFramesPerFrame = loopFrames / numFrames;

                      // error check!
                      if( animFramesPerFrame == 0 )
                      {
                        animFramesPerFrame = 1;
                      }
                    }
                  }
                }
              }
            }

            unsigned int lastFrame = 0;
            if( fgroup->lastFrameDuration != 0.f )
            {
              lastFrame = static_cast<unsigned int>(fgroup->lastFrameDuration * 60.f);
            }

            idx->reserveSteps( checked_cast<unsigned int>(loopFrames + lastFrame) );

            for( unsigned int c = 0; c < (loopFrames/animFramesPerFrame); c++ )
            {
              for( unsigned int j=0;j<animFramesPerFrame;j++)
              {
                // extend final frame if loop duration and number frames was
                // not an even multiple
                idx->addStep((c>=numFrames)?numFrames-1:c );
              }
            }

            // repeat the last frame some number of times, if requested
            for( unsigned int c = 0; c < lastFrame; c++ )
            {
              idx->addStep( numFrames-1 );
            }
          }
          // set the FramedIndexAnimationDescription description into an IndexAnimation
          IndexAnimationSharedPtr iah = IndexAnimation::create();
          {
            IndexAnimationLock wai( iah );
            wai->setDescription( hIdx );
            wai->start();   // and finally, start it on the user's behalf.
          }

          flip->setAnimation( iah );
        }
#endif

        if (parent)
        {
          parent->addChild( hGroup );
        }

      }
      else
      {
        // there is only 1 child and no comments, do not create an extra group node
        collectGeometry( parent, node, flt );
        if( !collect( node->child[0] ) )
        {
          buildScene(parent, node->child[0], flt, level+1);
        }
      }
    }

    return;
  }

  else if (node->type == FLTRECORD_HEADER )
  {
    FltHeader * hdr = (FltHeader * )node;

    // check for RGB (packed) color flag
    // (I think we need to check the format level too)
    if( hdr->flags & FLTHDRFLAGS_RGB_COLOR )
    {
      m_rgbColorMode = true;
    }

    // add comment to parent if it is the header
    addAnnotation( parent, node );

    if( node->numChildren == 0 )
    {
      // improperly formatted file.  try to see if there are any siblings.
      if( node->next && getWorkaround( "CHILDLESS_HEADER" ) )
      {
        FltNode * nn = node->next;
        while( nn )
        {
          buildScene( parent, nn, flt, level );
          nn = nn->next;
        }
      }
      else
      {
        // this file is empty
        INVOKE_CALLBACK(onFileEmpty( flt->fileName ) );
        return;
      }
    }
    else
    {
      // just descend as usual
      for (unsigned int i=0;i<node->numChildren;i++)
      {
        buildScene(parent, node->child[i], flt, level+1);
      }
    }

    return;
  }

  // descend hier for everything else 
#if 0
  NVSG_TRACE_OUT_F(("%d: Unknown node: %s\n", level, fltSafeNodeName( node )));
#endif

  for (unsigned int i=0;i<node->numChildren;i++)
  {
    buildScene(parent, node->child[i], flt, level+1);
  }
}

SceneSharedPtr
FLTLoader::load(  std::string const& filename,
                  std::vector<std::string> const& searchPaths,
                  dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  DP_ASSERT( m_textureCache.empty() );

  FltFile * flt = 0;
  
  // create the normalization cube map
  m_normalizationCubeMap = createNormalizationCubeMap( 128 );

  // set locale temporarily to standard "C" locale
  TempLocale tl("C");

  // start by setting workaround mode
  if( const char * ev = getenv( "NVSG_FLT_WORKAROUNDS" ) )
  {
    setWorkaroundMode( ev );
  }

  // open, parse, close file
  flt = fltOpen(filename.c_str());
  if ( !flt )
  {
    throw std::runtime_error( std::string( "Failed to load: " ) + filename );
  }
   
  SceneSharedPtr hScene;
  // set fileID first
  fltSetFileID( flt, nextFileID() );

  fltParse(flt, 0);
  fltClose(flt);

  //
  // Set these first so that the files path will appear first
  // in the list.
  //
  for(unsigned int i=0;i<searchPaths.size();i++)
  {
    fltAddSearchPath( flt, const_cast<char *>(searchPaths[i].c_str()));
  }

  // also copy them for later
  m_searchPaths = searchPaths;

  // check to see if there is a texture atlas file 
  m_taiManager = new TaiManager();
  if( m_taiManager->loadFile( filename + std::string(".tai") ) == false )
  {
    // no atlas file
    delete m_taiManager;
    m_taiManager = 0;

#if 0
    printf("NO Texture Atlas named: %s\n", 
                              std::string( filename + ".tai" ).c_str());
#endif
  }
  else
  {
#if 0
    printf("Using Texture Atlas: %s\n", 
                              std::string( filename + ".tai" ).c_str());
#endif
  }

  // check to see if we should disable stackedatlas support
  bool saScale = false;
  {
    std::string result;
    if( this->getWorkaround( "USE_STACKEDATLAS", &result ) )
    {
      if( result == "scale" )
      {
        m_useStackedAtlasManager = true;
        saScale = true;
      }
      else
      {
        m_useStackedAtlasManager = ( result == "true" );
      }
    }
  }

  // check to see if we should build stacked atlases
  if( m_useStackedAtlasManager )
  {
    m_stackedAtlasManager = new StackedAtlasManager();

    m_stackedAtlasManager->setSearchPaths( searchPaths );
    m_stackedAtlasManager->setRescaleTextures( saScale, 512, 512 );
  }


  // create toplevel group
  GroupSharedPtr hGroup = Group::create();
  hGroup->setName( filename );

  // build scene
  buildScene( hGroup, (FltNode *)flt->header, flt, 0, 0 );

  // free openflight file
  fltFileFree( flt );

  // create toplevel scene
  hScene = Scene::create();

  // add group as scene's toplevel
  hScene->setRootNode( hGroup );

  m_normalizationCubeMap.reset();
  m_materialMap.clear();
  m_materialCache.clear();
  m_textureCache.clear();

  return hScene;
}

//
// Removes degenerate triangles and duplicate verts where possible.
//
// Returns true to ignore this vertex list all together.
//
bool 
FLTLoader::removeDegenerates( std::vector<unsigned int> & indexList, 
                              std::vector<Vec3f> & verts )
{
  if( verts.size() < 3 || indexList.size() < 3 )
    return true;

  std::vector<unsigned int>::iterator outer = indexList.begin();
  std::vector<unsigned int>::iterator inner;

  while( 1 )
  {
    bool remove = false;

    inner = outer + 1;

    if( inner == indexList.end() )
      break;

    while( inner != indexList.end() )
    {
      if( verts[(*inner)] == verts[(*outer)] )
      {
        remove = true;
        break;
      }

      ++inner;
    }

    if( remove )
    {
      indexList.erase( inner );
      // start over if we erased one
      outer = indexList.begin();
    }
    else
    {
      ++outer;
    }
  }

  // if we have removed points such that they have created a degenerate
  // then remove the polygon
  if( indexList.size() < 3 )
    return true;
  else
    return false;
}

void 
FLTLoader::addAnnotation( NodeSharedPtr const& nNode, FltNode * fNode )
{
  FltComment * comm = (FltComment *)
          fltFindFirstAttrNode( fNode, FLTRECORD_COMMENT );

  if( comm )
  {
    if(comm->text == "TERRAIN:DYNAMICBOX")
    {
      int y = 2;
    }
    nNode->setAnnotation( comm->text ); 
  }
}

bool 
FLTLoader::hasComment( FltNode *fNode )
{
  return (fltFindFirstAttrNode( fNode, FLTRECORD_COMMENT ) != NULL);
}

bool 
FLTLoader::faceNormal( Vec3f & result, FltVertex ** vlist, unsigned int numVerts )
{
  // note that we have not determined whether this triangle is degenerate yet
  // so we try to generate a good normal
  unsigned int i = 0;
  while( i < (numVerts-2) )
  {
    Vec3f v0((float)vlist[i+0]->x, (float)vlist[i+0]->y, (float)vlist[i+0]->z);
    Vec3f v1((float)vlist[i+1]->x, (float)vlist[i+1]->y, (float)vlist[i+1]->z);
    Vec3f v2((float)vlist[i+2]->x, (float)vlist[i+2]->y, (float)vlist[i+2]->z);

    Vec3f c0 = v1 - v0;
    Vec3f c1 = v2 - v0;

    // compute cross product
    result = c0 ^ c1;

    if( length( result ) < FLT_EPSILON )
    {
      // move on
      i++;
      continue;
    }

    // normalize result
    result.normalize();

    return true;
  }

  return false;
}

void
FLTLoader::setWorkaroundMode( const std::string & mode )
{
  std::stringstream ss( mode );

  //
  // Set workaround mode from a colon separated list, with optional values:
  // KEY=VALUE:KEY:[...]
  //
  while( ss.good() )
  {
    char tmp[1024] = {0};

    ss.getline( tmp, 1024, ':' );

    // skip empty entries
    if( !tmp[0] )
    {
      continue;
    }

    std::stringstream entry( tmp );

    char key[256]={0}, value[256]={0};

    entry.getline( key, 256, '=' );

    if( entry.good() )
    {
      entry.getline( value, 256 );
    }

    m_workarounds[ std::string(key) ] = std::string(value);

    printf("Workaround: '%s' = '%s'\n", key, value );
  }
}

bool
FLTLoader::getWorkaround( const std::string & e, std::string * result )
{
  if( result )
  {
    result->clear();
  }

  std::map<std::string,std::string>::iterator iter = m_workarounds.find( e );

  if( iter != m_workarounds.end() )
  {
    if( result )
    {
      *result = (*iter).second;
    }

    return true;
  }
  else
  {
    // not found
    return false;
  }
}

bool 
FLTLoader::createExtRef( const std::string & file, const GroupSharedPtr & grp )
{
  // assume refcount has already been incremented earlier
  m_extRefs[ file ] = grp;

  return true;
}

GroupSharedPtr FLTLoader::lookupExtRef( const std::string & file )
{
  std::map<std::string, GroupSharedPtr >::iterator iter = m_extRefs.find( file );

  if( iter != m_extRefs.end() )
  {
    return (*iter).second;
  }
  else
  {
    return 0;
  }
}
