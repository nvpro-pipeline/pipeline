// Copyright NVIDIA Corporation 2002-2010
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


#include <stdio.h>
#include <sstream>
#include <dp/sg/core/Scene.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>
#include "OBJSaver.h"
#include "ExtractGeometryTraverser.h"

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;
using dp::util::PlugIn;
using dp::util::UPIID;
using dp::util::UPITID;
using namespace std;

// supported Plug Interface ID
const UPITID PITID_SCENE_SAVER(UPITID_SCENE_SAVER, UPITID_VERSION); // plug-in type
const UPIID  PIID_OBJ_SCENE_SAVER(".OBJ", PITID_SCENE_SAVER); // plug-in ID

#if defined( _WIN32 )
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    int i=0;
  }

  return TRUE;
}
#elif defined( LINUX )
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  if ( piid==PIID_OBJ_SCENE_SAVER )
  {
    pi = OBJSaver::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_OBJ_SCENE_SAVER);
}

class ExtractToOBJ : public ExtractGeometryTraverser
{
public:

  ExtractToOBJ() {}
  ~ExtractToOBJ() {}

  void submitIndexedTriangleSet( 
            const std::vector<unsigned int> & indices, 
            const std::vector<Vec3f> & verts,
            const std::vector<Vec3f> & normals,
            const std::vector<Vec2f> & texcoords,
            const OBJMaterial & material )
  {
    unsigned int startIndex = dp::checked_cast<unsigned int>(m_verts.size());

    if ( indices.size() && verts.size() )
    {
      for (size_t i = 0; i < verts.size(); i++)
      {
        m_verts.push_back( verts[i] );
        m_normals.push_back( normals[i] );
        m_texcoords.push_back( texcoords[i] );
      }

      // save materials
      m_materials.push_back( material );
      m_materialMappings.push_back( 
        make_pair<unsigned int,unsigned int>( dp::checked_cast<unsigned int>( m_indices.size() ),
                                              dp::checked_cast<unsigned int>( m_materials.size() - 1) ) );

      for (size_t i = 0; i < indices.size(); i++)
      {
        m_indices.push_back( indices[i] + startIndex );
      }
    }
    else
    {
      printf("ERR: indices.size() = %d - verts.size() = %d\n", 
             dp::checked_cast<unsigned int>(indices.size()),
             dp::checked_cast<unsigned int>(verts.size()) );
    }
  }

  std::vector<Vec3f> & getVerts() { return m_verts; }
  std::vector<Vec3f> & getNormals() { return m_normals; }
  std::vector<Vec2f> & getTexCoords() { return m_texcoords; }
  std::vector<unsigned int> & getIndices() { return m_indices; }
  std::vector< std::pair<unsigned int, unsigned int> > & getMaterialMappings() { return m_materialMappings; }
  std::vector< OBJMaterial > & getMaterials() { return m_materials; }

  std::vector<Vec3f> m_verts;
  std::vector<Vec3f> m_normals;
  std::vector<Vec2f> m_texcoords;
  std::vector<unsigned int>  m_indices;
  std::vector< OBJMaterial > m_materials;
  std::vector< std::pair< unsigned int, unsigned int > > m_materialMappings;
};

static std::string fixslashes( const std::string & orig )
{
  std::string result( orig );

  size_t idx;
  while( (idx = result.find_first_of( '\\' )) != string::npos )
  {
    result[idx] = '/';
  }

  return result;
}

static void 
writeMaterialEntry( FILE * f, unsigned int index, OBJMaterial & m )
{
  fprintf( f, "newmtl %d\n", index );

  if( m.isTexture )
  {
    fprintf( f, "map_Kd %s\n", fixslashes( m.filename ).c_str() );
  }

  if( m.isMaterial )
  {
    fprintf( f, "Ka %f %f %f\n", m.ambient[0], m.ambient[1], m.ambient[2] );
    fprintf( f, "Kd %f %f %f\n", m.diffuse[0], m.diffuse[1], m.diffuse[2]);
    fprintf( f, "Ks %f %f %f\n", m.specular[0], m.specular[1], m.specular[2]);
    fprintf( f, "Tr %f\n", m.opacity );
    fprintf( f, "d %f\n", m.opacity );
    fprintf( f, "Ns %f\n", m.exponent );
    fprintf( f, "illum 2\n" );
  }

  fprintf( f, "\n" );
}

OBJSaverSharedPtr OBJSaver::create()
{
  return( std::shared_ptr<OBJSaver>( new OBJSaver() ) );
}

OBJSaver::OBJSaver()
{
}

OBJSaver::~OBJSaver()
{
}

bool  OBJSaver::save( SceneSharedPtr const& scene, dp::sg::ui::ViewStateSharedPtr const& viewState, string const& filename )
{
  FILE *fh = fopen( filename.c_str(), "w" );
  FILE *mh = fopen( string( filename + ".mtl" ).c_str() , "w" );
  
  // set locale temporarily to standard "C" locale
  dp::util::TempLocale tl("C");

  if ( fh && mh )
  {
    ExtractToOBJ extractor;
    size_t materialIndex = 0;

    // extract all geometry from the scene and add it to the triangle mesh
    extractor.apply( scene );

    // these are copied out because they have to be saved
    vector<Vec3f> & verts          = extractor.getVerts();
    vector<Vec3f> & norms          = extractor.getNormals();
    vector<Vec2f> & texcoords      = extractor.getTexCoords();
    vector<unsigned int> & indices = extractor.getIndices();
    vector<pair<unsigned int, unsigned int> > & materialMapping = extractor.getMaterialMappings();
    vector<OBJMaterial> & materials = extractor.getMaterials();

    size_t last = filename.find_last_of( "\\" );
    if( last == string::npos )
    {
      // check unix
      last = filename.find_last_of( "/" );
    }

    string mtlname = filename;

    if( last != string::npos )
    {
      mtlname = filename.substr( last+1 );
    }

    fprintf(fh, "mtllib %s.mtl\n", mtlname.c_str() );

    for( size_t i = 0; i < verts.size(); i ++ )
    {
      fprintf(fh, "v %f %f %f\n", verts[i][0], verts[i][1], verts[i][2] );
      fprintf(fh, "vt %f %f\n", texcoords[i][0], texcoords[i][1] );
      fprintf(fh, "vn %f %f %f\n", norms[i][0], norms[i][1], norms[i][2] );
    }

    for( size_t i = 0; i < indices.size(); i += 3 )
    {
      // first, check for materials on this index

      if ( materialIndex < materialMapping.size() && 
          i >= materialMapping[ materialIndex ].first )
      {
        fprintf(fh, "usemtl %d\n", materialMapping[ materialIndex ].second );

        writeMaterialEntry( mh, 
                            materialMapping[ materialIndex ].second, 
                            materials[ materialMapping[ materialIndex ].second ] );
            
        // move on to next one
        materialIndex++;
      }

      // indices are 1-based
      unsigned int idx1 = indices[i    ] + 1;
      unsigned int idx2 = indices[i + 1] + 1;
      unsigned int idx3 = indices[i + 2] + 1;
      fprintf(fh, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                     idx1, idx1, idx1,
                     idx2, idx2, idx2,
                     idx3, idx3, idx3 );
    }

    fclose( fh );
    fclose( mh );
  }

  return( !!fh );
}
