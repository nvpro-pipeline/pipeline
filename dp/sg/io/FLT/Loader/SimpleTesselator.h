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
// SimpleTesselator.h
//

#ifndef _SIMPLETESSELATOR_H_
#define _SIMPLETESSELATOR_H_

#include <GL/gl.h>
#include <GL/glu.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/nvsgapi.h>
#include <vector>

#if defined( WIN32 ) || defined( _WIN32 )
#define CALLBACK __stdcall
#define GLU_TESS_CALLBACK void (__stdcall *)()
#else
#define CALLBACK
#define GLU_TESS_CALLBACK void (*)()
#endif

class SimpleTesselator
{
public:
  SimpleTesselator() : m_tess( gluNewTess() ) 
  {
    DP_ASSERT( m_tess );

    // register callbacks
    //
    // if we register an edge flag callback then everything
    // will come as triangle lists
    //
    //gluTessCallback( m_tess, GLU_TESS_EDGE_FLAG_DATA,  
    //                         (GLU_TESS_CALLBACK) edgeCB );
    gluTessCallback( m_tess, GLU_TESS_VERTEX_DATA, 
                            (GLU_TESS_CALLBACK) vertexCB );
    gluTessCallback( m_tess, GLU_TESS_BEGIN_DATA,  
                            (GLU_TESS_CALLBACK) beginCB );
    gluTessCallback( m_tess, GLU_TESS_END_DATA,    
                            (GLU_TESS_CALLBACK) endCB );
    //gluTessCallback( m_tess, GLU_TESS_COMBINE_DATA,
    //                        (GLU_TESS_CALLBACK) combineCB );
    gluTessCallback( m_tess, GLU_TESS_ERROR_DATA,  
                             (GLU_TESS_CALLBACK) errorCB );

    // no merging!!
    gluTessProperty( m_tess, GLU_TESS_TOLERANCE, 0.0 );
  }

  virtual ~SimpleTesselator(void) 
  { 
    gluDeleteTess( m_tess );
  }

  bool tesselate( const std::vector<dp::math::Vec3f> & vertsIn,
                  std::vector<unsigned int> & indexListInOut )
  {
    // must have at least one poly
    if( vertsIn.size() < 3 || indexListInOut.size() < 3 )
      return false;
    
    std::vector<GLdouble> verts( 3* indexListInOut.size() );

    // clear error
    m_error = 0;
    m_primtype = 0xffffffff;

    gluTessBeginPolygon( m_tess, this );
    //gluTessNormal( m_tess, n[0], n[1], n[2] );
    gluTessBeginContour( m_tess );

    for(unsigned int i=0;i<indexListInOut.size();i++)
    {
      double x, y, z;

      x = (double)vertsIn[indexListInOut[i]][0];
      y = (double)vertsIn[indexListInOut[i]][1];
      z = (double)vertsIn[indexListInOut[i]][2];

      verts[3*i+0] = x;
      verts[3*i+1] = y;
      verts[3*i+2] = z;

      gluTessVertex( m_tess, &verts[3*i], (void *)&indexListInOut[i] ); 
    }

    gluTessEndContour( m_tess );
    gluTessEndPolygon( m_tess );

    convertTriList();

    //
    // Copy out the faces
    //
    indexListInOut = m_triList;

    return (m_error == 0 && m_triList.size());
  }

  std::vector<unsigned int> m_triList;
  std::vector<unsigned int> m_drawList;
  GLenum m_error;
  GLenum m_primtype;

private:
  GLUtesselator * m_tess;

  void convertTriList();
  static void CALLBACK edgeCB(GLboolean flag, void* userData);
  static void CALLBACK beginCB(GLenum which, void* userData);
  static void CALLBACK vertexCB(GLvoid *data, void* userData);
  static void CALLBACK combineCB(GLdouble coords[3], void* vertex_data[4],
                        GLfloat weight[4], void** outData,
                        void* useData);
  static void CALLBACK endCB(void* userData);
  static void CALLBACK errorCB(GLenum errorCode, void* userData);

};

void SimpleTesselator::convertTriList( void )
{
  GLenum mode = GL_INVALID_VALUE;
  int count = 0;

  // walk through draw commands and decompose them back into triangles
  for( unsigned int i=0; i < m_drawList.size(); i++ )
  {
    // end command
    if( m_drawList[i] == 0xffffffff )
    {
      mode = GL_INVALID_VALUE;
      count = 0;
      continue;
    }

    switch( mode )
    {
      case GL_INVALID_VALUE:
        // expect draw command
        mode = m_drawList[i];
        break;

      case GL_TRIANGLES:
        m_triList.push_back( m_drawList[i] );
        break;

      case GL_TRIANGLE_STRIP:
        if( count < 3 )
        {
          m_triList.push_back( m_drawList[i] );
        }
        else
        {
          if( count & 1 )
          {
            // odd
            m_triList.push_back( m_drawList[i-1] );
            m_triList.push_back( m_drawList[i-2] );
            m_triList.push_back( m_drawList[i] );
          }
          else
          {
            // even
            m_triList.push_back( m_drawList[i-2] );
            m_triList.push_back( m_drawList[i-1] );
            m_triList.push_back( m_drawList[i] );
          }
        }
        count++;
        break;

      case GL_TRIANGLE_FAN:
        if( count < 3 )
        {
          m_triList.push_back( m_drawList[i] );
        }
        else
        {
          m_triList.push_back( m_drawList[i-count] );
          m_triList.push_back( m_drawList[i-1] );
          m_triList.push_back( m_drawList[i] );
        }
        count++;
        break;

      default:
        DP_ASSERT(0);
    }
  }
}

void SimpleTesselator::edgeCB( GLboolean flag, void * data )
{
  // do nothing
}

void SimpleTesselator::beginCB(GLenum which, void* userData)
{
  ((SimpleTesselator*)userData)->m_drawList.push_back( which );
}

void SimpleTesselator::vertexCB(GLvoid *data, void* userData)
{
  DP_ASSERT( userData != 0 );

  ((SimpleTesselator*)userData)->m_drawList.push_back( *(unsigned int*)data );
}

void SimpleTesselator::combineCB(GLdouble coords[3], void * vertex_data[4],
                      GLfloat w[4], void** outData,
                      void* userData)
{
  //DP_ASSERT(0);
  GLdouble *n = new GLdouble[3];
  GLdouble * d = (GLdouble *)vertex_data;

  n[0] =  coords[0];
  n[1] =  coords[1];
  n[2] =  coords[2];

  *outData = (void *)n;
}

void SimpleTesselator::endCB(void* userData)
{
  ((SimpleTesselator*)userData)->m_drawList.push_back( 0xffffffff );
}

void SimpleTesselator::errorCB(GLenum errorCode, void* userData)
{
  switch( errorCode )
  {
    case GLU_TESS_COORD_TOO_LARGE:
      break;
    case GLU_TESS_NEED_COMBINE_CALLBACK:
      break;
    case GLU_OUT_OF_MEMORY:
      break;
  }

  ((SimpleTesselator*)userData)->m_error = errorCode;
}

#endif /* _SIMPLETESSELATOR_H_ */

