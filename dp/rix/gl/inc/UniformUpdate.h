// Copyright NVIDIA Corporation 2012
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


#include "GL/glew.h"

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      /************************************************************************/
      /* setUniform without conversion                                        */
      /************************************************************************/
      // glUniform* template.
      template<unsigned int n, typename T>
      void setUniform( int uniformLocation, unsigned int arraySize, const void *data ); // GENERAL CASE NOT SUPPORTED

      template<unsigned int n, typename T, typename SourceType >
      void setUniform( int uniformLocation, unsigned int arraySize, const void *data ); // GENERAL CASE NOT SUPPORTED

      // Specializations

      // glUniform{1,2,3,4}fv
      template<> inline void setUniform<1, float>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform1fv( uniformLocation, arraySize, static_cast<const float *>( data ));
      }
      template<> inline void setUniform<2, float>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform2fv( uniformLocation, arraySize, static_cast<const float *>( data ));
      }
      template<> inline void setUniform<3, float>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform3fv( uniformLocation, arraySize, static_cast<const float *>( data ));
      }
      template<> inline void setUniform<4, float>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform4fv( uniformLocation, arraySize, static_cast<const float *>( data ));
      }

      //// glUniform{1,2,3,4}dv
      template<> inline void setUniform<1, double>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform1dv( uniformLocation, arraySize, static_cast<const double *>( data ));
      }
      template<> inline void setUniform<2, double>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform2dv( uniformLocation, arraySize, static_cast<const double *>( data ));
      }
      template<> inline void setUniform<3, double>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform3dv( uniformLocation, arraySize, static_cast<const double *>( data ));
      }
      template<> inline void setUniform<4, double>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform4dv( uniformLocation, arraySize, static_cast<const double *>( data ));
      }

      // glUniform{1,2,3,4}iv
      template<> inline void setUniform<1, dp::Int32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform1iv( uniformLocation, arraySize, static_cast<const int *>( data ));
      }
      template<> inline void setUniform<2, dp::Int32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform2iv( uniformLocation, arraySize, static_cast<const int *>( data ));
      }
      template<> inline void setUniform<3, dp::Int32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform3iv( uniformLocation, arraySize, static_cast<const int *>( data ));
      }
      template<> inline void setUniform<4, dp::Int32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform4iv( uniformLocation, arraySize, static_cast<const int *>( data ));
      }

      // glUniform{1,2,3,4}uiv
      template<> inline void setUniform<1, dp::Uint32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform1uiv( uniformLocation, arraySize, static_cast<const unsigned int *>( data ));
      }
      template<> inline void setUniform<2, dp::Uint32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform2uiv( uniformLocation, arraySize, static_cast<const unsigned int *>( data ));
      }
      template<> inline void setUniform<3, dp::Uint32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform3uiv( uniformLocation, arraySize, static_cast<const unsigned int *>( data ));
      }
      template<> inline void setUniform<4, dp::Uint32>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform4uiv( uniformLocation, arraySize, static_cast<const unsigned int *>( data ));
      }

      // glUniform{1,2,3,4}i64v
      template<> inline void setUniform<1, GLint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform1i64vNV( uniformLocation, arraySize, static_cast<const GLint64EXT *>( data ));
      }
      template<> inline void setUniform<2, GLint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform2i64vNV( uniformLocation, arraySize, static_cast<const GLint64EXT *>( data ));
      }
      template<> inline void setUniform<3, GLint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform3i64vNV( uniformLocation, arraySize, static_cast<const GLint64EXT *>( data ));
      }
      template<> inline void setUniform<4, GLint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform4i64vNV( uniformLocation, arraySize, static_cast<const GLint64EXT *>( data ));
      }

      // glUniform{1,2,3,4}ui64v
      template<> inline void setUniform<1, GLuint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform1ui64vNV( uniformLocation, arraySize, static_cast<const GLuint64EXT *>( data ));
      }
      template<> inline void setUniform<2, GLuint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform2ui64vNV( uniformLocation, arraySize, static_cast<const GLuint64EXT *>( data ));
      }
      template<> inline void setUniform<3, GLuint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform3ui64vNV( uniformLocation, arraySize, static_cast<const GLuint64EXT *>( data ));
      }
      template<> inline void setUniform<4, GLuint64EXT>(int uniformLocation, unsigned int arraySize, const void *data )
      {
        glUniform4ui64vNV( uniformLocation, arraySize, static_cast<const GLuint64EXT *>( data ));
      }


      /************************************************************************/
      /* glUniform for matrices                                               */
      /************************************************************************/
      template<unsigned int n, unsigned int m, typename T>
      void setUniformMatrix( int uniformLocation, unsigned int arraySize, bool transpose, const void *data ); // GENERAL CASE NOT SUPPORTED

      // mat{2,3,4}x{2,3,4}
      template<> inline void setUniformMatrix<2, 2, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix2fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }
      template<> inline void setUniformMatrix<2, 3, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix2x3fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }
      template<> inline void setUniformMatrix<2, 4, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3x4fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }

      template<> inline void setUniformMatrix<3, 2, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3x2fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }
      template<> inline void setUniformMatrix<3, 3, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }
      template<> inline void setUniformMatrix<3, 4, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3x4fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }

      template<> inline void setUniformMatrix<4, 2, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix4x2fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }
      template<> inline void setUniformMatrix<4, 3, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix4x3fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }
      template<> inline void setUniformMatrix<4, 4, float>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix4fv( uniformLocation, arraySize, transpose, static_cast<const float *>( data ));
      }

      // dmat{2,3,4}x{2,3,4}
      template<> inline void setUniformMatrix<2, 2, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix2dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }
      template<> inline void setUniformMatrix<2, 3, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix2x3dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }
      template<> inline void setUniformMatrix<2, 4, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3x4dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }

      template<> inline void setUniformMatrix<3, 2, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3x2dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }
      template<> inline void setUniformMatrix<3, 3, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }
      template<> inline void setUniformMatrix<3, 4, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix3x4dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }

      template<> inline void setUniformMatrix<4, 2, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix4x2dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }
      template<> inline void setUniformMatrix<4, 3, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix4x3dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }
      template<> inline void setUniformMatrix<4, 4, double>(int uniformLocation, unsigned int arraySize, bool transpose, const void *data )
      {
        glUniformMatrix4dv( uniformLocation, arraySize, transpose, static_cast<const double *>( data ));
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
