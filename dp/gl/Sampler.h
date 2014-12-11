// Copyright NVIDIA Corporation 2015
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

#include <dp/gl/Config.h>
#include <dp/gl/Buffer.h>
#include <dp/gl/Object.h>
#include <dp/math/math.h>

// TODO Image capturing is currently not implemented. The code sequences using 
// SamplerHost have been disabled with #if 0.

namespace dp
{
  namespace gl
  {
    /*! \brief Base class to represent an OpenGL sampler.
     */
    class Sampler : public Object
    {
      public:
        DP_GL_API static SamplerSharedPtr create();
        DP_GL_API virtual ~Sampler();

      public:
        DP_GL_API void bind( GLuint unit );
        DP_GL_API void unbind( GLuint unit );

        DP_GL_API void setBorderColor( float color[4] );
        DP_GL_API void setBorderColor( unsigned int color[4] );
        DP_GL_API void setBorderColor( int color[4] );

        DP_GL_API void setCompareParameters( GLenum mode, GLenum func );

        /*! \brief Set filtering modes for this texture
         *  \param minFilter The filter to use for minification
         *  \param magFilter The filter to use for magnification
        **/
        DP_GL_API void setFilterParameters( GLenum minFilter, GLenum magFilter );

        DP_GL_API void setLODParameters( float minLOD, float maxLOD );

        DP_GL_API void setWrapParameters( GLenum wrapS, GLenum wrapT, GLenum wrapR );

      protected:
        /*! \brief Constructor for sampler class. */
        DP_GL_API Sampler();

      private:
#if !defined(NDEBUG)
        std::set<GLuint>  m_boundUnits;
#endif
    };

  } // namespace gl
} // namespace dp