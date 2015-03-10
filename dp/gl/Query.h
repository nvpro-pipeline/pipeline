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
#include <dp/gl/Object.h>

namespace dp
{
  namespace gl
  {
    class Query : public Object
    {
      public:
        DP_GL_API static QuerySharedPtr create();
        DP_GL_API virtual ~Query();

        DP_GL_API void begin( GLenum target );
        DP_GL_API void end();

        template <typename T> T getResult() const;
        DP_GL_API bool isResultAvailable() const;

      protected:
        DP_GL_API Query();

      private:
        template <typename T> T get( GLenum param ) const;

      private:
        GLenum  m_target;
    };


    template <typename T>
    inline T Query::getResult() const
    {
      return( get<T>( GL_QUERY_RESULT ) );
    }


    template <typename T>
    inline T Query::get( GLenum param ) const
    {
      DP_STATIC_ASSERT( !"no specialization for type T available!" );
    }

    template <>
    inline GLint Query::get<GLint>( GLenum param ) const
    {
      DP_ASSERT( m_target == GL_INVALID_ENUM );
      GLint result;
      glGetQueryObjectiv( m_id, param, &result );
      return( result );
    }

    template <>
    inline GLuint Query::get<GLuint>( GLenum param ) const
    {
      DP_ASSERT( m_target == GL_INVALID_ENUM );
      GLuint result;
      glGetQueryObjectuiv( m_id, param, &result );
      return( result );
    }

    template <>
    inline GLint64 Query::get<GLint64>( GLenum param ) const
    {
      DP_ASSERT( m_target == GL_INVALID_ENUM );
      GLint64 result;
      glGetQueryObjecti64v( m_id, param, &result );
      return( result );
    }

    template <>
    inline GLuint64 Query::get<GLuint64>( GLenum param ) const
    {
      DP_ASSERT( m_target == GL_INVALID_ENUM );
      GLuint64 result;
      glGetQueryObjectui64v( m_id, param, &result );
      return( result );
    }

  } // namespace gl
} // namespace dp
