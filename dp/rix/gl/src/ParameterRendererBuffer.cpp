// Copyright NVIDIA Corporation 2013
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


#include <dp/rix/gl/inc/ParameterRendererBuffer.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      ParameterRendererBuffer::ParameterRendererBuffer()
      {
      }

      ParameterRendererBuffer::ParameterRendererBuffer( ParameterCacheEntryStreamBuffers const& parameterCacheEntries, dp::gl::SmartBuffer const& ubo, GLenum target, size_t uboBinding, size_t uboOffset, GLsizeiptr uboBlockSize )
        : m_parameters( parameterCacheEntries )
        , m_ubo( ubo )
        , m_target( target )
        , m_uboBinding( GLint(uboBinding) )
        , m_uboOffset( GLsizeiptr(uboOffset) )
        , m_uboBlockSize( uboBlockSize )
      {
        DP_ASSERT( !m_parameters.empty() );
      }

      void ParameterRendererBuffer::activate()
      {
        glBindBufferRange( m_target, m_uboBinding, m_ubo->getGLId(), m_uboOffset, m_uboBlockSize );
      }

      void ParameterRendererBuffer::render( void const* cache )
      { 
        m_ubo->setSubData( m_target, m_uboOffset, m_uboBlockSize, cache );
      }

      void ParameterRendererBuffer::update( void* cache, void const* container )
      {
        // TODO ensure that update is called only on non-empty containers
        size_t numParameters = m_parameters.size();
        if ( numParameters )
        {
          ParameterCacheEntryStreamBufferSharedPtr const* parameterObjects = &m_parameters[0];
          ParameterCacheEntryStreamBufferSharedPtr const* const parameterObjectsEnd = parameterObjects + numParameters;

          for( ParameterCacheEntryStreamBufferSharedPtr const* parameterObject = parameterObjects; parameterObject != parameterObjectsEnd
            ; ++parameterObject )
          {
            (*parameterObject)->update( cache, container );
          }
        }
      }

      size_t ParameterRendererBuffer::getCacheSize( ) const
      {
        return m_uboBlockSize;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp

