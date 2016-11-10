// Copyright NVIDIA Corporation 2013-2015
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


#include <dp/rix/gl/inc/ParameterRendererUniform.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      ParameterRendererUniform::ParameterRendererUniform()
      {
      }

      ParameterRendererUniform::ParameterRendererUniform( ParameterCacheEntryStreams const& parameterCacheEntries )
        : m_numParameterObjects( parameterCacheEntries.size() )
        , m_parameters( parameterCacheEntries )
      {
      }

      void ParameterRendererUniform::activate( )
      {
        if (m_parameters.size())
        {
          // TODO ensure that render is called only on non-empty containers
          size_t numParameters = m_numParameterObjects;
          ParameterCacheEntryStreamSharedPtr const* parameterObject = m_parameters.data();
          ParameterCacheEntryStreamSharedPtr const* const parameterObjectEnd = parameterObject + numParameters;

          do
          {
            (*parameterObject)->resetState();
            ++parameterObject;
          } while (parameterObject != parameterObjectEnd);
        }
      }

      void ParameterRendererUniform::render( void const* cache)
      {
        if ( m_parameters.size() )
        {
          // TODO ensure that render is called only on non-empty containers
          size_t numParameters = m_numParameterObjects;
          ParameterCacheEntryStreamSharedPtr const* parameterObject = m_parameters.data();
          ParameterCacheEntryStreamSharedPtr const* const parameterObjectEnd = parameterObject + numParameters;

          do 
          {
            (*parameterObject)->render( cache );
            ++parameterObject;
          } while (parameterObject != parameterObjectEnd);
        }
      }

      void ParameterRendererUniform::update( void* cache, void const* container )
      {
        // TODO ensure that update is called only on non-empty containers
        if ( m_parameters.size() )
        {
          ParameterCacheEntryStreamSharedPtr const* parameterObject = m_parameters.data();
          ParameterCacheEntryStreamSharedPtr const* const parameterObjectEnd = parameterObject + m_parameters.size();
          do 
          {
            (*parameterObject)->update(cache, container);
            ++parameterObject;
          } while (parameterObject != parameterObjectEnd);
        }
      }

      size_t ParameterRendererUniform::getCacheSize( ) const
      {
        size_t cacheSize = 0;
        for( ParameterCacheEntryStreams::const_iterator it = m_parameters.begin(); it != m_parameters.end(); ++it )
        {
          cacheSize += (*it)->getSize( );
        }

        return cacheSize;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp

