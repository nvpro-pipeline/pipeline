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


#include <dp/rix/gl/inc/ParameterCacheEntryStreamBuffer.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      ParameterCacheEntryStreamBuffer::ParameterCacheEntryStreamBuffer( size_t cacheOffset, size_t containerOffset, size_t size )
        : m_cacheOffset( cacheOffset)
        , m_containerOffset( containerOffset)
        , m_size( size )
      {
      }

      template <int n, int m, typename SourceType, typename DestType>
      class CacheEntryMatrix : public ParameterCacheEntryStreamBuffer
      {
      public:
        static dp::util::SharedPtr<CacheEntryMatrix> create( dp::gl::Program::Uniform const& uniformInfo, size_t containerOffset, size_t size );
        virtual void update( void * cache, void const * container ) const;

        size_t getSize() const { return m_size; }

      protected:
        CacheEntryMatrix( dp::gl::Program::Uniform const& uniformInfo, size_t containerOffset, size_t size );

      protected:
        size_t m_arraySize;
        size_t m_arrayStride;
        size_t m_matrixStride;
      };

      template <int n, int m, typename SourceType, typename DestType>
      dp::util::SharedPtr<CacheEntryMatrix<n,m,SourceType, DestType>> CacheEntryMatrix<n, m, SourceType, DestType>::create( dp::gl::Program::Uniform const& uniformInfo, size_t containerOffset, size_t size )
      {
        return( std::shared_ptr<CacheEntryMatrix<n,m,SourceType,DestType>>( new CacheEntryMatrix<n,m,SourceType,DestType>( uniformInfo, containerOffset, size ) ) );
      }

      template <int n, int m, typename SourceType, typename DestType>
      CacheEntryMatrix<n,m,SourceType, DestType>::CacheEntryMatrix( dp::gl::Program::Uniform const& uniformInfo, size_t containerOffset, size_t size )
        : ParameterCacheEntryStreamBuffer( uniformInfo.offset, containerOffset, 0 )
        , m_arraySize( size )
        , m_arrayStride( uniformInfo.arrayStride )
        , m_matrixStride( uniformInfo.matrixStride )
      {
        DP_ASSERT( (uniformInfo.arraySize == 0 && size == 1) || (uniformInfo.arraySize == dp::checked_cast<GLint>(size)) );

        if ( n > 1 )
        {
          DP_ASSERT( m_matrixStride );
          m_size = n * m_matrixStride;
        }
        else
        {
          m_size = m * sizeof( DestType );
        }
        if ( m_arraySize > 1 )
        {
          DP_ASSERT( m_arrayStride );
          m_size *= m_arrayStride;
        }
      }

      template <int n, int m, typename SourceType, typename DestType>
      void CacheEntryMatrix<n,m,SourceType, DestType>::update( void *cache, void const* container ) const
      {
        SourceType const* containerData = reinterpret_cast<SourceType const*>( reinterpret_cast<const char*>(container) + m_containerOffset );
        for ( size_t arrayIndex = 0; arrayIndex < m_arraySize; ++arrayIndex )
        {
          for (size_t row = 0;row < n;++row )
          {
            DestType* cacheData = reinterpret_cast<DestType*>(reinterpret_cast<char*>(cache) + m_cacheOffset + arrayIndex * m_arrayStride + row * m_matrixStride);
            for (size_t column = 0;column < m; ++column )
            {
              cacheData[column] = *containerData;
              ++containerData;
            }
          }
        }
      }

      ParameterCacheEntryStreamBufferSharedPtr createParameterCacheEntryStreamBuffer( dp::rix::gl::ProgramGLHandle /*program*/
        , dp::rix::core::ContainerParameterType containerParameterType
        , dp::gl::Program::Uniform uniformInfo
        , size_t /*cacheOffset*/
        , size_t containerOffset )
      {
        ParameterCacheEntryStreamBufferSharedPtr parameterCacheEntry;
#if !defined(NDEBUG)
        GLint const& uniformType = uniformInfo.type;
#endif
        size_t newArraySize = 1;
        switch( containerParameterType )
        {
        case dp::rix::core::CPT_FLOAT:
          DP_ASSERT( uniformType == GL_FLOAT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_FLOAT2:
          DP_ASSERT( uniformType == GL_FLOAT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_FLOAT3:
          DP_ASSERT( uniformType == GL_FLOAT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_FLOAT4:
          DP_ASSERT( uniformType == GL_FLOAT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_INT_8:
          DP_ASSERT( uniformType == GL_INT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int8, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT2_8:
          DP_ASSERT( uniformType == GL_INT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int8, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT3_8:
          DP_ASSERT( uniformType == GL_INT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int8, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT4_8:
          DP_ASSERT( uniformType == GL_INT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int8, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_INT_16:
          DP_ASSERT( uniformType == GL_INT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int16, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT2_16:
          DP_ASSERT( uniformType == GL_INT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int16, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT3_16:
          DP_ASSERT( uniformType == GL_INT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int16, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT4_16:
          DP_ASSERT( uniformType == GL_INT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int16, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_INT_32:
          DP_ASSERT( uniformType == GL_INT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int32, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT2_32:
          DP_ASSERT( uniformType == GL_INT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int32, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT3_32:
          DP_ASSERT( uniformType == GL_INT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int32, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT4_32:
          DP_ASSERT( uniformType == GL_INT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int32, dp::Int32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_INT_64:
          DP_ASSERT( uniformType == GL_INT64_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int64, dp::Int64>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT2_64:
          DP_ASSERT( uniformType == GL_INT64_VEC2_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int64, dp::Int64>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT3_64:
          DP_ASSERT( uniformType == GL_INT64_VEC3_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int64, dp::Int64>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_INT4_64:
          DP_ASSERT( uniformType == GL_INT64_VEC4_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int64, dp::Int64>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_UINT_8:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT2_8:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT3_8:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT4_8:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_UINT_16:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int16, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT2_16:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int16, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT3_16:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int16, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT4_16:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int16, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_UINT_32:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Uint32, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT2_32:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Uint32, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT3_32:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Uint32, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT4_32:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Uint32, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_UINT_64:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT64_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Uint64, dp::Uint64>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT2_64:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT64_VEC2_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Uint64, dp::Uint64>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT3_64:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT64_VEC3_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Uint64, dp::Uint64>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_UINT4_64:
          DP_ASSERT( uniformType == GL_UNSIGNED_INT64_VEC4_NV );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Uint64, dp::Uint64>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_BOOL:
          DP_ASSERT( uniformType == GL_BOOL );
          parameterCacheEntry = CacheEntryMatrix<1, 1, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_BOOL2:
          DP_ASSERT( uniformType == GL_BOOL_VEC2 );
          parameterCacheEntry = CacheEntryMatrix<1, 2, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_BOOL3:
          DP_ASSERT( uniformType == GL_BOOL_VEC3 );
          parameterCacheEntry = CacheEntryMatrix<1, 3, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_BOOL4:
          DP_ASSERT( uniformType == GL_BOOL_VEC4 );
          parameterCacheEntry = CacheEntryMatrix<1, 4, dp::Int8, dp::Uint32>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_MAT2X2:
          DP_ASSERT( uniformType == GL_FLOAT_MAT2 );
          parameterCacheEntry = CacheEntryMatrix<2,2,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_MAT2X3:
          DP_ASSERT( uniformType == GL_FLOAT_MAT2x3 );
          parameterCacheEntry = CacheEntryMatrix<2,3,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_MAT2X4:
          DP_ASSERT( uniformType == GL_FLOAT_MAT2x4 );
          parameterCacheEntry = CacheEntryMatrix<2,4,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_MAT3X2:
          DP_ASSERT( uniformType == GL_FLOAT_MAT3x2 );
          parameterCacheEntry = CacheEntryMatrix<3,2,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_MAT3X3:
          DP_ASSERT( uniformType == GL_FLOAT_MAT3 );
          parameterCacheEntry = CacheEntryMatrix<3,3,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_MAT3X4:
          DP_ASSERT( uniformType == GL_FLOAT_MAT3x4 );
          parameterCacheEntry = CacheEntryMatrix<3,4,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;

        case dp::rix::core::CPT_MAT4X2:
          DP_ASSERT( uniformType == GL_FLOAT_MAT4x2 );
          parameterCacheEntry = CacheEntryMatrix<4,2,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_MAT4X3:
          DP_ASSERT( uniformType == GL_FLOAT_MAT4x3 );
          parameterCacheEntry = CacheEntryMatrix<4,3,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        case dp::rix::core::CPT_MAT4X4:
          DP_ASSERT( uniformType == GL_FLOAT_MAT4 );
          parameterCacheEntry = CacheEntryMatrix<4,4,float, float>::create( uniformInfo, containerOffset, newArraySize );
          break;
        default:
          DP_ASSERT( !"unknown type" );
        }
        return parameterCacheEntry;
      }

      ParameterCacheEntryStreamBuffers createParameterCacheEntriesStreamBuffer( dp::rix::gl::ProgramGLHandle program
        , dp::rix::gl::ContainerDescriptorGLHandle descriptor
        , dp::rix::gl::ProgramGL::UniformInfos const& uniformInfos )
      {
        std::vector<ParameterCacheEntryStreamBufferSharedPtr> parameterCacheEntries;
        size_t cacheOffset = 0;
        for ( dp::rix::gl::ProgramGL::UniformInfos::const_iterator it = uniformInfos.begin(); it != uniformInfos.end(); ++it )
        {
          size_t parameterIndex = descriptor->getIndex( it->first );
          parameterCacheEntries.push_back( createParameterCacheEntryStreamBuffer( program, descriptor->m_parameterInfos[parameterIndex].m_type, it->second
                                                                                , cacheOffset, descriptor->m_parameterInfos[parameterIndex].m_offset ) );
          cacheOffset += parameterCacheEntries.back()->getSize();
        }
        return parameterCacheEntries;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
