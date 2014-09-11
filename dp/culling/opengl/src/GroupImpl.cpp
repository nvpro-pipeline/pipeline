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


#include <GL\glew.h>
#if defined(GL_VERSION_4_3)

#include <dp/culling/opengl/inc/GroupImpl.h>
#include <dp/util/FrameProfiler.h>

namespace
{
  struct ShaderObject
  {
    // TODO pack matrix into 4th component of extent!
    //dp::math::Mat44f matrix;
    dp::util::Uint32 matrix;
    dp::util::Uint32 pad0;
    dp::util::Uint32 pad1;
    dp::util::Uint32 pad2;
    dp::math::Vec4f  lowerLeft;
    dp::math::Vec4f  extent;
  };
}

namespace dp
{
  namespace culling
  {
    namespace opengl
    {
      GroupImpl::GroupImpl()
        : m_outputBufferSize( ~0 )
      {
      }

      GroupImpl::~GroupImpl()
      {
      }

      void GroupImpl::update( size_t workGroupSize )
      {
        updateInputBuffer( workGroupSize );
        updateOutputBuffer( workGroupSize );
        updateMatrices();
      }

      void GroupImpl::updateInputBuffer( size_t workgroupSize )
      {
        dp::util::ProfileEntry p("cull::updateInputBuffer");

        if ( m_inputChanged )
        {
          if ( ! m_inputBuffer )
          {
            m_inputBuffer = dp::gl::Buffer::create();
          }

          size_t const numberOfObjects = getObjectCount();
          size_t const numberOfWorkingGroups = (numberOfObjects + workgroupSize - 1) / workgroupSize;
          m_inputBuffer->setData( GL_SHADER_STORAGE_BUFFER, numberOfWorkingGroups * workgroupSize * sizeof(ShaderObject), nullptr, GL_STATIC_DRAW );
          dp::gl::MappedBuffer<ShaderObject> inputs( m_inputBuffer, GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY );
          DP_ASSERT( inputs );

          // generate list of objects to cull from shader
          for ( size_t index = 0;index < numberOfObjects; ++index )
          {
            const ObjectBitSetHandle& objectImpl = getObject( index );
            ShaderObject &object = inputs[index];
            object.matrix = static_cast<dp::util::Uint32>(objectImpl->getTransformIndex());
            object.lowerLeft = objectImpl->getLowerLeft();
            object.extent = objectImpl->getExtent();
          }

          // initialize unused objects in workgroup
          for ( size_t index = numberOfObjects; index < numberOfWorkingGroups * workgroupSize; ++index )
          {
            ShaderObject &object = inputs[index];
            object.matrix = 0;
            object.extent = dp::math::Vec4f(0.0, 0.0f, 0.0f, 0.0f);
            object.lowerLeft = dp::math::Vec4f( 0.0f, 0.0f, 0.0f, 0.0f);
          }

          m_inputChanged = false;
        }
      }

      void GroupImpl::updateOutputBuffer( size_t workGroupSize )
      {
        size_t const numberOfObjects = getObjectCount();
        size_t const numberOfWorkingGroups = (numberOfObjects + workGroupSize - 1) / workGroupSize;
        size_t const numberOfResults = numberOfWorkingGroups * workGroupSize;

        GLsizei newBufferOutputSize = static_cast<GLsizei>(numberOfResults / 8);

        if ( m_outputBufferSize != newBufferOutputSize )
        {
          m_outputBufferSize = newBufferOutputSize;

          if ( ! m_outputBuffer )
          {
            m_outputBuffer = dp::gl::Buffer::create();
          }

          m_outputBuffer->setData( GL_SHADER_STORAGE_BUFFER, m_outputBufferSize, nullptr, GL_STATIC_DRAW );
        }
      }

      void GroupImpl::updateMatrices( )
      {
        dp::util::ProfileEntry p("cull::updateMatrices");
        if ( m_matricesChanged )
        {
          if ( ! m_matricesBuffer )
          {
            m_matricesBuffer = dp::gl::Buffer::create();
          }

          // copy over matrices
          m_matricesBuffer->setData( GL_SHADER_STORAGE_BUFFER, getMatricesCount() * sizeof( dp::math::Mat44f ), nullptr, GL_STATIC_DRAW );
          dp::gl::MappedBuffer<dp::math::Mat44f> matrices( m_matricesBuffer, GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY );
          char const* basePtr = reinterpret_cast<char const*>(getMatrices());
          for ( size_t index = 0; index < getMatricesCount(); ++index )
          {
            dp::math::Mat44f const& modelView = reinterpret_cast<dp::math::Mat44f const&>(*(basePtr + index * getMatricesStride()));
            matrices[index] = modelView;
          }

          m_matricesChanged = false;
        }
        else
        {
          struct MatrixUpdater
          {
            MatrixUpdater( char const* matricesInBasePtr, size_t matricesInStride )
              : m_matricesInBasePtr( matricesInBasePtr )
              , m_matricesInStride( matricesInStride )
            {
            }

            void operator()( size_t index )
            {
              glBufferSubData( GL_SHADER_STORAGE_BUFFER, index * sizeof(dp::math::Mat44f), sizeof(dp::math::Mat44f), m_matricesInBasePtr + index * m_matricesInStride );
            }

          private:
            char const*       m_matricesInBasePtr;
            size_t            m_matricesInStride;
          };

          bind( GL_SHADER_STORAGE_BUFFER, m_matricesBuffer );
          MatrixUpdater matrixUpdater( reinterpret_cast<char const*>(getMatrices()), getMatricesStride() );
          m_dirtyMatrices.traverseBits( matrixUpdater );
        }
        m_dirtyMatrices.clear();
      }

    } // namespace opengl
  } // namespace culling
} // namespace dp

// GL_VERSION_4_3
#endif
