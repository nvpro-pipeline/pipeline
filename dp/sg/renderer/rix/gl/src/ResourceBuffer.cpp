// Copyright NVIDIA Corporation 2011
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


#include <dp/sg/renderer/rix/gl/inc/ResourceBuffer.h>
#include <dp/sg/core/Buffer.h>
#include <dp/sg/gl/BufferGL.h>
#include <dp/rix/gl/RiXGL.h>

namespace dp
{
  namespace sg
  {
    namespace renderer
    {
      namespace rix
      {
        namespace gl
        {

          ResourceBufferSharedPtr ResourceBuffer::get( const dp::sg::core::BufferSharedPtr &buffer, const ResourceManagerSharedPtr& resourceManager )
          {
            assert( buffer );
            assert( !!resourceManager );
    
            ResourceBufferSharedPtr resourceBuffer = resourceManager->getResource<ResourceBuffer>( reinterpret_cast<size_t>(buffer.operator->()) );   // Big Hack !!
            if ( !resourceBuffer )
            {
              resourceBuffer = std::shared_ptr<ResourceBuffer>( new ResourceBuffer( buffer, resourceManager ) );
              if ( buffer.isPtrTo<dp::sg::gl::BufferGL>() )
              {
                dp::sg::gl::BufferGLSharedPtr const& buffergl = buffer.staticCast<dp::sg::gl::BufferGL>();
                dp::rix::gl::BufferDescriptionGL bufferDescription( dp::rix::gl::UH_STATIC_DRAW, buffergl->getBuffer() );

                resourceBuffer->m_bufferHandle = resourceManager->getRenderer()->bufferCreate( bufferDescription );
                resourceBuffer->m_isNativeBuffer = true;
              }
              else
              {
                resourceBuffer->m_bufferHandle = resourceManager->getRenderer()->bufferCreate();
              }
              resourceBuffer->update();
            }
    
            return resourceBuffer;
          }

          ResourceBuffer::ResourceBuffer( const dp::sg::core::BufferSharedPtr &buffer, const ResourceManagerSharedPtr& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( buffer.operator->() ), resourceManager )   // Big Hack !!
            , m_buffer( buffer )
            , m_isNativeBuffer( false )
            , m_bufferSize( 0 )
          {
            resourceManager->subscribe( this );
          }

          ResourceBuffer::~ResourceBuffer()
          {
            if ( m_resourceManager )
            {
              m_resourceManager->unsubscribe( this );
            }
          }

          const dp::sg::core::HandledObjectSharedPtr& ResourceBuffer::getHandledObject() const
          {
            return m_buffer.inplaceCast<dp::sg::core::HandledObject>();
          }

          void ResourceBuffer::update()
          {
            if ( m_isNativeBuffer)
            {
              return;
            }

            size_t bufferSize = m_buffer->getSize();
            dp::sg::core::Buffer::DataReadLock drl( m_buffer );
            assert( m_resourceManager );
            assert( m_resourceManager->getRenderer() );
            if( bufferSize != m_bufferSize )
            {
              m_bufferSize = bufferSize;
              m_resourceManager->getRenderer()->bufferSetSize( m_bufferHandle, m_bufferSize );
            }
            m_resourceManager->getRenderer()->bufferUpdateData( m_bufferHandle, 0, drl.getPtr(), bufferSize );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

