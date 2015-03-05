// Copyright NVIDIA Corporation 2011-2012
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


#include <BufferGL.h>
#include <typeinfo>

#include "DataTypeConversionGL.h"

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      /************************************************************************/
      /* BufferGL::Event                                                      */
      /************************************************************************/
      BufferGL::Event::Event( BufferGL *buffer, Event::EventType eventType )
        : m_buffer( buffer )
        , m_eventType( eventType )
      {
      }

      BufferGL::BufferGL( const dp::rix::core::BufferDescription& bufferDescription )
        : m_format( dp::rix::core::BF_UNKNOWN )
        , m_elementSize( 1 )  // NOTE: set to one for now ( see setElementSize )
        , m_usageHint( GL_STATIC_DRAW )
        , m_width( 0 )
        , m_height( 0 )
        , m_depth( 0 )
#if !defined(NDEBUG)
        , m_accessType( dp::rix::core::AT_NONE )
#endif
#if 0
        , m_numBufferSlots( 0 )
        , m_numSamplerSlots( 0 )
        , m_bufferReferences( nullptr )
        , m_samplerReferences( nullptr )
#endif
      {
        if ( bufferDescription.m_type == dp::rix::core::BDT_NATIVE )
        {
          DP_ASSERT( dynamic_cast<const BufferDescriptionGL*>( &bufferDescription ) );
          const BufferDescriptionGL& bd = static_cast<const BufferDescriptionGL&>( bufferDescription );

          m_buffer      = bd.m_buffer;
          m_usageHint   = getGLUsage( bd.m_usageHint );  // It's possible to set only the usage hint through this path!
#if !defined(NDEBUG)
          m_managesData = false;
#endif
        }
        
        // For dp::rix::core::BufferDescription or BufferdescriptionGL default values the OpenGL buffer ID is zero.
        // That means to generate one here. Ownership automatically belongs to this BufferGL then. 
        if ( !m_buffer )
        {
          m_buffer = dp::gl::Buffer::create(dp::gl::Buffer::CORE, m_usageHint);
          DP_ASSERT( m_buffer && "Couldn't create OpenGL buffer object" );
#if !defined(NDEBUG)
          m_managesData     = true;
#endif
        }
      }

      BufferGL::~BufferGL()
      {
#if 0
        resetBufferReferences();
        resetSamplerReferences();
#endif
      }

      BufferGLHandle BufferGL::create( const dp::rix::core::BufferDescription& bufferDescription )
      {
        return new BufferGL( bufferDescription );
      }

      void BufferGL::setSize( size_t width, size_t height /*= 0*/, size_t depth /*= 0 */ )
      {
        DP_ASSERT( m_accessType == dp::rix::core::AT_NONE );

        m_width  = width;
        m_height = height;
        m_depth  = depth;

        // calculate dimensionality: W -> 1, WH -> 2, WHD -> 3
        unsigned int dimensionality = !m_width ? 0 : ( !m_height ? 1 : ( !m_depth ? 2 : 3 ) ); 
        DP_ASSERT( dimensionality != 0 );

        // Calculate the size in bytes to verify later setData() calls.
        size_t size = m_elementSize * m_width;
        if (dimensionality > 1)
        {
          size *= m_height;
          if (dimensionality > 2)
          {
            size *= m_depth;
          }
        }

        if ( size != m_buffer->getSize() )
        {
          m_buffer->setSize(size);
          notify( Event( this, Event::DATA_AND_SIZE_CHANGED ) ); // Data is undefined after glBufferData with nullptr.
        }
      }

      void BufferGL::setElementSize( size_t elementSize )
      {
        DP_ASSERT( !"Not supported" );
        DP_ASSERT( m_accessType == dp::rix::core::AT_NONE );
        m_elementSize = elementSize;
      }

      void BufferGL::setFormat( dp::rix::core::BufferFormat bufferFormat )
      {
        DP_ASSERT( !"Not supported" );
        DP_ASSERT( m_accessType == dp::rix::core::AT_NONE );
        m_format = bufferFormat;
      }

      void BufferGL::updateData( size_t offset, const void *data, size_t size )
      {
        DP_ASSERT( m_accessType == dp::rix::core::AT_NONE );
        DP_ASSERT( offset + size <= m_buffer->getSize() );
        DP_ASSERT( m_managesData );
        // DP_ASSERT( m_format != dp::rix::core::BF_UNKNOWN ); // TODO: check this when setFormat and setElementsize are implemented

        m_buffer->update(data, offset, size);
        notify( Event( this, Event::DATA_CHANGED ) );
      }

#if 0
      void BufferGL::resetBufferReferences()
      {
        if ( m_bufferReferences )
        {
          if ( m_samplerReferences )
          {
            delete [] m_samplerReferences;
            m_numSamplerSlots = 0;
          }
        }
      }

      void BufferGL::resetSamplerReferences()
      {
        if ( m_samplerReferences )
        {
          delete [] m_samplerReferences;
          m_numSamplerSlots = 0;
        }
      }

      inline void BufferGL::initReferences( const BufferReferencesBuffer &infoData )
      {
        resetBufferReferences();
        m_numBufferSlots = infoData.m_numSlots;
        m_bufferReferences = new BufferReference[infoData.m_numSlots];
      }

      inline void BufferGL::initReferences( const BufferReferencesSampler &infoData )
      {
        resetSamplerReferences();
        m_numSamplerSlots  = infoData.m_numSlots;
        m_samplerReferences = new SamplerReference[infoData.m_numSlots];
      }

      void BufferGL::initReferences( const BufferReferences &infoData )
      {
        switch ( infoData.getBufferReferenceType() )
        {
        case dp::rix::core::BRT_BUFFER:
          DP_ASSERT( dynamic_cast<const dp::rix::core::BufferReferencesBuffer*>(&infoData) );
          initReferences( static_cast<const dp::rix::core::BufferReferencesBuffer&>( infoData) );
          break;
        case dp::rix::core::BRT_SAMPLER:
          DP_ASSERT( dynamic_cast<const dp::rix::core::BufferReferencesSampler*>(&infoData) );
          initReferences( static_cast<const dp::rix::core::BufferReferencesSampler&>( infoData ) );
          break;
        default:
          DP_ASSERT(!"Unsupported BufferReferenceType type.");
          break;
        }
      }

      inline void BufferGL::setReference( size_t slot, const ContainerDataBuffer &data, BufferStoredReferenceGL &refstore )
      {
        DP_ASSERT ( slot < m_numBufferSlots );

        if ( data.m_bufferHandle )
        {
          DP_ASSERT( handleIsTypeOf<BufferGL>( data.m_bufferHandle ) );
          BufferGLHandle bufferHandle = handleCast<BufferGL>( data.m_bufferHandle );
          m_bufferReferences[ slot ].m_bufferHandle = bufferHandle;
          refstore.address = bufferHandle->getAddress() + data.m_offset;
        }
        else
        {
          m_bufferReferences[ slot ].m_bufferHandle.reset();
          refstore.address = 0;
        }
      }

      inline void BufferGL::setReference( size_t slot, const ContainerDataSampler &data, BufferStoredReferenceGL &refstore )
      {
        if ( data.m_textureHandle )
        {
          DP_ASSERT( handleIsTypeOf<TextureGL>( data.m_textureHandle ) );
          TextureGLHandle textureHandle = handleCast<TextureGL>( data.m_textureHandle );

          SamplerStateGLHandle samplerStateHandle;
          if ( data.m_samplerStateHandle )
          {
            DP_ASSERT( handleIsTypeOf<SamplerStateGL>( data.m_samplerStateHandle) );
            samplerStateHandle = handleCast<SamplerStateGL>(data.m_samplerStateHandle);
          }
          else
          {
            samplerStateHandle = nullptr;
          }
          TextureGL::BindlessReferenceHandle bindlessHandle = textureHandle->getBindlessTextureHandle( samplerStateHandle );
          m_samplerReferences[ slot ].set(textureHandle, samplerStateHandle, bindlessHandle);

          refstore.address = bindlessHandle->getHandle();
        }
        else
        {
          m_samplerReferences[ slot ].set(nullptr,nullptr,nullptr);

          refstore.address = 0;
        }
      }

      void BufferGL::setReference( size_t slot, const ContainerData &containerData, BufferStoredReferenceGL &refstore )
      {
        switch ( containerData.getContainerDataType() )
        {
        case dp::rix::core::CDT_BUFFER:
          DP_ASSERT( dynamic_cast<const dp::rix::core::ContainerDataBuffer*>(&containerData) );
          setReference( slot, static_cast<const dp::rix::core::ContainerDataBuffer&>( containerData), refstore );
          break;
        case dp::rix::core::CDT_SAMPLER:
          DP_ASSERT( dynamic_cast<const dp::rix::core::ContainerDataSampler*>(&containerData) );
          setReference( slot, static_cast<const dp::rix::core::ContainerDataSampler&>( containerData), refstore );
          break;
        default:
          DP_ASSERT(!"Unsupported ContainerData type.");
          break;
        }
      }
#endif

      dp::gl::BufferSharedPtr const& BufferGL::getBuffer() const
      {
        return( m_buffer );
      }

      void* BufferGL::map( dp::rix::core::AccessType accessType )
      {
        DP_ASSERT( !"never passed this path!" );
        DP_ASSERT( m_buffer->getGLId() );
        DP_ASSERT( m_accessType == dp::rix::core::AT_NONE ); // Assert that the buffer is not mapped.
#if !defined(NDEBUG)
        m_accessType = accessType;
#endif

        return( m_buffer->map( getGLAccessBitField( accessType ) ) );
      }

      bool BufferGL::unmap()
      {
        DP_ASSERT( !"never passed this path!" );
        DP_ASSERT( m_buffer->getGLId() );
        DP_ASSERT( m_accessType != dp::rix::core::AT_NONE );

        bool success = !!m_buffer->unmap();

#if !defined(NDEBUG)
        m_accessType = dp::rix::core::AT_NONE; // Not mapped.
#endif
        return success;
      }

      /**************************************************************************************/
      /* The following code abstracts the call to glBindBufferRange vs. glBindBuffersRange. */
      /* glBindBufferRange updates the indexed and the non-indexed binding where usually    */
      /* only the indexed binding is of interest. glBindBuffersRange does not update the    */
      /* non-indexed binding and thus is a little bit faster. This feature is available     */
      /* starting with GL 4.4.                                                              */
      /**************************************************************************************/
      BindBufferRangePtr bindBufferRangeInternal = 0;

#if defined(GL_VERSION_4_4)
      void bindBufferRangeFast( GLenum target,  GLuint index,  GLuint buffer,  GLintptr offset,  GLsizeiptr size )
      {
        glBindBuffersRange( target, index, 1, &buffer, &offset, &size );
      }
#endif

      void initBufferBinding()
      {
#if defined(GL_VERSION_4_4)
        if ( GLEW_VERSION_4_4 || GLEW_ARB_multi_bind )
        {
          bindBufferRangeInternal = &bindBufferRangeFast;
        }
        else
#endif
        {
          bindBufferRangeInternal = glBindBufferRange;
        }
      }

    } // namespace gl
  } // namespace rix
} // namespace dp

