// Copyright NVIDIA Corporation 2010-2015
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
    /** \brief Buffer is a class around the OpenGL buffer object.
               It tries to utilize DSA for operations if possible. If DSA
               is not available it'll use a configurable standard-target for
               all operations. The target can be changed after any call in this class

               setSize() is also supported for immutable buffers. In this case setSize()
               will destroy the current buffer and create a new one.
    **/
    class Buffer : public Object
    {
      public:
        enum Mode_Core {CORE};
        enum Mode_Persistent_Buffer {PERSISTENT_BUFFER};

        /** \brief Create a new standard OpenGL buffer
            \param mode         Must be CORE
            \param usage        Usage flags passed to glBufferData
            \param defaultTaget Target used for operations if DSA is not available
        **/
        DP_GL_API static BufferSharedPtr create(Mode_Core, GLenum usage, GLenum defaultTarget = GL_UNIFORM_BUFFER);

        /** \brief Create a new OpenGL buffer
            \param mode Must be PERSISTENT_BUFFER creates an immutable buffer.
            \param mode GLbitfield from OpenGL (GL_MAP_WRITE_BIT, GL_MAP_READ_BIT, ...)
        **/
        DP_GL_API static BufferSharedPtr create(Mode_Persistent_Buffer, GLbitfield modeBits);
        DP_GL_API virtual ~Buffer();

      public:
        /** \brief Update the content of the buffer. 
            \param data The data to put into this buffer
            \param offset The offset where to put the data into this buffer
            \param length The number of bytes to update. If size is ~0 this function will update
                   (getSize() - offset) bytes.
        **/
        DP_GL_API virtual void update(void const *data, size_t offset = 0, size_t length = ~0) = 0;

        /** \brief Map a subrange of the buffer on the host.
            \param access The access bits when mapping the buffer.
            \param offset The first byte of the buffer to map
            \param length The number of bytes of the buffer to map. If the length is ~0 
                   (getSize() - offset) bytes will be mapped.
        **/
        DP_GL_API virtual void* map(GLbitfield access, size_t offset = 0, size_t length = ~0) = 0;

        /** \brief Remove the buffer mapping from the host
        **/
        DP_GL_API virtual GLboolean unmap() = 0;

        /** \brief Get the currently mapped address of the buffer.
            \return The currently mapped address of the buffer. If the buffer is not 
                    mapped a std::runtime error will be thrown.
        **/
        DP_GL_API void* getMappedAddress() const;

        /** \brief Query if the buffer is currently mapped on the host side
            \return true if the buffer is currently mapped on the host side
        **/
        DP_GL_API bool isMapped() const;

        /** \brief Set the new size of the buffer.
            \param size The new size of the buffer.
            If the buffer is a CORE buffer all data is lost and the buffer id is the same.
            If the buffer is a PERSISTENT_BUFFER all data is lost and the buffer id will change.
        **/
        DP_GL_API virtual void setSize(size_t size) = 0;

        /** \brief Invalidate the content of the buffer.
        **/
        DP_GL_API virtual void invalidate() = 0;

        /** \brief Get the current size of the buffer.
            \return The current size of the buffer.
        **/
        DP_GL_API size_t getSize() const;

        // unified memory

        /** \brief Make the buffer resident on the current context. If the buffer is already resident
                   in any context a std::runtime error will be thrown.
        **/
        DP_GL_API virtual void makeResident() = 0;

        /** \brief Make the buffer non-resident. This function does not check if the buffer is
                   resident in the current context, thus it is not valid to call makeNonResident
                   in another context than makeResident. If the buffer is not resident in any
                   context a std::runtime error will be thrown.
        **/
        DP_GL_API virtual void makeNonResident() = 0;

        /** \brief Get the GPU address of the buffer. If the buffer is not resident 
                   \sa Buffer::makeResident() will be called.
                   \return The current GPU address
        **/
        DP_GL_API GLuint64EXT getAddress();

        /** \brief Check if the buffer is resident on the GPU
            \return true if the buffer buffer is resident on the GPU.
        **/
        DP_GL_API bool        isResident() const;

      protected:
        DP_GL_API Buffer();

      protected:
        mutable GLuint64EXT m_address; // 64-bit bindless address
        size_t              m_size;
        void*               m_mappedAddress;
    };

    inline size_t Buffer::getSize() const
    {
      return m_size;
    }


    inline void* Buffer::getMappedAddress() const
    {
      if (!m_mappedAddress) {
        throw std::runtime_error("buffer is not mapped");
      }
      return m_mappedAddress;
    }

    inline bool Buffer::isMapped() const
    {
      return !!m_mappedAddress;
    }

    inline GLuint64EXT Buffer::getAddress()
    {
      if (!m_address)
      {
        makeResident();
      }
      DP_ASSERT(m_address);
      return m_address;
    }

    inline bool Buffer::isResident() const
    {
      return !!m_address;
    }

    DP_GL_API void bind( GLenum target, BufferSharedPtr const& buffer );
    DP_GL_API void copy( BufferSharedPtr const& src, BufferSharedPtr const& dst, size_t srcOffset, size_t dstOffset, size_t size );


    template <typename T>
    class MappedBuffer
    {
      public:
        MappedBuffer( BufferSharedPtr const& buffer, GLbitfield access );
        ~MappedBuffer();

      public:
        operator T*() const;

      private:
        BufferSharedPtr   m_buffer;
        T               * m_ptr;
    };


    template <typename T>
    inline MappedBuffer<T>::MappedBuffer( BufferSharedPtr const& buffer, GLbitfield access )
      : m_buffer( buffer )
    {
      m_ptr = reinterpret_cast<T*>(m_buffer->map(access));
    }

    template <typename T>
    inline MappedBuffer<T>::~MappedBuffer()
    {
      DP_VERIFY(m_buffer->unmap());
    }

    template <typename T>
    inline MappedBuffer<T>::operator T*() const
    {
      return( m_ptr );
    }

  } // namespace gl
} // namespace dp
