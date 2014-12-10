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


#pragma once

#include <GL/glew.h>
#if defined(GL_VERSION_4_3)

#include <dp/culling/GroupBitSet.h>
#include <dp/gl/Buffer.h>
#include <boost/shared_array.hpp>

namespace dp
{
  namespace culling
  {
    namespace opengl
    {
      DEFINE_PTR_TYPES( GroupImpl );

      class GroupImpl : public GroupBitSet
      {
      public:
        static GroupImplSharedPtr create();
        virtual ~GroupImpl();

        void update( size_t workGroupSize );

        dp::gl::BufferSharedPtr const & getInputBuffer() { return m_inputBuffer; }
        dp::gl::BufferSharedPtr const & getMatrixBuffer() { return m_matricesBuffer; }
        dp::gl::BufferSharedPtr const & getOutputBuffer() { return m_outputBuffer; }

      protected:
        GroupImpl();
        void updateMatrices( );
        void updateInputBuffer( size_t workGroupSize );
        void updateOutputBuffer( size_t workGroupSize );

      private:
        dp::gl::BufferSharedPtr  m_inputBuffer;
        dp::gl::BufferSharedPtr  m_matricesBuffer;
        dp::gl::BufferSharedPtr  m_outputBuffer;
        GLsizei               m_outputBufferSize;
      };

    } // namespace opengl
  } // namespace culling
} // namespace dp

// GL_VERSION_4_3
#endif
