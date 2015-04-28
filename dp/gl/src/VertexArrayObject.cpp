// Copyright NVIDIA Corporation 2014
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

#include <dp/gl/VertexArrayObject.h>

namespace dp
{
  namespace gl
  {
    VertexArrayObjectSharedPtr VertexArrayObject::create()
    {
      return( std::shared_ptr<VertexArrayObject>( new VertexArrayObject() ) );
    }

    VertexArrayObject::VertexArrayObject()
    {
      glGenVertexArrays( 1, &m_id );
    }

    VertexArrayObject::~VertexArrayObject()
    {
      if ( m_id )
      {
        if ( getShareGroup() )
        {
          DEFINE_PTR_TYPES( CleanupTask );
          class CleanupTask : public ShareGroupTask
          {
            public:
              static CleanupTaskSharedPtr create( GLuint id )
              {
                return( std::shared_ptr<CleanupTask>( new CleanupTask( id ) ) );
              }

              virtual void execute() { glDeleteVertexArrays( 1, &m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
          };

          // make destructor exception safe
          try
          {
            getShareGroup()->executeTask( CleanupTask::create( m_id ) );
          } catch (...) {}
        }
        else
        {
          glDeleteVertexArrays( 1, &m_id );
        }
      }
    }

    void VertexArrayObject::setIndices()
    {
      glBindVertexArray( m_id );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_indices->getGLId() );
    }

  } // namespace gl
} // namespace dp
