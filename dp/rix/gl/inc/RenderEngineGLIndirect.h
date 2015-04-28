// Copyright NVIDIA Corporation 2012-2015
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

#include <dp/rix/gl/inc/RenderEngineGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class RenderEngineGLIndirect : public RenderEngineGL
      {
      public:
        struct RenderGroupGLCache : RenderGroupGL::Cache
        {
          // TODO Indirect Engine not supported atm
          RenderGroupGLCache()
            : RenderGroupGL::Cache( nullptr, nullptr )
            , m_initialized(false)
          {
            assert( !"update me to new interface" );
          }

          ~RenderGroupGLCache()
          {
          }

          bool m_initialized;

          struct IndirectEntry
          {
            GLuint64EXT position;
            GLuint64EXT normal;
            GLuint64EXT transform;
            GLuint64EXT material;
          };

          /** MultiDrawElementsIndirect **/
          struct DrawArraysIndirectCommand {
            GLuint count;
            GLuint primCount;
            GLuint first;
            GLuint baseInstance;
          };

          struct DrawArrayListEntry
          {
            GLenum                                   m_primitiveType;
            std::vector< IndirectEntry >             m_indirectEntries;
            BufferGLHandle                           m_indirectEntriesBuffer;
            BufferGLHandle                           m_indirectPointersBuffer;
            std::vector< DrawArraysIndirectCommand > m_indirectCommands;
            BufferGLHandle                           m_indirectCommandsBuffer;
          };

          std::vector<DrawArrayListEntry>            m_drawArrayList;

          /** MultiDrawElementsIndirect **/
          struct DrawElementsIndirectCommand
          {
            GLuint count;
            GLuint primCount;
            GLuint firstIndex;
            GLuint baseVertex;
            GLuint baseInstance;
          };

          struct BindlessPointer
          {
            GLuint64EXT m_address;
            GLuint64EXT m_range;
          };

          struct DrawElementsListEntry
          {
            GLenum                                   m_primitiveType;
            std::vector< IndirectEntry >             m_indirectEntries;
            std::vector< BindlessPointer >           m_indexPointers;
            BufferGLHandle                           m_indirectEntriesBuffer;
            BufferGLHandle                           m_indirectPointersBuffer;
            GLuint64EXT                              m_indirectPointersBufferAddress;
            GLuint64EXT                              m_indirectPointersBufferRange;
            std::vector< DrawElementsIndirectCommand > m_indirectCommands;
            BufferGLHandle                           m_indirectCommandsBuffer;

            //GLuint                                   m_indexBuffer;
            //GLuint64EXT                              m_indexBufferAddress;
            //GLuint64EXT                              m_indexBufferRange;
          };

          std::vector<DrawElementsListEntry>         m_drawElementsList;
        };


        RenderEngineGLIndirect();
        virtual ~RenderEngineGLIndirect();

        virtual void beginRender();
        virtual void render( RenderGroupGLHandle groupHandle, const dp::rix::core::RenderOptions& renderOptions );
        virtual void render( RenderGroupGLHandle groupHandle, const dp::rix::core::GeometryInstanceHandle* gis, size_t numGIs, const dp::rix::core::RenderOptions& renderOptions );
        virtual void endRender();
        virtual RenderGroupGL::SmartCache createCache( RenderGroupGLHandle );

      };
    } // namespace gl
  } // namespace rix
} // namespace dp
