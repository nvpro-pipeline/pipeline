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


#include <dp/rix/core/RiX.h>

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

          class BufferAllocator {
          public:
              BufferAllocator( dp::rix::core::Renderer *renderer );
              dp::rix::core::VertexAttributesSharedHandle getVertexAttributes(dp::rix::core::Renderer *renderer, dp::sg::core::VertexAttributeSetSharedPtr const & vertexAttributeSet, unsigned int &baseVertex);
              dp::rix::core::IndicesSharedHandle getIndices(dp::rix::core::Renderer *renderer, dp::sg::core::IndexSetSharedPtr const & indexSet, unsigned int &baseIndex);

              size_t allocateVertices(dp::rix::core::Renderer *renderer, dp::rix::core::VertexFormatSharedHandle, unsigned int elementSize, size_t numElements, dp::rix::core::VertexAttributesSharedHandle &vertexAttributesHandle, dp::rix::core::BufferSharedHandle &buffer);
              size_t allocateIndices(dp::rix::core::Renderer *renderer, dp::util::DataType dataType, size_t numElements, dp::rix::core::IndicesSharedHandle &indices, dp::rix::core::BufferSharedHandle &buffer);

          private:
            struct BufferInfo {
              BufferInfo(size_t elementSize = 0);

              void createBuffer(dp::rix::core::Renderer *renderer, size_t numElements);
              /************************************************************************/
              /* allocate numElements in the buffer, return ~0 if allocation fails    */
              /************************************************************************/
              size_t allocate(size_t numElements);

              dp::rix::core::BufferSharedHandle m_buffer;
              size_t m_elementSize; // size of one element in the buffer
              size_t m_remaining; // remaining elements 
              size_t m_offset; // current allocation offset
            };
            
            struct VertexAttributeInfo {
                  BufferInfo bufferInfo;
                  dp::rix::core::VertexAttributesSharedHandle vertexAttributes;
            };

            typedef std::map<dp::rix::core::VertexFormatSharedHandle, VertexAttributeInfo> VertexAttributeInfos;

            // indices suballocator
            struct IndicesInfo {
                BufferInfo bufferInfo;
                dp::rix::core::IndicesSharedHandle indices;
            };

            typedef std::map<dp::util::DataType, IndicesInfo> IndicesInfos;

            VertexAttributeInfos m_vertexAttributeInfos;
            IndicesInfos m_indicesInfos;

            dp::rix::core::Renderer*            m_renderer;
          };


        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

