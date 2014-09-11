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


#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/renderer/rix/gl/inc/BufferAllocator.h>
#include <dp/util/Memory.h>

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

                    static const size_t AllocationSize = 500000;
                    static const size_t AllocationMaxChunkSize = 100000;

                    BufferAllocator::BufferInfo::BufferInfo(size_t elementSize)
                        : m_elementSize(elementSize)
                        , m_remaining(0)
                        , m_offset(0)
                    {

                    }

                    void BufferAllocator::BufferInfo::createBuffer(dp::rix::core::Renderer *renderer, size_t numElements)
                    {
                        m_buffer = renderer->bufferCreate();
                        m_remaining = AllocationSize;
                        m_offset = 0;
                        renderer->bufferSetSize(m_buffer, m_remaining * m_elementSize);
                    }

                    size_t BufferAllocator::BufferInfo::allocate(size_t numElements)
                    {
                        if (numElements <= m_remaining)
                        {
                            size_t offset = m_offset;
                            m_remaining -= numElements;
                            m_offset += numElements;
                            return offset;
                        }
                        return ~0;
                    }

                    BufferAllocator::BufferAllocator( dp::rix::core::Renderer *renderer )
                        : m_renderer( renderer )
                    {

                    }

                    dp::rix::core::VertexAttributesSharedHandle BufferAllocator::getVertexAttributes(dp::rix::core::Renderer *renderer, dp::sg::core::VertexAttributeSetSharedPtr const & vertexAttributeSet, unsigned int & baseVertex)
                    {
                        dp::rix::core::VertexAttributesSharedHandle rixVA;

                        std::vector<dp::rix::core::VertexFormatInfo>  vertexInfos;
                        std::vector<unsigned int> sourceAttributes;

                        unsigned int offset = 0;
                        baseVertex = 0;

                        // TODO hack, magic number!
                        for (unsigned int index = 0;index < 16;++index)
                        {
                            if (vertexAttributeSet->isEnabled(index))
                            {
                                dp::sg::core::VertexAttribute const & va = vertexAttributeSet->getVertexAttribute(index);

                                dp::rix::core::VertexFormatInfo vfi( index, 
                                    va.getVertexDataType(), 
                                    va.getVertexDataSize(),
                                    false, 
                                    0, // reducing to a single stream always!
                                    offset, // offset, fill in later
                                    0 // stride, fill in later
                                    );
                                vertexInfos.push_back(vfi);
                                sourceAttributes.push_back(index);

                                offset += (unsigned int)(getSizeOf(va.getVertexDataType()) * va.getVertexDataSize());
                            }
                        }

                        if (!sourceAttributes.empty()) {
                            unsigned int stride = offset;
                            size_t numberOfVertices = vertexAttributeSet->getNumberOfVertexData(sourceAttributes.front());

                            std::vector<dp::util::Uint8> vertices(numberOfVertices * stride);
                            for (size_t index = 0;index < sourceAttributes.size();++index) {
                                dp::sg::core::VertexAttribute const & va = vertexAttributeSet->getVertexAttribute(sourceAttributes[index]);

                                vertexInfos[index].m_stride = stride;

                                dp::sg::core::BufferSharedPtr const & buffer = vertexAttributeSet->getVertexBuffer(sourceAttributes[index]);
                                dp::sg::core::Buffer::DataReadLock drl(buffer);
                                dp::util::stridedMemcpy(vertices.data(), vertexInfos[index].m_offset, vertexInfos[index].m_stride, 
                                    drl.getPtr(), va.getVertexDataOffsetInBytes(), va.getVertexDataStrideInBytes(), va.getVertexDataBytes(),
                                    numberOfVertices);
                            }

                            dp::rix::core::VertexFormatDescription vertexFormatDescription(vertexInfos.empty() ? nullptr : &vertexInfos[0], vertexInfos.size());
                            dp::rix::core::VertexFormatSharedHandle vertexFormat = renderer->vertexFormatCreate( vertexFormatDescription );

                            dp::rix::core::BufferSharedHandle buffer;
                            if (numberOfVertices < AllocationMaxChunkSize) {
                                baseVertex = (unsigned int)(allocateVertices(renderer, vertexFormat, stride, numberOfVertices, rixVA, buffer));
                            }
                            else {
                                buffer = renderer->bufferCreate();
                                renderer->bufferSetSize(buffer, numberOfVertices * stride);

                                dp::rix::core::VertexFormatDescription vertexFormatDescription(vertexInfos.empty() ? nullptr : &vertexInfos[0], vertexInfos.size());
                                dp::rix::core::VertexFormatSharedHandle vertexFormat = renderer->vertexFormatCreate( vertexFormatDescription );

                                dp::rix::core::VertexDataSharedHandle vertexData = renderer->vertexDataCreate();
                                renderer->vertexDataSet(vertexData, 0, buffer, 0, numberOfVertices);
                                rixVA = renderer->vertexAttributesCreate();
                                renderer->vertexAttributesSet( rixVA, vertexData, vertexFormat );
                                baseVertex = 0;
                            }

                            renderer->bufferUpdateData(buffer, baseVertex * stride, vertices.data(), vertices.size());

                        }

                        return rixVA;
                    }

                    dp::rix::core::IndicesSharedHandle BufferAllocator::getIndices(dp::rix::core::Renderer *renderer, dp::sg::core::IndexSetSharedPtr const & indexSet, unsigned int &baseIndex)
                    {
                        dp::rix::core::IndicesSharedHandle indices;
                        dp::rix::core::BufferSharedHandle buffer;
                        size_t numberOfIndices = indexSet->getNumberOfIndices();
                        size_t elementSize = getSizeOf(indexSet->getIndexDataType());

                        if (numberOfIndices < AllocationMaxChunkSize) {
                            baseIndex = (unsigned int)(allocateIndices(renderer, indexSet->getIndexDataType(), numberOfIndices, indices, buffer));
                        }
                        else {
                            buffer = renderer->bufferCreate();
                            renderer->bufferSetSize(buffer, numberOfIndices * elementSize);

                            indices = renderer->indicesCreate();
                            renderer->indicesSetData(indices, indexSet->getIndexDataType(), buffer, 0, numberOfIndices);

                            baseIndex = 0;
                        }

                        dp::sg::core::Buffer::DataReadLock drl(indexSet->getBuffer());
                        renderer->bufferUpdateData(buffer, baseIndex * elementSize, drl.getPtr() /* TODO offset? */, numberOfIndices * elementSize);

                        return indices;
                    }

                    size_t BufferAllocator::allocateVertices(dp::rix::core::Renderer *renderer, dp::rix::core::VertexFormatSharedHandle vertexFormat, unsigned int elementSize, size_t numElements, dp::rix::core::VertexAttributesSharedHandle &vertexAttributesHandle, dp::rix::core::BufferSharedHandle &buffer)
                    {
                        VertexAttributeInfos::iterator it = m_vertexAttributeInfos.find(vertexFormat);
                        if (it == m_vertexAttributeInfos.end()) {
                            it = m_vertexAttributeInfos.insert(std::make_pair(vertexFormat, VertexAttributeInfo())).first;
                            it->second.bufferInfo = BufferInfo(elementSize);
                        }
                        size_t offset = it->second.bufferInfo.allocate(numElements);

                        if (offset == ~0) {
                            size_t numberOfVertices = AllocationSize;
                            it->second.bufferInfo.createBuffer(renderer, numberOfVertices);
                            offset = it->second.bufferInfo.allocate(numElements);
                            dp::rix::core::VertexDataSharedHandle vertexData = renderer->vertexDataCreate();
                            renderer->vertexDataSet(vertexData, 0, it->second.bufferInfo.m_buffer, 0, numberOfVertices);
                            it->second.vertexAttributes = renderer->vertexAttributesCreate();
                            renderer->vertexAttributesSet( it->second.vertexAttributes, vertexData, vertexFormat );
                        }
                        vertexAttributesHandle = it->second.vertexAttributes;
                        buffer = it->second.bufferInfo.m_buffer;

                        return offset;
                    }

                    size_t BufferAllocator::allocateIndices(dp::rix::core::Renderer *renderer, dp::util::DataType dataType, size_t numElements, dp::rix::core::IndicesSharedHandle &indices, dp::rix::core::BufferSharedHandle &buffer)
                    {
                        IndicesInfos::iterator it = m_indicesInfos.find(dataType);
                        if (it == m_indicesInfos.end()) {
                            it = m_indicesInfos.insert(std::make_pair(dataType, IndicesInfo())).first;
                            it->second.bufferInfo = BufferInfo(getSizeOf(dataType));
                        }
                        size_t offset = it->second.bufferInfo.allocate(numElements);

                        if (offset == ~0) {
                            size_t numberOfIndices = AllocationSize;
                            it->second.bufferInfo.createBuffer(renderer, numberOfIndices);
                            offset = it->second.bufferInfo.allocate(numElements);
                            it->second.indices = renderer->indicesCreate();
                            renderer->indicesSetData(it->second.indices, dataType, it->second.bufferInfo.m_buffer, 0, numberOfIndices);
                        }
                        indices = it->second.indices;
                        buffer = it->second.bufferInfo.m_buffer;

                        return offset;
                    }

                } // namespace gl
            } // namespace rix
        } // namespace renderer
    } // namespace sg
} // namespace dp
