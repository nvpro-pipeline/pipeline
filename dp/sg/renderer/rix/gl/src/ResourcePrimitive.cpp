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


#include <dp/sg/renderer/rix/gl/inc/ResourcePrimitive.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/core/IndexSet.h>

//#define USE_SUBALLOCATOR 1

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

          class ResourcePrimitiveStandard : public ResourcePrimitive {
          public:
            ResourcePrimitiveStandard(const dp::sg::core::PrimitiveSharedPtr &primitive, const ResourceManagerSharedPtr& resourceManager);

          protected:
            virtual void updateVertexAttributesAndIndices();
            virtual dp::rix::core::VertexAttributesSharedHandle getVertexAttributes() const;
            virtual dp::rix::core::IndicesSharedHandle getIndices() const;
            virtual unsigned int getBaseVertex() const;
            virtual unsigned int getFirstIndex() const;

          private:
            ResourceIndexSetSharedPtr           m_resourceIndexSet;
            ResourceVertexAttributeSetSharedPtr m_resourceVertexAttributeSet;
          };

          ResourcePrimitiveStandard::ResourcePrimitiveStandard(const dp::sg::core::PrimitiveSharedPtr &primitive, const ResourceManagerSharedPtr& resourceManager)
            : ResourcePrimitive(primitive, resourceManager)
          {
          }

          dp::rix::core::VertexAttributesSharedHandle ResourcePrimitiveStandard::getVertexAttributes() const
          {
            return m_resourceVertexAttributeSet ? m_resourceVertexAttributeSet->m_vertexAttributesHandle : nullptr;
          }

          dp::rix::core::IndicesSharedHandle ResourcePrimitiveStandard::getIndices() const
          {
            return m_resourceIndexSet ? m_resourceIndexSet->m_indicesHandle : nullptr;
          }

          void ResourcePrimitiveStandard::updateVertexAttributesAndIndices()
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            /** copy over vertex data **/
            {
              dp::sg::core::VertexAttributeSetSharedPtr vas = m_primitive->getVertexAttributeSet();
              m_resourceVertexAttributeSet = ResourceVertexAttributeSet::get( vas, m_resourceManager );
            }

            /** copy over indices **/
            {
              dp::sg::core::IndexSetSharedPtr indexSet = m_primitive->getIndexSet();
              if ( indexSet )
              {
                m_resourceIndexSet = ResourceIndexSet::get( indexSet, m_resourceManager );
              }
              else
              {
                m_resourceIndexSet.reset();
              }
            }
          }

          unsigned int ResourcePrimitiveStandard::getBaseVertex() const
          {
            return 0;
          }

          unsigned int ResourcePrimitiveStandard::getFirstIndex() const
          {
            return 0;
          }

          /************************************************************************/
          /* Use a simple buffer suballocator for the indices and vertices        */
          /* There's no freelist reuse, defragmentation or other clever logic yet */
          /* The vertex format during the upload is always changes to interleaved */
          /************************************************************************/
          class ResourcePrimitiveSubAllocator : public ResourcePrimitive {
          public:
            ResourcePrimitiveSubAllocator(const dp::sg::core::PrimitiveSharedPtr &primitive, const ResourceManagerSharedPtr& resourceManager);

          protected:
            virtual void updateVertexAttributesAndIndices();
            virtual dp::rix::core::VertexAttributesSharedHandle getVertexAttributes() const;
            virtual dp::rix::core::IndicesSharedHandle getIndices() const;
            virtual unsigned int getBaseVertex() const;
            virtual unsigned int getFirstIndex() const;

          private:
            dp::rix::core::IndicesSharedHandle          m_resourceIndices;
            dp::rix::core::VertexAttributesSharedHandle m_resourceVertexAttributeSet;
            unsigned int                                m_baseVertex;
            unsigned int                                m_firstIndex;
          };

          ResourcePrimitiveSubAllocator::ResourcePrimitiveSubAllocator(const dp::sg::core::PrimitiveSharedPtr &primitive, const ResourceManagerSharedPtr& resourceManager)
            : ResourcePrimitive(primitive, resourceManager)
          {
          }

          dp::rix::core::VertexAttributesSharedHandle ResourcePrimitiveSubAllocator::getVertexAttributes() const
          {
            return m_resourceVertexAttributeSet;
          }

          dp::rix::core::IndicesSharedHandle ResourcePrimitiveSubAllocator::getIndices() const
          {
            return m_resourceIndices;
          }

          void ResourcePrimitiveSubAllocator::updateVertexAttributesAndIndices()
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            BufferAllocator &bufferAllocator = m_resourceManager->getBufferAllocator();

            /** copy over vertex data **/
            m_resourceVertexAttributeSet = bufferAllocator.getVertexAttributes(m_resourceManager->getRenderer(), m_primitive->getVertexAttributeSet(), m_baseVertex);

            dp::sg::core::IndexSetSharedPtr indexSet = m_primitive->getIndexSet();
            if ( indexSet )
            {
              m_resourceIndices = bufferAllocator.getIndices(m_resourceManager->getRenderer(), indexSet, m_firstIndex);
            }
            else
            {
              m_resourceIndices.reset();
            }

          }

          unsigned int ResourcePrimitiveSubAllocator::getBaseVertex() const
          {
            return m_baseVertex;
          }

          unsigned int ResourcePrimitiveSubAllocator::getFirstIndex() const
          {
            return m_firstIndex;
          }

          ResourcePrimitiveSharedPtr ResourcePrimitive::get( const dp::sg::core::PrimitiveSharedPtr &primitive, const ResourceManagerSharedPtr& resourceManager )
          {
            assert( primitive );
            assert( resourceManager );

            ResourcePrimitiveSharedPtr resourcePrimitive = resourceManager->getResource<ResourcePrimitive>( reinterpret_cast<size_t>(primitive.operator->()) );   // Big Hack !!
            if ( !resourcePrimitive )
            {
#if defined(USE_SUBALLOCATOR)
              resourcePrimitive = std::shared_ptr<ResourcePrimitive>( new ResourcePrimitiveSubAllocator( primitive, resourceManager ) );
#else
              resourcePrimitive = std::shared_ptr<ResourcePrimitive>( new ResourcePrimitiveStandard( primitive, resourceManager ) );
#endif
              resourcePrimitive->m_geometryHandle = resourceManager->getRenderer()->geometryCreate();
              resourcePrimitive->update();
            }
            return resourcePrimitive;
          }

          ResourcePrimitive::ResourcePrimitive( const dp::sg::core::PrimitiveSharedPtr &primitive, const ResourceManagerSharedPtr& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( primitive.operator->() ), resourceManager )    // Big Hack !!
            , m_primitive( primitive )
          {
            resourceManager->subscribe( this );
          }

          ResourcePrimitive::~ResourcePrimitive()
          {
            m_resourceManager->unsubscribe( this );
          }

          const dp::sg::core::HandledObjectSharedPtr& ResourcePrimitive::getHandledObject() const
          {
            return m_primitive.inplaceCast<dp::sg::core::HandledObject>();
          }

          void ResourcePrimitive::update()
          {
            updateVertexAttributesAndIndices();

            dp::GeometryPrimitiveType primitiveType;
            switch (m_primitive->getPrimitiveType())
            {
            case dp::sg::core::PrimitiveType::POINTS:
                primitiveType = dp::GeometryPrimitiveType::POINTS;
            break;
            case dp::sg::core::PrimitiveType::LINE_STRIP:
                primitiveType = dp::GeometryPrimitiveType::LINE_STRIP;
                break;
            case dp::sg::core::PrimitiveType::LINE_LOOP:
                primitiveType = dp::GeometryPrimitiveType::LINE_LOOP;
                break;
            case dp::sg::core::PrimitiveType::LINES:
                primitiveType = dp::GeometryPrimitiveType::LINES;
                break;
            case dp::sg::core::PrimitiveType::TRIANGLE_STRIP:
                primitiveType = dp::GeometryPrimitiveType::TRIANGLE_STRIP;
                break;
            case dp::sg::core::PrimitiveType::TRIANGLE_FAN:
                primitiveType = dp::GeometryPrimitiveType::TRIANGLE_FAN;
                break;
            case dp::sg::core::PrimitiveType::TRIANGLES:
                primitiveType = dp::GeometryPrimitiveType::TRIANGLES;
                break;
            case dp::sg::core::PrimitiveType::QUAD_STRIP:
                primitiveType = dp::GeometryPrimitiveType::QUAD_STRIP;
                break;
            case dp::sg::core::PrimitiveType::QUADS:
                primitiveType = dp::GeometryPrimitiveType::QUADS;
                break;
            case dp::sg::core::PrimitiveType::POLYGON:
                primitiveType = dp::GeometryPrimitiveType::POLYGON;
                break;
            case dp::sg::core::PrimitiveType::PATCHES:
                primitiveType = dp::GeometryPrimitiveType::PATCHES;
                break;
            default:
              assert( 0 && "unknown primitive type" );
            }

            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            dp::rix::core::GeometryDescriptionSharedHandle geometryDescription = renderer->geometryDescriptionCreate();

            unsigned int restart = ~0;
            unsigned int indexCount = 0;
            dp::sg::core::IndexSetSharedPtr indexSet = m_primitive->getIndexSet();
            if (indexSet)
            {
              restart = indexSet->getPrimitiveRestartIndex();
              indexCount = indexSet->getNumberOfIndices();
            }

            renderer->geometryDescriptionSet( geometryDescription, primitiveType, restart );
            renderer->geometryDescriptionSetBaseVertex( geometryDescription, getBaseVertex());
            renderer->geometryDescriptionSetIndexRange( geometryDescription, getFirstIndex(), indexCount );

            renderer->geometrySetData( m_geometryHandle, geometryDescription
              , getVertexAttributes()
              , getIndices()
              );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
