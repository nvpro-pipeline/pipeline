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


#pragma once

#include <dp/sg/renderer/rix/gl/Config.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceVertexAttributeSet.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceIndexSet.h>

#include <dp/sg/core/CoreTypes.h>

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

          class ResourcePrimitive;
          typedef dp::util::SmartPtr<ResourcePrimitive> SmartResourcePrimitive;

          class ResourcePrimitive : public ResourceManager::Resource
          {
          public:
            /** \brief Fetch resource for the given object/resourceManager. If no resource exists it'll be created **/
            static SmartResourcePrimitive get( const dp::sg::core::PrimitiveSharedPtr &primitive, const SmartResourceManager& resourceManager );
            virtual void update();

            ~ResourcePrimitive();

            virtual const dp::sg::core::HandledObjectSharedPtr& getHandledObject() const;

            dp::rix::core::GeometrySharedHandle m_geometryHandle;

          protected:
            virtual void updateVertexAttributesAndIndices() = 0;
            virtual dp::rix::core::VertexAttributesSharedHandle getVertexAttributes() const = 0;
            virtual dp::rix::core::IndicesSharedHandle getIndices() const = 0;
            virtual unsigned int getBaseVertex() const = 0;
            virtual unsigned int getFirstIndex() const = 0;

            dp::sg::core::PrimitiveSharedPtr m_primitive;
            ResourcePrimitive( const dp::sg::core::PrimitiveSharedPtr &primitive, const SmartResourceManager& resourceManager );
          };

          typedef dp::util::SmartPtr<ResourcePrimitive> SmartResourcePrimitive;

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
