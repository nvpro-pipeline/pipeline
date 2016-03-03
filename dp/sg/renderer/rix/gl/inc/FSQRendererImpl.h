// Copyright (c) 2010-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/renderer/rix/gl/FSQRenderer.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceEffectDataRiXFx.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourcePrimitive.h>
#include <dp/math/Vecnt.h>
#include <dp/rix/fx/Manager.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/util/DynamicLibrary.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/ui/Renderer.h>
#include <dp/sg/gl/TextureGL.h>
#include <dp/gl/RenderTarget.h>

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
          DEFINE_PTR_TYPES( RendererFSQImpl );

          /*! \brief Renderer to draw a "Full Screen" (ie: viewport-filling) quad, using a supplied pair of geometry effect and material effect.
           *  \par Namespace: nvgl
           *  \remarks Many rendering effects in modern applications require a Viewport-filling quad to realize certain screen-space
           *  or compositing effects.  This class will render this type of quad, using the supplied effects.  Additionally, applications may
           *  supply per vertex data as texture coordinates to be used by the effects while rendering.\n
           *  This class also provides several helper methods to "present" (render) a texture in a full screen quad.
           *
           *  \note Draws a viewport filling quad on Z = 0, with the following characteristics:
           *  \code
           *         Vertex Coords                 Texture Coords (units 0 and 1)
           *
           *     (-1,1) 3-------2 (1,1)          0:(0,1)   3-------2 0:(1,1)
           *            |       |                1:(x,y+h) |       | 1:(x+w,y+h)
           *            |       |                          |       |
           *    (-1,-1) 0-------1 (1,-1)         0:(0,0)   0-------1 0:(1,0)
           *                                     1:(x,y)             1:(x+w,y)
           *
           *                                      Where: X,Y,W,H are either Viewport Parameters or
           *                                             X,Y=0 and W,H are output RenderTarget dimensions
           *  \endcode
           *
           *  \note Here is an example effect that could be used to render a checkerboard pattern over the scene and tint it
           *  Red.
           *  In xml, the effect could be described as
           *  \code
           *
           *    <?xml version="1.0"?>
           *    <library>
           *      <effect type="Geometry" id="FSQGeometry">
           *        <technique type="forward">
           *          <glsl domain="vertex" signature="v3f">
           *            <source file="FSQGeometry.glsl" />
           *          </glsl>
           *        </technique>
           *      </effect>
           *      <effect type="Material" id="FSQMaterial">
           *        <technique type="forward">
           *          <glsl domain="fragment" signature="v3f">
           *            <source file="FSQMaterial.glsl"/>
           *          </glsl>
           *        </technique>
           *      </effect>
           *    </library>
           *
           *  \endcode
           *
           *  The FSQGeometry.glsl could simply look like
           *
           *  \code
           *
           *  layout(location =  0) in vec4 attrPosition;
           *
           *  void main(void)
           *  {
           *     // Note, vertices are already appropriately transformed.  Typically it will not be necessary to
           *     // further transform them.  However, any TEXCOORDS that are used should be passed to the fragment program.
           *     gl_Position = attrPosition;
           *  }
           *
           *  \endcode
           *
           *  And the FSQMaterial.glsl:
           *
           *  \code
           *
           *  in int2 varPosition;
           *
           *  layout(location = 0, index = 0) out vec4 Color;
           *
           *  void main(void)
           *  {
           *     //make a checkerboard pattern
           *     bool xp = varPosition.x & 4;
           *     bool yp = varPosition.y & 4;
           *     Color = (xp != yp) ? vec4( 1.0f, 0.0f, 0.0f, 0.5f ) : vec4(0.0f, 0.0f, 0.0f, 0.0f);
           *  }
           *
           *  \endcode
           *
           *  \sa dp::sg::core::EffectSpec, dp::sg::core::PipelineData, dp::sg::ui::Renderer */
          class RendererFSQImpl : public FSQRenderer
          {
            public:
              static RendererFSQImplSharedPtr create( const dp::gl::RenderTargetSharedPtr &renderTarget = dp::gl::RenderTargetSharedPtr() );
              virtual ~RendererFSQImpl(void);

              void setPipeline( const dp::sg::core::PipelineDataSharedPtr & effect );
              const dp::sg::core::PipelineDataSharedPtr & getPipeline() const;

              /*! \brief Add or remove texture coordinate attributes.
               *  \remarks Adds or removes texture coordinate attributes from this FSQ.
               *  \param unit The texture attribute identifier on which to modify the texcoords.  If adding texcoords, they will be available
               *  as TEXCOORDunit, where unit can range from 2 to 7, as TEXCOORD0 and TEXCOORD1 are used internally by the FSQ.
               *  \param coords A vector of Vec4f's to use as the data.  Note the ordering of the vertex data, as described above.  If this vector
               *  is empty, any texcoords assigned to this texcoord unit are removed.
               **/
              virtual void setTexCoords( unsigned int unit, const std::vector<dp::math::Vec4f> &coords );

              /*! \brief Set a sampler by name
               *  \remarks This is a convenience function to set the TextureGL or TextureHost of a Sampler.
               *  This could also be accomplished using the EffectData/ParameterGroupData API as well.
               *  \param samplerName The Sampler name in the effect.
               *  \param sampler The sampler to assign to the Sampler name samplerName.
               *  \return True if the named sampler was found and the sampler was assigned.  False otherwise.
               *  \note An example would be the following:
               *  \code
               *   Effect:
               *   <parameter type="sampler2D" name="selection";/>
               *   C++:
               *   fsq->setSamplerByName( "selection", theSampler );
               *  \endcode
               *
               * \sa setSamplerTextureBySemantic */
              virtual bool setSamplerByName( std::string const & samplerName, dp::sg::core::SamplerSharedPtr const & sampler );

              /*! \brief Fill the viewport with the given 2D GL textureID.
               *  \remarks This is a convenience function to render the given OpenGL texture in a viewport-filling quad.
               *  \param textureId The GLuint representing an OpenGL texture resource.
               *  \param target The RenderTarget to fill with the texture.
               *  \param callRTBeginEnd Whether to wrap the Quad rendering with target->beginRendering() and target->endRendering().  In some
               *  cases the RenderTarget may be current and calling begin/endRendering may be either unnecessary or detremental.
               **/
              static void presentTextureGL2D( GLuint textureId, const dp::gl::RenderTargetSharedPtr &target, bool callRTBeginEnd = true );

              /*! \brief Fill the viewport with the given TextureGL2D.
               *  \remarks This is a convenience function to render the given TextureGL2D in a viewport-filling quad.
               *  \param tex2d The TextureGL2D resource.
               *  \param target The RenderTarget to fill with the texture.
               *  \param callRTBeginEnd Whether to wrap the Quad rendering with target->beginRendering() and target->endRendering().  In some
               *  cases the RenderTarget may be current and calling begin/endRendering may be either unnecessary or detremental.
               **/
              static void presentTextureGL2D( const dp::gl::Texture2DSharedPtr &tex2d, const dp::gl::RenderTargetSharedPtr &target, bool callRTBeginEnd = true );

              /*! \brief Fill the viewport with the given TextureGLRectangle.
               *  \remarks This is a convenience function to render the given TextureGLRectangle in a viewport-filling quad.
               *  \param tex2d The TextureGLRectangle resource.
               *  \param target The RenderTarget to fill with the texture.
               *  \param callRTBeginEnd Whether to wrap the Quad rendering with target->beginRendering() and target->endRendering().  In some
               *  cases the RenderTarget may be current and calling begin/endRendering may be either unnecessary or detremental.
               **/
              static void presentTextureGLRectangle( const dp::gl::TextureRectangleSharedPtr &tex2d, const dp::gl::RenderTargetSharedPtr &target, bool callRTBeginEnd = true );

              REFLECTION_INFO( RendererFSQImpl );
              BEGIN_DECLARE_STATIC_PROPERTIES
              END_DECLARE_STATIC_PROPERTIES

            protected:
              RendererFSQImpl( const dp::gl::RenderTargetSharedPtr &target );
              virtual void doRender( const dp::ui::RenderTargetSharedPtr &renderTarget );

            private:
              void setTexCoord1( const dp::gl::RenderTargetSharedPtr & target );

            private:
              int                                           m_targetX;
              int                                           m_targetY;
              unsigned int                                  m_targetW;
              unsigned int                                  m_targetH;
              dp::sg::core::VertexAttributeSetSharedPtr     m_vertexAttributeSet;
              dp::sg::core::PrimitiveSharedPtr              m_primitive;
              dp::sg::core::PipelineDataSharedPtr           m_pipelineData;
              bool                                          m_pipelinesValid;
              dp::util::DynamicLibrarySharedPtr             m_rendererGLLib;
              dp::rix::core::Renderer                     * m_renderer;
              dp::rix::core::RenderGroupSharedHandle        m_renderGroup;
              dp::rix::core::GeometryInstanceSharedHandle   m_geometryInstance;
              dp::rix::fx::ManagerSharedPtr                 m_rixFxManager;
              dp::rix::fx::InstanceSharedHandle             m_instance;
              ResourceManagerSharedPtr            m_resourceManager;
              ResourceEffectDataRiXFxSharedPtr    m_resourceEffectData;
              ResourcePrimitiveSharedPtr          m_resourcePrimitive;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
