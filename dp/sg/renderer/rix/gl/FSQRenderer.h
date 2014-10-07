// Copyright NVIDIA Corporation 2010-2012
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
#include <dp/sg/ui/Renderer.h>
#include <dp/gl/RenderTarget.h>
#include <dp/gl/Texture.h>
#include <dp/sg/core/EffectData.h>

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
           *  \sa dp::sg::core::EffectSpec, dp::sg::core::EffectData, dp::sg::ui::Renderer */
          class FSQRenderer : public dp::sg::ui::Renderer
          {
            public:
              DP_SG_RDR_RIX_GL_API static dp::util::SmartPtr<FSQRenderer> create( const dp::gl::SharedRenderTarget &renderTarget = dp::gl::SharedRenderTarget() );
              DP_SG_RDR_RIX_GL_API virtual ~FSQRenderer(void);

              DP_SG_RDR_RIX_GL_API virtual void setEffect( const dp::sg::core::EffectDataSharedPtr & effect ) = 0;
              DP_SG_RDR_RIX_GL_API virtual const dp::sg::core::EffectDataSharedPtr & getEffect() const = 0;

              /*! \brief Add or remove texture coordinate attributes.
               *  \remarks Adds or removes texture coordinate attributes from this FSQ.
               *  \param unit The texture attribute identifier on which to modify the texcoords.  If adding texcoords, they will be available 
               *  as TEXCOORDunit, where unit can range from 2 to 7, as TEXCOORD0 and TEXCOORD1 are used internally by the FSQ.
               *  \param coords A vector of Vec4f's to use as the data.  Note the ordering of the vertex data, as described above.  If this vector
               *  is empty, any texcoords assigned to this texcoord unit are removed.
               **/
              DP_SG_RDR_RIX_GL_API virtual void setTexCoords( unsigned int unit, const std::vector<dp::math::Vec4f> &coords ) = 0;

              /*! \brief Set a Sampler by name
               *  \remarks This is a convenience function to set the dp::gl::Texture or TextureHost of a Sampler.
               *  This could also be accomplished using the EffectData/ParameterGroupData API as well.
               *  \param samplerName The Sampler name in the effect.
               *  \param sampler The sampler to assign to the Sampler named samplerName.
               *  \return True if the named sampler was found and the texture was assigned.  False otherwise.
               *  \note An example would be the following:
               *  \code
               *   Effect:
               *   <parameter type="sampler2D" name="selection";/>
               *   C++:
               *   fsq->setSamplerByName( "selection", theSampler );
               *  \endcode
               *
               * \sa setSamplerTextureBySemantic */
              DP_SG_RDR_RIX_GL_API virtual bool setSamplerByName( std::string const & samplerName, dp::sg::core::SamplerSharedPtr const & sampler ) = 0;

              /*! \brief Fill the viewport with the given dp::gl::Texture2D.
               *  \remarks This is a convenience function to render the given dp::gl::Texture2D in a viewport-filling quad.
               *  \param tex2d The dp::gl::Texture2D resource.
               *  \param target The RenderTarget to fill with the texture.
               *  \param callRTBeginEnd Whether to wrap the Quad rendering with target->beginRendering() and target->endRendering().  In some 
               *  cases the RenderTarget may be current and calling begin/endRendering may be either unnecessary or detremental.
               **/
              DP_SG_RDR_RIX_GL_API static void presentTexture2D( const dp::gl::SharedTexture2D &tex2d, const dp::gl::SharedRenderTarget &target, bool callRTBeginEnd = true );

              /*! \brief Fill the viewport with the given dp::gl::TextureRectangle.
               *  \remarks This is a convenience function to render the given dp::gl::TextureRectangle in a viewport-filling quad.
               *  \param tex2d The dp::gl::TextureRectangle resource.
               *  \param target The RenderTarget to fill with the texture.
               *  \param callRTBeginEnd Whether to wrap the Quad rendering with target->beginRendering() and target->endRendering().  In some 
               *  cases the RenderTarget may be current and calling begin/endRendering may be either unnecessary or detremental.
               **/
              DP_SG_RDR_RIX_GL_API static void presentTextureRectangle( const dp::gl::SharedTextureRectangle &tex2d, const dp::gl::SharedRenderTarget &target, bool callRTBeginEnd = true );

              REFLECTION_INFO_API( DP_SG_RDR_RIX_GL_API, RendererGLFSQImpl );
              BEGIN_DECLARE_STATIC_PROPERTIES
              END_DECLARE_STATIC_PROPERTIES

            protected:
              DP_SG_RDR_RIX_GL_API FSQRenderer( const dp::gl::SharedRenderTarget &target );
          };

          typedef dp::util::SmartPtr<FSQRenderer> SmartFSQRenderer;

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
