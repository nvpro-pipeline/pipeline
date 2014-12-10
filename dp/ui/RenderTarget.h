// Copyright NVIDIA Corporation 2010-2011
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

#include <dp/ui/Config.h>
#include <dp/util/Image.h>

namespace dp
{
  namespace ui
  {
    DEFINE_PTR_TYPES( RenderTarget );

    /** \brief An dp::ui::RenderTarget specifies a render surface to be used in conjunction with an dp::sg::ui::Renderer.
               devtech platform supports RenderTargets for OpenGL (dp::gl::RenderTargetFB, dp::gl::RenderTargetFBO) .

               After a Renderer's render command has been executed, the rendered image can be queried from the 
               RenderTarget as an dp::sg::core::TextureHost by calling RenderTarget::getTextureHost.

               A RenderTarget is also prepared to support stereoscopic rendering in conjunction with dp::sg::ui::SceneRenderer.
        \remarks For mono rendering the LEFT surface is used as default. 
    **/
    class RenderTarget
    {
    public:
      enum StereoTarget { LEFT = 0,         //!< Left eye only. Mono is aliased to the left eye.
                          RIGHT,            //!< Right eye only
                          LEFT_AND_RIGHT    //!< Left and right eye at the same time, may not work on all targets
                        };

      DP_UI_API virtual ~RenderTarget();

      /** \brief This function will be called by renderers when they start to render a frame.
          \return true If RenderTarget is ready for rendering, false otherwise.
      **/
      DP_UI_API virtual bool beginRendering();

      /** \brief This function will be called by renderers when they finished a frame. **/
      DP_UI_API virtual void endRendering();

      /** \brief Update the dimensions of the RenderTarget
          \param width New width of the RenderTarget
          \param height New height of the RenderTarget
      **/
      DP_UI_API virtual void setSize( unsigned int width, unsigned int height ) = 0;

      /** \brief Retrieve the dimensions of the RenderTarget.
          \param width Returns current width of the RenderTarget.
          \param height Returns current height of the RenderTarget.
      **/
      DP_UI_API virtual void getSize( unsigned int &width, unsigned int &height ) const = 0;

      /** \brief Get the current width of the RenderTarget
          \return current width of the RenderTarget
      **/
      unsigned int getWidth() const;

      /** \brief Get the current heightof the RenderTarget
          \return current heightof the RenderTarget
      **/
      unsigned int getHeight() const;

      /** \brief Get the current aspect ratio of the RenderTarget. The default implementation
                 returns width/height. If a derived class supports a RenderTarget with non-square
                 pixels this function should be overriden.
          \return Aspect ratio of the surface
      **/
      virtual float getAspectRatio() const;

  #if 0
      /** \brief Fetch pixels of the surface in a TextureHost.
          \param pixelFormat Pixel format to use when grabbing the pixels.
          \param pixelDataType Data type to use for each pixel component.
          \return A TextureHostSharedPtr containing a texture with the content of the surface.
          \remarks If a RenderTarget cannot support this operation it returns a null object.
      **/
      DP_UI_API virtual dp::sg::core::TextureHostSharedPtr getImage( 
            dp::sg::core::Image::PixelFormat pixelFormat = dp::sg::core::Image::IMG_BGR, 
            dp::sg::core::Image::PixelDataType pixelDataType = dp::sg::core::Image::IMG_UNSIGNED_BYTE ) = 0;
  #else
      /** \brief Fetch pixels of the surface in a TextureHost.
          \param pixelFormat Pixel format to use when grabbing the pixels.
          \param pixelDataType Data type to use for each pixel component.
          \return A TextureHostSharedPtr containing a texture with the content of the surface.
          \remarks If a RenderTarget cannot support this operation it returns a null object.
      **/
      DP_UI_API virtual dp::util::ImageSharedPtr getImage( 
            dp::util::PixelFormat pixelFormat = dp::util::PF_BGRA, 
            dp::util::DataType pixelDataType = dp::util::DT_UNSIGNED_INT_8,
            unsigned int index = 0) = 0;
  #endif


      /** \brief Check if the RenderTarget is valid and can be used for rendering.
          \return true If the RenderTarget is ready for rendering, false otherwise.
      **/
      DP_UI_API virtual bool isValid() = 0;

      /** \brief Check if stereo is enabled.
          \return true If the RenderTarget has stereo enabled, false otherwise.
      **/
      DP_UI_API virtual bool isStereoEnabled() const;

      /** \brief Choose stereo surface to render on.
          \param target LEFT, RIGHT or LEFT_AND_RIGHT for the corresponding surface.
          \return true If the surface had been selected and false otherwise.
      **/
      DP_UI_API virtual bool setStereoTarget( StereoTarget target );

      /** \brief Retrieve which stereo surface is currently active.
          \return Currently active surface for stereo rendering.
      **/
      DP_UI_API virtual StereoTarget getStereoTarget() const;

    protected:
      DP_UI_API RenderTarget()
      {}
    };

    inline unsigned int RenderTarget::getWidth() const
    {
      unsigned int width, height;
      getSize(width, height);
      return width;
    }

    inline unsigned int RenderTarget::getHeight() const
    {
      unsigned int width, height;
      getSize(width, height);
      return height;
    }

    inline float RenderTarget::getAspectRatio() const
    {
      unsigned int width, height;
      getSize(width, height);
      float ratio = 1.0;
      if ( width && height )
      {
        ratio = float(width) / float(height);
      }
      return ratio;
    }

  } // namespace ui
} // namespace dp
