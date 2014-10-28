// Copyright NVIDIA Corporation 2011-2012
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

#include <dp/util/Image.h>

namespace dp
{
  namespace util
  {
    namespace ui
    {
      SMART_TYPES( RenderTarget );

      /** \brief An interface to a memory buffer to which the frame is rendered.
          \remarks For mono rendering the LEFT surface is used as default. 
      **/
      class RenderTarget
      {
      public:
        enum StereoTarget
        { 
            LEFT             //!< Left eye only. Mono is aliased to the left eye.
          , RIGHT            //!< Right eye only
          , LEFT_AND_RIGHT   //!< Left and right eye at the same time, may not work on all targets
        };

        DP_UTIL_API virtual ~RenderTarget();

        /** \brief This function will be called by renderers when they start to render a frame.
            \return true if RenderTarget is ready for rendering, false otherwise.
        **/
        DP_UTIL_API virtual bool beginRendering();

        /** \brief This function will be called by renderers when they finished a frame. **/
        DP_UTIL_API virtual void endRendering();

        /** \brief Get the current width of the RenderTarget
            \return current width of the RenderTarget
        **/
        DP_UTIL_API unsigned int getWidth() const;

        /** \brief Get the current height of the RenderTarget
            \return current height of the RenderTarget
        **/
        DP_UTIL_API unsigned int getHeight() const;

        /** \brief Get the current aspect ratio of the RenderTarget. The default implementation
                   returns width/height. If a derived class supports a RenderTarget with non-square
                   pixels this function should be overridden.
            \return Aspect ratio of the surface
        **/
        DP_UTIL_API virtual float getAspectRatio() const;

        /** \brief Fetch pixels of the surface in a Bitmap.
        **/
        DP_UTIL_API virtual SmartImage getScreenshot(unsigned int layer = 0) = 0;

        /** \brief Check if the RenderTarget is valid and can be used for rendering.
            \return true If the RenderTarget is ready for rendering, false otherwise.
        **/
        DP_UTIL_API virtual bool isValid() = 0;

        /** \brief Check if stereo is enabled.
            \return true If the RenderTarget has stereo enabled, false otherwise.
        **/
        DP_UTIL_API virtual bool isStereoEnabled() const;

        /** \brief Choose stereo surface to render on.
            \param target LEFT, RIGHT or LEFT_AND_RIGHT for the corresponding surface.
            \return true If the surface had been selected and false otherwise.
        **/
        DP_UTIL_API virtual bool setStereoTarget( StereoTarget target );

        /** \brief Retrieve which stereo surface is currently active.
            \return Currently active surface for stereo rendering.
        **/
        DP_UTIL_API virtual StereoTarget getStereoTarget() const;

        /** \brief Set the width of the RenderTarget
            \param width the new width
        **/
        DP_UTIL_API void setWidth(unsigned int width);

        /** \brief Set the height of the RenderTarget
            \param height the new height
        **/
        DP_UTIL_API void setHeight(unsigned int height);

      protected:
        DP_UTIL_API RenderTarget();

      protected:
        unsigned int m_width;
        unsigned int m_height;
      };

    } // namespace ui
  } // namespace util
} // namespace dp
