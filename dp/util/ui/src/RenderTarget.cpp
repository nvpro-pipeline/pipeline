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


#include <dp/util/ui/RenderTarget.h>

namespace dp
{
  namespace util
  {
    namespace ui
    {

      RenderTarget::RenderTarget()
        : m_width(0)
        , m_height(0)
      {
      }

      RenderTarget::~RenderTarget()
      {
      }

      bool RenderTarget::beginRendering()
      {
        return( true );
      }

      void RenderTarget::endRendering()
      {
      }

      unsigned int RenderTarget::getWidth() const
      {
        return m_width;
      }

      unsigned int RenderTarget::getHeight() const
      {
        return m_height;
      }

      float RenderTarget::getAspectRatio() const
      {
        float ratio = 0.0f;
        if ( m_width && m_height )
        {
          ratio = float(m_width) / float(m_height);
        }
        return ratio;
      }

      bool RenderTarget::isStereoEnabled() const
      {
        return false;
      }

      bool RenderTarget::setStereoTarget( StereoTarget target )
      {
        return target == LEFT;
      }

      RenderTarget::StereoTarget RenderTarget::getStereoTarget() const
      {
        return LEFT;
      }

      void RenderTarget::setWidth( unsigned int width )
      {
        m_width = width;
      }

      void RenderTarget::setHeight( unsigned int height )
      {
        m_height = height;
      }

    } // namespace ui
  } // namespace util
} // namespace dp
