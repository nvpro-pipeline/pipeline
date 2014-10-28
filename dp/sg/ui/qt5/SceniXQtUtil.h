// Copyright NVIDIA Corporation 2010-2013
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

#include <QImage>

#include <dp/sg/ui/qt5/Config.h>
#include <dp/util/Image.h>
#include <dp/sg/core/CoreTypes.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace qt5
      {
        /** \brief Creates an QImage out of a TextureHost
            \param textureImage The TextureHost which should be used as source for the QImage.
                   Currently the following formats are supported
                   <pre>IMG_RGB, IMG_RGBA, IMG_BGR, IMG_BGRA</pre>
            \param image Image to use from \a textureImage
            \param mipmap Mipmap to use from \a textureImage
            \return A QImage created out of the pixel data from \a textureImage
        **/
        DP_SG_UI_QT5_API QImage createQImage( const dp::sg::core::TextureHostSharedPtr &textureImage, int image = 0, int mipmap = 0 );
        DP_SG_UI_QT5_API QImage createQImage( const dp::util::SmartImage& image /*, int image, int mipmap */ );

      } // namespace qt5
    } // namespace ui
  } // namespace sg
} // namespace dp

