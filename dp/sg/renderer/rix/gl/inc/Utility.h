// Copyright NVIDIA Corporation 2012
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

#include <dp/rix/core/RiX.h>
#include <dp/sg/core/TextureHost.h>

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

          /************************************************************************/
          /* Utility functions                                                    */
          /************************************************************************/
          inline dp::DataType getRiXDataType( dp::sg::core::Image::PixelDataType scenixDataType )
          {
            switch( scenixDataType )
            {
            case dp::sg::core::Image::IMG_BYTE:
              return dp::DataType::INT_8;
            case dp::sg::core::Image::IMG_UNSIGNED_BYTE:
              return dp::DataType::UNSIGNED_INT_8;
            case dp::sg::core::Image::IMG_SHORT:
              return dp::DataType::INT_16;
            case dp::sg::core::Image::IMG_UNSIGNED_SHORT:
              return dp::DataType::UNSIGNED_INT_16;
            case dp::sg::core::Image::IMG_INT:
              return dp::DataType::INT_32;
            case dp::sg::core::Image::IMG_UNSIGNED_INT:
              return dp::DataType::UNSIGNED_INT_32;
            case dp::sg::core::Image::IMG_FLOAT16:
              return dp::DataType::FLOAT_16;
            case dp::sg::core::Image::IMG_FLOAT32:
              return dp::DataType::FLOAT_32;
            default:
              DP_ASSERT( false && "unknown datatype");
              return dp::DataType::UNKNOWN;
            }
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

