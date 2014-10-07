// Copyright NVIDIA Corporation 2013
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

#include <dp/gl/Program.h>
#include <dp/gl/RenderContext.h>
#include <dp/gl/Texture.h>
#include <dp/util/SmartPtr.h>
#include <map>

class TextureTransfer;
typedef dp::util::SmartPtr<TextureTransfer> SmartTextureTransfer;
class TextureTransfer : public dp::util::RCObject
{
public:
  TextureTransfer( dp::gl::SharedRenderContext const& dstContext, dp::gl::SharedRenderContext const& srcContext );
  ~TextureTransfer();

  void setTileSize( size_t width, size_t height );
  void setMaxIndex( size_t maxIndex );

  void transfer( size_t index
               , dp::gl::SharedTexture2D dstTexture
               , dp::gl::SharedTexture2D srcTexture );

private:
  void constructComputeShaders();
  void destroyComputeShaders();
  dp::gl::SharedProgram compileShader( dp::gl::SharedRenderContext const& context, char const* source );

  // get a texture for the given context 
  dp::gl::SharedTexture2D const& getTmpTexture( dp::gl::SharedRenderContext const& context, size_t width, size_t height );

private:
  typedef std::map< dp::gl::SharedRenderContext, dp::gl::SharedTexture2D > Textures;

private:
  dp::gl::SharedRenderContext m_dstContext;
  dp::gl::SharedRenderContext m_srcContext;

  size_t m_tileWidth;
  size_t m_tileHeight;
  size_t m_maxIndex;

  dp::gl::SharedProgram  m_compressProgram;
  dp::gl::SharedProgram  m_decompressProgram;
  dp::gl::SharedProgram  m_copyProgram;
  bool m_shadersInitialized;

  Textures m_tmpTextures;
};

