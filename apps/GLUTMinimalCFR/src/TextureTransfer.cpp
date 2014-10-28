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


#include <inc/TextureTransfer.h>

#include <dp/util/FrameProfiler.h>

// #define ENABLE_PROFILING 0
// #include <dp/util/Profile.h>

#include <boost/scoped_array.hpp>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define OBJECT_BINDING 0
#define MATRIX_BINDING 1
#define VISIBILITY_BINDING 2

static const char* shaderHeader =
  "#version 430\n"
  ;

static const char* compressShader =
  "layout( local_size_x = TILEWIDTH, local_size_y = TILEHEIGHT) in;\n"
  "uniform layout( binding = 0, r32ui ) restrict writeonly uimage2D tmp;\n"
  "uniform layout( binding = 1, r32ui ) restrict readonly  uimage2D srcImg;\n"
  "uniform layout( location = 0 ) int maxIndex;\n"
  "uniform layout( location = 1 ) int index;\n"
  "\n"
  "void main()\n"
  "{\n"
  "  // x src pos: find right tile (mind, skew with y invocation ID), add local position in tile\n"
  "  // y src pos: tiles are packed tightly\n"
  "  ivec2 srcPos = ivec2( (gl_WorkGroupID.x * maxIndex + (index + gl_WorkGroupID.y) % maxIndex ) * TILEWIDTH  + gl_LocalInvocationID.x\n"
  "                      ,  gl_WorkGroupID.y                                                      * TILEHEIGHT + gl_LocalInvocationID.y );\n"
  "  // x and y tmp position: tiles are packed tightly\n"
  "  ivec2 tmpPos = ivec2(  gl_GlobalInvocationID.xy );\n"
  "  if( all( lessThan( srcPos, imageSize( srcImg ) ) ) )\n"
  "  {\n"
  "    imageStore( tmp, tmpPos, imageLoad( srcImg, srcPos ) );\n"
  "  }\n"
  "}\n"
  ;

static const char* decompressShader =
  "layout( local_size_x = TILEWIDTH, local_size_y = TILEHEIGHT) in;\n"
  "uniform layout( binding = 0, r32ui ) restrict /*writeonly*/ uimage2D dstImg;\n"
  "uniform layout( binding = 1, r32ui ) restrict readonly  uimage2D tmp;\n"
  "uniform layout( location = 0 ) int maxIndex;\n"
  "uniform layout( location = 1 ) int index;\n"
  "\n"
  "void main()\n"
  "{\n"
  "  // x src pos: find right tile (mind, skew with y invocation ID), add local position in tile\n"
  "  // y src pos: tiles are packed tightly\n"
  "  ivec2 dstPos = ivec2( (gl_WorkGroupID.x * maxIndex + (index + gl_WorkGroupID.y) % maxIndex ) * TILEWIDTH  + gl_LocalInvocationID.x\n"
  "                      ,  gl_WorkGroupID.y                                                      * TILEHEIGHT + gl_LocalInvocationID.y );\n"
  "  // x and y dstImg position: tiles are packed tightly\n"
  "  ivec2 tmpPos = ivec2(  gl_GlobalInvocationID.xy );\n"
  "  if( all( lessThan( dstPos, imageSize( dstImg ) ) ) )\n"
  "  {\n"
  "    imageStore( dstImg, dstPos, imageLoad( tmp, tmpPos ) );\n"
//   "    imageStore( dstImg, dstPos, uvec4( 0xFF000000 | dstPos.y ) );\n"
  "  }\n"
  "}\n"
  ;

static const char* copyShader =
  "layout( local_size_x = TILEWIDTH, local_size_y = TILEHEIGHT) in;\n"
  "uniform layout( binding = 0, r32ui ) restrict /*writeonly*/ uimage2D dstImg;\n"
  "uniform layout( binding = 1, r32ui ) restrict readonly  uimage2D srcImg;\n"
  "uniform layout( location = 0 ) int maxIndex;\n"
  "uniform layout( location = 1 ) int index;\n"
  "\n"
  "void main()\n"
  "{\n"
  "  ivec2 pos = ivec2( (gl_WorkGroupID.x * maxIndex + (index + gl_WorkGroupID.y) % maxIndex ) * TILEWIDTH  + gl_LocalInvocationID.x\n"
  "                   ,  gl_WorkGroupID.y                                                      * TILEHEIGHT + gl_LocalInvocationID.y );\n"
  "  if( all( lessThan( pos, imageSize( dstImg ) ) ) )\n"
  "  {\n"
  "    imageStore( dstImg, pos, imageLoad( srcImg, pos ) );\n"
//    "    imageStore( dstImg, pos, uvec4( 0xFF000000 | index * 32 ) );\n"
  "  }\n"
  "}\n"
  ;

SmartTextureTransfer TextureTransfer::create( dp::gl::SharedRenderContext const& dstContext, dp::gl::SharedRenderContext const& srcContext )
{
  return( std::shared_ptr<TextureTransfer>( new TextureTransfer( dstContext, srcContext ) ) );
}

TextureTransfer::TextureTransfer( dp::gl::SharedRenderContext const& dstContext, dp::gl::SharedRenderContext const& srcContext )
  : m_dstContext( dstContext )
  , m_srcContext( srcContext )
  , m_tileWidth( 0 )
  , m_tileHeight( 0 )
  , m_maxIndex( 0 )
  , m_shadersInitialized( 0 )
{
}

TextureTransfer::~TextureTransfer()
{
  destroyComputeShaders();
}

void TextureTransfer::setTileSize( size_t width, size_t height )
{
  m_tileWidth = width;
  m_tileHeight = height;

  destroyComputeShaders();
}

void TextureTransfer::setMaxIndex( size_t maxIndex )
{
  m_maxIndex = maxIndex;
}

void TextureTransfer::transfer( size_t index
                              , dp::gl::SharedTexture2D dstTexture
                              , dp::gl::SharedTexture2D srcTexture )
{
  //PROFILE( "Copy " );

#if 1

  if( !m_shadersInitialized )
  {
    constructComputeShaders();
  }
  size_t width  = dstTexture->getWidth();
  size_t height = dstTexture->getHeight();

  DP_ASSERT( width == srcTexture->getWidth() && height == srcTexture->getHeight() );

  size_t tilesX = (width  + m_tileWidth  - 1) / m_tileWidth;  // number of horizontal tiles
  size_t tilesY = (height + m_tileHeight - 1) / m_tileHeight; // number of vert1ical tiles
  size_t tilesXPerGpu = ( tilesX + m_maxIndex - 1 ) / m_maxIndex;

  dp::gl::RenderContextStack contextStack;

  contextStack.push( m_srcContext );

  if( m_srcContext == m_dstContext )
  {
    m_copyProgram->setImageTexture( "dstImg", dstTexture, GL_WRITE_ONLY );
    m_copyProgram->setImageTexture( "srcImg", srcTexture, GL_READ_ONLY );

    dp::gl::ProgramUseGuard pug( m_copyProgram );

    glUniform1i( 0, dp::util::checked_cast<GLint>(m_maxIndex) );
    glUniform1i( 1, dp::util::checked_cast<GLint>(index) );

    glDispatchCompute( dp::util::checked_cast<GLuint>(tilesXPerGpu), dp::util::checked_cast<GLuint>(tilesY), 1 );
  }
  else
  {
    size_t tmpTexWidth  = m_tileWidth * tilesXPerGpu;
    size_t tmpTexHeight = height;

    // compress the image data into src tmp texture
    dp::gl::SharedTexture2D srcTmpTexture = getTmpTexture( m_srcContext, tmpTexWidth, tmpTexHeight );
    m_compressProgram->setImageTexture( "tmp", srcTmpTexture, GL_WRITE_ONLY );
    m_compressProgram->setImageTexture( "srcImg", srcTexture, GL_READ_ONLY );
    {
      dp::gl::ProgramUseGuard pug( m_compressProgram );

      glUniform1i( 0, dp::util::checked_cast<GLint>(m_maxIndex) );
      glUniform1i( 1, dp::util::checked_cast<GLint>(index) );

      glDispatchCompute( dp::util::checked_cast<GLuint>(tilesXPerGpu), dp::util::checked_cast<GLuint>(tilesY), 1 );
    }

    contextStack.pop();

    contextStack.push( m_dstContext );
    dp::gl::SharedTexture2D dstTmpTexture = getTmpTexture( m_dstContext, tmpTexWidth, tmpTexHeight );

    // copy src into dst tmp texture
    GLuint idSrc = srcTmpTexture->getGLId();
    GLuint idDst = dstTmpTexture->getGLId();

    HGLRC hglrcSrc = m_srcContext->getHGLRC();
    HGLRC hglrcDst = m_dstContext->getHGLRC();


    {
      dp::util::ProfileEntry p("wglCopyImageSubDataNV");
      DP_VERIFY( wglCopyImageSubDataNV( hglrcSrc, idSrc, GL_TEXTURE_2D, 0, 0, 0, 0
        , hglrcDst, idDst, GL_TEXTURE_2D, 0, 0, 0, 0
        , dp::util::checked_cast<GLsizei>(tmpTexWidth)
        , dp::util::checked_cast<GLsizei>(tmpTexHeight)
        , 1 ) );
    }

    // decompress image data from dst tmp texture
    m_decompressProgram->setImageTexture( "dstImg", dstTexture, GL_WRITE_ONLY );
    m_decompressProgram->setImageTexture( "tmp", dstTmpTexture, GL_READ_ONLY );

    dp::gl::ProgramUseGuard pug( m_decompressProgram );

    glUniform1i( 0, dp::util::checked_cast<GLint>(m_maxIndex) );
    glUniform1i( 1, dp::util::checked_cast<GLint>(index) );

    glDispatchCompute( dp::util::checked_cast<GLuint>(tilesXPerGpu), dp::util::checked_cast<GLuint>(tilesY), 1 );
  }

  contextStack.pop();

#else

  // just copy the image, don't take the pattern into account (debug code)
  size_t width  = dstTexture->getWidth();
  size_t height = dstTexture->getHeight();

  GLuint idSrc = srcTexture->getGLId();
  GLuint idDst = dstTexture->getGLId();

  HGLRC hglrcSrc = m_srcContext->getHGLRC();
  HGLRC hglrcDst = m_dstContext->getHGLRC();

  wglCopyImageSubDataNV( hglrcSrc, idSrc, GL_TEXTURE_2D, 0, 0, 0, 0
                       , hglrcDst, idDst, GL_TEXTURE_2D, 0, 0, 0, 0
                       , dp::util::checked_cast<GLsizei>(width)
                       , dp::util::checked_cast<GLsizei>(height)
                       , 1 );

#endif
}

void TextureTransfer::constructComputeShaders()
{
  if ( !m_shadersInitialized )
  {
    m_compressProgram   = compileShader( m_srcContext, compressShader );
    m_decompressProgram = compileShader( m_dstContext, decompressShader );
    m_copyProgram       = compileShader( m_dstContext, copyShader );
    DP_ASSERT( m_compressProgram && m_decompressProgram && m_copyProgram );
    m_shadersInitialized = true;
  }

  // m_uniformViewProjection = m_program->getUniformLocation( "viewProjection" );
}

void TextureTransfer::destroyComputeShaders()
{
  if ( m_shadersInitialized )
  {
    dp::gl::RenderContextStack contextStack;
    contextStack.push( m_srcContext );
    m_compressProgram.reset();
    contextStack.pop();

    contextStack.push( m_dstContext );
    m_decompressProgram.reset();
    contextStack.pop();

    m_shadersInitialized = false;

    Textures::iterator it;
    Textures::iterator itEnd = m_tmpTextures.end();
    for( it = m_tmpTextures.begin(); it != itEnd; ++it )
    {
      contextStack.push(it->first);
      it->second.reset();
      contextStack.pop();
    }
    m_tmpTextures.clear();
  }
}

dp::gl::SharedProgram TextureTransfer::compileShader( dp::gl::SharedRenderContext const& context, char const* source )
{
  dp::gl::RenderContextStack contextStack;
  contextStack.push( context );

  std::stringstream ss;
  ss << shaderHeader;
  ss << "#define TILEWIDTH  " << m_tileWidth  << "\n";
  ss << "#define TILEHEIGHT " << m_tileHeight << "\n";
  ss << source;

  dp::gl::SharedProgram program = dp::gl::Program::create( dp::gl::ComputeShader::create( ss.str() ) );

  contextStack.pop();
  return( program );
}

dp::gl::SharedTexture2D const& TextureTransfer::getTmpTexture( dp::gl::SharedRenderContext const& context, size_t width, size_t height )
{
  DP_ASSERT( context == dp::gl::RenderContext::getCurrentRenderContext() );
  
  Textures::iterator it = m_tmpTextures.find( context );
  if ( it == m_tmpTextures.end() || it->second->getWidth() != width || it->second->getHeight() != height )
  {
    // element doesn't exist yet, or size does not match, create a new texture with the correct size
    dp::gl::SharedTexture2D tex = dp::gl::Texture2D::create( GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, dp::util::checked_cast<GLsizei>(width), dp::util::checked_cast<GLsizei>(height) );

    if ( it == m_tmpTextures.end() )
    {
      // element isn't in the map yet, insert the element
      it = m_tmpTextures.insert( std::make_pair(context, tex)).first;
    }
    else
    {
      // update the element
      it->second = tex;
    }
  }

  // here we are guaranteed to have a valid texture
  return it->second;
}