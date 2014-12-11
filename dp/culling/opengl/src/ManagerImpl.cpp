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


#include <dp/culling/opengl/inc/ManagerImpl.h>

#if defined(GL_VERSION_4_3)
#include <dp/culling/ObjectBitSet.h>
#include <dp/culling/ResultBitSet.h>
#include <dp/culling/opengl/inc/GroupImpl.h>
#include <dp/gl/Program.h>
#include <dp/util/BitArray.h>
#include <dp/util/FrameProfiler.h>
#include <boost/scoped_array.hpp>
#include <GL/glew.h>
#include <iostream>
#include <cstring>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// TODO This small workgroup size reduces the #parallel threads to 8*32 on a single SM.
// It turns out to be enough at the time (scenes ~10k elements). Benchmark again with
// scenes containing ~100k elements.
#define WORKGROUP_SIZE 32

DP_STATIC_ASSERT( (WORKGROUP_SIZE % 32) == 0 );

#define OBJECT_BINDING 0
#define MATRIX_BINDING 1
#define VISIBILITY_BINDING 2

//#define NO_GLSL_BARRIER

static const char* shader =
"#version 430\n"
"struct Object {\n"
"  uint matrix;\n"
"  vec4 lowerLeft;\n"
"  vec4 extent;\n"
"};\n"
"\n"
"shared uint sharedVisible[" TOSTRING(WORKGROUP_SIZE) "];\n"
"\n"
"layout( std430, binding = " TOSTRING(OBJECT_BINDING) ") buffer Objects {\n"
"  Object objects[];\n"
"};\n"
"layout( std430, binding = " TOSTRING(MATRIX_BINDING) " ) buffer Matrices {\n"
"  mat4 matrices[];\n"
"};\n"
"layout( std430, binding = " TOSTRING(VISIBILITY_BINDING) " ) buffer BufferVisible {\n"
"  uint visible[];\n"
"};\n"
#if defined(NO_GLSL_BARRIER)
"layout( local_size_x = 32, local_size_y = 1, local_size_z = 1) in;\n"
#else
"layout( local_size_x = " TOSTRING(WORKGROUP_SIZE) ", local_size_y = 1, local_size_z = 1) in;\n"
#endif
"\n"
"void determineCullFlags( in vec4 p, inout uint cfo, inout uint cfa )\n"
"{\n"
"  uint cf = 0;\n"
"\n"
"  if ( p.x <= -p.w )\n"
"  {\n"
"    cf |= 0x01;\n" 
"  }\n"
"  else if ( p.w <= p.x )\n"
"  {\n"
"    cf |= 0x02;\n" 
"  }\n"
"  if ( p.y <= -p.w )\n"
"  {\n"
"    cf |= 0x04;\n" 
"  }\n"
"  else if ( p.w <= p.y )\n"
"  {\n"
"    cf |= 0x08;\n" 
"  }\n"
"  if ( p.z <= -p.w )\n"
"  {\n"
"    cf |= 0x10;\n" 
"  }\n"
"  else if ( p.w <= p.z )\n"
"  {\n"
"    cf |= 0x20;\n" 
"  }\n"
"  cfo = cfo | cf;\n"
"  cfa = cfa & cf;\n"
"}\n"
"\n"
"bool isVisible( in mat4 viewProjection, in mat4 modelView, in vec4 lower, in vec4 extent)\n"
"{\n"
"  uint cfo = 0;\n"
"  uint cfa = 0xffffffff;\n"
"\n"
"  mat4 modelViewProjection = viewProjection * modelView;\n"
"  vec4 vectors[8];\n"
"  vectors[0] = modelViewProjection * lower;\n"
"\n"
"  vec4 x = extent.x * modelViewProjection[0];\n"
"  vec4 y = extent.y * modelViewProjection[1];\n"
"  vec4 z = extent.z * modelViewProjection[2];\n"
"\n"
"  vectors[1] = vectors[0] + x;\n"
"  vectors[2] = vectors[0] + y;\n"
"  vectors[3] = vectors[1] + y;\n"
"  vectors[4] = vectors[0] + z;\n"
"  vectors[5] = vectors[1] + z;\n"
"  vectors[6] = vectors[2] + z;\n"
"  vectors[7] = vectors[3] + z;\n"
"\n"
"  for ( uint i = 0;i < 8; ++i )\n"
"  {\n"
"    determineCullFlags( vectors[i], cfo, cfa );\n"
"  }\n"

"\n"
"  return (cfo == 0) || (cfa == 0);\n"
"}\n"
"\n"
"uniform mat4 viewProjection;\n"
"void main() {\n"
#if defined(NO_GLSL_BARRIER)
"  uint index = gl_GlobalInvocationID.x * 32;\n"
"  uint visibleMask = 0;\n"
"  uint bit = 1u;\n"
//"  for ( uint count = 0;count < 32;++count )\n"
"  {\n"
"    uint globalIndex = index + count;\n"
"    visibleMask = visibleMask | (isVisible( viewProjection, matrices[objects[globalIndex].matrix], objects[globalIndex].lowerLeft, objects[globalIndex].extent ) ? bit : 0);\n"
"    bit = bit << 1u;\n"
"  }\n"
"  visible[gl_GlobalInvocationID.x] = visibleMask;\n"
#else
"  uint index = gl_GlobalInvocationID.x;\n"
"  uint localId = index % " TOSTRING(WORKGROUP_SIZE) ";\n"
"  uint bit = 1 << (localId & 31);\n"
"  sharedVisible[localId] = isVisible( viewProjection, matrices[objects[index].matrix], objects[index].lowerLeft, objects[index].extent ) ? bit : 0;\n"
"  barrier();\n"
"  memoryBarrierShared();\n"
"  if ( (localId % 2) == 0 ) {sharedVisible[localId] |= sharedVisible[localId + 1]; } memoryBarrierShared();barrier();\n"
"  if ( (localId % 4) == 0 ) {sharedVisible[localId] |= sharedVisible[localId + 2]; } memoryBarrierShared();barrier();\n"
"  if ( (localId % 8) == 0 ) {sharedVisible[localId] |= sharedVisible[localId + 4]; } memoryBarrierShared();barrier();\n"
"  if ( (localId % 16) == 0 ) {sharedVisible[localId] |= sharedVisible[localId + 8]; } memoryBarrierShared();barrier();\n"
"  if ( (localId % 32) == 0 ) { visible[index / 32] = sharedVisible[localId] | sharedVisible[localId + 16]; }\n"
#endif
"}\n"
;


namespace dp
{
  namespace culling
  {
    namespace opengl
    {

      ManagerImpl::ManagerImpl()
        : m_uniformViewProjection(0)
        , m_shaderInitialized( false )
      {

      }
      ManagerImpl::~ManagerImpl()
      {

      }

      GroupSharedPtr ManagerImpl::groupCreate()
      {
        return GroupImpl::create();
      }

      ObjectSharedPtr ManagerImpl::objectCreate( PayloadSharedPtr const& userData )
      {
        return ObjectBitSet::create( userData );
      }


      ResultSharedPtr ManagerImpl::groupCreateResult( GroupSharedPtr const& group )
      {
        return( ResultBitSet::create( group.staticCast<GroupImpl>() ) );
      }

      void ManagerImpl::initializeComputeShader()
      {
        if ( !m_shaderInitialized )
        {
          glewInit();

          dp::gl::ProgramSharedPtr program = dp::gl::Program::create( dp::gl::ComputeShader::create( shader ) );
          m_shaderInitialized = true;
          m_uniformViewProjection = program->getActiveUniform( program->getActiveUniformIndex( "viewProjection" ) ).location;
          m_program = dp::gl::ProgramInstance::create( program );
        }
      }

      void ManagerImpl::cull( const GroupSharedPtr& group, const ResultSharedPtr& result, const dp::math::Mat44f& viewProjection )
      {
        dp::util::ProfileEntry p("cull");

        const GroupImplSharedPtr& groupImpl = group.staticCast<GroupImpl>();

        dp::math::Mat44f vp = viewProjection;
        dp::math::Mat44f modelViewProjection;

        initializeComputeShader();
        groupImpl->update( WORKGROUP_SIZE );

        // initialize output buffer
        size_t numberOfWorkingGroups = (groupImpl->getObjectCount() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        m_program->apply();
        glUniformMatrix4fv( m_uniformViewProjection, 1, false, viewProjection[0].getPtr() );

        glBindBufferBase( GL_SHADER_STORAGE_BUFFER, OBJECT_BINDING, groupImpl->getInputBuffer()->getGLId() );
        glBindBufferBase( GL_SHADER_STORAGE_BUFFER, MATRIX_BINDING, groupImpl->getMatrixBuffer()->getGLId() );
        glBindBufferBase( GL_SHADER_STORAGE_BUFFER, VISIBILITY_BINDING, groupImpl->getOutputBuffer()->getGLId() );
        glDispatchCompute( static_cast<GLuint>(numberOfWorkingGroups), 1, 1 );
        glMemoryBarrier( GL_BUFFER_UPDATE_BARRIER_BIT ); // TODO This is way too slow to use, but correct.
        glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
        dp::gl::MappedBuffer<dp::Uint32> visibleShader( groupImpl->getOutputBuffer(), GL_MAP_READ_BIT );
        result.staticCast<ResultBitSet>()->updateChanged( visibleShader );
      }

      Manager* Manager::create()
      {
        return new ManagerImpl;
      }

    } // namespace opengl
  } // namespace culling
} // namespace dp

#else
namespace dp
{
  namespace culling
  {
    namespace opengl
    {

      Manager* Manager::create()
      {
  DP_ASSERT( !"Compiled without OpenGL 4.3 support" );
        return nullptr;
      }

    } // namespace opengl
  } // namespace culling
} // namespace dp


// GL_VERSION_4_3
#endif
