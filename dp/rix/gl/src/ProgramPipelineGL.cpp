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


#include "ProgramPipelineGL.h"

#include "ContainerGL.h"
#include "ProgramGL.h"
#include "DataTypeConversionGL.h"

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

#if RIX_GL_SEPARATE_SHADER_OBJECTS_SUPPORT == 1
      ProgramPipelineGL::ProgramPipelineGL( const ProgramHandle* programs, unsigned int numPrograms )
        : m_activeAttributeMask( 0 )
      {
        glGenProgramPipelines( 1, &m_pipelineId );
        DP_ASSERT( m_pipelineId );

        for( unsigned int i=0; i < numPrograms; ++i )
        {
          DP_ASSERT( handleIsTypeOf<ProgramGL>(programs[i]) );
          const ProgramGLHandle& program = handleCast<ProgramGL>(programs[i]);
          m_programPositions[program] = static_cast<unsigned int>(m_programs.size());
          m_programs.push_back(program);
          m_activeAttributeMask |= program->getActiveAttributeMask();

          //
          // assemble the program pipeline
          //
          GLbitfield stages = 0;
          for( size_t i = 0; i < program->m_shaders.size(); ++i )
          {
            stages |= getGLShaderBits(program->m_shaders[i].m_shaderType);
          }
          glUseProgramStages( m_pipelineId, stages, program->m_programId );

          //
          // collect all unique container descriptors
          // save their position in the descriptor vector
          // build up a list of programs that use this descriptor
          //
          ProgramGL::DescriptorIndexMap::const_iterator it;
          ProgramGL::DescriptorIndexMap::const_iterator it_end = program->m_descriptorToPosition.end();
          for( it = program->m_descriptorToPosition.begin(); it != it_end; ++it )
          {
            ContainerDescriptorGLHandle containerDescriptor = it->first;

            ContainerDescriptorPositions::iterator cdp = m_containerDescriptorPositions.find( containerDescriptor );
            if( cdp == m_containerDescriptorPositions.end() )
            {
              // container descriptor not yet encountered, collect all needed information
              DescriptorData descriptorData;
              descriptorData.m_descriptor          = containerDescriptor;
              descriptorData.m_numParameterObjects = static_cast<unsigned int>(containerDescriptor->m_parameterInfos.size());
              descriptorData.m_parameterObjects    = program->m_descriptorParameters[it->second];
              descriptorData.m_programs.push_back( program );

              size_t pos = m_containerDescriptorData.size();
              m_containerDescriptorData.push_back( descriptorData );
              m_containerDescriptorPositions[containerDescriptor] = static_cast<unsigned int>(pos);
            }
            else
            {
              // container descriptor already known, just add the program to its users
              m_containerDescriptorData[cdp->second].m_programs.push_back(program);
            }
          }
        }

        // calculate number of active attributes (bits in mask set to 1)
        m_numberOfActiveAttributes = 0;
        unsigned int tmp = m_activeAttributeMask;
        while ( tmp )
        {
          ++m_numberOfActiveAttributes;
          tmp &= tmp - 1; // this sets the rightmost 1 bit to 0
        }

        glValidateProgramPipeline( m_pipelineId );
      }

      ProgramPipelineGL::~ProgramPipelineGL()
      {
        glDeleteProgramPipelines( 1, &m_pipelineId );
      }
#else
      ProgramPipelineGL::ProgramPipelineGL( ProgramSharedHandle const * xprograms, unsigned int xnumPrograms )
        : m_activeAttributeMask( 0 )
      {
        ProgramGLSharedHandle program = nullptr;
        if ( xnumPrograms == 1)
        {
          program = handleCast<ProgramGL>(xprograms[0]);
        }
        else
        {
          std::vector<std::string> sources;
          std::vector<ShaderType> shaderTypes;
          std::vector<dp::rix::core::ContainerDescriptorSharedHandle> descriptors;
          std::set<dp::rix::core::ContainerDescriptorSharedHandle> existingDescriptors;

          for ( size_t programIndex = 0; programIndex < xnumPrograms; ++programIndex )
          {
            ProgramGLHandle programGLHandle = handleCast<ProgramGL>(xprograms[programIndex].get());
            std::vector<dp::gl::SmartShader> const& shaders = programGLHandle->getProgram()->getShaders();
            for ( size_t shaderIndex = 0;shaderIndex < shaders.size(); ++shaderIndex )
            {
              sources.push_back( shaders[shaderIndex]->getSource() );
              shaderTypes.push_back( getShaderType( shaders[shaderIndex]->getType() ) );
            }

            std::vector<ContainerDescriptorGLSharedHandle> const & newDescriptors = programGLHandle->getDescriptors();
            for ( size_t descriptorIndex = 0;descriptorIndex < newDescriptors.size(); ++descriptorIndex )
            {
              if ( existingDescriptors.insert( handleCast<ContainerDescriptor>(newDescriptors[descriptorIndex]) ).second )
              {
                descriptors.push_back( handleCast<ContainerDescriptor>(newDescriptors[descriptorIndex]) );
              }
            }
          }

          std::vector<char const *> stringPointers;
          for (size_t index = 0;index < sources.size();++index)
          {
            stringPointers.push_back(sources[index].c_str());
          }
          const char** sc = stringPointers.data();
          ShaderType* st = shaderTypes.data();
          ContainerDescriptorSharedHandle* dh = descriptors.empty() ? nullptr : &descriptors[0];

          // TODO this fails during the dynamic_cast type check. figure out why
          //rix::core::ProgramDescription pd( dp::rix::core::ProgramShaderCode( sources.size(), sc, st ), dh, descriptors.size() );

          // TODO this is working nicely
          dp::rix::core::ProgramShaderCode psc( sources.size(), sc, st );
          rix::core::ProgramDescription pd( psc, dh, descriptors.size() );

          program = new ProgramGL( pd );
        }


        m_programPositions[program.get()] = static_cast<unsigned int>(m_programs.size());
        m_programs.push_back(program);
        m_activeAttributeMask |= program->getProgram()->getActiveAttributesMask();

        //
        // collect all unique container descriptors
        // save their position in the descriptor vector
        // build up a list of programs that use this descriptor
        //
        ProgramGL::DescriptorIndexMap::const_iterator it;
        ProgramGL::DescriptorIndexMap::const_iterator it_end = program->getDescriptorToIndexMap().end();
        for( it = program->getDescriptorToIndexMap().begin(); it != it_end; ++it )
        {
          ContainerDescriptorGLHandle containerDescriptor = it->first;

          ContainerDescriptorPositions::iterator cdp = m_containerDescriptorPositions.find( containerDescriptor );
          if( cdp == m_containerDescriptorPositions.end() )
          {
            // container descriptor not yet encountered, collect all needed information
            DescriptorData descriptorData;
            descriptorData.m_descriptor          = containerDescriptor;
            descriptorData.m_programs.push_back( program.get() );

            size_t pos = m_containerDescriptorData.size();
            m_containerDescriptorData.push_back( descriptorData );
            m_containerDescriptorPositions[containerDescriptor] = static_cast<unsigned int>(pos);
          }
          else
          {
            // container descriptor already known, just add the program to its users
            m_containerDescriptorData[cdp->second].m_programs.push_back( program.get() );
          }
        }

        // calculate number of active attributes (bits in mask set to 1)
        m_numberOfActiveAttributes = 0;
        unsigned int tmp = m_activeAttributeMask;
        while ( tmp )
        {
          ++m_numberOfActiveAttributes;
          tmp &= tmp - 1; // this sets the rightmost 1 bit to 0
        }
      }

      ProgramPipelineGL::~ProgramPipelineGL()
      {
      }

#endif

    } // namespace gl
  } // namespace rix
} // namespace dp
