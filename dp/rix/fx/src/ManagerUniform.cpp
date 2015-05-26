// Copyright NVIDIA Corporation 2012-2015
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

#include <assert.h>
#include <iostream>
#include <GroupDataUniform.h>
#include <GroupDataBuffered.h>
#include <GroupDataBufferedCombined.h>
#include <ManagerUniform.h>
#include <ParameterGroupSpecInfo.h>

#include <dp/rix/fx/inc/BufferManagerIndexed.h>
#include <dp/rix/fx/inc/BufferManagerOffset.h>

#include <dp/fx/inc/ExtensionSnippet.h>
#include <dp/fx/inc/EnumSpecSnippet.h>
#include <dp/fx/inc/FileSnippet.h>
#include <dp/fx/inc/ParameterGroupDataPrivate.h>
#include <dp/fx/inc/ParameterGroupSnippet.h>
#include <dp/fx/inc/StringSnippet.h>
#include <dp/fx/inc/VersionSnippet.h>

#include <dp/util/Array.h>
#include <dp/util/HashGeneratorMurMur.h>
#include <dp/util/File.h>
#include <dp/util/Memory.h>
#include <boost/scoped_array.hpp>

#include <typeinfo>

using namespace dp::rix::core;
using dp::util::array;

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      /************************************************************************/
      /* ManagerUniform::Program                                              */
      /************************************************************************/
      class ManagerUniform::Program : public dp::rix::fx::Program
      {
      public:
        Program( ManagerUniform *manager, dp::fx::EffectSpecSharedPtr const & effectSpec
               , Manager::SystemSpecs const & systemSpecs
               , char const * technique, dp::rix::core::ContainerDescriptorHandle * userDescriptors, size_t numDescriptors
               , SourceFragments const & sourceFragments );

        static void getShaderSource( dp::fx::ShaderPipelineConfiguration const & configuration
                                   , dp::fx::ShaderPipelineSharedPtr const & pipeline
                                   , dp::rix::fx::Manager::SystemSpecs const & systemSpecs
                                   , dp::fx::Domain domain
                                   , std::string& source
                                   , std::string& entrypoint );

        dp::rix::core::ProgramPipelineSharedHandle      m_programPipeline;
        std::vector<SmartParameterGroupSpecInfoHandle> m_parameterGroupSpecInfos;
      };

      ManagerUniform::Program::Program( ManagerUniform *manager
                                      , dp::fx::EffectSpecSharedPtr const & effectSpec
                                      , Manager::SystemSpecs const & systemSpecs
                                      , char const * technique
                                      , dp::rix::core::ContainerDescriptorHandle * userDescriptors
                                      , size_t numDescriptors
                                      , SourceFragments const & sourceFragments )
      {
        DP_ASSERT( manager && effectSpec );
        DP_ASSERT( effectSpec->getType() == dp::fx::EffectSpec::EST_PIPELINE );
        DP_ASSERT( userDescriptors || numDescriptors == 0 );

        {
          /************************************************************************/
          /* create the program                                                   */
          /************************************************************************/
          dp::fx::ShaderPipelineConfiguration configuration( effectSpec->getName() );
          configuration.setManager( manager->getManagerType() );
          configuration.setTechnique( technique );  // Required to find the matching signature of the vertex shader.
          for ( SourceFragments::const_iterator it = sourceFragments.begin(); it != sourceFragments.end(); ++it )
          {
            configuration.addSourceCode( it->first, it->second );
          }
          dp::fx::ShaderPipelineSharedPtr shaderPipeline = dp::fx::EffectLibrary::instance()->generateShaderPipeline( configuration );
         
#if !defined(NDEBUG)
          const dp::fx::ShaderPipeline::iterator itStageVertex   = shaderPipeline->getStage( dp::fx::DOMAIN_VERTEX );
          const dp::fx::ShaderPipeline::iterator itStageFragment = shaderPipeline->getStage( dp::fx::DOMAIN_FRAGMENT );

          DP_ASSERT( itStageVertex   != shaderPipeline->endStages() );
          DP_ASSERT( itStageFragment != shaderPipeline->endStages() );
#endif

          std::vector<std::string> shaderStringSources;
          std::vector<char const*> shaderSources;
          std::vector<dp::rix::core::ShaderType> shaderEnums;
          for ( dp::fx::ShaderPipeline::iterator it = shaderPipeline->beginStages();it != shaderPipeline->endStages(); ++it )
          {
            std::string source;
            std::string entryPoint;
            getShaderSource(configuration, shaderPipeline, systemSpecs, (*it).domain, source, entryPoint );

            switch ( (*it).domain )
            {
            case dp::fx::DOMAIN_VERTEX:
              shaderEnums.push_back( dp::rix::core::ST_VERTEX_SHADER );
              break;
            case dp::fx::DOMAIN_FRAGMENT:
              shaderEnums.push_back( dp::rix::core::ST_FRAGMENT_SHADER );
              break;
            case dp::fx::DOMAIN_GEOMETRY:
              shaderEnums.push_back( dp::rix::core::ST_GEOMETRY_SHADER );
              break;
            case dp::fx::DOMAIN_TESSELLATION_CONTROL:
              shaderEnums.push_back( dp::rix::core::ST_TESS_CONTROL_SHADER );
              break;
            case dp::fx::DOMAIN_TESSELLATION_EVALUATION:
              shaderEnums.push_back( dp::rix::core::ST_TESS_EVALUATION_SHADER );
              break;
            default:
              throw std::runtime_error( "unexpected shader type" );
              break;
            }

            shaderStringSources.push_back( source );

            // For each SystemSpec of the current stage
            for ( std::vector<std::string>::const_iterator itSystemSpec = (*it).systemSpecs.begin(); itSystemSpec != (*it).systemSpecs.end(); ++itSystemSpec )
            {
              // check if there's a systemSpec for the given string
              Manager::SystemSpecs::const_iterator itSystemSpec2 = systemSpecs.find( *itSystemSpec );
              if ( itSystemSpec2 != systemSpecs.end() )
              {
                // Global SystemSpecs are being used to generate the parameter header for the source code, but they're not supposed to generate
                // ContainerDescriptors for the GeometryInstances. Thus skip them here.
                if ( !itSystemSpec2->second.m_isGlobal )
                {
                  dp::fx::EffectSpecSharedPtr const & effectSpec = itSystemSpec2->second.m_effectSpec;
                  for ( dp::fx::EffectSpec::iterator itParameterGroupSpec = effectSpec->beginParameterGroupSpecs(); itParameterGroupSpec != effectSpec->endParameterGroupSpecs(); ++itParameterGroupSpec )
                  {
                    m_parameterGroupSpecInfos.push_back(manager->getParameterGroupSpecInfo(*itParameterGroupSpec));
                  }
                }
              }
            }
          }

          // generate vector of char const* for sources
          for ( size_t index = 0; index < shaderStringSources.size(); ++index )
          {
            shaderSources.push_back( shaderStringSources[index].c_str() );
          }

          DP_ASSERT( !shaderSources.empty() );
          
          dp::rix::core::ProgramShaderCode programShaderCode( shaderSources.size(), &shaderSources[0], &shaderEnums[0] );

          /************************************************************************/
          /* Add descriptors                                                      */
          /************************************************************************/
          std::vector<dp::rix::core::ContainerDescriptorSharedHandle> descriptors;
          std::copy( userDescriptors, userDescriptors + numDescriptors, std::back_inserter(descriptors));

          // ensure that a spec gets inserted only once even if it is being used by multiple domains.
          std::set<dp::fx::ParameterGroupSpecSharedPtr> usedSpecs;

          for ( dp::fx::EffectSpec::iterator it = effectSpec->beginParameterGroupSpecs(); it != effectSpec->endParameterGroupSpecs(); ++it )
          {
            if ( usedSpecs.insert(*it).second )
            {
              m_parameterGroupSpecInfos.push_back(manager->getParameterGroupSpecInfo(*it));
            }
          }

          for ( size_t index = 0;index < m_parameterGroupSpecInfos.size(); ++index )
          {
            // TODO get array of descriptors here...
            if ( m_parameterGroupSpecInfos[index]->m_bufferManager )
            {
              if ( typeid(*m_parameterGroupSpecInfos[index]->m_bufferManager.get()) == typeid(BufferManagerIndexed) )
              {
                BufferManagerIndexed* bufferManager = dp::rix::core::handleCast<BufferManagerIndexed>(m_parameterGroupSpecInfos[index]->m_bufferManager.get());

                descriptors.push_back(bufferManager->getBufferDescriptor());
                descriptors.push_back(bufferManager->getIndexDescriptor());
              }
              else if ( typeid(*m_parameterGroupSpecInfos[index]->m_bufferManager.get()) == typeid(BufferManagerOffset) )
              {
                BufferManagerOffset* bufferManager = dp::rix::core::handleCast<BufferManagerOffset>(m_parameterGroupSpecInfos[index]->m_bufferManager.get());

                descriptors.push_back(bufferManager->getBufferDescriptor());
              }
              else
              {
                DP_ASSERT( !"unsupported BufferManager" );
              }
            }
            else
            {
              descriptors.push_back(m_parameterGroupSpecInfos[index]->m_descriptor);
            }
          }

          dp::rix::core::ProgramDescription description( programShaderCode, descriptors.empty() ? nullptr : &descriptors[0], descriptors.size() );
          dp::rix::core::ProgramSharedHandle programs[] = { manager->getRenderer()->programCreate( description ) };
          m_programPipeline = manager->getRenderer()->programPipelineCreate( programs, sizeof array( programs ) );
        }
      }

    void ManagerUniform::Program::getShaderSource( dp::fx::ShaderPipelineConfiguration const & configuration
                                                 , dp::fx::ShaderPipelineSharedPtr const & pipeline
                                                 , dp::rix::fx::Manager::SystemSpecs const & systemSpecs
                                                 , dp::fx::Domain domain
                                                 , std::string& source
                                                 , std::string& /*entrypoint*/ )
    {
      std::vector<dp::fx::SnippetSharedPtr> shaderSnippets;

      dp::fx::ShaderPipeline::Stage const & stage = *pipeline->getStage( domain );
      
      // generate header only if a body had been generated
      if ( stage.source )
      {
        std::vector<dp::fx::SnippetSharedPtr> headerSnippets;
        headerSnippets.push_back( std::make_shared<dp::fx::VersionSnippet>() );
        headerSnippets.push_back( std::make_shared<dp::fx::ExtensionSnippet>() );

        std::vector<dp::fx::SnippetSharedPtr> pgsSnippets;

        // gather all parameter groups
        for ( std::vector<dp::fx::ParameterGroupSpecSharedPtr>::const_iterator it = stage.parameterGroupSpecs.begin();
          it != stage.parameterGroupSpecs.end();
          ++it )
        {
          pgsSnippets.push_back( std::make_shared<dp::fx::ParameterGroupSnippet>( *it ) );
        }

        for( std::vector<std::string>::const_iterator it = stage.systemSpecs.begin(); it != stage.systemSpecs.end(); ++it )
        {
          dp::rix::fx::Manager::SystemSpecs::const_iterator itSystemSpec = systemSpecs.find(*it);
          if ( itSystemSpec == systemSpecs.end() )
          {
            throw std::runtime_error( std::string("Missing SystemSpec " + *it + " for effect " + configuration.getName() ) );
          }
          for ( std::vector<dp::fx::ParameterGroupSpecSharedPtr>::const_iterator it = itSystemSpec->second.m_effectSpec->beginParameterGroupSpecs();
            it != itSystemSpec->second.m_effectSpec->endParameterGroupSpecs();
            ++it )
          {
            pgsSnippets.push_back( std::make_shared<dp::fx::ParameterGroupSnippet>( *it ) );
          }
        }

        // get the enums from shader and pgs snippets
        std::set<dp::fx::EnumSpecSharedPtr> enumSpecs;
        for ( std::vector<dp::fx::SnippetSharedPtr>::const_iterator it = shaderSnippets.begin(); it != shaderSnippets.end(); ++it )
        {
          std::set<dp::fx::EnumSpecSharedPtr> const & snippetEnums = (*it)->getRequiredEnumSpecs();
          if ( ! snippetEnums.empty() )
          {
            enumSpecs.insert( snippetEnums.begin(), snippetEnums.end() );
          }
        }

        for ( std::vector<dp::fx::SnippetSharedPtr>::const_iterator it = pgsSnippets.begin(); it != pgsSnippets.end(); ++it )
        {
          const std::set<dp::fx::EnumSpecSharedPtr> & snippetEnums = (*it)->getRequiredEnumSpecs();
          if ( ! snippetEnums.empty() )
          {
            enumSpecs.insert( snippetEnums.begin(), snippetEnums.end() );
          }
        }

        // insert enumSpecs from source body
        enumSpecs.insert( stage.source->getRequiredEnumSpecs().begin(), stage.source->getRequiredEnumSpecs().end() );

        for ( std::set<dp::fx::EnumSpecSharedPtr>::const_iterator it = enumSpecs.begin(); it != enumSpecs.end(); ++it )
        {
          headerSnippets.push_back( std::make_shared<dp::fx::EnumSpecSnippet>( *it ) );
        }

        // add snippets from renderer
        std::vector<dp::fx::SnippetSharedPtr> sources;
        std::vector<std::string> const& codeSnippets = configuration.getSourceCodes(domain);
        for( size_t index = 0;index < codeSnippets.size(); ++index )
        {
          sources.push_back( std::make_shared<dp::fx::StringSnippet>( codeSnippets[index] ) );
        }

        // generate code from header, pgs, and shader
        dp::fx::GeneratorConfiguration config;
        config.manager = configuration.getManager();

        sources.push_back( stage.source );

        source = generateSnippets( headerSnippets, config )
               + generateSnippets( pgsSnippets,    config )
               + generateSnippets( sources,   config );
      }
        }

    class ManagerUniform::Instance : public dp::rix::fx::Instance 
    {
    public:
      virtual void setProgram( dp::rix::core::Renderer * renderer, ManagerUniform::ProgramSharedHandle ) = 0;
      virtual bool useGroupData( dp::rix::core::Renderer * renderer, dp::rix::fx::GroupDataHandle groupHandle ) = 0;
    };

      /************************************************************************/
      /* InstanceGeometryInstance                                             */
      /************************************************************************/
      class InstanceGeometryInstance : public ManagerUniform::Instance
      {
      public:
        InstanceGeometryInstance(core::GeometryInstanceHandle gi) 
          : m_gi(gi)
        {
        }

        virtual void setProgram( dp::rix::core::Renderer * renderer, ManagerUniform::ProgramSharedHandle program);
        virtual bool useGroupData( dp::rix::core::Renderer * renderer, dp::rix::fx::GroupDataHandle groupHandle );

      private:
        ManagerUniform::ProgramSharedHandle  m_program;
        core::GeometryInstanceSharedHandle   m_gi;
      };

      void InstanceGeometryInstance::setProgram( dp::rix::core::Renderer * renderer, ManagerUniform::ProgramSharedHandle program )
      {
        m_program = program;
        renderer->geometryInstanceSetProgramPipeline( m_gi, m_program->m_programPipeline );
      }

      bool InstanceGeometryInstance::useGroupData( dp::rix::core::Renderer * renderer, dp::rix::fx::GroupDataHandle groupHandle )
      {
        GroupDataBaseHandle groupData = handleCast<GroupDataBase>(groupHandle);

        if (handleIsTypeOf<GroupDataBufferedCombined>(groupData ) )
        {
          ((GroupDataBufferedCombined*)(groupData))->useContainers( m_gi );
        }
        else
        {
          renderer->geometryInstanceUseContainer( m_gi, groupData->m_container );
        }

        return true;
      }

      class InstanceRenderGroup : public ManagerUniform::Instance
      {
      public:
        InstanceRenderGroup(core::RenderGroupHandle renderGroup) 
          : m_renderGroup(renderGroup)
        {
        }

        virtual void setProgram( dp::rix::core::Renderer* renderer, ManagerUniform::ProgramSharedHandle);
        virtual bool useGroupData( dp::rix::core::Renderer* renderer, dp::rix::fx::GroupDataHandle groupHandle );

      private:
        ManagerUniform::ProgramSharedHandle  m_program;
        dp::rix::core::RenderGroupHandle    m_renderGroup;
      };

      void InstanceRenderGroup::setProgram( dp::rix::core::Renderer * renderer, ManagerUniform::ProgramSharedHandle program )
      {
        m_program = program;
        renderer->renderGroupSetProgramPipeline( m_renderGroup, m_program->m_programPipeline );
      }

      bool InstanceRenderGroup::useGroupData( dp::rix::core::Renderer * renderer, dp::rix::fx::GroupDataHandle groupHandle )
      {
        GroupDataBaseHandle groupData = handleCast<GroupDataBase>(groupHandle);

        if (handleIsTypeOf<GroupDataBufferedCombined>(groupData ) )
        {
          ((GroupDataBufferedCombined*)(groupData))->useContainers( m_renderGroup );
        }
        else
        {
          renderer->renderGroupUseContainer( m_renderGroup, groupData->m_container );
        }

        return true;
      }

      //////////////////////////////////////////////////////////////////////////
      // System
      ManagerUniformSharedPtr ManagerUniform::create( core::Renderer* renderer, dp::fx::Manager managerType )
      {
        return( std::shared_ptr<ManagerUniform>( new ManagerUniform( renderer, managerType ) ) );
      }

      ManagerUniform::ManagerUniform( core::Renderer* renderer, dp::fx::Manager managerType )
        : m_renderer( renderer )
        , m_managerType( managerType )
      {
      }

      ManagerUniform::~ManagerUniform()
      {

      }

      void ManagerUniform::runPendingUpdates()
      {
        for ( DirtyGroupContainer::iterator it = m_dirtyGroups.begin(); it != m_dirtyGroups.end(); ++it )
        {
          DP_ASSERT( handleIsTypeOf<GroupDataBase>( it->get() ) );
          GroupDataBaseHandle groupData = handleCast<GroupDataBase>( it->get() );

          groupData->update();
        }
        m_dirtyGroups.clear();

        for ( std::vector<SmartBufferManager>::iterator it = m_bufferManagers.begin(); it != m_bufferManagers.end(); ++it )
        {
          (*it)->update();
        }
      }

      /************************************************************************/
      /* GroupData                                                            */
      /************************************************************************/

      dp::rix::fx::GroupDataSharedHandle ManagerUniform::groupDataCreate( dp::fx::ParameterGroupSpecSharedPtr const& group )
      {
        SmartParameterGroupSpecInfoHandle pgsi = getParameterGroupSpecInfo( group );
        dp::fx::ParameterGroupLayoutSharedPtr layout = pgsi->m_groupLayout;
        switch( layout->getManager() )
        {
        case dp::fx::MANAGER_UNIFORM:
        case dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX:
        case dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT_RIX:
          return new GroupDataUniform( this, pgsi );

        case dp::fx::MANAGER_SHADERBUFFER:
          return new GroupDataBufferedCombined( this, pgsi );

        case dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT:
          if ( layout->isInstanced() )
          {
            return new GroupDataBufferedCombined( this, pgsi );
          }
          else
          {
            return new GroupDataBuffered( this, pgsi );
          }
        case dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX:
          return new GroupDataBufferedCombined( this, pgsi );

        default:
          DP_ASSERT( !"unsupported manager" );
          return nullptr;
        }
      }

      void ManagerUniform::groupDataSetValue( dp::rix::fx::GroupDataSharedHandle const& groupDataHandle
                                            , const dp::fx::ParameterGroupSpec::iterator& parameter
                                            , const dp::rix::core::ContainerData& data )
      {
        GroupDataBaseHandle groupData = handleCast<GroupDataBase>(groupDataHandle.get());

        if ( groupData->setValue( parameter, data ) )
        {
          m_dirtyGroups.push_back( groupDataHandle );
        }
      }

      //////////////////////////////////////////////////////////////////////////
      // Instance

      dp::rix::fx::InstanceHandle ManagerUniform::instanceCreate( dp::rix::core::GeometryInstanceHandle gi )
      {
        InstanceHandle instance = new InstanceGeometryInstance( gi );

        return instance;
      }

      dp::rix::fx::InstanceHandle ManagerUniform::instanceCreate( dp::rix::core::RenderGroupHandle renderGroup )
      {
        InstanceHandle instance = new InstanceRenderGroup( renderGroup );

        return instance;
      }


      dp::rix::fx::ProgramSharedHandle ManagerUniform::programCreate( const dp::fx::EffectSpecSharedPtr& effectSpec
                                                                    , Manager::SystemSpecs const & systemSpecs
                                                                    ,  const char *technique
                                                                    ,  dp::rix::core::ContainerDescriptorHandle *userDescriptors
                                                                    ,  size_t numDescriptors
                                                                    ,  SourceFragments const& sourceFragments )
      {
        std::string key = effectSpec->getName() + "@" + technique;
        ProgramMap::iterator it = m_programs.find( key );
        if ( it == m_programs.end() )
        {
          dp::rix::fx::ProgramSharedHandle handle;
          DP_ASSERT( m_managerType != dp::fx::MANAGER_UNKNOWN );
          if ( dp::fx::EffectLibrary::instance()->effectHasTechnique( effectSpec, technique, true ) )
          {
            handle = new ManagerUniform::Program( this, effectSpec, systemSpecs, technique, userDescriptors, numDescriptors, sourceFragments );
          }
          m_programs[key] = handle;
          return handle;
        }
        return it->second;
      }

      std::map<dp::fx::Domain,std::string> ManagerUniform::getShaderSources( dp::fx::EffectSpecSharedPtr const & effectSpec
                                                                           , bool depthPass
                                                                           , dp::rix::fx::Manager::SystemSpecs const & systemSpecs
                                                                           , SourceFragments const & sourceFragments ) const
      {
        DP_ASSERT( effectSpec->getType() == dp::fx::EffectSpec::EST_PIPELINE );

        dp::fx::ShaderPipelineConfiguration configuration( effectSpec->getName() );

        configuration.setManager( m_managerType );
        for ( SourceFragments::const_iterator it = sourceFragments.begin(); it != sourceFragments.end(); ++it )
        {
          configuration.addSourceCode( it->first, it->second );
        }
        
        if ( depthPass )
        {
          configuration.setTechnique( "depthPass" );
        }
        else
        {
          configuration.setTechnique( "forward" ); // Required to find the matching signature of the vertex shader.
        }

        dp::fx::ShaderPipelineSharedPtr shaderPipeline = dp::fx::EffectLibrary::instance()->generateShaderPipeline( configuration );

        std::map<dp::fx::Domain,std::string> sources;
        for ( dp::fx::ShaderPipeline::iterator it = shaderPipeline->beginStages() ; it != shaderPipeline->endStages() ; ++it )
        {
          std::string source;
          std::string entryPoint;

          Program::getShaderSource( configuration, shaderPipeline, systemSpecs, (*it).domain, source, entryPoint );
          sources[it->domain] = source;
        }

        return( sources );
      }

      void ManagerUniform::instanceSetProgram( dp::rix::fx::InstanceHandle instanceHandle, dp::rix::fx::ProgramHandle programHandle )
      {
        DP_ASSERT( handleIsTypeOf<Instance>( instanceHandle ) );
        InstanceHandle instance = handleCast<Instance>(instanceHandle);

        DP_ASSERT( handleIsTypeOf<Program>( programHandle ) );
        ProgramHandle program = handleCast<Program>( programHandle );

        instance->setProgram( getRenderer(), program );
      }

      bool ManagerUniform::instanceUseGroupData( dp::rix::fx::InstanceHandle instanceHandle, dp::rix::fx::GroupDataHandle groupHandle )
      {
        InstanceHandle  instance  = handleCast<Instance>(instanceHandle);

        return instance->useGroupData( getRenderer(), groupHandle );
      }

      SmartParameterGroupSpecInfoHandle ManagerUniform::getParameterGroupSpecInfo( const dp::fx::ParameterGroupSpecSharedPtr& spec ) 
      {
        ParameterGroupSpecInfoMap::iterator it = m_groupInfos.find(spec);
        if ( it == m_groupInfos.end() )
        {
          SmartParameterGroupSpecInfoHandle specInfo = new ParameterGroupSpecInfo( spec, getManagerType(), getRenderer() );
          m_groupInfos[spec] = specInfo;

          // keep track of BufferManagers for update function
          if ( specInfo->m_bufferManager )
          {
            m_bufferManagers.push_back( specInfo->m_bufferManager ); 
          }

          return specInfo;
        }

        return it->second;
      }

      dp::rix::core::Renderer* ManagerUniform::getRenderer() const
      {
        return m_renderer;
      }

      dp::fx::Manager ManagerUniform::getManagerType() const
      {
        return m_managerType;
      }

    }
  }
}
