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

#include <dp/rix/fx/Manager.h>
#include <BufferManager.h>
#include <ParameterGroupSpecInfo.h>
#include <map>
#include <string>

namespace dp 
{
  namespace rix
  {
    namespace fx 
    {

      class ManagerUniform :  public Manager {
      public:
#define DEFINE_CLASS( name )\
        class name; \
        typedef name*   name##Handle; \
        typedef name*   name##WeakHandle; \
        typedef dp::rix::core::SmartHandle<name> name##SharedHandle;

        DEFINE_CLASS(Instance);
        DEFINE_CLASS(Program);

#undef DEFINE_CLASS

        ManagerUniform( dp::rix::core::Renderer* renderer, dp::fx::Manager managerType );
        ~ManagerUniform();

        void runPendingUpdates();

        dp::rix::fx::ProgramSharedHandle programCreate( dp::fx::SmartEffectSpec const & effectSpec
                                                      , Manager::SystemSpecs const & systemSpecs
                                                      , char const * technique
                                                      , dp::rix::core::ContainerDescriptorHandle *userDescriptors
                                                      , size_t numDescriptors
                                                      , SourceFragments const & sourceFragments = SourceFragments() );

        std::map<dp::fx::Domain,std::string> getShaderSources( const dp::fx::SmartEffectSpec & effectSpec
                                                             , bool depthPass
                                                             , dp::rix::fx::Manager::SystemSpecs const & systemSpecs
                                                             , SourceFragments const & sourceFragments = SourceFragments() ) const;

        dp::rix::fx::GroupDataSharedHandle groupDataCreate  ( dp::fx::SmartParameterGroupSpec const& groupSpec );
        void                              groupDataSetValue( dp::rix::fx::GroupDataSharedHandle const& groupData, const dp::fx::ParameterGroupSpec::iterator& parameter, const dp::rix::core::ContainerData& data );

        virtual dp::rix::fx::InstanceHandle instanceCreate       (core::GeometryInstanceHandle gi) ;
        virtual dp::rix::fx::InstanceHandle instanceCreate       (core::RenderGroupHandle renderGroup) ;
        virtual void                        instanceSetProgram   ( dp::rix::fx::InstanceHandle instanceHandle, dp::rix::fx::ProgramHandle programHandle );
        virtual bool                        instanceUseGroupData ( dp::rix::fx::InstanceHandle instanceHandle, dp::rix::fx::GroupDataHandle groupHandle);

        dp::rix::core::Renderer*  getRenderer() const;
        virtual dp::fx::Manager   getManagerType() const;

      private:
        SmartParameterGroupSpecInfoHandle getParameterGroupSpecInfo( const dp::fx::SmartParameterGroupSpec& spec );

        dp::rix::core::Renderer*   m_renderer;
        dp::fx::Manager            m_managerType;

        typedef std::map<std::string, dp::rix::fx::ProgramSharedHandle> ProgramMap;
        ProgramMap m_programs;

        typedef std::map<dp::fx::SmartParameterGroupSpec, SmartParameterGroupSpecInfoHandle> ParameterGroupSpecInfoMap;
        ParameterGroupSpecInfoMap m_groupInfos;

        typedef std::vector<dp::rix::fx::GroupDataSharedHandle> DirtyGroupContainer;
        DirtyGroupContainer m_dirtyGroups;
        std::vector<SmartBufferManager> m_bufferManagers;
      };

    } // namespace fx
  } // namespace rix
} // namespace dp

