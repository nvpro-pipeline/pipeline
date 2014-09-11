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

#include <dp/fx/EffectLibrary.h>
#include <dp/fx/EnumSpec.h>
#include <dp/fx/EffectSpec.h>
#include <dp/fx/EffectData.h>
#include <dp/fx/inc/Snippet.h>
#include <stack>

namespace dp
{
  namespace fx
  {

    class EffectLoader;

    class EffectLibraryImpl : public EffectLibrary
    {
    public:
      EffectLibraryImpl();
      virtual ~EffectLibraryImpl();

      static EffectLibrary* instance();

      // public interface
      virtual bool loadEffects(const std::string& filename, const std::vector<std::string> &searchPaths );
      virtual bool save( const SmartEffectData& effectData, const std::string& filename );
      virtual void getEffectNames(std::vector<std::string>& names );
      virtual void getEffectNames( const std::string & filename, EffectSpec::Type type, std::vector<std::string> & names ) const;
      virtual const SmartEffectSpec& getEffectSpec(const std::string& effectName) const;
      virtual std::string const& getEffectFile( std::string const& effectName ) const;
      virtual const SmartParameterGroupSpec& getParameterGroupSpec( const std::string & pgsName ) const;
      virtual const SmartEnumSpec& getEnumSpec( const std::string& name ) const;

      virtual const SmartParameterGroupData& getParameterGroupData( const std::string& name ) const;
      virtual const SmartEffectData& getEffectData( const std::string& name ) const;

      SmartShaderPipeline generateShaderPipeline( const ShaderPipelineConfiguration& configuration );

      // interface for backends
      virtual SmartEnumSpec registerSpec( const SmartEnumSpec& enumSpec );
      virtual SmartEffectSpec registerSpec( const SmartEffectSpec& enumSpec, EffectLoader* effectLoader );
      virtual SmartParameterGroupSpec registerSpec( const SmartParameterGroupSpec& enumSpec );
      virtual SmartParameterGroupData registerParameterGroupData( const SmartParameterGroupData& parameterGroupData );
      virtual SmartEffectData registerEffectData( const SmartEffectData& effectData );
      virtual std::vector<std::string> getRegisteredExtensions() const;
      virtual bool effectHasTechnique( SmartEffectSpec const& effectSpec, std::string const& techniqueName, bool rasterizer ) const;

      void registerEffectLoader( const boost::shared_ptr<EffectLoader>& effectLoader, const std::string& extension );

    private:

      typedef std::vector< std::string> SearchPaths;
      SearchPaths m_searchPaths;

      typedef std::map<std::string, boost::shared_ptr<EffectLoader> > EffectLoaders;
      EffectLoaders m_effectLoaders;

      typedef std::map<std::string, SmartParameterGroupSpec> ParameterGroupSpecs;
      ParameterGroupSpecs m_parameterGroupSpecs;

      // List of EffectSpecs
      struct EffectSpecInfo
      {
        EffectSpecInfo()
          : effectLoader( nullptr )
        {
        }

        EffectSpecInfo( const SmartEffectSpec& pEffectSpec, EffectLoader* pEffectLoader, const std::string & file )
          : effectSpec( pEffectSpec)
          , effectLoader( pEffectLoader )
          , effectFile( file )
        {
        }

        SmartEffectSpec   effectSpec;
        EffectLoader    * effectLoader;
        std::string       effectFile;
      };

      typedef std::map<std::string, EffectSpecInfo> EffectSpecs;
      EffectSpecs m_effectSpecs; 

      typedef std::map<std::string, SmartEnumSpec> EnumSpecs;
      EnumSpecs m_enumSpecs;

      typedef std::map<std::string, SmartParameterGroupData> ParameterGroupDatas;
      ParameterGroupDatas m_parameterGroupDatas;

      typedef std::map<std::string, SmartEffectData> EffectDatas;
      EffectDatas m_effectDatas;

      std::stack<std::string> m_currentFile;

      std::set<std::string> m_loadedFiles;
    };



  } // namespace fx
} // namespace dp