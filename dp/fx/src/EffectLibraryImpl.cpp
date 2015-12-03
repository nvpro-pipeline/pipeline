// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/fx/inc/EffectDataPrivate.h>
#include <dp/fx/inc/EffectLibraryImpl.h>
#include <dp/fx/inc/ExtensionSnippet.h>
#include <dp/fx/inc/EnumSpecSnippet.h>
#include <dp/fx/inc/FileSnippet.h>
#include <dp/fx/inc/ParameterGroupDataPrivate.h>
#include <dp/fx/inc/ParameterGroupSnippet.h>
#include <dp/fx/inc/StringSnippet.h>
#include <dp/fx/inc/VersionSnippet.h>

#include <dp/fx/xml/EffectLoader.h>

#include <dp/util/HashGeneratorMurMur.h>
#include <dp/util/File.h>
#include <dp/DP.h>

#include <string>

namespace dp
{
  namespace fx
  {

    EffectLibrary* EffectLibraryImpl::instance()
    {
      static EffectLibraryImpl wrapper;
      return &wrapper;
    }

    EffectLibraryImpl::EffectLibraryImpl()
    {
      // set default search paths
      m_fileFinder.addSearchPath( dp::home() + "/media/dpfx" );
      m_fileFinder.addSearchPath( dp::home() + "/media/effects" );
    }

    EffectLibraryImpl::~EffectLibraryImpl()
    {
    }

    void EffectLibraryImpl::registerEffectLoader( EffectLoaderSharedPtr const& effectLoader, const std::string& extension )
    {
      DP_ASSERT( m_effectLoaders.find(extension) == m_effectLoaders.end() );
      m_effectLoaders[extension] = effectLoader;
    }

    std::vector<std::string> EffectLibraryImpl::getRegisteredExtensions() const
    {
      std::vector<std::string> extensions;
      for ( EffectLoaders::const_iterator it = m_effectLoaders.begin() ; it != m_effectLoaders.end() ; ++it )
      {
        extensions.push_back( it->first );
      }
      return( extensions );
    }

    bool EffectLibraryImpl::effectHasTechnique( EffectSpecSharedPtr const& effectSpec, std::string const& techniqueName, bool rasterizer ) const
    {
      EffectSpecs::const_iterator it = m_effectSpecs.find( effectSpec->getName() );
      if ( it != m_effectSpecs.end() )
      {
        return( it->second.effectLoader->effectHasTechnique( effectSpec, techniqueName, rasterizer ) );
      }
      return( false );
    }

    bool EffectLibraryImpl::loadEffects( const std::string& filename, dp::util::FileFinder const& fileFinder )
    {
      if ( m_loadedFiles.find( filename ) != m_loadedFiles.end() )
      {
        return( true );
      }

      std::string file = fileFinder.findRecursive( filename );
      if ( file.empty() )
      {
        file = m_fileFinder.findRecursive( filename );
      }

      if ( file.empty() )
      {
        // if it could not be found in the search directories, look relative to dp::home()
        if ( !dp::util::isAbsolutePath( filename ) )
        {
          std::string dpFile = dp::home() + std::string( "/" ) + filename;
          if ( dp::util::fileExists( dpFile ) )
          {
            file = dpFile;
          }
        }
      }
      if ( file.empty() )
      {
        std::cerr << "File not found: " << filename << std::endl;
        return false;
      }

      std::string extension = dp::util::getFileExtension( filename );

      EffectLoaders::iterator it = m_effectLoaders.find( extension );
      if ( it != m_effectLoaders.end() )
      {
        m_currentFile.push( file );
        dp::util::convertPath( m_currentFile.top() );
        bool success = it->second->loadEffects( m_currentFile.top(), fileFinder );
        m_currentFile.pop();
        m_loadedFiles.insert( filename );

        return success;
      }
      else
      {
        std::cerr << "no loader for effect file " << file << " available." << std::endl;
        return false;
      }
    }

    void EffectLibraryImpl::getEffectNames( std::vector<std::string> & names )
    {
      names.clear();
      for ( EffectSpecs::iterator it = m_effectSpecs.begin(); it != m_effectSpecs.end(); ++it )
      {
        names.push_back( it->first );
      }
    }

    void EffectLibraryImpl::getEffectNames( std::string const& filename, EffectSpec::Type type, std::vector<std::string> & names ) const
    {
      std::string fn( filename );
      dp::util::convertPath( fn );
      for ( EffectSpecs::const_iterator it = m_effectSpecs.begin(); it != m_effectSpecs.end(); ++it )
      {
        if ( ( it->second.effectSpec->getType() == type ) && ( it->second.effectFile == fn ) )
        {
          names.push_back( it->first );
        }
      }
    }

    const EffectSpecSharedPtr& EffectLibraryImpl::getEffectSpec(const std::string& effectName ) const
    {
      EffectSpecs::const_iterator it = m_effectSpecs.find( effectName );
      if ( it != m_effectSpecs.end() )
      {
        return it->second.effectSpec;
      }
      else
      {
        return( EffectSpecSharedPtr::null );
      }
    }

    std::string const& EffectLibraryImpl::getEffectFile( std::string const& effectName ) const
    {
      EffectSpecs::const_iterator it = m_effectSpecs.find( effectName );
      if ( it != m_effectSpecs.end() )
      {
        return( it->second.effectFile );
      }
      else
      {
        static std::string dummy;
        return( dummy );
      }
    }

    const ParameterGroupSpecSharedPtr& EffectLibraryImpl::getParameterGroupSpec( const std::string & pgsName ) const
    {
      ParameterGroupSpecs::const_iterator it = m_parameterGroupSpecs.find( pgsName );
      if ( it != m_parameterGroupSpecs.end() )
      {
        return it->second;
      }
      else
      {
        return( ParameterGroupSpecSharedPtr::null );
      }
    }

    ShaderPipelineSharedPtr EffectLibraryImpl::generateShaderPipeline( const ShaderPipelineConfiguration& configuration )
    {
      EffectSpecs::const_iterator itEffectSpec = m_effectSpecs.find( configuration.getName() );

      return itEffectSpec->second.effectLoader->generateShaderPipeline( configuration );
    }

    EnumSpecSharedPtr EffectLibraryImpl::registerSpec( const EnumSpecSharedPtr& enumSpec )
    {
      DP_ASSERT( enumSpec );
      EnumSpecs::iterator it = m_enumSpecs.find( enumSpec->getType() );
      if ( it != m_enumSpecs.end() )
      {
        DP_ASSERT( it->second->isEquivalent( enumSpec, false, false ) );
        return it->second;
      }
      else
      {
        m_enumSpecs[ enumSpec->getType() ] = enumSpec;
        return enumSpec;
      }
    }

    EffectSpecSharedPtr EffectLibraryImpl::registerSpec( const EffectSpecSharedPtr& effectSpec, EffectLoader* effectLoader )
    {
      DP_ASSERT( effectSpec );
      EffectSpecs::iterator it = m_effectSpecs.find( effectSpec->getName() );
      if ( it != m_effectSpecs.end() )
      {
        DP_ASSERT( it->second.effectLoader == effectLoader );
        DP_ASSERT( it->second.effectSpec->isEquivalent( effectSpec, false, false ) );
        DP_ASSERT( !m_currentFile.empty() && ( it->second.effectFile == m_currentFile.top() ) );
        return it->second.effectSpec;
      }
      else
      {
        m_effectSpecs[ effectSpec->getName()] = EffectSpecInfo( effectSpec, effectLoader, m_currentFile.top() );

        // create EffectData for default values of EffectSpec
        EffectDataPrivateSharedPtr effectData = EffectDataPrivate::create( effectSpec, effectSpec->getName() );
        for ( EffectSpec::iterator it = effectSpec->beginParameterGroupSpecs(); it != effectSpec->endParameterGroupSpecs(); ++it )
        {
          effectData->setParameterGroupData( it, getParameterGroupData( (*it)->getName() ) );
        }
        registerEffectData( effectData );

        return effectSpec;
      }
    }

    ParameterGroupSpecSharedPtr EffectLibraryImpl::registerSpec( const ParameterGroupSpecSharedPtr& parameterGroupSpec )
    {
      DP_ASSERT( parameterGroupSpec );
      ParameterGroupSpecs::const_iterator it = m_parameterGroupSpecs.find( parameterGroupSpec->getName() );
      if ( it != m_parameterGroupSpecs.end() )
      {
        DP_ASSERT( it->second->isEquivalent( parameterGroupSpec, false, false ) );
        return it->second;
      }
      else
      {
        m_parameterGroupSpecs[ parameterGroupSpec->getName() ] = parameterGroupSpec;

        // register default ParameterGroupData for parameterGroupSpec
        registerParameterGroupData( ParameterGroupDataPrivate::create( parameterGroupSpec, parameterGroupSpec->getName() ) );
        return parameterGroupSpec;
      }
    }

    ParameterGroupDataSharedPtr EffectLibraryImpl::registerParameterGroupData( const ParameterGroupDataSharedPtr& parameterGroupData )
    {
      DP_ASSERT( parameterGroupData );
      ParameterGroupDatas::const_iterator it = m_parameterGroupDatas.find( parameterGroupData->getName() );
      if ( it != m_parameterGroupDatas.end() )
      {
        DP_ASSERT( *(it->second) == *parameterGroupData );
        return it->second;
      }
      else
      {
        m_parameterGroupDatas[ parameterGroupData->getName() ] = parameterGroupData;
        return parameterGroupData;
      }
    }

    EffectDataSharedPtr EffectLibraryImpl::registerEffectData( const EffectDataSharedPtr& effectData )
    {
      DP_ASSERT( effectData );
      EffectDatas::const_iterator it = m_effectDatas.find( effectData->getName() );
      if ( it != m_effectDatas.end() )
      {
        //DP_ASSERT( it->second->isEquivalent( effectData, false, false ) );
        return it->second;
      }
      else
      {
        m_effectDatas[ effectData->getName() ] = effectData;
        return effectData;
      }
    }

    const EnumSpecSharedPtr& EffectLibraryImpl::getEnumSpec( const std::string& name ) const
    {
      EnumSpecs::const_iterator it = m_enumSpecs.find( name );
      if ( it != m_enumSpecs.end() )
      {
        return it->second;
      }
      else
      {
        return( EnumSpecSharedPtr::null );
      }
    }

    const ParameterGroupDataSharedPtr& EffectLibraryImpl::getParameterGroupData( const std::string& name ) const
    {
      ParameterGroupDatas::const_iterator it = m_parameterGroupDatas.find( name );
      if ( it != m_parameterGroupDatas.end() )
      {
        return it->second;
      }
      else
      {
        return( ParameterGroupDataSharedPtr::null );
      }
    }

    const EffectDataSharedPtr& EffectLibraryImpl::getEffectData( const std::string& name ) const
    {
      EffectDatas::const_iterator it = m_effectDatas.find( name );
      if ( it != m_effectDatas.end() )
      {
        return it->second;
      }
      else
      {
        return( EffectDataSharedPtr::null );
      }
    }

    bool EffectLibraryImpl::save( const EffectDataSharedPtr& effectData, const std::string& filename )
    {
      EffectLoaders::iterator it = m_effectLoaders.find( dp::util::getFileExtension( filename ) );
      if ( it != m_effectLoaders.end() )
      {
        bool success = it->second->save( effectData, filename );
        DP_ASSERT( success && "failed to save effectData" );

        return success;
      }
      else
      {
        return false;
      }
    }

  } // namespace fx
} // namespace dp
