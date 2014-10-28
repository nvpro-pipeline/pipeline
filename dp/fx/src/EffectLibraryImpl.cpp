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
      // determine default search paths
      m_searchPaths.push_back( dp::home() + "/media/dpfx" );
      dp::util::convertPath( m_searchPaths.back() );
      m_searchPaths.push_back( dp::home() + "/media/effects" );
      dp::util::convertPath( m_searchPaths.back() );
    }

    EffectLibraryImpl::~EffectLibraryImpl()
    {
    }

    void EffectLibraryImpl::registerEffectLoader( const boost::shared_ptr<EffectLoader>& effectLoader, const std::string& extension )
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

    bool EffectLibraryImpl::effectHasTechnique( SmartEffectSpec const& effectSpec, std::string const& techniqueName, bool rasterizer ) const
    {
      EffectSpecs::const_iterator it = m_effectSpecs.find( effectSpec->getName() );
      if ( it != m_effectSpecs.end() )
      {
        return( it->second.effectLoader->effectHasTechnique( effectSpec, techniqueName, rasterizer ) );
      }
      return( false );
    }

    bool EffectLibraryImpl::loadEffects( const std::string& filename, const std::vector<std::string> &searchPaths )
    {
      if ( m_loadedFiles.find( filename ) != m_loadedFiles.end() )
      {
        return( true );
      }

      // filter out paths, that already are part of the search paths
      // as we're going to findFileRecursive, also filter out paths that are more specific than what we already have
      // but don't just use the outer-most part we can find, as more specific paths listed before more general paths
      // should be searched through first, even though they'd be searched through a second time, then.
      std::vector<std::string> fullSearchPaths = m_searchPaths;
      for ( std::vector<std::string>::const_iterator it = searchPaths.begin() ; it != searchPaths.end() ; ++it )
      {
        std::vector<std::string>::const_iterator fit = fullSearchPaths.begin();
        for ( ; fit != fullSearchPaths.end() ; ++fit )
        {
          // check if the new search path starts with one of those we already have
          if ( it->compare( 0, fit->length(), *fit, 0, fit->length() ) == 0 )
          {
            break;
          }
        }
        if ( fit == fullSearchPaths.end() )
        {
          fullSearchPaths.push_back( *it );
        }
      }

      std::string file = dp::util::findFileRecursive( filename, fullSearchPaths );
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

      std::string extension = dp::util::getFileExtension( file );

      EffectLoaders::iterator it = m_effectLoaders.find( extension );
      if ( it != m_effectLoaders.end() )
      {
        m_currentFile.push( file );
        dp::util::convertPath( m_currentFile.top() );
        bool success = it->second->loadEffects( m_currentFile.top() );
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

    const SmartEffectSpec& EffectLibraryImpl::getEffectSpec(const std::string& effectName ) const
    {
      EffectSpecs::const_iterator it = m_effectSpecs.find( effectName );
      if ( it != m_effectSpecs.end() )
      {
        return it->second.effectSpec;
      }
      else
      {
        static SmartEffectSpec dummy;
        return dummy;
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

    const SmartParameterGroupSpec& EffectLibraryImpl::getParameterGroupSpec( const std::string & pgsName ) const
    {
      ParameterGroupSpecs::const_iterator it = m_parameterGroupSpecs.find( pgsName );
      if ( it != m_parameterGroupSpecs.end() )
      {
        return it->second;          
      }
      else
      {
        static SmartParameterGroupSpec dummy;
        return( dummy );
      }
    }

    SmartShaderPipeline EffectLibraryImpl::generateShaderPipeline( const ShaderPipelineConfiguration& configuration )
    {
      EffectSpecs::const_iterator itEffectSpec = m_effectSpecs.find( configuration.getName() );

      return itEffectSpec->second.effectLoader->generateShaderPipeline( configuration );
    }

    SmartEnumSpec EffectLibraryImpl::registerSpec( const SmartEnumSpec& enumSpec )
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

    SmartEffectSpec EffectLibraryImpl::registerSpec( const SmartEffectSpec& effectSpec, EffectLoader* effectLoader )
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
        SmartEffectDataPrivate effectData( new EffectDataPrivate( effectSpec, effectSpec->getName() ) );
        for ( EffectSpec::iterator it = effectSpec->beginParameterGroupSpecs(); it != effectSpec->endParameterGroupSpecs(); ++it )
        {
          effectData->setParameterGroupData( it, getParameterGroupData( (*it)->getName() ) );
        }
        registerEffectData( effectData );

        return effectSpec;
      }
    }

    SmartParameterGroupSpec EffectLibraryImpl::registerSpec( const SmartParameterGroupSpec& parameterGroupSpec )
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
        registerParameterGroupData( std::make_shared<ParameterGroupDataPrivate>( parameterGroupSpec, parameterGroupSpec->getName() ) );
        return parameterGroupSpec;
      }
    }

    SmartParameterGroupData EffectLibraryImpl::registerParameterGroupData( const SmartParameterGroupData& parameterGroupData )
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

    SmartEffectData EffectLibraryImpl::registerEffectData( const SmartEffectData& effectData )
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

    const SmartEnumSpec& EffectLibraryImpl::getEnumSpec( const std::string& name ) const
    {
      EnumSpecs::const_iterator it = m_enumSpecs.find( name );
      if ( it != m_enumSpecs.end() )
      {
        return it->second;
      }
      else
      {
        static SmartEnumSpec dummy;
        return( dummy );
      }
    }

    const SmartParameterGroupData& EffectLibraryImpl::getParameterGroupData( const std::string& name ) const
    {
      ParameterGroupDatas::const_iterator it = m_parameterGroupDatas.find( name );
      if ( it != m_parameterGroupDatas.end() )
      {
        return it->second;
      }
      else
      {
        static SmartParameterGroupData dummy;
        return( dummy );
      }
    }

    const SmartEffectData& EffectLibraryImpl::getEffectData( const std::string& name ) const
    {
      EffectDatas::const_iterator it = m_effectDatas.find( name );
      if ( it != m_effectDatas.end() )
      {
        return it->second;
      }
      else
      {
        static SmartEffectData dummy;
        return( dummy );
      }
    }

    bool EffectLibraryImpl::save( const SmartEffectData& effectData, const std::string& filename )
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
