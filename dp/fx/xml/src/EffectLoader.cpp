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


#include <dp/util/File.h>
#include <dp/util/SharedPtr.h>
#include <dp/util/Singleton.h>
#include <dp/util/Tokenizer.h>
#include <dp/fx/xml/EffectLoader.h>
#include <dp/fx/ParameterConversion.h>
#include <dp/fx/ParameterGroupLayout.h>
#include <dp/fx/inc/EffectDataPrivate.h>
#include <dp/fx/inc/EffectLibraryImpl.h>
#include <dp/fx/inc/FileSnippet.h>
#include <dp/fx/inc/ParameterGroupDataPrivate.h>
#include <dp/fx/inc/ParameterGroupSnippet.h>
#include <dp/fx/inc/SnippetListSnippet.h>
#include <dp/fx/inc/StringSnippet.h>
#include <dp/DP.h>
#include <boost/bind.hpp>

#include <tinyxml.h>
#include <iostream>
#include <map>
#include <utility>
#include <exception>
#include <boost/make_shared.hpp>

using namespace dp::util;
using std::auto_ptr;
using std::map;
using std::string;
using std::vector;

namespace dp
{
  namespace fx
  {
    namespace xml
    {

      /************************************************************************/
      /* ShaderPipelineImpl                                                   */
      /************************************************************************/
      DEFINE_PTR_TYPES( ShaderPipelineImpl );

      class ShaderPipelineImpl : public ShaderPipeline
      {
      public:
        static ShaderPipelineImplSharedPtr create()
        {
          return( std::shared_ptr<ShaderPipelineImpl>( new ShaderPipelineImpl() ) );
        }

        void addStage( const Stage& stage )
        {
          DP_ASSERT( std::find_if( m_stages.begin(), m_stages.end(), boost::bind( &Stage::domain, _1) == stage.domain ) == m_stages.end() );
          m_stages.push_back( stage );
        }

      protected:
        ShaderPipelineImpl()
        {}
      };

      /************************************************************************/
      /* Technique                                                            */
      /************************************************************************/
      TechniqueSharedPtr Technique::create( std::string const& type )
      {
        return( std::shared_ptr<Technique>( new Technique( type ) ) );
      }

      Technique::Technique( std::string const & type)
        : m_type( type )
      {
      }

      void Technique::addDomainSnippet( dp::fx::Domain domain, std::string const & signature, dp::fx::SnippetSharedPtr const & snippet )
      {
        if ( m_domainSignatures.find( domain ) != m_domainSignatures.end() 
          && m_domainSignatures[domain].find( signature ) != m_domainSignatures[domain].end() )
        {
          throw std::runtime_error( std::string( "signature " + signature + " has already been added to the domain" ) );
        }
        m_domainSignatures[domain][signature] = snippet;
      }

      Technique::SignatureSnippets const & Technique::getSignatures( dp::fx::Domain domain ) const
      {
        DomainSignatures::const_iterator it = m_domainSignatures.find( domain );
        if (it == m_domainSignatures.end() )
        {
          throw std::runtime_error( "There're no signatures for the given domain" );
        }
        return it->second;
      }

      std::string const & Technique::getType() const
      {
        return m_type;
      }

      /************************************************************************/
      /* DomainSpec                                                           */
      /************************************************************************/
      DomainSpecSharedPtr DomainSpec::create( std::string const & name, dp::fx::Domain domain, ParameterGroupSpecsContainer const & specs, bool transparent, Techniques const & techniques )
      {
        return( std::shared_ptr<DomainSpec>( new DomainSpec( name, domain, specs, transparent, techniques ) ) );
      }

      DomainSpec::DomainSpec( std::string const & name, dp::fx::Domain domain, ParameterGroupSpecsContainer const & specs, bool transparent, Techniques const & techniques )
        : m_name( name )
        , m_domain( domain )
        , m_parameterGroups( specs )
        , m_transparent( transparent )
        , m_techniques( techniques)
      {

      }

      TechniqueSharedPtr DomainSpec::getTechnique( std::string const & name )
      {
        Techniques::const_iterator it = m_techniques.find( name );
        if ( it != m_techniques.end() )
        {
          return it->second;
        }
        // The DomainSpec's technique doesn't match the queried one.
        // Return nullptr so that it's going to be ignored. 
        return TechniqueSharedPtr::null;
      }

      DomainSpec::ParameterGroupSpecsContainer const & DomainSpec::getParameterGroups() const
      {
        return m_parameterGroups;
      }

      dp::fx::Domain DomainSpec::getDomain() const
      {
        return m_domain;
      }

      bool DomainSpec::isTransparent() const
      {
        return m_transparent;
      }

      /************************************************************************/
      /* DomainData                                                           */
      /************************************************************************/
      DomainDataSharedPtr DomainData::create( DomainSpecSharedPtr const & domainSpec, std::string const & name, std::vector<dp::fx::ParameterGroupDataSharedPtr> const & parameterGroupDatas, bool transparent )
      {
        return( std::shared_ptr<DomainData>( new DomainData( domainSpec, name, parameterGroupDatas, transparent ) ) );
      }

      DomainData::DomainData( DomainSpecSharedPtr const & domainSpec, std::string const & name, std::vector<dp::fx::ParameterGroupDataSharedPtr> const & parameterGroupDatas, bool transparent )
        : m_domainSpec( domainSpec )
        , m_parameterGroupDatas( new ParameterGroupDataSharedPtr[domainSpec->getParameterGroups().size()] )
        , m_name( name )
        , m_transparent( transparent )
      {
        DomainSpec::ParameterGroupSpecsContainer const & parameterGroupSpecs = domainSpec->getParameterGroups();
        for ( std::vector<dp::fx::ParameterGroupDataSharedPtr>::const_iterator it = parameterGroupDatas.begin(); it != parameterGroupDatas.end(); ++it )
        {
          dp::fx::ParameterGroupSpecSharedPtr parameterGroupSpec = (*it)->getParameterGroupSpec();
          
          DomainSpec::ParameterGroupSpecsContainer::const_iterator itParameterGroupSpec = std::find( parameterGroupSpecs.begin(), parameterGroupSpecs.end(), parameterGroupSpec );
          if ( itParameterGroupSpec == parameterGroupSpecs.end() )
          {
            throw std::runtime_error( "ParameterGroupSpec does not exist for given ParameterGroupData" );
          }

          m_parameterGroupDatas[ std::distance( parameterGroupSpecs.begin(), itParameterGroupSpec ) ] = *it;
        }
      }

      DomainSpecSharedPtr const & DomainData::getDomainSpec() const
      {
        return m_domainSpec;
      }

      ParameterGroupDataSharedPtr const & DomainData::getParameterGroupData( DomainSpec::ParameterGroupSpecsContainer::const_iterator it ) const
      {
        return m_parameterGroupDatas[ std::distance( m_domainSpec->getParameterGroups().begin(), it ) ];
      }

      std::string const & DomainData::getName() const
      {
        return m_name;
      }

      bool DomainData::getTransparent() const
      {
        return m_transparent;
      }

      /************************************************************************/
      /* EffectSpec                                                           */
      /************************************************************************/
      EffectSpecSharedPtr EffectSpec::create( std::string const & name, DomainSpecs const & domainSpecs )
      {
        return( std::shared_ptr<EffectSpec>( new EffectSpec( name, domainSpecs ) ) );
      }

      EffectSpec::EffectSpec( std::string const & name, DomainSpecs const & domainSpecs )
        : dp::fx::EffectSpec( name, EST_PIPELINE, gatherParameterGroupSpecs( domainSpecs ), gatherTransparency( domainSpecs ) )
        , m_domainSpecs( domainSpecs )
      {
      }

      EffectSpec::DomainSpecs const & EffectSpec::getDomainSpecs() const
      {
        return m_domainSpecs;
      }

      DomainSpecSharedPtr const & EffectSpec::getDomainSpec( dp::fx::Domain domainSpec ) const
      {
        DomainSpecs::const_iterator it = m_domainSpecs.find( domainSpec );
        if ( m_domainSpecs.end() == it )
        {
          throw std::runtime_error( "missing DomainSpec for given domain" );
        }

        return it->second;
      }

      EffectSpec::ParameterGroupSpecsContainer EffectSpec::gatherParameterGroupSpecs( DomainSpecs const & domainSpecs )
      {
        // gather all ParameterGroupSpecs and ensure that each one exists only once
        std::set<dp::fx::ParameterGroupSpecSharedPtr> gatheredSpecs;
        for ( DomainSpecs::const_iterator it = domainSpecs.begin(); it != domainSpecs.end(); ++it )
        {
          for (DomainSpec::ParameterGroupSpecsContainer::const_iterator it2 = it->second->getParameterGroups().begin(); it2 != it->second->getParameterGroups().end(); ++it2 )
          {
            gatheredSpecs.insert(*it2);
          }
        }

        ParameterGroupSpecsContainer returnValue;
        std::copy( gatheredSpecs.begin(), gatheredSpecs.end(), std::back_inserter( returnValue ) );
        return returnValue;
      }

      bool EffectSpec::gatherTransparency( DomainSpecs const & domainSpecs )
      {
        bool transparency = false;
        for ( DomainSpecs::const_iterator it = domainSpecs.begin(); it != domainSpecs.end(); ++it )
        {
          transparency |= it->second->isTransparent();
        }

        return transparency;
      }

      // Public interface functions ########################################

      dp::fx::Domain getDomainFromString( std::string const & domain )
      {
        // OpenGL
        if ( domain == "vertex" )                 return dp::fx::DOMAIN_VERTEX;
        if ( domain == "fragment" )               return dp::fx::DOMAIN_FRAGMENT;
        if ( domain == "geometry" )               return dp::fx::DOMAIN_GEOMETRY;
        if ( domain == "tessellation_control" )   return dp::fx::DOMAIN_TESSELLATION_CONTROL;
        if ( domain == "tessellation_evaluation") return dp::fx::DOMAIN_TESSELLATION_EVALUATION;

        throw std::runtime_error("unknown domain type: " + domain );
      }

      EffectLoaderSharedPtr EffectLoader::create( EffectLibraryImpl * effectLibrary )
      {
        return( std::shared_ptr<EffectLoader>( new EffectLoader( effectLibrary ) ) );
      }

      EffectLoader::EffectLoader( EffectLibraryImpl * effectLibrary )
        : dp::fx::EffectLoader( effectLibrary )
      {
        // Map for getContainerParameterTypeFromGLSLType()
        // These are the built-in GLSL variables types:
        m_mapGLSLtoPT.insert(std::make_pair("float",  static_cast<unsigned int>(PT_FLOAT32)));
        m_mapGLSLtoPT.insert(std::make_pair("vec2",   PT_FLOAT32 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("vec3",   PT_FLOAT32 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("vec4",   PT_FLOAT32 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("int",    static_cast<unsigned int>(PT_INT32)));
        m_mapGLSLtoPT.insert(std::make_pair("ivec2",  PT_INT32 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("ivec3",  PT_INT32 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("ivec4",  PT_INT32 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("uint",   static_cast<unsigned int>(PT_UINT32)));
        m_mapGLSLtoPT.insert(std::make_pair("uvec2",  PT_UINT32 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("uvec3",  PT_UINT32 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("uvec4",  PT_UINT32 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("bool",   static_cast<unsigned int>(PT_BOOL)));
        m_mapGLSLtoPT.insert(std::make_pair("bvec2",  PT_BOOL | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("bvec3",  PT_BOOL | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("bvec4",  PT_BOOL | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("mat2",   PT_FLOAT32 | PT_MATRIX2x2));
        m_mapGLSLtoPT.insert(std::make_pair("mat2x2", PT_FLOAT32 | PT_MATRIX2x2));
        m_mapGLSLtoPT.insert(std::make_pair("mat2x3", PT_FLOAT32 | PT_MATRIX2x3));
        m_mapGLSLtoPT.insert(std::make_pair("mat2x4", PT_FLOAT32 | PT_MATRIX2x4));
        m_mapGLSLtoPT.insert(std::make_pair("mat3x2", PT_FLOAT32 | PT_MATRIX3x2));
        m_mapGLSLtoPT.insert(std::make_pair("mat3",   PT_FLOAT32 | PT_MATRIX3x3));
        m_mapGLSLtoPT.insert(std::make_pair("mat3x3", PT_FLOAT32 | PT_MATRIX3x3));
        m_mapGLSLtoPT.insert(std::make_pair("mat3x4", PT_FLOAT32 | PT_MATRIX3x4));
        m_mapGLSLtoPT.insert(std::make_pair("mat4x2", PT_FLOAT32 | PT_MATRIX4x2));
        m_mapGLSLtoPT.insert(std::make_pair("mat4x3", PT_FLOAT32 | PT_MATRIX4x3));
        m_mapGLSLtoPT.insert(std::make_pair("mat4",   PT_FLOAT32 | PT_MATRIX4x4));
        m_mapGLSLtoPT.insert(std::make_pair("mat4x4", PT_FLOAT32 | PT_MATRIX4x4));
        m_mapGLSLtoPT.insert(std::make_pair("double",  static_cast<unsigned int>(PT_FLOAT64)));
        m_mapGLSLtoPT.insert(std::make_pair("dvec2",   PT_FLOAT64 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("dvec3",   PT_FLOAT64 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("dvec4",   PT_FLOAT64 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("dmat2",   PT_FLOAT64 | PT_MATRIX2x2));
        m_mapGLSLtoPT.insert(std::make_pair("dmat2x2", PT_FLOAT64 | PT_MATRIX2x2));
        m_mapGLSLtoPT.insert(std::make_pair("dmat2x3", PT_FLOAT64 | PT_MATRIX2x3));
        m_mapGLSLtoPT.insert(std::make_pair("dmat2x4", PT_FLOAT64 | PT_MATRIX2x4));
        m_mapGLSLtoPT.insert(std::make_pair("dmat3x2", PT_FLOAT64 | PT_MATRIX3x2));
        m_mapGLSLtoPT.insert(std::make_pair("dmat3",   PT_FLOAT64 | PT_MATRIX3x3));
        m_mapGLSLtoPT.insert(std::make_pair("dmat3x3", PT_FLOAT64 | PT_MATRIX3x3));
        m_mapGLSLtoPT.insert(std::make_pair("dmat3x4", PT_FLOAT64 | PT_MATRIX3x4));
        m_mapGLSLtoPT.insert(std::make_pair("dmat4x2", PT_FLOAT64 | PT_MATRIX4x2));
        m_mapGLSLtoPT.insert(std::make_pair("dmat4x3", PT_FLOAT64 | PT_MATRIX4x3));
        m_mapGLSLtoPT.insert(std::make_pair("dmat4",   PT_FLOAT64 | PT_MATRIX4x4));
        m_mapGLSLtoPT.insert(std::make_pair("dmat4x4", PT_FLOAT64 | PT_MATRIX4x4));
        m_mapGLSLtoPT.insert(std::make_pair("sampler1D",              PT_SAMPLER_PTR | PT_SAMPLER_1D));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2D",              PT_SAMPLER_PTR | PT_SAMPLER_2D));
        m_mapGLSLtoPT.insert(std::make_pair("sampler3D",              PT_SAMPLER_PTR | PT_SAMPLER_3D));
        m_mapGLSLtoPT.insert(std::make_pair("samplerCube",            PT_SAMPLER_PTR | PT_SAMPLER_CUBE));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DRect",          PT_SAMPLER_PTR | PT_SAMPLER_2D_RECT));
        m_mapGLSLtoPT.insert(std::make_pair("sampler1DArray",         PT_SAMPLER_PTR | PT_SAMPLER_1D_ARRAY));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DArray",         PT_SAMPLER_PTR | PT_SAMPLER_2D_ARRAY));
        m_mapGLSLtoPT.insert(std::make_pair("samplerBuffer",          PT_SAMPLER_PTR | PT_SAMPLER_BUFFER));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DMS",            PT_SAMPLER_PTR | PT_SAMPLER_2D_MULTI_SAMPLE));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DMSArray",       PT_SAMPLER_PTR | PT_SAMPLER_2D_MULTI_SAMPLE_ARRAY));
        m_mapGLSLtoPT.insert(std::make_pair("samplerCubeArray",       PT_SAMPLER_PTR | PT_SAMPLER_CUBE_ARRAY)); // DAR Only exists in GLSL 4.00++
        m_mapGLSLtoPT.insert(std::make_pair("sampler1DShadow",        PT_SAMPLER_PTR | PT_SAMPLER_1D_SHADOW));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DShadow",        PT_SAMPLER_PTR | PT_SAMPLER_2D_SHADOW));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DRectShadow",    PT_SAMPLER_PTR | PT_SAMPLER_2D_RECT_SHADOW));
        m_mapGLSLtoPT.insert(std::make_pair("sampler1DArrayShadow",   PT_SAMPLER_PTR | PT_SAMPLER_1D_ARRAY_SHADOW));
        m_mapGLSLtoPT.insert(std::make_pair("sampler2DArrayShadow",   PT_SAMPLER_PTR | PT_SAMPLER_2D_ARRAY_SHADOW));
        m_mapGLSLtoPT.insert(std::make_pair("samplerCubeShadow",      PT_SAMPLER_PTR | PT_SAMPLER_CUBE_SHADOW));
        m_mapGLSLtoPT.insert(std::make_pair("samplerCubeArrayShadow", PT_SAMPLER_PTR | PT_SAMPLER_CUBE_ARRAY_SHADOW));
        // m_mapGLSLtoPT.insert(std::make_pair("RiXInternal?", PT_NATIVE));
            
        // These are invented variable types and not valid GLSL variables

        // These are used to support OptiX rtBuffer<format>, rtBuffer<format, 2>, rtBuffer<format, 3>.
        // Not really needed. The developer programming the render pipeline must define buffers.
        //m_mapGLSLtoPT.insert(std::make_pair("buffer1D", PT_BUFFER_PTR | PT_BUFFER_1D));
        //m_mapGLSLtoPT.insert(std::make_pair("buffer2D", PT_BUFFER_PTR | PT_BUFFER_2D));
        //m_mapGLSLtoPT.insert(std::make_pair("buffer3D", PT_BUFFER_PTR | PT_BUFFER_3D));
        // These are added to be able to handle smaller data types than int 
        // inside EffectSpecs parameter groups on scene graph side.
        // Code generation will cast them to the next bigger supported format in GLSL.
        m_mapGLSLtoPT.insert(std::make_pair("enum", static_cast<unsigned int>(PT_ENUM)));
        m_mapGLSLtoPT.insert(std::make_pair("char", static_cast<unsigned int>(PT_INT8)));
        m_mapGLSLtoPT.insert(std::make_pair("char2", PT_INT8 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("char3", PT_INT8 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("char4", PT_INT8 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("uchar",  static_cast<unsigned int>(PT_UINT8)));
        m_mapGLSLtoPT.insert(std::make_pair("uchar2", PT_UINT8 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("uchar3", PT_UINT8 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("uchar4", PT_UINT8 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("short", static_cast<unsigned int>(PT_INT16)));
        m_mapGLSLtoPT.insert(std::make_pair("short2", PT_INT16 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("short3", PT_INT16 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("short4", PT_INT16 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("ushort", static_cast<unsigned int>(PT_UINT16)));
        m_mapGLSLtoPT.insert(std::make_pair("ushort2", PT_UINT16 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("ushort3", PT_UINT16 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("ushort4", PT_UINT16 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("longlong",  static_cast<unsigned int>(PT_INT64)));
        m_mapGLSLtoPT.insert(std::make_pair("longlong2", PT_INT64 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("longlong3", PT_INT64 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("longlong4", PT_INT64 | PT_VECTOR4));
        m_mapGLSLtoPT.insert(std::make_pair("ulonglong",  static_cast<unsigned int>(PT_UINT64)));
        m_mapGLSLtoPT.insert(std::make_pair("ulonglong2", PT_UINT64 | PT_VECTOR2));
        m_mapGLSLtoPT.insert(std::make_pair("ulonglong3", PT_UINT64 | PT_VECTOR3));
        m_mapGLSLtoPT.insert(std::make_pair("ulonglong4", PT_UINT64 | PT_VECTOR4));

        // generate inverse map for writer
        for ( std::map<std::string, unsigned int>::iterator it = m_mapGLSLtoPT.begin(); it != m_mapGLSLtoPT.end(); ++it )
        {
          m_mapPTtoGLSL.insert( make_pair(it->second, it->first) );
        }

        // some files still reside in dpfx
        m_fileFinder.addSearchPath( dp::home() + "/media/dpfx" );
        m_fileFinder.addSearchPath( dp::home() + "/media/textures" );
      }

      EffectLoader::~EffectLoader()
      {
      }

      bool EffectLoader::loadEffects(const string &inputFilename )
      {
        // Make sure Windows partition letters are always the same case.
        std::string filename = inputFilename;
        if ( ( 1 < filename.length() ) && ( filename[1] == ':' ) )
        {
          filename[0] = ::toupper( filename[0] );
        }
        DP_ASSERT( dp::util::fileExists( filename ) );

        if ( m_loadedFiles.find( filename ) == m_loadedFiles.end() )
        {
          // This search path is going to be used to find the files referenced inside the XML effects description.
          std::string dir = dp::util::getFilePath( filename );
          bool addedSearchPath = m_fileFinder.addSearchPath( dir );

          std::unique_ptr<TiXmlDocument> doc(new TiXmlDocument(filename.c_str()));
          if ( !doc )
          {
            return false;
          }

          if ( !doc->LoadFile() )
          {
            std::cerr << "failed to load " << filename << "!" << std::endl;
            return false;
          }

          TiXmlHandle libraryHandle = doc->FirstChildElement( "library" );   // The required XML root node.
          TiXmlElement * root = libraryHandle.Element();
          if ( ! root )
          {
            return false;
          }

          m_loadedFiles.insert( filename );
          parseLibrary( root );

          if ( addedSearchPath )
          {
            m_fileFinder.removeSearchPath( dir );
          }
        }

        return true;
      }

      bool EffectLoader::getShaderSnippets( const dp::fx::ShaderPipelineConfiguration& configuration
                                          , dp::fx::Domain domain
                                          , std::string& entrypoint // TODO remove entrypoint
                                          , std::vector<dp::fx::SnippetSharedPtr>& snippets )
      {
        snippets.clear();

        dp::fx::xml::EffectSpecSharedPtr effectSpec = dp::util::shared_cast<EffectSpec>( dp::fx::EffectLibrary::instance()->getEffectSpec( configuration.getName() ) );

        // All other domains have only one set of code snippets per technique and ignore the signature.
        
        dp::fx::Domain signatureDomain = DOMAIN_FRAGMENT;
        std::string signature;

        switch ( domain )
        {
        case DOMAIN_VERTEX:
        case DOMAIN_GEOMETRY:
        case DOMAIN_TESSELLATION_CONTROL:
        case DOMAIN_TESSELLATION_EVALUATION:
          {
            // Geometry shaders need to be matched to the underlying domain. (See FIXME above.)
            DomainSpecSharedPtr const & signatureDomainSpec = effectSpec->getDomainSpec( signatureDomain );
            DP_ASSERT( signatureDomainSpec );
            dp::fx::xml::TechniqueSharedPtr const & signatureTechnique = signatureDomainSpec->getTechnique( configuration.getTechnique() );
            if ( !signatureTechnique )
            {
              return false;  // Ignore domains which don't have a matching technique.
            }
            Technique::SignatureSnippets const & signatureSnippets = signatureTechnique->getSignatures( signatureDomain );
            DP_ASSERT( signatureSnippets.size() == 1 );
            signature = signatureSnippets.begin()->first;
          }
          // Intentional fall through!
        case DOMAIN_FRAGMENT:
          {
            DomainSpecSharedPtr const & domainSpec = effectSpec->getDomainSpec( domain );
            dp::fx::xml::TechniqueSharedPtr const & technique = domainSpec->getTechnique( configuration.getTechnique() );
            if ( !technique )
            {
              return false;  // Ignore domains which don't have a matching technique.
            }
            Technique::SignatureSnippets const & signatureSnippets = technique->getSignatures( domain );
            // signature.empty() means we didn't ask for a geometry shader, but one of the system or fragment level domains. Latter have a single snippets block per technique.
            Technique::SignatureSnippets::const_iterator it = (signature.empty()) ? signatureSnippets.begin() : signatureSnippets.find( signature );
            DP_ASSERT( it != signatureSnippets.end() );
            snippets.push_back( it->second );
          }
          break;

        default:
          DP_ASSERT(!" EffectLoader::getShaderSnippets(): Unexpected domain." );
          return false;
        }

        return true;
      }


      // Private helper functions ########################################

      EffectLoader::EffectElementType EffectLoader::getTypeFromElement(TiXmlElement *element)
      {
        if (!element)
        {
          return EET_NONE;
        }

        string value = element->Value();

        // DAR TODO Use a map.
        if (value == "enum")               return EET_ENUM;
        if (value == "effect")             return EET_EFFECT;
        if (value == "parameterGroup")     return EET_PARAMETER_GROUP;
        if (value == "parameter")          return EET_PARAMETER;
        if (value == "parameterGroupData") return EET_PARAMETER_GROUP_DATA;
        if (value == "technique")          return EET_TECHNIQUE;
        if (value == "glsl")               return EET_GLSL;
        if (value == "source")             return EET_SOURCE;
        if (value == "include")            return EET_INCLUDE;
        if (value == "PipelineSpec")       return EET_PIPELINE_SPEC;
        if (value == "PipelineData")       return EET_PIPELINE_DATA;
      
        return EET_UNKNOWN;
      }
    
      unsigned int EffectLoader::getParameterTypeFromGLSLType( const string &glslType )
      {
        std::map<string, unsigned int>::const_iterator it = m_mapGLSLtoPT.find(glslType);
        if (it != m_mapGLSLtoPT.end())
        {
          return (*it).second;
        }
        DP_ASSERT( !"EffectLoader::getParameterTypeFromGLSLType(): Type not found." );
        return PT_UNDEFINED; // Return something which indicates an error and isn't going to work further down.
      }

      SnippetSharedPtr EffectLoader::getSourceSnippet( string const & filename )
      {
        std::string name = m_fileFinder.find( filename );
        if ( name.empty() )
        {
          throw std::runtime_error( std::string("EffectLoader::loadSource(): File " + filename + "not found." ) );
        }
        return( std::make_shared<FileSnippet>( name ) );
      }

      SnippetSharedPtr EffectLoader::getParameterSnippet( string const & inout, string const & type, TiXmlElement *element )
      {
        std::ostringstream oss;

        const char * location = element->Attribute( "location" );
        if ( location )
        {
          oss << "layout(location = " + string( location ) + ") ";
        }
        DP_ASSERT( element->Attribute( "name" ) );
        oss << inout + string( " " ) + type + string( " " ) + string( element->Attribute( "name" ) ) + string( ";\n" );

        return( std::make_shared<StringSnippet>( oss.str() ) );
      }

      void EffectLoader::parseLibrary( TiXmlElement * root )
      {
        TiXmlHandle xmlHandle = root->FirstChildElement();
        TiXmlElement *element = xmlHandle.Element();

        while ( element )
        {
          switch( getTypeFromElement( element ) )
          {
            case EET_ENUM :
              parseEnum( element );
              break;
            case EET_EFFECT :
              parseEffect( element );
              break;
            case EET_PARAMETER_GROUP :
              parseParameterGroup( element );
              break;
            case EET_PARAMETER_GROUP_DATA :
              parseParameterGroupData( element );
              break;
            case EET_INCLUDE:
              parseInclude( element );
              break;
            case EET_PIPELINE_SPEC:
              parsePipelineSpec( element );
              break;
            case EET_PIPELINE_DATA:
              parsePipelineData( element );
              break;
            default :
              DP_ASSERT( !"Unknown element type in Library" );
              break;
          }
          element = element->NextSiblingElement();
        }
      }

      void EffectLoader::parseEnum( TiXmlElement * element )
      {
        DP_ASSERT( element->Attribute( "type" ) );
        string type = element->Attribute( "type" );

        const char * values = element->Attribute( "values" );
        DP_ASSERT( values );

        vector<string> valueVector;
        dp::util::StrTokenizer tokenizer( " " );
        tokenizer.setInput( values );
        for ( int i=0 ; tokenizer.hasMoreTokens() ; i++ )
        {
          valueVector.push_back( tokenizer.getNextToken() );
        }

        EnumSpecSharedPtr spec = EnumSpec::create( type, valueVector );
        getEffectLibrary()->registerSpec( spec );

        // add this enum type to the GLSL->PT map
        m_mapGLSLtoPT[type] = PT_ENUM;
      }

      void EffectLoader::parseLightEffect( TiXmlElement * effect)
      {
        DP_ASSERT( effect->Attribute( "id" ) );
        string id = effect->Attribute( "id" );

        const char * transparent = effect->Attribute( "transparent" );
        bool isTransparent = ( transparent && ( strcmp( transparent, "true" ) == 0 ) );

        TiXmlHandle xmlHandle = effect->FirstChildElement();
        TiXmlElement *element = xmlHandle.Element();

        EffectSpec::ParameterGroupSpecsContainer pgsc;
        while ( element )
        {
          EffectElementType eet = getTypeFromElement( element );
          switch( eet )
          {
          case EET_PARAMETER_GROUP:
            {
              ParameterGroupSpecSharedPtr pgs = parseParameterGroup( element );
              DP_ASSERT( find( pgsc.begin(), pgsc.end(), pgs ) == pgsc.end() );
              pgsc.push_back( pgs );
            }
            break;
          default:
            DP_ASSERT( !"Unknown element type in Effect" );
            break;
          }
          element = element->NextSiblingElement();
        }

        dp::fx::EffectSpecSharedPtr es = dp::fx::EffectSpec::create( id, dp::fx::EffectSpec::EST_LIGHT, pgsc, isTransparent );
        dp::fx::EffectSpecSharedPtr registeredEffectSpec = getEffectLibrary()->registerSpec( es, this );
        if ( es == registeredEffectSpec )
        {
          EffectDataPrivateSharedPtr effectData = EffectDataPrivate::create( registeredEffectSpec, registeredEffectSpec->getName() );
          getEffectLibrary()->registerEffectData( effectData );
          for ( EffectSpec::iterator it = registeredEffectSpec->beginParameterGroupSpecs(); it != registeredEffectSpec->endParameterGroupSpecs(); ++it )
          {
            effectData->setParameterGroupData( it, getEffectLibrary()->getParameterGroupData( (*it)->getName() ) );
          }
        }
      }

      void EffectLoader::parseDomainSpec( TiXmlElement * effect )
      {
        DP_ASSERT( effect->Attribute( "id" ) );
        DP_ASSERT( effect->Attribute( "domain" ) );

        std::string id           = effect->Attribute( "id" );
        std::string domainString = effect->Attribute( "domain" );
        
        dp::fx::Domain domain = getDomainFromString( domainString );

        DomainSpecs::const_iterator it = m_domainSpecs.find( id );
        if ( it != m_domainSpecs.end() )
        {
          DomainSpecSharedPtr const & domainSpec = it->second;
          if ( domainSpec->getDomain() == domain )
          {
            throw std::runtime_error( "DomainSpec " + id + " with domain " + domainString + " has already been registered" );
          }
        }

        const char * transparent = effect->Attribute( "transparent" );
        bool isTransparent = ( transparent && ( strcmp( transparent, "true" ) == 0 ) );

        TiXmlHandle xmlHandle = effect->FirstChildElement();
        TiXmlElement *element = xmlHandle.Element();

        DomainSpec::Techniques                   techniques;
        DomainSpec::ParameterGroupSpecsContainer pgsc;

        while ( element )
        {
          EffectElementType eet = getTypeFromElement( element );
          switch( eet )
          {
          case EET_PARAMETER_GROUP:
            {
              ParameterGroupSpecSharedPtr pgs = parseParameterGroup( element );
              DP_ASSERT( find( pgsc.begin(), pgsc.end(), pgs ) == pgsc.end() );
              pgsc.push_back( pgs );
            }
            break;
          case EET_TECHNIQUE:
            {
              TechniqueSharedPtr technique = parseTechnique( element, domain );
              techniques[technique->getType()] = technique;
            }
            break;
          default:
            DP_ASSERT( !"Unknown element type in Effect" );
            break;
          }
          element = element->NextSiblingElement();
        }

        // register DomainSpec
        DomainSpecSharedPtr domainSpec = DomainSpec::create( id, domain, pgsc, isTransparent, techniques );
        m_domainSpecs[id] = domainSpec;
        
        // register DomainData
        std::vector<dp::fx::ParameterGroupDataSharedPtr> parameterGroupDatas;
        DomainSpec::ParameterGroupSpecsContainer const & parameterGroupSpecs = domainSpec->getParameterGroups();
        for ( DomainSpec::ParameterGroupSpecsContainer::const_iterator it = parameterGroupSpecs.begin(); it != parameterGroupSpecs.end(); ++it )
        {
          parameterGroupDatas.push_back( getEffectLibrary()->getParameterGroupData( (*it)->getName() ) );
        }
        m_domainDatas[id] = DomainData::create( domainSpec, id, parameterGroupDatas, false );
      }

      void EffectLoader::parseEffect( TiXmlElement * effect )
      {
        if ( !effect->Attribute( "id" ) )
        {
          throw std::runtime_error( "DomainSpec is missing attribute 'id'" );
        }
        if ( !effect->Attribute( "domain" ) )
        {
          throw std::runtime_error( "DomainSpec is missing attribute 'domain'" );
        }

        std::string domain = effect->Attribute( "domain" );
        if ( domain == "light" ) // Invented domain "light shader" to special case this.
        {
          parseLightEffect( effect );
        }
        else
        {
          parseDomainSpec( effect );
        }
      }

      TechniqueSharedPtr EffectLoader::parseTechnique( TiXmlElement *technique, dp::fx::Domain domain )
      {
        char const * type = technique->Attribute( "type" );
        if ( type )
        {
          TechniqueSharedPtr newTechnique = Technique::create( type );

          TiXmlHandle xmlHandle = technique->FirstChildElement();
          TiXmlElement *element = xmlHandle.Element();

          while ( element )
          {
            EffectElementType elementType = getTypeFromElement( element );
            switch ( elementType )
            {
            case EET_GLSL:
              {
                char const * signature = element->Attribute( "signature" );
                if ( signature )
                {
                  SnippetSharedPtr snippet = parseSources(element);
                  newTechnique->addDomainSnippet( domain, signature, snippet );
                }
                else
                {
                  throw std::runtime_error("glsl tag is missing signature attribute");
                }
              }
              break;
         
            default:
              throw std::runtime_error( std::string("Expected glsl or cuda tag. Found invalid tag ") + element->Value() );
              break;
            }
            element = element->NextSiblingElement();
          }
          return newTechnique;
        }
        else
        {
          throw std::runtime_error("Technique tag is missing type attribute");
        }
        return TechniqueSharedPtr();
      }

      ParameterGroupSpecSharedPtr EffectLoader::parseParameterGroup( TiXmlElement * pg )
      {
        if ( pg->Attribute( "ref" ) )
        {
          string ref = pg->Attribute( "ref" );
          const ParameterGroupSpecSharedPtr& spec = getEffectLibrary()->getParameterGroupSpec( ref );
          DP_ASSERT( spec && "invalid ParameterGroupSpec reference");
          return spec;
        }
        else
        {
          DP_ASSERT( pg->Attribute( "id" ) );
          string id = pg->Attribute( "id" );

          TiXmlHandle xmlHandle = pg->FirstChildElement();
          TiXmlElement *element = xmlHandle.Element();

          vector<ParameterSpec> psc;
          while ( element )
          {
            switch( getTypeFromElement( element ) )
            {
              case EET_PARAMETER :
                parseParameter( element, psc );
                break;
              default :
                DP_ASSERT( !"Unknown element type in ParameterGroup" );
                break;
            }
            element = element->NextSiblingElement();
          }

          ParameterGroupSpecSharedPtr spec = ParameterGroupSpec::create( id, psc );
          ParameterGroupSpecSharedPtr registeredSpec = getEffectLibrary()->registerSpec( spec );
          if( registeredSpec == spec ) // a new spec had been added
          {
            ParameterGroupDataSharedPtr data = ParameterGroupDataPrivate::create( registeredSpec, registeredSpec->getName() );
            getEffectLibrary()->registerParameterGroupData( data );
          }
          return registeredSpec;
        }
      }

      void EffectLoader::parseParameter( TiXmlElement * param, vector<ParameterSpec> & psc )
      {
        DP_ASSERT( param->Attribute( "type" ) );
        string type = param->Attribute( "type" );

        DP_ASSERT( param->Attribute( "name" ) );
        string name = param->Attribute( "name" );

        DP_ASSERT( param->Attribute( "semantic" ) );
        Semantic semantic = stringToSemantic( param->Attribute( "semantic" ) );

        std::string value = param->Attribute( "value" ) ? param->Attribute( "value" ) : "";
        std::string annotation = param->Attribute( "annotation" ) ? param->Attribute( "annotation" ) : "";

        int arraySize = 0;
        // Ok, this sets the integer to zero if the attribute is not found.
        param->Attribute("size", &arraySize);

        unsigned int typeId = getParameterTypeFromGLSLType( type );
        if ( typeId == PT_ENUM )
        {
          DP_ASSERT( semantic == SEMANTIC_VALUE );
          psc.push_back( ParameterSpec( name, getEffectLibrary()->getEnumSpec( type ), arraySize, value, annotation ) );
        }
        else if ( ( ( typeId & PT_POINTER_TYPE_MASK ) == PT_SAMPLER_PTR ) && !value.empty() )
        {
          DP_ASSERT( arraySize == 0 );
          std::string fileName = m_fileFinder.findRecursive( value );
          psc.push_back( ParameterSpec( name, typeId, semantic, arraySize, fileName.empty() ? value : fileName, annotation ) );
        }
        else
        {
          psc.push_back( ParameterSpec( name, typeId, semantic, arraySize, value, annotation ) );
        }
      }

      void EffectLoader::parseInclude( TiXmlElement* include )
      {
        const char* file = include->Attribute("file");
        if ( !file )
        {
          std::cerr << "include not found: " << include << std::endl;
          DP_ASSERT( include->Attribute("include") );
        }
        else
        {
          getEffectLibrary()->loadEffects( file, m_fileFinder );
        }
      }

      SnippetSharedPtr EffectLoader::parseSources( TiXmlElement * effect )
      {
        TiXmlHandle xmlHandle = effect->FirstChildElement();
        TiXmlElement *element = xmlHandle.Element();

        std::vector<SnippetSharedPtr> snippets;
        while (element)
        {
          switch ( getTypeFromElement( element ) )
          {
            case EET_SOURCE:
              {
                if ( element->Attribute( "file" ) )
                {
                  char const *filename = element->Attribute( "file" );
                  std::string name = m_fileFinder.find( filename );
                  if ( !name.empty() )
                  {
                    snippets.push_back( std::make_shared<FileSnippet>( name ) );
                  }
                  else
                  {
                    throw std::runtime_error( std::string("source not found: ") + filename );
                  }
                }
                else if ( element->Attribute( "input" ) )
                {
                  snippets.push_back( getParameterSnippet( "in", element->Attribute( "input" ), element ) );
                }
                else if ( element->Attribute( "output" ) )
                {
                  snippets.push_back( getParameterSnippet( "out", element->Attribute( "output" ), element ) );
                }
                else if ( element->Attribute( "string" ) )
                {
                  snippets.push_back( std::make_shared<StringSnippet>( element->Attribute( "string" ) ) );
                }
                else
                {
                  DP_ASSERT( !"EffectLibrary::getShaderSource: unknown Attribute in EffectElementType EET_SOURCE" );
                }
              }
              break;
            default:
              throw std::runtime_error( std::string("invalid source tag: ") + element->Value() );
              break;
          }

          element = element->NextSiblingElement();
        }
        return( std::make_shared<SnippetListSnippet>( snippets ) );
      }

      ParameterGroupDataSharedPtr EffectLoader::parseParameterGroupData( TiXmlElement * pg )
      {
        ParameterGroupDataPrivateSharedPtr parameterGroupData;

        // Newly defined paramererGroupData.
        if ( pg->Attribute( "id" ) && pg->Attribute( "spec" ) )
        {
          std::string id   = pg->Attribute( "id" );
          std::string spec = pg->Attribute( "spec" );

          const ParameterGroupSpecSharedPtr& parameterGroupSpec = getEffectLibrary()->getParameterGroupSpec( spec );
          if ( !parameterGroupSpec )
          {
            throw std::runtime_error( "Couldn't find parameterGroupSpec " + spec + " for parameterGroupData " + id );
          }
          if ( m_parameterGroupDataLookup.find( id ) != m_parameterGroupDataLookup.end() )
          {
            throw std::runtime_error( "parameterGroupData for " + id + " already exists" );
          }

          parameterGroupData = ParameterGroupDataPrivate::create( parameterGroupSpec, id );

          TiXmlHandle xmlHandle = pg->FirstChildElement();
          TiXmlElement *element = xmlHandle.Element();

          while ( element )
          {
            switch( getTypeFromElement( element ) )
            {
            case EET_PARAMETER :
              parseParameterData( element, parameterGroupData );
              break;
            default :
              std::cerr << "Unknown element type " << element->Value() << "in ParameterGroupData" << std::endl;
              DP_ASSERT( !"Unknown element type in ParameterGroup" );
              break;
            }
            element = element->NextSiblingElement();
          }
          m_parameterGroupDataLookup[id] = dp::util::shared_cast<ParameterGroupData>( parameterGroupData );
        }
        else if ( pg->Attribute( "ref" ) )  // Reference existing parameterGroupData defined before.
        {
          std::string ref = pg->Attribute( "ref" );
          // Verify that the ParameterGroupData has been seen before.
          ParameterGroupDataLookup::const_iterator itpgd = m_parameterGroupDataLookup.find( ref );
          if ( itpgd == m_parameterGroupDataLookup.end() )
          {
            throw std::runtime_error( "parameterGroupData for " + ref + " not found in global scope." );
          }
          parameterGroupData = dp::util::shared_cast<ParameterGroupDataPrivate>( itpgd->second );
        }
        else
        {
          DP_ASSERT( !"Missing 'id', 'spec', or 'ref' for parameterGroupData." );
        }
        return parameterGroupData;
      }

      void EffectLoader::parseParameterData( TiXmlElement * param, const dp::fx::ParameterGroupDataPrivateSharedPtr& pgd )
      {
        DP_ASSERT( param->Attribute( "name" ) );
        DP_ASSERT( param->Attribute( "value" ) );

        string name  = param->Attribute( "name" );
        string value = param->Attribute( "value" );

        ParameterGroupSpec::iterator it = pgd->getParameterGroupSpec()->findParameterSpec( name );
        if ( it == pgd->getParameterGroupSpec()->endParameterSpecs() )
        {
          std::cerr << "Parameter " << name << " unknown in ParameterGroupSpec " << pgd->getParameterGroupSpec()->getName() << std::endl;
          DP_ASSERT( !"unknown parameter type" );
        }
        else
        {
          if ( isParameterPointer(it->first) )
          {
            pgd->setParameter( it, value.c_str() );
          }
          else if ( isParameterEnum(it->first) )
          {
            getValueFromString( it->first.getEnumSpec(), it->first.getArraySize(), value, reinterpret_cast<dp::fx::EnumSpec::StorageType*>(pgd->getValuePointer( it ) ) );
          }
          else
          {
            getValueFromString( it->first.getType(), it->first.getArraySize(), value, pgd->getValuePointer( it ) );
          }
        }
      }

      // pipeline
      void EffectLoader::parsePipelineSpec( TiXmlElement * effect )
      {
        // DAR FIXME Add all domain names.
        static const char *domains[] =
        {
          // OpenGL
            "vertex"
          , "tessellation_control"
          , "tessellation_evaluation"
          , "geometry"
          , "fragment"
        };

        static const size_t numShaderDomains = sizeof(domains) / sizeof(domains[0]);

        if ( effect->Attribute( "id" ) )
        {
          string id( effect->Attribute( "id" ) );

          EffectSpec::DomainSpecs domainSpecs;

          for (size_t i = 0; i < numShaderDomains; ++i)
          {
            const char *pName = effect->Attribute( domains[i] );
            if ( pName )
            {
              string name( pName );

              DomainSpecs::iterator itDomainSpec = m_domainSpecs.find( name );
              if ( itDomainSpec == m_domainSpecs.end() )
              {
                throw std::runtime_error( std::string("Pipeline '" + id + "' references undefined DomainSpec '" + name + "'") );
              }
              DomainSpecSharedPtr const & domainSpec = itDomainSpec->second;
              if ( !domainSpecs.insert( std::make_pair( domainSpec->getDomain(), domainSpec ) ).second ) // DAR FIXME Do we need the domain definition inside this domain spec? It's defined by the PipelineSpec now.
              {
                throw std::runtime_error( std::string("There's already another EffectSpec for the domain specified by '") + name + "' in the pipeline '" + id + "'" );
              }
            }
          }

          dp::fx::EffectSpecSharedPtr es = dp::util::shared_cast<dp::fx::EffectSpec>( EffectSpec::create( id, domainSpecs ) );
          dp::fx::EffectSpecSharedPtr registeredEffectSpec = getEffectLibrary()->registerSpec( es, this );
          if ( es == registeredEffectSpec )
          {
            EffectDataPrivateSharedPtr effectData = EffectDataPrivate::create( registeredEffectSpec, registeredEffectSpec->getName() );
            getEffectLibrary()->registerEffectData( effectData );
            for ( EffectSpec::iterator it = registeredEffectSpec->beginParameterGroupSpecs(); it != registeredEffectSpec->endParameterGroupSpecs(); ++it )
            {
              effectData->setParameterGroupData( it, getEffectLibrary()->getParameterGroupData( (*it)->getName() ) );
            }
          }
        }
        else
        {
          throw std::runtime_error( "Pipeline tag is missing attribute 'id'" );
        }
      }

      void EffectLoader::parsePipelineData( TiXmlElement * effect )
      {
        if ( effect->Attribute( "id") && effect->Attribute( "spec" ) )
        {
          string id   = effect->Attribute( "id" );
          string spec = effect->Attribute( "spec" );

          dp::fx::EffectSpecSharedPtr effectSpec = dp::fx::EffectLibrary::instance()->getEffectSpec( spec );
          EffectDataPrivateSharedPtr newEffectData = dp::fx::EffectDataPrivate::create( effectSpec, id );

          // Default initialize the newEffectData! Only the ParameterGroupData specified inside the XML will be overwritten below.
          for ( EffectSpec::iterator it = effectSpec->beginParameterGroupSpecs(); it != effectSpec->endParameterGroupSpecs(); ++it )
          {
            newEffectData->setParameterGroupData( it, dp::fx::EffectLibrary::instance()->getParameterGroupData( (*it)->getName() ) );
          }

          TiXmlHandle xmlHandle = effect->FirstChildElement();
          TiXmlElement *element = xmlHandle.Element();

          while ( element )
          {
            EffectElementType eet = getTypeFromElement( element );
            switch( eet )
            {
            case EET_PARAMETER_GROUP_DATA:
              {
                const ParameterGroupDataSharedPtr& parameterGroupData = parseParameterGroupData( element ); // This handles newly defined and referenced programParameterGroupData
                DP_ASSERT( parameterGroupData ); 
  
                // Find the ParameterGroupSpec name the ParameterGroupData should be written to.
                const ParameterGroupSpecSharedPtr& pgs = parameterGroupData->getParameterGroupSpec();
                dp::fx::EffectSpec::iterator ites = effectSpec->findParameterGroupSpec( pgs->getName() );
                if ( ites == effectSpec->endParameterGroupSpecs() )
                {
                  throw std::runtime_error( std::string( "Couldn't find parameterGroupSpec ") + spec );
                }
                // Overwrite the default ParameterGroupData of the effect with the instanced values defined in the PipelineData.
                newEffectData->setParameterGroupData( ites, parameterGroupData );
                break;
              }
            default :
              DP_ASSERT( !"Unknown element type in PipelineData" );
              break;
            }
            element = element->NextSiblingElement();
          }
          // Make the EffectData known to the effect library.
          getEffectLibrary()->registerEffectData( newEffectData );
        }
        else
        {
          throw std::runtime_error( "PipelineData tag is missing attribute 'id' or 'spec'" );
        }
      }

      ShaderPipelineSharedPtr EffectLoader::generateShaderPipeline( const ShaderPipelineConfiguration& configuration )
      {
        ShaderPipelineImplSharedPtr shaderPipeline = ShaderPipelineImpl::create();

        EffectSpecSharedPtr effectSpec = dp::util::shared_cast<EffectSpec>( dp::fx::EffectLibrary::instance()->getEffectSpec( configuration.getName() ) );
        EffectSpec::DomainSpecs const & domainSpecs = effectSpec->getDomainSpecs();

        for ( EffectSpec::DomainSpecs::const_iterator it = domainSpecs.begin(); it != domainSpecs.end(); ++it ) 
        {
          DP_ASSERT( it->first != DOMAIN_PIPELINE );

          ShaderPipeline::Stage stage;
          stage.domain = it->first;
          stage.parameterGroupSpecs = it->second->getParameterGroups();

          // generate snippets
          std::vector<dp::fx::SnippetSharedPtr> snippets;
          if ( getShaderSnippets( configuration, stage.domain, stage.entrypoint, snippets ) )
          {
            stage.source = std::make_shared<SnippetListSnippet>( snippets );

            if ( stage.domain == DOMAIN_VERTEX
              || stage.domain == DOMAIN_GEOMETRY
              || stage.domain == DOMAIN_TESSELLATION_CONTROL
              || stage.domain == DOMAIN_TESSELLATION_EVALUATION)
            {
              stage.systemSpecs.push_back( "sys_matrices" );
              stage.systemSpecs.push_back( "sys_camera" );
            }
            else if ( stage.domain == DOMAIN_FRAGMENT )
            {
              stage.systemSpecs.push_back( "sys_Fragment" );
            }

            shaderPipeline->addStage( stage );
          }
        }

        return shaderPipeline;
      }

      bool EffectLoader::effectHasTechnique( dp::fx::EffectSpecSharedPtr const& effectSpec, std::string const& techniqueName, bool /*rasterizer*/ )
      {
        bool hasTechnique = true;
        EffectSpecSharedPtr xmlEffectSpec = dp::util::shared_cast<EffectSpec>( effectSpec );
        for ( EffectSpec::DomainSpecs::const_iterator it = xmlEffectSpec->getDomainSpecs().begin() ; it != xmlEffectSpec->getDomainSpecs().end() && hasTechnique ; ++it )
        {
            switch( it->second->getDomain() )
            {
              case DOMAIN_VERTEX :
              case DOMAIN_TESSELLATION_CONTROL :
              case DOMAIN_TESSELLATION_EVALUATION :
              case DOMAIN_GEOMETRY :
              case DOMAIN_FRAGMENT :
                hasTechnique = !!it->second->getTechnique( techniqueName );
                break;
              default :
                break;
            }
          }
        return( hasTechnique );
      }


      /************************************************************************/
      /*                                                                      */
      /************************************************************************/

      bool EffectLoader::save( const EffectDataSharedPtr& effectData, const std::string& filename )
      {
        // DAR FIXME!!!!
        DP_ASSERT( !"Re-implement the new format!" );

        /**************************************************************************
        <?xml version="1.0"?>
        <library>
          <effectData id="..." spec="..." >
            <parameterGroupData id="..." spec="..." >
              <parameter name="faceWindingCCW" value="true" />
            </parameterGroupData>
            <parameterGroupData id="..." spec="standardDirectedLightParameters">
              <parameter name="ambient" value="0.0 0.0 0.0" />
              <parameter name="diffuse" value="1.0 1.0 1.0" />
              <parameter name="specular" value="1.0 1.0 1.0" />
              <parameter name="direction" value="0.0 0.0 -1.0" />
            </parameterGroup>
          </effectdata>
        </library>
        **************************************************************************/
        TiXmlDocument xmlDocument;
        TiXmlDeclaration* declaration = new TiXmlDeclaration( "1.0", "", "" );
        xmlDocument.LinkEndChild( declaration );

        TiXmlElement* elementLibrary = new TiXmlElement("library");

        TiXmlElement* elementEffectData = new TiXmlElement("effectData");

        elementEffectData->SetAttribute( "id", effectData->getName().c_str() );
        elementEffectData->SetAttribute( "spec", effectData->getEffectSpec()->getName().c_str() );

        dp::fx::EffectSpecSharedPtr const & effectSpec = effectData->getEffectSpec();
        for ( EffectSpec::iterator itPgs = effectSpec->beginParameterGroupSpecs(); itPgs != effectSpec->endParameterGroupSpecs(); ++itPgs )
        {
          const dp::fx::ParameterGroupDataSharedPtr& parameterGroupData = effectData->getParameterGroupData( itPgs );
          TiXmlElement* elementParameterGroupData = new TiXmlElement("parameterGroupData");
          elementParameterGroupData->SetAttribute( "id", parameterGroupData->getName().c_str() );
          elementParameterGroupData->SetAttribute( "spec", (*itPgs)->getName().c_str() );
          for ( ParameterGroupSpec::iterator itPs = (*itPgs)->beginParameterSpecs(); itPs != (*itPgs)->endParameterSpecs(); ++itPs )
          {
            TiXmlElement* elementParameter = new TiXmlElement("parameter");
            elementParameter->SetAttribute( "name", itPs->first.getName().c_str() );

            unsigned int type = itPs->first.getType();
            if ( isParameterEnum( itPs->first ) )
            {
              elementParameter->SetAttribute( "value", 
                                              dp::fx::getStringFromValue( itPs->first.getEnumSpec(), itPs->first.getArraySize(),
                                                                          reinterpret_cast<const dp::fx::EnumSpec::StorageType*>(parameterGroupData->getParameter( itPs ) ) ).c_str() );
            }
            else if ( isParameterPointer( itPs->first ) )
            {
              const char* value = reinterpret_cast<const char *>( parameterGroupData->getParameter( itPs ) );
              if ( value )
              {
                elementParameter->SetAttribute( "value", value);
              }
              else
              {
                elementParameter->SetAttribute( "value", "");
              }
            }
            else
            {
              elementParameter->SetAttribute( "value", dp::fx::getStringFromValue( type, itPs->first.getArraySize(), parameterGroupData->getParameter( itPs ) ).c_str() );
            }
            elementParameterGroupData->LinkEndChild( elementParameter );
          }
          elementEffectData->LinkEndChild( elementParameterGroupData );
        }

        elementLibrary->LinkEndChild( elementEffectData );
        xmlDocument.LinkEndChild( elementLibrary );

        xmlDocument.SaveFile( filename.c_str() );
        return true;
      }

    } // xml
  } // fx
} // dp

namespace
{
  bool registerEffectLibrary()
  {
    dp::fx::EffectLibraryImpl *eli = dynamic_cast<dp::fx::EffectLibraryImpl*>( dp::fx::EffectLibrary::instance() );
    DP_ASSERT( eli );
    if ( eli )
    {
      eli->registerEffectLoader( dp::fx::xml::EffectLoader::create( eli ), ".xml" );
      return true;
    }
    return false;
  }
} // namespace anonymous

extern "C"
{
  DP_FX_API bool dp_fx_xml_initialized = registerEffectLibrary();
}
