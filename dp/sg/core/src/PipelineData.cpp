// Copyright (c) 2011-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/fx/EffectLibrary.h>

using namespace dp::math;
using namespace dp::fx;

namespace
{
  class ParameterGroupDataLocal : public dp::fx::ParameterGroupData
  {
    public:
      ParameterGroupDataLocal( dp::sg::core::ParameterGroupDataSharedPtr const& parameterGroupData )
        : dp::fx::ParameterGroupData( parameterGroupData->getParameterGroupSpec(), parameterGroupData->getName() )
      {
        for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if ( isParameterPointer( it->first ) )
          {
            DP_ASSERT( ( it->first.getType() & PT_POINTER_TYPE_MASK ) == PT_SAMPLER_PTR );

            const dp::sg::core::SamplerSharedPtr & sampler = parameterGroupData->getParameter<dp::sg::core::SamplerSharedPtr>( it );
            DP_ASSERT( sampler );

            const dp::sg::core::TextureSharedPtr & texture = sampler->getTexture();
            DP_ASSERT( texture );
            DP_ASSERT( texture.isPtrTo<dp::sg::core::TextureFile>() );
            dp::sg::core::TextureFileSharedPtr const& textureFile = texture.staticCast<dp::sg::core::TextureFile>();
            std::string filename = textureFile->getFilename();

            char*& destination = *reinterpret_cast<char**>(&m_data[it->second]);
            delete[] destination;

            if ( !filename.empty() )
            {
              destination = new char[filename.length() + 1];
              strcpy( destination, filename.c_str() );
            }
            else
            {
              destination = nullptr;
            }
          }
          else
          {
            const char * value = reinterpret_cast<const char *>(parameterGroupData->getParameter( it ));
            unsigned int size = it->first.getElementSizeInBytes();
            if ( value )
            {
              memcpy( &m_data[it->second], value, size );
            }
            else
            {
              memset( &m_data[it->second], 0, size );
            }
          }
        }
      }
  };

  DEFINE_PTR_TYPES( EffectDataLocal );

  class EffectDataLocal : public dp::fx::EffectData
  {
    public:
      static EffectDataLocalSharedPtr create( dp::sg::core::PipelineData const* effectData )
      {
        return( std::shared_ptr<EffectDataLocal>( new EffectDataLocal( effectData ) ) );
      }

    protected:
      EffectDataLocal( const dp::sg::core::PipelineData * effectData )
        : dp::fx::EffectData( effectData->getEffectSpec(), effectData->getName() )
      {
        int i = 0;
        for ( dp::fx::EffectSpec::iterator it = m_effectSpec->beginParameterGroupSpecs() ; it != m_effectSpec->endParameterGroupSpecs() ; ++it, i++ )
        {
          m_parameterGroupDatas[i] = std::make_shared<ParameterGroupDataLocal>( effectData->getParameterGroupData( it ) );
        }
        setTransparent( effectData->getTransparent() );
      }
  };

}

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( PipelineData, Transparent);
      BEGIN_REFLECTION_INFO ( PipelineData )
        DERIVE_STATIC_PROPERTIES( PipelineData, Object );
        INIT_STATIC_PROPERTY_RW( PipelineData, Transparent, bool, SEMANTIC_VALUE, value, value );
      END_REFLECTION_INFO

      PipelineDataSharedPtr PipelineData::create( const EffectSpecSharedPtr & effectSpec )
      {
        return( std::shared_ptr<PipelineData>( new PipelineData( effectSpec ) ) );
      }

      PipelineDataSharedPtr PipelineData::create( const dp::fx::EffectDataSharedPtr& effectData )
      {
        return( std::shared_ptr<PipelineData>( effectData ? new PipelineData( effectData ) : nullptr ) );
      }

      HandledObjectSharedPtr PipelineData::clone() const
      {
        return( std::shared_ptr<PipelineData>( new PipelineData( *this ) ) );
      }

      PipelineData::PipelineData( const EffectSpecSharedPtr& effectSpec )
        : m_effectSpec( effectSpec )
        , m_transparent( effectSpec->getTransparent() )
      {
        DP_ASSERT( m_effectSpec );

        m_objectCode = OC_PIPELINE_DATA;
        m_parameterGroupData.reset( new ParameterGroupDataSharedPtr[effectSpec->getNumberOfParameterGroupSpecs()] );
      }

      PipelineData::PipelineData( const dp::fx::EffectDataSharedPtr& effectData )
        : m_effectSpec( effectData->getEffectSpec() )
        , m_transparent( effectData->getTransparent() )
      {
        DP_ASSERT( m_effectSpec );
        m_objectCode = OC_PIPELINE_DATA;

        setName( effectData->getName() );
        m_parameterGroupData.reset( new ParameterGroupDataSharedPtr[m_effectSpec->getNumberOfParameterGroupSpecs()] );

        for ( EffectSpec::iterator it = m_effectSpec->beginParameterGroupSpecs(); it != m_effectSpec->endParameterGroupSpecs(); ++it )
        {
          size_t index = std::distance( m_effectSpec->beginParameterGroupSpecs(), it );
          m_parameterGroupData[ index ] = ParameterGroupData::create( effectData->getParameterGroupData(it) );
          m_parameterGroupData[index]->attach( this );
        }
      }

      PipelineData::PipelineData( const PipelineData &rhs )
        : Object( rhs )
        , m_effectSpec( rhs.m_effectSpec )
        , m_parameterGroupData( new ParameterGroupDataSharedPtr[rhs.m_effectSpec->getNumberOfParameterGroupSpecs()] )
        , m_transparent( rhs.m_transparent )
      {
        m_objectCode = OC_PIPELINE_DATA;
        unsigned int nopgs = rhs.m_effectSpec->getNumberOfParameterGroupSpecs();
        for ( unsigned int i=0 ; i<nopgs ; i++ )
        {
          if ( rhs.m_parameterGroupData[i] )
          {
            m_parameterGroupData[i] = rhs.m_parameterGroupData[i].clone();
            m_parameterGroupData[i]->attach( this );
          }
          else
          {
            m_parameterGroupData[i].reset();
          }
        }
      }

      PipelineData::~PipelineData()
      {
        clearParameterGroupData();
      }

      void PipelineData::setParameterGroupData( EffectSpec::iterator const& it, const ParameterGroupDataSharedPtr & pgd )
      {
        DP_ASSERT( it != m_effectSpec->endParameterGroupSpecs() );
        DP_ASSERT( !pgd || ( *it == pgd->getParameterGroupSpec() ) );
        size_t idx = std::distance( m_effectSpec->beginParameterGroupSpecs(), it );
        if ( m_parameterGroupData[idx] != pgd )
        {
          if ( m_parameterGroupData[idx] )
          {
            m_parameterGroupData[idx]->detach( this );
          }
          m_parameterGroupData[idx] = pgd;
          if ( pgd )
          {
            pgd->attach( this );
          }
        }
      }

      bool PipelineData::setParameterGroupData( const ParameterGroupDataSharedPtr & pgd )
      {
        const ParameterGroupSpecSharedPtr & pgs = pgd->getParameterGroupSpec();
        EffectSpec::iterator it = m_effectSpec->findParameterGroupSpec( pgs );
        if ( it != m_effectSpec->endParameterGroupSpecs() )
        {
          setParameterGroupData( it, pgd );
          return( true );
        }
        return( false );
      }

      unsigned int PipelineData::getNumberOfParameterGroupData() const
      {
        unsigned int count = 0;
        unsigned int n = m_effectSpec->getNumberOfParameterGroupSpecs();
        for ( unsigned int i=0 ; i<n ; i++ )
        {
          if ( m_parameterGroupData[i] )
          {
            count++;
          }
        }
        return( count );
      }

      const ParameterGroupDataSharedPtr & PipelineData::findParameterGroupData( const ParameterGroupSpecSharedPtr & spec ) const
      {
        EffectSpec::iterator it = m_effectSpec->findParameterGroupSpec( spec );
        if ( it != m_effectSpec->endParameterGroupSpecs() )
        {
          return( getParameterGroupData( it ) );
        }
        return( ParameterGroupDataSharedPtr::null );
      }

      const ParameterGroupDataSharedPtr & PipelineData::findParameterGroupData( const std::string & name ) const
      {
        EffectSpec::iterator it = m_effectSpec->findParameterGroupSpec( name );
        if ( it != m_effectSpec->endParameterGroupSpecs() )
        {
          return( getParameterGroupData( it ) );
        }
        return( ParameterGroupDataSharedPtr::null );
      }

      PipelineData & PipelineData::operator=( const PipelineData & rhs )
      {
        if ( this != &rhs )
        {
          Object::operator=( rhs );
          m_effectSpec          = rhs.m_effectSpec;
          clearParameterGroupData();
          unsigned int nopgs = m_effectSpec->getNumberOfParameterGroupSpecs();
          m_parameterGroupData.reset( new ParameterGroupDataSharedPtr[nopgs] );
          for ( unsigned int i=0 ; i<nopgs ; i++ )
          {
            m_parameterGroupData[i] = rhs.m_parameterGroupData[i];
            if ( m_parameterGroupData[i] )
            {
              m_parameterGroupData[i]->attach( this );
            }
          }
        }
        return( *this );
      }

      bool PipelineData::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<PipelineData>() && Object::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          PipelineDataSharedPtr const& ed = object.staticCast<PipelineData>();

          equi = ( m_effectSpec->getNumberOfParameterGroupSpecs() == ed->m_effectSpec->getNumberOfParameterGroupSpecs() );
          if ( equi )
          {
            if ( deepCompare )
            {
              equi = m_effectSpec->isEquivalent( ed->m_effectSpec, ignoreNames, true );
              unsigned int nopgs = m_effectSpec->getNumberOfParameterGroupSpecs();
              for ( unsigned int i=0 ; equi && i<nopgs ; i++ )
              {
                equi =    ( !m_parameterGroupData[i] == !ed->m_parameterGroupData[i] )
                      &&  ( !m_parameterGroupData[i] || m_parameterGroupData[i]->isEquivalent( ed->m_parameterGroupData[i], ignoreNames, true ) );
              }
            }
            else
            {
              equi = ( m_effectSpec == ed->m_effectSpec );
              unsigned int nopgs = m_effectSpec->getNumberOfParameterGroupSpecs();
              for ( unsigned int i=0 ; equi && i<nopgs ; i++ )
              {
                equi = ( m_parameterGroupData[i] == ed->m_parameterGroupData[i] );
              }
            }
          }
        }
        return( equi );
      }

      void PipelineData::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Object::feedHashGenerator( hg );
        hg.update( m_effectSpec );
        unsigned int nopgs = m_effectSpec->getNumberOfParameterGroupSpecs();
        for ( unsigned int i=0 ; i<nopgs ; i++ )
        {
          if ( m_parameterGroupData[i] )
          {
            hg.update( m_parameterGroupData[i] );
          }
        }
      }

      void PipelineData::clearParameterGroupData()
      {
        unsigned int nopgs = m_effectSpec->getNumberOfParameterGroupSpecs();
        for ( unsigned int i=0 ; i<nopgs ; i++ )
        {
          if ( m_parameterGroupData[i] )
          {
            m_parameterGroupData[i]->detach( this );
            m_parameterGroupData[i].reset();
          }
        }
        notify( Event(this) );
      }

      void PipelineData::setTransparent( bool transparent )
      {
        if ( m_transparent != transparent )
        {
          m_transparent = transparent;
          notify( PropertyEvent( this, PID_Transparent ) );
        }
      }

      bool PipelineData::getTransparent() const
      {
        return( m_transparent );
      }

      bool PipelineData::save( const std::string & filename ) const
      {
        EffectDataLocalSharedPtr edl = EffectDataLocal::create( this );
        return( EffectLibrary::instance()->save( edl.inplaceCast<dp::fx::EffectData>(), filename ) );
      }

      const EffectSpecSharedPtr& getStandardGeometrySpec()
      {
        EffectSpecSharedPtr const & es = EffectLibrary::instance()->getEffectSpec( std::string("standardGeometryEffect") );
        DP_ASSERT( !es->getTransparent() );
        return( es );
      }

      const EffectSpecSharedPtr& getStandardDirectedLightSpec()
      {
        EffectSpecSharedPtr const & es = EffectLibrary::instance()->getEffectSpec( std::string("standardDirectedLightEffect") );
        DP_ASSERT( !es->getTransparent() );
        return( es );
      }

      const EffectSpecSharedPtr& getStandardPointLightSpec()
      {
        EffectSpecSharedPtr const & es = EffectLibrary::instance()->getEffectSpec( std::string("standardPointLightEffect") );
        DP_ASSERT( !es->getTransparent() );
        return( es );
      }

      const EffectSpecSharedPtr& getStandardSpotLightSpec()
      {
        EffectSpecSharedPtr const & es = EffectLibrary::instance()->getEffectSpec( std::string("standardSpotLightEffect") );
        DP_ASSERT( !es->getTransparent() );
        return( es );
      }

      const EffectSpecSharedPtr& getStandardMaterialSpec()
      {
        EffectSpecSharedPtr const & es = EffectLibrary::instance()->getEffectSpec( std::string("standardMaterial") );
        DP_ASSERT( es->getTransparent() );
        return( es );
      }

      PipelineDataSharedPtr createStandardGeometryData()
      {
        const EffectSpecSharedPtr & standardSpec = getStandardGeometrySpec();
        DP_ASSERT( standardSpec );
        EffectSpec::iterator groupSpecIt = standardSpec->findParameterGroupSpec( std::string( "standardGeometryParameters" ) );
        DP_ASSERT( groupSpecIt != standardSpec->endParameterGroupSpecs() );

        PipelineDataSharedPtr effectData = PipelineData::create( standardSpec );
        effectData->setParameterGroupData( groupSpecIt, ParameterGroupData::create( *groupSpecIt ) );

        return( effectData );
      }

      PipelineDataSharedPtr createStandardDirectedLightData( const Vec3f & direction, const Vec3f & ambient
                                                         , const Vec3f & diffuse, const Vec3f & specular )
      {
        const EffectSpecSharedPtr & standardSpec = getStandardDirectedLightSpec();
        DP_ASSERT( standardSpec );
        EffectSpec::iterator groupSpecIt = standardSpec->findParameterGroupSpec( std::string( "standardDirectedLightParameters" ) );
        DP_ASSERT( groupSpecIt != standardSpec->endParameterGroupSpecs() );

        ParameterGroupDataSharedPtr groupData = ParameterGroupData::create( *groupSpecIt );
        DP_VERIFY( groupData->setParameter( "ambient", ambient ) );
        DP_VERIFY( groupData->setParameter( "diffuse", diffuse ) );
        DP_VERIFY( groupData->setParameter( "specular", specular ) );
        DP_VERIFY( groupData->setParameter( "direction", direction ) );

        PipelineDataSharedPtr effectData = PipelineData::create( standardSpec );
        effectData->setParameterGroupData( groupSpecIt, groupData );

        return( effectData );
      }

      PipelineDataSharedPtr createStandardPointLightData( const Vec3f & position, const Vec3f & ambient
                                                      , const Vec3f & diffuse, const Vec3f & specular
                                                      , const std::array<float,3> & attenuations )
      {
        const EffectSpecSharedPtr & standardSpec = getStandardPointLightSpec();
        DP_ASSERT( standardSpec );
        EffectSpec::iterator groupSpecIt = standardSpec->findParameterGroupSpec( std::string( "standardPointLightParameters" ) );
        DP_ASSERT( groupSpecIt != standardSpec->endParameterGroupSpecs() );

        ParameterGroupDataSharedPtr groupData = ParameterGroupData::create( *groupSpecIt );
        DP_VERIFY( groupData->setParameter( "ambient", ambient ) );
        DP_VERIFY( groupData->setParameter( "diffuse", diffuse ) );
        DP_VERIFY( groupData->setParameter( "specular", specular ) );
        DP_VERIFY( groupData->setParameter( "position", position ) );
        DP_VERIFY(( groupData->setParameterArray<float,3>( "attenuations", attenuations ) ));

        PipelineDataSharedPtr effectData = PipelineData::create( standardSpec );
        effectData->setParameterGroupData( groupSpecIt, groupData );

        return( effectData );
      }

      PipelineDataSharedPtr createStandardSpotLightData( const Vec3f & position, const Vec3f & direction
                                                     , const Vec3f & ambient, const Vec3f & diffuse
                                                     , const Vec3f & specular
                                                     , const std::array<float,3> & attenuations
                                                     , float exponent, float cutoff )
      {
        const EffectSpecSharedPtr & standardSpec = getStandardSpotLightSpec();
        DP_ASSERT( standardSpec );
        EffectSpec::iterator groupSpecIt = standardSpec->findParameterGroupSpec( std::string( "standardSpotLightParameters" ) );
        DP_ASSERT( groupSpecIt != standardSpec->endParameterGroupSpecs() );

        ParameterGroupDataSharedPtr groupData = ParameterGroupData::create( *groupSpecIt );
        DP_VERIFY( groupData->setParameter( "ambient", ambient ) );
        DP_VERIFY( groupData->setParameter( "diffuse", diffuse ) );
        DP_VERIFY( groupData->setParameter( "specular", specular ) );
        DP_VERIFY( groupData->setParameter( "position", position ) );
        DP_VERIFY(( groupData->setParameterArray<float,3>( "attenuations", attenuations ) ));
        DP_VERIFY( groupData->setParameter( "direction", direction ) );
        DP_VERIFY( groupData->setParameter( "exponent", exponent ) );
        DP_VERIFY( groupData->setParameter( "cutoff", cutoff ) );

        PipelineDataSharedPtr effectData = PipelineData::create( standardSpec );
        effectData->setParameterGroupData( groupSpecIt, groupData );

        return( effectData );
      }

      PipelineDataSharedPtr createStandardMaterialData( const Vec3f & ambientColor, const Vec3f & diffuseColor
                                                    , const Vec3f & specularColor, const float specularExponent
                                                    , const Vec3f & emissiveColor, const float opacity
                                                    , const float reflectivity, const float indexOfRefraction )
      {
        const EffectSpecSharedPtr & standardSpec = getStandardMaterialSpec();
        DP_ASSERT( standardSpec );
        EffectSpec::iterator materialSpecIt = standardSpec->findParameterGroupSpec( std::string( "standardMaterialParameters" ) );
        DP_ASSERT( materialSpecIt != standardSpec->endParameterGroupSpecs() );

        ParameterGroupDataSharedPtr materialData = ParameterGroupData::create( *materialSpecIt );
        DP_VERIFY( materialData->setParameter( "frontEmissiveColor", emissiveColor ) );
        DP_VERIFY( materialData->setParameter( "frontAmbientColor", ambientColor ) );
        DP_VERIFY( materialData->setParameter( "frontDiffuseColor", diffuseColor ) );
        DP_VERIFY( materialData->setParameter( "frontSpecularColor", specularColor ) );
        DP_VERIFY( materialData->setParameter( "frontSpecularExponent", specularExponent ) );
        DP_VERIFY( materialData->setParameter( "frontOpacity", opacity ) );
        DP_VERIFY( materialData->setParameter( "frontReflectivity", reflectivity ) );
        DP_VERIFY( materialData->setParameter( "frontIOR", indexOfRefraction ) );

        DP_VERIFY( materialData->setParameter( "backEmissiveColor", emissiveColor ) );
        DP_VERIFY( materialData->setParameter( "backAmbientColor", ambientColor ) );
        DP_VERIFY( materialData->setParameter( "backDiffuseColor", diffuseColor ) );
        DP_VERIFY( materialData->setParameter( "backSpecularColor", specularColor ) );
        DP_VERIFY( materialData->setParameter( "backSpecularExponent", specularExponent ) );
        DP_VERIFY( materialData->setParameter( "backOpacity", opacity ) );
        DP_VERIFY( materialData->setParameter( "backReflectivity", reflectivity ) );
        DP_VERIFY( materialData->setParameter( "backIOR", indexOfRefraction ) );

        DP_VERIFY( materialData->setParameter( "unlitColor", Vec4f( diffuseColor, opacity ) ) );

        PipelineDataSharedPtr effectData = PipelineData::create( standardSpec );
        {
          effectData->setParameterGroupData( materialSpecIt, materialData );
          effectData->setTransparent( opacity < 1.0f );
        }

        return( effectData );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
