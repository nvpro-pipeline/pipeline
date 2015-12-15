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


#include <dp/fx/EffectLibrary.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/util/File.h>
#include <dp/util/WeakPtr.h>
#include <boost/algorithm/string.hpp>

using namespace dp::fx;
using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( ParameterGroupData )
        DERIVE_STATIC_PROPERTIES( ParameterGroupData, Object )
      END_REFLECTION_INFO

      typedef std::map<ParameterGroupSpec const*,std::pair<unsigned int,dp::util::PropertyListChain *> > SpecToPropertyListChainMap;
      static SpecToPropertyListChainMap gSpecToPropertyMap;

      ParameterGroupDataSharedPtr ParameterGroupData::create( const ParameterGroupSpecSharedPtr & parameterGroupSpec )
      {
        return( std::shared_ptr<ParameterGroupData>( new ParameterGroupData( parameterGroupSpec ) ) );
      }

      ParameterGroupDataSharedPtr ParameterGroupData::create( const dp::fx::ParameterGroupDataSharedPtr & parameterGroupData )
      {
        return( std::shared_ptr<ParameterGroupData>( new ParameterGroupData( parameterGroupData ) ) );
      }

      HandledObjectSharedPtr ParameterGroupData::clone() const
      {
        return( std::shared_ptr<ParameterGroupData>( new ParameterGroupData( *this ) ) );
      }

      template <typename ValueType>
      class TypedPropertyParameter : public dp::util::TypedProperty<ValueType>
      {
        public:
          TypedPropertyParameter( ParameterGroupSpec::iterator it );
          virtual ~TypedPropertyParameter();

          virtual void getValue( const dp::util::Reflection * owner, ValueType & value ) const;
          virtual void setValue( dp::util::Reflection * owner, const ValueType & value );

          virtual std::string const & getAnnotation() const;
          virtual dp::util::Semantic getSemantic() const;
          virtual bool isEnum() const;
          virtual std::string const & getEnumTypeName() const;
          virtual unsigned int getEnumsCount() const;
          virtual std::string const & getEnumName( unsigned int idx ) const;

          virtual void addRef();
          virtual void destroy();

        private:
          ParameterGroupSpec::iterator m_it;
          unsigned int                 m_refCount;
      };

      template <typename ValueType>
      class TypedPropertyParameterArray : public dp::util::TypedProperty<ValueType>
      {
        public:
          TypedPropertyParameterArray( ParameterGroupSpec::iterator it, unsigned int index );
          virtual ~TypedPropertyParameterArray();

          virtual void getValue( const dp::util::Reflection * owner, ValueType & value ) const;
          virtual void setValue( dp::util::Reflection * owner, const ValueType & value );

          virtual std::string const & getAnnotation() const;
          virtual dp::util::Semantic getSemantic() const;
          virtual bool isEnum() const;
          virtual std::string const & getEnumTypeName() const;
          virtual unsigned int getEnumsCount() const;
          virtual std::string const & getEnumName( unsigned int idx ) const;

          virtual void addRef();
          virtual void destroy();

        private:
          unsigned int                 m_index;
          ParameterGroupSpec::iterator m_it;
          unsigned int                 m_refCount;
      };


      template <typename ValueType>
      TypedPropertyParameter<ValueType>::TypedPropertyParameter( ParameterGroupSpec::iterator it )
        : m_it(it)
        , m_refCount(1)
      {
        DP_ASSERT( m_it->first.getArraySize() == 0 );
      }

      template <typename ValueType>
      TypedPropertyParameter<ValueType>::~TypedPropertyParameter()
      {
      }

      template <typename ValueType>
      void TypedPropertyParameter<ValueType>::getValue( const dp::util::Reflection * owner, ValueType & value ) const
      {
        value = static_cast<const ParameterGroupData *>(owner)->getParameter<ValueType>( m_it );
      }

      template <typename ValueType>
      void TypedPropertyParameter<ValueType>::setValue( dp::util::Reflection * owner, const ValueType & value )
      {
        static_cast<ParameterGroupData *>(owner)->setParameter<ValueType>( m_it, value );
      }

      template <typename ValueType>
      std::string const & TypedPropertyParameter<ValueType>::getAnnotation() const
      {
        return( m_it->first.getAnnotation() );
      }

      template <typename ValueType>
      util::Semantic TypedPropertyParameter<ValueType>::getSemantic() const
      {
        return( m_it->first.getSemantic() );
      }

      template <typename ValueType>
      bool TypedPropertyParameter<ValueType>::isEnum() const
      {
        return( !!m_it->first.getEnumSpec() );
      }

      template <typename ValueType>
      std::string const & TypedPropertyParameter<ValueType>::getEnumTypeName() const
      {
        static std::string emptyString;
        return( m_it->first.getEnumSpec() ? m_it->first.getEnumSpec()->getType() : emptyString );
      }

      template <typename ValueType>
      unsigned int TypedPropertyParameter<ValueType>::getEnumsCount() const
      {
        return( m_it->first.getEnumSpec() ? m_it->first.getEnumSpec()->getValueCount() : 0 );
      }

      template <typename ValueType>
      std::string const & TypedPropertyParameter<ValueType>::getEnumName( unsigned int idx ) const
      {
        static std::string emptyString;
        return( m_it->first.getEnumSpec() ? m_it->first.getEnumSpec()->getValueName( idx ) : emptyString );
      }

      template <typename ValueType>
      void TypedPropertyParameter<ValueType>::addRef()
      {
        ++m_refCount;
      }

      template <typename ValueType>
      void TypedPropertyParameter<ValueType>::destroy()
      {
        --m_refCount;
        if ( !m_refCount )
        {
          delete this;
        }
      }

      template <typename ValueType>
      TypedPropertyParameterArray<ValueType>::TypedPropertyParameterArray( ParameterGroupSpec::iterator it, unsigned int index )
        : m_index(index)
        , m_it(it)
        , m_refCount(1)
      {
        DP_ASSERT( m_it->first.getArraySize() != 0 );
      }

      template <typename ValueType>
      TypedPropertyParameterArray<ValueType>::~TypedPropertyParameterArray()
      {
      }

      template <typename ValueType>
      void TypedPropertyParameterArray<ValueType>::getValue( const dp::util::Reflection * owner, ValueType & value ) const
      {
        value = static_cast<const ParameterGroupData *>(owner)->getParameterArrayElement<ValueType>( m_it, m_index );
      }

      template <typename ValueType>
      void TypedPropertyParameterArray<ValueType>::setValue( dp::util::Reflection * owner, const ValueType & value )
      {
        static_cast<ParameterGroupData *>(owner)->setParameterArrayElement<ValueType>( m_it, m_index, value );
      }

      template <typename ValueType>
      std::string const & TypedPropertyParameterArray<ValueType>::getAnnotation() const
      {
        return( m_it->first.getAnnotation() );
      }

      template <typename ValueType>
      util::Semantic TypedPropertyParameterArray<ValueType>::getSemantic() const
      {
        return( m_it->first.getSemantic() );
      }

      template <typename ValueType>
      bool TypedPropertyParameterArray<ValueType>::isEnum() const
      {
        return( !!m_it->first.getEnumSpec() );
      }

      template <typename ValueType>
      std::string const & TypedPropertyParameterArray<ValueType>::getEnumTypeName() const
      {
        static std::string emptyString;
        return( m_it->first.getEnumSpec() ? m_it->first.getEnumSpec()->getType() : emptyString );
      }

      template <typename ValueType>
      unsigned int TypedPropertyParameterArray<ValueType>::getEnumsCount() const
      {
        return( m_it->first.getEnumSpec() ? m_it->first.getEnumSpec()->getValueCount() : 0 );
      }

      template <typename ValueType>
      std::string const & TypedPropertyParameterArray<ValueType>::getEnumName( unsigned int idx ) const
      {
        static std::string emptyString;
        return( m_it->first.getEnumSpec() ? m_it->first.getEnumSpec()->getValueName( idx ) : emptyString );
      }

      template <typename ValueType>
      void TypedPropertyParameterArray<ValueType>::addRef()
      {
        ++m_refCount;
      }

      template <typename ValueType>
      void TypedPropertyParameterArray<ValueType>::destroy()
      {
        --m_refCount;
        if ( !m_refCount )
        {
          delete this;
        }
      }

      template <typename T>
      void addParameterPropertyT( dp::util::PropertyListImpl * pil, ParameterGroupSpec::iterator it )
      {
        unsigned int n = it->first.getArraySize();
        if ( 0 < n )
        {
          for ( unsigned int i=0 ; i<n ; i++ )
          {
            std::ostringstream oss;
            oss << it->first.getName() << i;
            pil->addProperty( oss.str(), new TypedPropertyParameterArray<T>( it, i ) );
          }
        }
        else
        {
          pil->addProperty( it->first.getName(), new TypedPropertyParameter<T>( it ) );
        }
      }

      template<unsigned int n>
      void addParameterPropertyVN( dp::util::PropertyListImpl * pil, ParameterGroupSpec::iterator it )
      {
        unsigned int type = it->first.getType();
        DP_ASSERT( ( type & PT_SCALAR_MODIFIER_MASK ) && ( type & PT_SCALAR_TYPE_MASK ) );
        switch( type & PT_SCALAR_TYPE_MASK )
        {
          case PT_BOOL :
          case PT_ENUM :
          case PT_INT8 :
          case PT_UINT8 :
          case PT_INT16 :
          case PT_UINT16 :
          case PT_FLOAT64 :
          case PT_INT64 :
          case PT_UINT64 :
            //DP_ASSERT( !"vector type encountered without corresponding property type" );
            break;
          case PT_FLOAT32 :
            addParameterPropertyT<Vecnt<n,float> >( pil, it );
            break;
          case PT_INT32 :
            addParameterPropertyT<Vecnt<n,int> >( pil, it );
            break;
          case PT_UINT32 :
            addParameterPropertyT<Vecnt<n,unsigned int> >( pil, it );
            break;
          default :
            DP_ASSERT( false );
            break;
        }
      }

      template<unsigned int m, unsigned int n>
      void addParameterPropertyMN( dp::util::PropertyListImpl * pil, ParameterGroupSpec::iterator it )
      {
        unsigned int type = it->first.getType();
        DP_ASSERT( ( type & PT_SCALAR_MODIFIER_MASK ) && ( type & PT_SCALAR_TYPE_MASK ) );
        switch( type & PT_SCALAR_TYPE_MASK )
        {
          case PT_BOOL :
          case PT_ENUM :
          case PT_INT8 :
          case PT_UINT8 :
          case PT_INT16 :
          case PT_UINT16 :
          case PT_INT32 :
          case PT_UINT32 :
          case PT_FLOAT64 :
          case PT_INT64 :
          case PT_UINT64 :
            //DP_ASSERT( !"matrix type encountered without corresponding property type" );
            break;
          case PT_FLOAT32 :
            addParameterPropertyT<Matmnt<m,n,float> >( pil, it );
            break;
          default :
            DP_ASSERT( false );
            break;
        }
      }

      void addParameterProperty( dp::util::PropertyListImpl * pil, ParameterGroupSpec::iterator it )
      {
        unsigned int type = it->first.getType();
        if ( type & PT_SCALAR_MODIFIER_MASK )
        {
          switch( type & PT_SCALAR_MODIFIER_MASK )
          {
            case PT_VECTOR2 :
              addParameterPropertyVN<2>( pil, it );
              break;
            case PT_VECTOR3 :
              addParameterPropertyVN<3>( pil, it );
              break;
            case PT_VECTOR4 :
              addParameterPropertyVN<4>( pil, it );
              break;
            case PT_MATRIX2x2 :
            case PT_MATRIX2x3 :
            case PT_MATRIX2x4 :
            case PT_MATRIX3x2 :
            case PT_MATRIX3x4 :
            case PT_MATRIX4x2 :
            case PT_MATRIX4x3 :
              //DP_ASSERT( !"matrix size encountered without corresponding property type" );
              break;
            case PT_MATRIX3x3 :
              addParameterPropertyMN<3,3>( pil, it );
              break;
            case PT_MATRIX4x4 :
              addParameterPropertyMN<4,4>( pil, it );
              break;
            default :
              DP_ASSERT( false );
              break;
          }
        }
        else if ( type & PT_SCALAR_TYPE_MASK )
        {
          switch( type & PT_SCALAR_TYPE_MASK )
          {
            case PT_UINT8 :
            case PT_INT16 :
            case PT_UINT16 :
            case PT_FLOAT64 :
            case PT_INT64 :
            case PT_UINT64 :
              //DP_ASSERT( !"scalar type encountered without corresponding property type" );
              break;
            case PT_BOOL :
              addParameterPropertyT<bool>( pil, it );
              break;
            case PT_INT8 :
              addParameterPropertyT<char>( pil, it );
              break;
            case PT_FLOAT32 :
              addParameterPropertyT<float>( pil, it );
              break;
            case PT_ENUM :
            case PT_INT32 :
              addParameterPropertyT<int>( pil, it );
              break;
            case PT_UINT32 :
              addParameterPropertyT<unsigned int>( pil, it );
              break;
            default :
              DP_ASSERT( false );
              break;
          }
        }
        else if ( type & PT_POINTER_TYPE_MASK )
        {
          switch( type & PT_POINTER_TYPE_MASK )
          {
            case PT_BUFFER_PTR :
            case PT_SAMPLER_PTR :
              //DP_ASSERT( !"pointer type encountered without corresponding property type" );
              break;
            case PT_TEXTURE_PTR :
              addParameterPropertyT<TextureSharedPtr>( pil, it );
              break;
            default :
              DP_ASSERT( false );
              break;
          }
        }
        else
        {
          DP_ASSERT( !"unknown property type encountered" );
        }
      }

      /************************************************************************/
      /* ParameterGroupData                                                   */
      /************************************************************************/
      ParameterGroupData::ParameterGroupData( const dp::fx::ParameterGroupSpecSharedPtr& parameterGroupSpec)
      {
        m_objectCode = ObjectCode::PARAMETER_GROUP_DATA;
        //init ( parameterGroupSpec );
        initSpec( parameterGroupSpec );
        initData( dp::fx::EffectLibrary::instance()->getParameterGroupData( parameterGroupSpec->getName() ) );
      }

      ParameterGroupData::ParameterGroupData( const dp::fx::ParameterGroupDataSharedPtr& fxParameterGroupData)
      {
        m_objectCode = ObjectCode::PARAMETER_GROUP_DATA;
        initSpec( fxParameterGroupData->getParameterGroupSpec() );
        initData( fxParameterGroupData );
      }

      ParameterGroupData::ParameterGroupData( const ParameterGroupData & rhs )
        : Object( rhs )
      {
        m_objectCode = ObjectCode::PARAMETER_GROUP_DATA;
        initSpec( rhs.getParameterGroupSpec() );

        // fill the data area with the values from rhs
        for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if ( it->first.getType() & PT_POINTER_TYPE_MASK )
          {
            DP_ASSERT( ( it->first.getType() & PT_POINTER_TYPE_MASK ) == PT_SAMPLER_PTR );
            setParameter( it, rhs.getParameter<SamplerSharedPtr>( it ) );
          }
          else
          {
            setParameter( it, rhs.getParameter( it ) );
          }
        }
      }

      ParameterGroupData::~ParameterGroupData()
      {
        DP_STATIC_ASSERT( sizeof(HandledObjectSharedPtr) == sizeof(SamplerSharedPtr) );
        DP_STATIC_ASSERT( sizeof(HandledObjectSharedPtr) == sizeof(BufferSharedPtr) );

        // decrease refcount of textures and buffers
        for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if ( it->first.getType() & PT_POINTER_TYPE_MASK )
          {
            DP_ASSERT( it->second < m_data.size() );
            if ( it->first.getArraySize() )
            {
              DP_ASSERT( it->second + it->first.getArraySize() * sizeof(HandledObjectSharedPtr) <= m_data.size() );
              for ( unsigned int i=0 ; i<it->first.getArraySize() ; i++ )
              {
                HandledObjectSharedPtr & ho = *reinterpret_cast<HandledObjectSharedPtr *>(&m_data[it->second+i*sizeof(HandledObjectSharedPtr)]);
                if ( ho )
                {
                  if ( ho.isPtrTo<Sampler>() )
                  {
                    ho.staticCast<Sampler>()->detach( this );
                  }
                  ho.reset();
                }
              }
            }
            else
            {
              DP_ASSERT( it->second + sizeof(HandledObjectSharedPtr) <= m_data.size() );
              HandledObjectSharedPtr & ho = *reinterpret_cast<HandledObjectSharedPtr *>(&m_data[it->second]);
              if ( ho )
              {
                if ( ho.isPtrTo<Sampler>() )
                {
                  ho.staticCast<Sampler>()->detach( this );
                }
                ho.reset();
              }
            }
          }
        }

        // If it's not the last ParameterGroupData to this ParameterGroupSpec, we don't want to delete the
        // PropertyListChain here, as the ParameterGroupSpec holds the master (no refcount!)
        SpecToPropertyListChainMap::iterator pit = gSpecToPropertyMap.find( m_parameterGroupSpec.operator->() );    // Big Hack !!
        DP_ASSERT( pit != gSpecToPropertyMap.end() && pit->second.first );
        pit->second.first--;
        if ( pit->second.first )
        {
          m_propertyLists = nullptr;
        }
        else
        {
          gSpecToPropertyMap.erase( pit );
        }
      }

      void ParameterGroupData::initSpec( const dp::fx::ParameterGroupSpecSharedPtr & spec )
      {
        setName( spec->getName() );
        m_parameterGroupSpec = spec;
        m_data.resize( m_parameterGroupSpec->getDataSize() );
        DP_ASSERT( ( reinterpret_cast<size_t>(m_data.data()) & 0x7 ) == 0 );
        DP_ASSERT( !m_propertyLists );

        // create the PropertyList, if it's not yet there
        SpecToPropertyListChainMap::iterator it = gSpecToPropertyMap.find( m_parameterGroupSpec.operator->() );   // Big Hack !!
        if ( it == gSpecToPropertyMap.end() )
        {
          dp::util::PropertyListImpl * pli = new dp::util::PropertyListImpl( false );   // create a non-static, local PropertyList
          for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
          {
            addParameterProperty( pli, it );
          }

          DP_ASSERT( !m_propertyLists );
          m_propertyLists = new dp::util::PropertyListChain( false );
          m_propertyLists->addPropertyList( pli );

          gSpecToPropertyMap[m_parameterGroupSpec.operator->()] = std::make_pair( 1, m_propertyLists );   // Big Hack !!
        }
        else
        {
          it->second.first++;
          m_propertyLists = it->second.second;
        }
      }

      void ParameterGroupData::initData( const dp::fx::ParameterGroupDataSharedPtr & data )
      {
        // fill the data area with the defaults from the spec
        for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          // If the parameter is a sampler it needs to be initialized.
          // For GLSL to silence the OpenGL driver debug output warning that a sampler is not assigned.

          // Both the undefined and the default filename initialization should be done via some scene graph global texture cache.
          if ( ( it->first.getType() & PT_POINTER_TYPE_MASK ) == PT_SAMPLER_PTR )
          {
            if ( it->first.getDefaultValue() )
            {
              // If a sampler has a default value it must be a texture image filename.
              const char * name = static_cast<const char *>( data->getParameter( it ) );
              TextureFileSharedPtr texture = TextureFile::create( name, textureTypeToTarget( it->first.getType() ) );
              DP_ASSERT( texture );
              if ( texture )
              {
                texture->incrementMipmapUseCount();

                SamplerSharedPtr sampler = Sampler::create( texture );

                std::string extension = dp::util::getFileExtension( name );
                boost::algorithm::to_lower(extension);
                if ( extension == ".mbsdf" )
                {
                  sampler->setName( it->first.getName() );
                  sampler->setMagFilterMode( TextureMagFilterMode::NEAREST );
                  sampler->setMinFilterMode( TextureMinFilterMode::NEAREST );
                  sampler->setWrapModes( TextureWrapMode::CLAMP_TO_EDGE, TextureWrapMode::CLAMP_TO_EDGE, TextureWrapMode::CLAMP_TO_EDGE );
                }
                else
                {
                  sampler->setName( it->first.getName() );
                  sampler->setMagFilterMode( TextureMagFilterMode::LINEAR );
                  sampler->setMinFilterMode( TextureMinFilterMode::LINEAR_MIPMAP_LINEAR );
                }
                setParameter( it, sampler );
              }
            }
            else
            {
              // If the sampler has no default initialization, find out their type and assign a matching default texture.
              // DAR FIXME Support for "sampler" in standardMaterialEffect standardTextureParameters for now.
              // 2x2 RGBA8 red, green, blue, yellow default texture.
              static const unsigned char texel[] =
              {
                0xFF, 0x00, 0x00, 0xFF,
                0x00, 0xFF, 0x00, 0xFF,
                0x00, 0x00, 0xFF, 0xFF,
                0xFF, 0xFF, 0x00, 0xFF
              };
              TextureHostSharedPtr textureHost = TextureHost::create();
              DP_ASSERT( textureHost );
              textureHost->setCreationFlags( TextureHost::F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD );
              unsigned int index = textureHost->addImage( 2, 2, 1, Image::PixelFormat::RGBA, Image::PixelDataType::UNSIGNED_BYTE );
              DP_ASSERT( index != -1 );
              textureHost->setImageData( index, (const void *) &texel[0] );
              textureHost->setTextureTarget( TextureTarget::TEXTURE_2D );

              SamplerSharedPtr sampler = Sampler::create( textureHost );
              DP_ASSERT( sampler );
              sampler->setName( "default_sampler2D" );
              sampler->setMagFilterMode( TextureMagFilterMode::NEAREST );
              sampler->setMinFilterMode( TextureMinFilterMode::NEAREST );
              setParameter( it, sampler );
            }
          }
          else
          {
            // No other pointer types than samplers are supported inside the EffectLibrary so far.
            DP_ASSERT( ( it->first.getType() & PT_POINTER_TYPE_MASK ) == 0 );

            if ( it->first.getDefaultValue() )
            {
              setParameter( it, data->getParameter( it ) );
            }
          }
        }
      }

      bool ParameterGroupData::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<ParameterGroupData>() && Object::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          ParameterGroupDataSharedPtr const& pgd = object.staticCast<ParameterGroupData>();
          if ( deepCompare )
          {
            equi = m_parameterGroupSpec->isEquivalent( pgd->m_parameterGroupSpec, ignoreNames, true );
            for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; equi && it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
            {
              unsigned int pointerType = it->first.getType() & PT_POINTER_TYPE_MASK;
              if ( pointerType )
              {
                if ( pointerType == PT_SAMPLER_PTR )
                {
                  equi = getParameter<SamplerSharedPtr>( it )->isEquivalent( pgd->getParameter<SamplerSharedPtr>( it ), ignoreNames, true );
                }
                else
                {
                  DP_ASSERT( pointerType == PT_BUFFER_PTR );
                  DP_ASSERT( false );
                }
              }
              else
              {
                equi = ( memcmp( getParameter( it ), pgd->getParameter( it ), it->first.getSizeInByte() ) == 0 );
              }
            }
          }
          else
          {
            equi = ( m_parameterGroupSpec == pgd->m_parameterGroupSpec ) && ( m_data == pgd->m_data );
          }
        }
        return( equi );
      }

      void ParameterGroupData::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Object::feedHashGenerator( hg );
        hg.update( m_parameterGroupSpec );
        hg.update( reinterpret_cast<const unsigned char *>(m_data.data()), dp::checked_cast<unsigned int>(m_data.size()) );
      }

      void ParameterGroupData::setParameterIntern( const dp::fx::ParameterGroupSpec::iterator& it, const SamplerSharedPtr & value )
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK )) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() == 0 );
        DP_ASSERT( it->second + sizeof(SamplerSharedPtr) <= m_data.size() );
        SamplerSharedPtr & dst = *reinterpret_cast<SamplerSharedPtr*>(&m_data[it->second]);
        if ( dst != value )
        {
          if ( dst )
          {
            dst->detach( this );
          }
          dst = value;
          if ( dst )
          {
            dst->attach( this );
          }
          notify( Event(it) );
        }
      }

      ParameterGroupDataSharedPtr createStandardTextureParameterData( const SamplerSharedPtr & sampler )
      {
        EffectSpecSharedPtr materialSpec = getStandardMaterialSpec();
        EffectSpec::iterator groupSpecIt = materialSpec->findParameterGroupSpec( std::string( "standardTextureParameters" ) );
        DP_ASSERT( groupSpecIt != materialSpec->endParameterGroupSpecs() );

        ParameterGroupDataSharedPtr parameterData = ParameterGroupData::create( *groupSpecIt );
        DP_VERIFY( parameterData->setParameter( "sampler", sampler ) );
        DP_VERIFY( parameterData->setParameter<bool>( "textureEnable", !!sampler ) );
        return( parameterData );
      }

      ParameterGroupDataSharedPtr createStandardBumpmapParameterData( const SamplerSharedPtr & sampler )
      {
        EffectSpecSharedPtr materialSpec = dp::fx::EffectLibrary::instance()->getEffectSpec("standardMaterialEffectBumped");
        EffectSpec::iterator groupSpecIt = materialSpec->findParameterGroupSpec( std::string( "standardBumpmapParameters" ) );
        DP_ASSERT( groupSpecIt != materialSpec->endParameterGroupSpecs() );

        ParameterGroupDataSharedPtr parameterData = ParameterGroupData::create( *groupSpecIt );
        DP_VERIFY( parameterData->setParameter( "bumpmapSampler", sampler ) );
        DP_VERIFY( parameterData->setParameter<bool>( "bumpmapEnable", !!sampler ) );
        return( parameterData );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
