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


#pragma once
/** @file */

#include <dp/fx/ParameterGroupSpec.h>
#include <dp/fx/ParameterGroupData.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/Sampler.h>
#include <dp/util/Observer.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {


      // The parameter data specified by a ParameterGroupSpec
      // This is owned by some EffectData objects.
      // NOTE: as soon as we derive from ParameterGroupData, and introduce additional dynamic Properties, we need to
      // change our current handling of m_propertyLists !!
      class ParameterGroupData : public Object
      {
        public:
          DP_SG_CORE_API static ParameterGroupDataSharedPtr create( const dp::fx::ParameterGroupDataSharedPtr & parameterGroupData );
          DP_SG_CORE_API static ParameterGroupDataSharedPtr create( const dp::fx::ParameterGroupSpecSharedPtr & parameterGroupSpec );

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~ParameterGroupData();

        public:
          DP_SG_CORE_API const dp::fx::ParameterGroupSpecSharedPtr & getParameterGroupSpec() const;

          template <typename T> const T & getParameter( const dp::fx::ParameterGroupSpec::iterator& it ) const;
          template <typename T> const T & getParameter( const std::string & name ) const;
          template <typename T> void setParameter( const dp::fx::ParameterGroupSpec::iterator& it, const T & value );
          template <typename T> bool setParameter( const std::string & name, const T & value );

          template <typename T> void getParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, std::vector<T> & data ) const;
          template <typename T> void setParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, const std::vector<T> & data );
          template <typename T> bool setParameterArray( const std::string & name, const std::vector<T> & data );

          template <typename T, unsigned int n> void getParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, std::array<T,n> & data ) const;
          template <typename T, unsigned int n> void setParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, const std::array<T,n> & data );
          template <typename T, unsigned int n> bool setParameterArray( const std::string & name, const std::array<T,n> & data );

          template <typename T> const T & getParameterArrayElement( const dp::fx::ParameterGroupSpec::iterator& it, unsigned int index ) const;
          template <typename T> void setParameterArrayElement( const dp::fx::ParameterGroupSpec::iterator& it, unsigned int index, const T & value );
          template <typename T> bool setParameterArrayElement( const std::string & name, unsigned int index, const T & value );

          DP_SG_CORE_API const void * getParameter( const dp::fx::ParameterGroupSpec::iterator& it ) const;
          DP_SG_CORE_API void setParameter( const dp::fx::ParameterGroupSpec::iterator& it, const void * value );
          DP_SG_CORE_API bool setParameter( const std::string & name, const void * value );

          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames = true, bool deepCompare = false ) const;

          /************************************************************************/
          /* New Observer interface                                               */
          /************************************************************************/
          // TODO pass source?
          class Event : public dp::sg::core::Event
          {
          public:
            Event( const dp::fx::ParameterGroupSpec::iterator& parameter )
              : dp::sg::core::Event( Event::Type::PARAMETER_GROUP_DATA )
              , m_parameter( parameter )
            {
            }

            const dp::fx::ParameterGroupSpec::iterator& getParameter() const { return m_parameter; }
          private:
            const dp::fx::ParameterGroupSpec::iterator& m_parameter;
          };

          REFLECTION_INFO_API( DP_SG_CORE_API, ParameterGroupData );

        protected:
          DP_SG_CORE_API ParameterGroupData( const ParameterGroupData &rhs );
          DP_SG_CORE_API ParameterGroupData( const dp::fx::ParameterGroupDataSharedPtr& fxParameterGroupData );
          DP_SG_CORE_API ParameterGroupData( const dp::fx::ParameterGroupSpecSharedPtr& parameterGroupSpec );
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

          // TODO require due to sampler/pgd cross references
          DP_SG_CORE_API void setParameterIntern( const dp::fx::ParameterGroupSpec::iterator& it, const SamplerSharedPtr & value );

        private:
          DP_SG_CORE_API void initSpec( const dp::fx::ParameterGroupSpecSharedPtr & spec );
          DP_SG_CORE_API void initData( const dp::fx::ParameterGroupDataSharedPtr & data );

          DP_SG_CORE_API ParameterGroupData & operator=( const ParameterGroupData & rhs );

        private:
          dp::util::Subject m_subject;

          dp::fx::ParameterGroupSpecSharedPtr m_parameterGroupSpec;
          std::vector<char>                   m_data;
      };


      DP_SG_CORE_API ParameterGroupDataSharedPtr createStandardTextureParameterData( const SamplerSharedPtr & sampler = SamplerSharedPtr() );
      DP_SG_CORE_API ParameterGroupDataSharedPtr createStandardBumpmapParameterData( const SamplerSharedPtr & sampler = SamplerSharedPtr() );

      inline const dp::fx::ParameterGroupSpecSharedPtr & ParameterGroupData::getParameterGroupSpec() const
      {
        return( m_parameterGroupSpec );
      }

      inline const void * ParameterGroupData::getParameter( const dp::fx::ParameterGroupSpec::iterator& it ) const
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() );
        DP_ASSERT( it->second + it->first.getSizeInByte() <= m_data.size() );
        return( &m_data[it->second] );
      }

      inline void ParameterGroupData::setParameter( const dp::fx::ParameterGroupSpec::iterator& it, const void * value )
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() );
        DP_ASSERT( it->second + it->first.getSizeInByte() <= m_data.size() );
        unsigned int size = it->first.getSizeInByte();
        if ( memcmp( &m_data[it->second], value, size ) != 0 )
        {
          memcpy( &m_data[it->second], value, size );
          notify( Event(it) );
        }
      }

      inline bool ParameterGroupData::setParameter( const std::string & name, const void * value )
      {
        for ( dp::fx::ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if (    ( it->first.getName() == name )
              &&  ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() ) )
          {
            setParameter( it, value );
            return( true );
          }
        }
        return( false );
      }

      template <typename T>
      inline const T & ParameterGroupData::getParameter( const dp::fx::ParameterGroupSpec::iterator& it ) const
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK ) ) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() == 0 );
        DP_ASSERT( it->second + sizeof(T) <= m_data.size() );
        return( *reinterpret_cast<const T *>(&m_data[it->second]) );
      }

      template <typename T>
      inline const T & ParameterGroupData::getParameter( const std::string & name ) const
      {
        dp::fx::ParameterGroupSpec::iterator it;
        for ( it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if (    ( it->first.getName() == name )
            &&  ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() ) )
          {
            break;
          }
        }
        return getParameter<T>( it );
      }

      template <typename T>
      inline void ParameterGroupData::setParameter( const dp::fx::ParameterGroupSpec::iterator& it, const T & value )
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK ) ) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() == 0 );
        DP_ASSERT( it->second + sizeof(T) <= m_data.size() );
        T * dst = reinterpret_cast<T*>(&m_data[it->second]);
        if ( *dst != value )
        {
          *dst = value;
          notify( Event(it) );
        }
      }

      template <>
      inline void ParameterGroupData::setParameter<SamplerSharedPtr>( const dp::fx::ParameterGroupSpec::iterator& it, const SamplerSharedPtr & value )
      {
        setParameterIntern( it, value );
      }

      template <typename T>
      inline bool ParameterGroupData::setParameter( const std::string & name, const T & value )
      {
        for ( dp::fx::ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if ( ( it->first.getName() == name ) && ( it->first.getArraySize() == 0 ) )
          {
            setParameter( it, value );
            return( true );
          }
        }
        return( false );
      }

      template <typename T>
      inline void ParameterGroupData::getParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, std::vector<T> & data ) const
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK )) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() != 0 );
        DP_ASSERT( it->second + it->first.getArraySize() * sizeof(T) <= m_data.size() );
        data.resize( it->first.getArraySize() );
        for ( unsigned int i=0 ; i<data.size() ; i++ )
        {
          data[i] = getParameterArrayElement<T>( it, i );
        }
      }

      template <typename T>
      inline void ParameterGroupData::setParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, const std::vector<T> & data )
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK )) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() != 0 );
        DP_ASSERT( it->second + data.size()*sizeof(T) <= m_data.size() );
        DP_ASSERT( it->first.getArraySize() <= data.size() );
        for ( unsigned int i=0 ; i<data.size() ; i++ )
        {
          setParameterArrayElement<T>( it, i, data[i] );
        }
      }

      template <typename T>
      inline bool ParameterGroupData::setParameterArray( const std::string & name, const std::vector<T> & data )
      {
        for ( dp::fx::ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if (    ( it->first.getName() == name )
              &&  ( it->first.getArraySize() != 0 ) )
          {
            setParameterArray( it, data );
            return( true );
          }
        }
        return( false );
      }

      template <typename T, unsigned int n>
      inline void ParameterGroupData::getParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, std::array<T,n> & data ) const
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK )) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() == n );
        DP_ASSERT( it->second + n*sizeof(T) <= m_data.size() );
        for ( unsigned int i=0 ; i<n ; i++ )
        {
          data[i] = getParameterArrayElement<T>( it, i );
        }
      }

      template <typename T, unsigned int n>
      inline void ParameterGroupData::setParameterArray( const dp::fx::ParameterGroupSpec::iterator& it, const std::array<T,n> & data )
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK )) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() == n );
        DP_ASSERT( it->second + n*sizeof(T) <= m_data.size() );
        for ( unsigned int i=0 ; i<n ; i++ )
        {
          setParameterArrayElement<T>( it, i, data[i] );
        }
      }

      template <typename T, unsigned int n>
      inline bool ParameterGroupData::setParameterArray( const std::string & name, const std::array<T,n> & data )
      {
        for ( dp::fx::ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if ( ( it->first.getName() == name ) && ( it->first.getArraySize() != 0 ) )
          {
            setParameterArray<T,n>( it, data );
            return( true );
          }
        }
        return( false );
      }

      template <typename T>
      inline const T & ParameterGroupData::getParameterArrayElement( const dp::fx::ParameterGroupSpec::iterator& it, unsigned int index ) const
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK )) == it->first.getType() );
        DP_ASSERT( it->first.getArraySize() != 0 );
        DP_ASSERT( it->second + it->first.getArraySize() * sizeof(T) <= m_data.size() );
        DP_ASSERT( index < it->first.getArraySize() );
        return( *reinterpret_cast<const T *>(&m_data[it->second+index*sizeof(T)]) );
      }

      template <typename T>
      inline void ParameterGroupData::setParameterArrayElement( const dp::fx::ParameterGroupSpec::iterator& it, unsigned int index, const T & value )
      {
        DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
        DP_ASSERT( ( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() )
                     ? ( ( dp::fx::ParameterTraits<T>::type == it->first.getType() ) || ( ( dp::fx::ParameterTraits<T>::type == dp::fx::PT_INT32 ) && ( it->first.getType() == dp::fx::PT_ENUM ) ) )
                     : ( ( it->first.getType() & (dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK) ) == it->first.getType() ) );
        DP_ASSERT( it->first.getArraySize() != 0 );
        DP_ASSERT( it->second + it->first.getArraySize() * sizeof(T) <= m_data.size() );
        DP_ASSERT( index < it->first.getArraySize() );
        T & dst = *reinterpret_cast<T*>(&m_data[it->second+index*sizeof(T)]);
        if ( dst != value )
        {
          dst = value;
          notify( Event( it ) );
        }
      }

      template <typename T>
      inline bool ParameterGroupData::setParameterArrayElement( const std::string & name, unsigned int index, const T & value )
      {
        for ( dp::fx::ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          if ( ( it->first.getName() == name ) && ( index < it->first.getArraySize() ) )
          {
            setParameterArrayElement<T>( it, index, value );
            return( true );
          }
        }
        return( false );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

