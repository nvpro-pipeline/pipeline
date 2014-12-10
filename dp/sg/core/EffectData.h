// Copyright NVIDIA Corporation 2011
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

#include <boost/scoped_array.hpp>
#include <dp/fx/EffectSpec.h>
#include <dp/fx/EffectData.h>
#include <dp/util/Array.h>
#include <dp/sg/core/OwnedObject.h>
#include <dp/sg/core/ParameterGroupData.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      // The complete set of data of an Effect
      // This is the base of MaterialEffect, LightEffect, GeometryEffect,...
      class EffectData : public OwnedObject<Object>
      {
        public:
          /*! \brief create an EffectData
           *  \param effectSpec The dp::fx::EffectSpec specifying the layout of the EffectData. */
          DP_SG_CORE_API static EffectDataSharedPtr create( const dp::fx::EffectSpecSharedPtr & effectSpec );

          /*! \brief create an EffectData
           *  \param effectData The dp::fx::EffectData specifying the layout and values of the newly created EffectData. */
          DP_SG_CORE_API static EffectDataSharedPtr create( const dp::fx::EffectDataSharedPtr & effectData);

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~EffectData();

        public:
          /*! \brief Get the dp::fx::EffectSpec of this EffectData
           *  \returns A constant reference to the dp::fx::EffectSpec of this EffectData. */
          DP_SG_CORE_API const dp::fx::EffectSpecSharedPtr & getEffectSpec() const;

          /*! \brief Get the ParameterGroupData corresponding to an iterator on an EffectSpec.
           *  \param it The iterator on an EffectSpec specifying the ParameterGroupData to get.
           *  \return A constant reference to the ParameterGroupData corresponding the iterator \a it on an EffectSpec.
           *  \note If this EffectData does not hold a ParameterGroupData corresponding to the iterator \a it, an empty dummy ParameterGroupData is returned.
           *  \sa setParameterGroupData, getNumberOfParameterGroupData, findParameterGroupData, clearParameterGroupData */
          const ParameterGroupDataSharedPtr & getParameterGroupData( dp::fx::EffectSpec::iterator const& it ) const;

          /*! \brief Set the ParameterGroupData corresponding to an iterator on an EffectSpec.
           *  \param it The iterator on an EffectSpec specifying the ParameterGroupData to set.
           *  \param pgd The ParameterGroupData to set.
           *  \note If \a pgd is empty, the ParameterGroupData corresponding to \a it is removed.
           *  \sa getParameterGroupData, getNumberOfParameterGroupData, findParameterGroupData, clearParameterGroupData */
          DP_SG_CORE_API void setParameterGroupData( dp::fx::EffectSpec::iterator const& it, const ParameterGroupDataSharedPtr & pgd );

          /*! \brief Set a ParameterGroupData.
           *  \param pgd The ParameterGroupData to set.
           *  \return \c true, if the ParameterGroupData has been set, otherwise \c false.
           *  \note This is a convenience function that eases setting a ParameterGroupData. It first finds the iterator into the EffectSpec, using the
           *  ParameterGroupSpec of \a pgd. If that could be found, \a pgd is set accordingly.
           *  \sa getParameterGroupData, getNumberOfParameterGroupData, findParameterGroupData, clearParameterGroupData */
          DP_SG_CORE_API bool setParameterGroupData( const ParameterGroupDataSharedPtr & pgd );

          /*! \brief Get the number of ParameterGroupData contained in this EffectData
           *  \returns The number of ParameterGroupData contained in this EffectData. */
          DP_SG_CORE_API unsigned int getNumberOfParameterGroupData() const;

          /*! \brief Find the element in this EffectData for the specified ParameterGroupSpec.
           *  \param spec A constant reference to a dp::fx::ParameterGroupSpecSharedPtr, that specifies the ParameterGroupData to find.
           *  \return The ParameterGroupData in this EffectData corresponding to the ParameterGroupSpec \a spec.
           *  \note If there is no ParameterGroupData corresponding the ParameterGroupSpec \a spec, an empty ParameterGroupData is returned. */
          DP_SG_CORE_API const ParameterGroupDataSharedPtr & findParameterGroupData( const dp::fx::ParameterGroupSpecSharedPtr & spec ) const;

          /*! \brief Find the element in this EffectData for the specified name of the ParameterGroupSpec.
           *  \param specName The name of the ParameterGroupSpec that specifies the ParameterGroupData to find.
           *  \return The ParameterGroupData in this EffectData corresponding to the name \a specName.
           *  \note If there is no ParameterGroupData corresponding the name \a specName, an empty ParameterGroupData is returned. */
          DP_SG_CORE_API const ParameterGroupDataSharedPtr & findParameterGroupData( const std::string & specName ) const;

          /*! \brief Remove all elements from this EffectData. */
          DP_SG_CORE_API void clearParameterGroupData();

          DP_SG_CORE_API bool getTransparent() const;
          DP_SG_CORE_API void setTransparent( bool transparent );

          DP_SG_CORE_API virtual bool determineTransparencyContainment() const;
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames = true, bool deepCompare = false ) const;

          DP_SG_CORE_API bool save( const std::string & filename ) const;

          REFLECTION_INFO_API( DP_SG_CORE_API, EffectData );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Transparent );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          friend class OwnedObject<EffectData>;

          DP_SG_CORE_API EffectData( const dp::fx::EffectSpecSharedPtr& effectSpec );
          DP_SG_CORE_API EffectData( const dp::fx::EffectDataSharedPtr& effectData );
          DP_SG_CORE_API EffectData( const EffectData& rhs );
          DP_SG_CORE_API EffectData & operator=( const EffectData & rhs );
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          dp::fx::EffectSpecSharedPtr                       m_effectSpec;
          boost::scoped_array<ParameterGroupDataSharedPtr>  m_parameterGroupData;
          bool                                              m_transparent;
      };


      // Helper function to get some "standard" spec
      DP_SG_CORE_API const dp::fx::EffectSpecSharedPtr& getStandardGeometrySpec();
      DP_SG_CORE_API const dp::fx::EffectSpecSharedPtr& getStandardDirectedLightSpec();
      DP_SG_CORE_API const dp::fx::EffectSpecSharedPtr& getStandardPointLightSpec();
      DP_SG_CORE_API const dp::fx::EffectSpecSharedPtr& getStandardSpotLightSpec();
      DP_SG_CORE_API const dp::fx::EffectSpecSharedPtr& getStandardMaterialSpec();

      DP_SG_CORE_API EffectDataSharedPtr createStandardGeometryData();
      DP_SG_CORE_API EffectDataSharedPtr createStandardDirectedLightData( const dp::math::Vec3f & direction = dp::math::Vec3f( 0.0f, 0.0f, -1.0f )
                                                                        , const dp::math::Vec3f & ambient = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                        , const dp::math::Vec3f & diffuse = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                        , const dp::math::Vec3f & specular = dp::math::Vec3f( 1.0f, 1.0f, 1.0f ) );
      DP_SG_CORE_API EffectDataSharedPtr createStandardPointLightData( const dp::math::Vec3f & position = dp::math::Vec3f( 0.0f, 0.0f, 1.0f )
                                                                     , const dp::math::Vec3f & ambient = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                     , const dp::math::Vec3f & diffuse = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                     , const dp::math::Vec3f & specular = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                     , const boost::array<float,3> & attenuations = dp::util::makeArray( 1.0f, 0.0f, 0.0f ) );
      DP_SG_CORE_API EffectDataSharedPtr createStandardSpotLightData( const dp::math::Vec3f & position = dp::math::Vec3f( 0.0f, 0.0f, 1.0f )
                                                                    , const dp::math::Vec3f & direction = dp::math::Vec3f( 0.0f, 0.0f, -1.0f )
                                                                    , const dp::math::Vec3f & ambient = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                    , const dp::math::Vec3f & diffuse = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                    , const dp::math::Vec3f & specular = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                    , const boost::array<float,3> & attenuations = dp::util::makeArray( 1.0f, 0.0f, 0.0f )
                                                                    , float exponent = 0.0f
                                                                    , float cutoff = 45.0f );
      DP_SG_CORE_API EffectDataSharedPtr createStandardMaterialData( const dp::math::Vec3f & ambientColor = dp::math::Vec3f( 0.2f, 0.2f, 0.2f )
                                                                   , const dp::math::Vec3f & diffuseColor = dp::math::Vec3f( 0.8f, 0.8f, 0.8f )
                                                                   , const dp::math::Vec3f & specularColor = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                   , const float specularExponent = 1.0f
                                                                   , const dp::math::Vec3f & emissiveColor = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                   , const float opacity = 1.0f
                                                                   , const float reflectivity = 0.0f
                                                                   , const float indexOfRefraction = 1.0f );


      inline const dp::fx::EffectSpecSharedPtr & EffectData::getEffectSpec() const
      {
        return( m_effectSpec );
      }

      inline const ParameterGroupDataSharedPtr & EffectData::getParameterGroupData( dp::fx::EffectSpec::iterator const& it ) const
      {
        DP_ASSERT( it != m_effectSpec->endParameterGroupSpecs() );
        return( m_parameterGroupData[ std::distance( m_effectSpec->beginParameterGroupSpecs(), it ) ] );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
