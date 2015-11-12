// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/sg/core/LightSource.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( LightSource, Enabled );
      DEFINE_STATIC_PROPERTY( LightSource, ShadowCasting );

      BEGIN_REFLECTION_INFO( LightSource )
        DERIVE_STATIC_PROPERTIES( LightSource, Node );
        INIT_STATIC_PROPERTY_RW_BOOL( LightSource, Enabled      , bool         , SEMANTIC_VALUE, value          , value );
        INIT_STATIC_PROPERTY_RW_BOOL( LightSource, ShadowCasting, bool         , SEMANTIC_VALUE, value          , value );
      END_REFLECTION_INFO
      /* FIXME MISSING
        * Animation
        * TransformationMatrix
        * Inverse
        **/

      LightSourceSharedPtr LightSource::create()
      {
        return( std::shared_ptr<LightSource>( new LightSource() ) );
      }

      HandledObjectSharedPtr LightSource::clone() const
      {
        return( std::shared_ptr<LightSource>( new LightSource( *this ) ) );
      }

      LightSource::LightSource()
      : Node()
      , m_shadowCasting(true)
      , m_enabled(true)
      {
        m_objectCode = OC_LIGHT_SOURCE;
      }

      LightSource::LightSource(const LightSource& rhs)
      : Node(rhs) // copy base class part
      , m_shadowCasting( rhs.m_shadowCasting)
      , m_enabled(rhs.m_enabled)
      {
        m_objectCode = OC_LIGHT_SOURCE;

        if ( rhs.m_lightPipeline )
        {
          m_lightPipeline = rhs.m_lightPipeline.clone();
          m_lightPipeline->attach( this );
        }
      }

      LightSource::~LightSource()
      {
        if ( m_lightPipeline )
        {
          m_lightPipeline->detach( this );
        }
      }

      void LightSource::setLightPipeline( const PipelineDataSharedPtr & effect )
      {
        if ( m_lightPipeline != effect )
        {
          if ( m_lightPipeline )
          {
            m_lightPipeline->detach( this );
          }
          m_lightPipeline = effect;
          if ( m_lightPipeline )
          {
            m_lightPipeline->attach( this );
          }
          notify( Event(this ) );
        }
      }


      LightSource & LightSource::operator=(const LightSource & rhs)
      {
        if (&rhs != this)
        {
          Node::operator=(rhs);

          if ( m_lightPipeline )
          {
            m_lightPipeline->detach( this );
            m_lightPipeline.reset();
          }
          if ( rhs.m_lightPipeline )
          {
            m_lightPipeline = rhs.m_lightPipeline.clone();
            m_lightPipeline->attach( this );
          }

          m_shadowCasting = rhs.m_shadowCasting;
          m_enabled       = rhs.m_enabled;

          notify( PropertyEvent( this, PID_Enabled ) );
          notify( PropertyEvent( this, PID_ShadowCasting ) );

        }
        return *this;
      }

      bool LightSource::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<LightSource>() && Node::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          LightSourceSharedPtr const& ls = object.staticCast<LightSource>();

          equi =    ( m_shadowCasting   == ls->m_shadowCasting )
                &&  ( m_enabled         == ls->m_enabled       )
                &&  ( !!m_lightPipeline == !!ls->m_lightPipeline );
          if ( equi )
          {
            if ( deepCompare )
            {
              if ( equi && m_lightPipeline )
              {
                equi = m_lightPipeline->isEquivalent( ls->m_lightPipeline, ignoreNames, true );
              }
            }
            else
            {
               equi = ( m_lightPipeline == ls->m_lightPipeline );
            }
          }
        }
        return( equi );
      }

      void LightSource::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Node::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_shadowCasting), sizeof(m_shadowCasting) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_enabled), sizeof(m_enabled) );
        if ( m_lightPipeline )
        {
          hg.update( m_lightPipeline );
        }
      }


      LightSourceSharedPtr createStandardDirectedLight( const Vec3f & direction, const Vec3f & ambient
                                                      , const Vec3f & diffuse, const Vec3f & specular )
      {
        Vec3f dir = direction;
        dir.normalize();

        dp::sg::core::PipelineDataSharedPtr lightPipeline = createStandardDirectedLightData( dir, ambient, diffuse, specular );
        LightSourceSharedPtr lightSource = LightSource::create();
        lightSource->setLightPipeline( lightPipeline );
        return( lightSource );
      }

      bool isStandardDirectedLight( const LightSourceSharedPtr & lightSource )
      {
        dp::sg::core::PipelineDataSharedPtr lightPipeline = lightSource->getLightPipeline();
        DP_ASSERT( lightPipeline );
        return( !!lightPipeline->findParameterGroupData( std::string( "standardDirectedLightParameters" ) ) );
      }

      LightSourceSharedPtr createStandardPointLight( const Vec3f & position, const Vec3f & ambient
                                                   , const Vec3f & diffuse, const Vec3f & specular
                                                   , const std::array<float,3> & attenuations )
      {
        dp::sg::core::PipelineDataSharedPtr lightPipeline = createStandardPointLightData( position, ambient, diffuse, specular, attenuations );
        LightSourceSharedPtr lightSource = LightSource::create();
        lightSource->setLightPipeline( lightPipeline );
        return( lightSource );
      }

      bool isStandardPointLight( const LightSourceSharedPtr & lightSource )
      {
        dp::sg::core::PipelineDataSharedPtr lightPipeline = lightSource->getLightPipeline();
        DP_ASSERT( lightPipeline );
        return( !!lightPipeline->findParameterGroupData( std::string( "standardPointLightParameters" ) ) );
      }

      LightSourceSharedPtr createStandardSpotLight( const Vec3f & position, const Vec3f & direction
                                                  , const Vec3f & ambient, const Vec3f & diffuse
                                                  , const Vec3f & specular, const std::array<float,3> & attenuations
                                                  , float exponent, float cutoff )
      {
        Vec3f dir = direction;
        dir.normalize();

        dp::sg::core::PipelineDataSharedPtr lightPipeline = createStandardSpotLightData( position, dir, ambient, diffuse, specular, attenuations, exponent, cutoff );
        LightSourceSharedPtr lightSource = LightSource::create();
        lightSource->setLightPipeline( lightPipeline );
        return( lightSource );
      }

      bool isStandardSpotLight( const LightSourceSharedPtr & lightSource )
      {
        dp::sg::core::PipelineDataSharedPtr lightPipeline = lightSource->getLightPipeline();
        DP_ASSERT( lightPipeline );
        return( !!lightPipeline->findParameterGroupData( std::string( "standardSpotLightParameters" ) ) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

