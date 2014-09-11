// Copyright NVIDIA Corporation 2002-2011
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

#include <dp/sg/core/nvsgapi.h> // commonly used stuff
#include <dp/math/Trafo.h>
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/Node.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Abstract base class for all light sources.
       *  \par Namespace: dp::sg::core
       *  A LightSource as a Node can be the child of any Group or Group-derived Node. The position in the
       *  scene hierarchy determines the position in the world.\n
       *  To enable a LightSource for a specified Node (and, recursively, for all its children), you have to
       *  add a reference of the LightSource to the Node, using Node::addLightSource(). That way, both global
       *  and local light are possible. */
      class LightSource : public Node
      {
        public:
          DP_SG_CORE_API static LightSourceSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~LightSource();

        public:
          const EffectDataSharedPtr & getLightEffect() const;

          DP_SG_CORE_API void setLightEffect( const EffectDataSharedPtr & effect );

          /*! \brief Query if this LightSource is enabled.
           *  \return \c true, if this LightSource is enabled, otherwise \c false.
           *  \sa setEnabled */
          DP_SG_CORE_API bool isEnabled() const;

          /*! \brief Query if this LightSource casts shadows.
           *  \return \c true, if this LightSource casts shadows, otherwise \c false.
           *  \remarks A LightSource might be excluded from the shadow casters by this flag. The
           *  default value is \c true.
           *  \sa setShadowCasting */
          DP_SG_CORE_API bool isShadowCasting() const;

          /*! \brief Set the enabled state of this LightSource.
           *  \param enable \c true, to switch the LightSource on, \c false to switch it off.
           *  \sa isEnabled */
          DP_SG_CORE_API void setEnabled( bool enable );

          /*! \brief Set the shadow casting state of this LightSource.
           *  \param cast \c true, to switch shadow casting on, \c false to switch it off.
           *  \remarks A LightSource might be excluded from the shadow casters by this flag. The
           *  default value is \c true.
           *  \sa isShadowCasting */
          DP_SG_CORE_API void setShadowCasting( bool cast );

          /*! \brief Test for equivalence with another LightSource.
           *  \param p A pointer to the LightSource to test for equivalence with.
           *  \param ignoreNames Optional parameter to ignore the names of the objects; default is \c
           *  true.
           *  \param deepCompare Optional parameter to perform a deep comparsion; default is \c false.
           *  \return \c true if the LightSource \a p is equivalent to \c this, otherwise \c false.
           *  \remarks If \a p and \c this are equivalent as Node, their Animation objects are tested
           *  for equivalence. If \a deepCompare is true, and there is an Animation object in both \c
           *  this and \a p, a full equivalence test on them is performed, otherwise they are
           *  considered to be equivalent if they are the same pointers. All the other LightSource
           *  data are tested for equality.
           *  \note The behavior is undefined if \a p is not a LightSource nor derived from one. */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          REFLECTION_INFO_API( DP_SG_CORE_API, LightSource );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Enabled );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( ShadowCasting );
          END_DECLARE_STATIC_PROPERTIES 

        protected:
          /*! \brief Protected default constructor to prevent explicit creation.
           *  \remarks The default values of the newly created LightSource are as follows:\n
           *    - ambient color is black (0.0,0.0,0.0)\n
           *    - animation is none (NULL)\n
           *    - diffuse color is white (1.0,1.0,1.0)\n
           *    - intensity is one (1.0)\n
           *    - shadow casting is on\n
           *    - specular color is white (1.0,1.0,1.0)\n
           *    - transform is the identity */
          DP_SG_CORE_API LightSource();

          /*! \brief Protected copy constructor to prevent explicit creation.
           *  \param rhs A reference to the constant LightSource to copy from.
           *  \remarks The newly created LightSource is copied from \a rhs. */
          DP_SG_CORE_API LightSource( const LightSource& rhs );

          /*! \brief Assignment operator
           *  \param rhs A reference to the constant LightSource to copy from.
           *  \return A reference to the assigned LightSource.
           *  \remarks The assignment operator calls the assignment operator of Node. The reference
           *  count of any currently attached Animation is decremented, all data specific to a
           *  LightSource is copied, and any Animation attached to \a rhs is cloned. */
          DP_SG_CORE_API LightSource & operator=(const LightSource & rhs);

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          EffectDataSharedPtr     m_lightEffect;
          bool                    m_shadowCasting;
          bool                    m_enabled;
      };

      /*! Generate a directed light with the direction \a direction, the ambient color \a ambientColor, the diffuse
       *  color \a diffuseColor, a specular color of (0.0f,0.0f,0.0f), a specular exponent of 0.0f */ 
      DP_SG_CORE_API LightSourceSharedPtr createStandardDirectedLight( const dp::math::Vec3f & direction = dp::math::Vec3f( 0.0f, 0.0f, -1.0f )
                                                                     , const dp::math::Vec3f & ambient = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                     , const dp::math::Vec3f & diffuse = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                     , const dp::math::Vec3f & specular = dp::math::Vec3f( 1.0f, 1.0f, 1.0f ) );
      DP_SG_CORE_API bool isStandardDirectedLight( const LightSourceSharedPtr & lightSource );

      DP_SG_CORE_API LightSourceSharedPtr createStandardPointLight( const dp::math::Vec3f & position = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                  , const dp::math::Vec3f & ambient = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                  , const dp::math::Vec3f & diffuse = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                  , const dp::math::Vec3f & specular = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                  , const boost::array<float,3> & attenuations = dp::util::makeArray( 1.0f, 0.0f, 0.0f ) );
      DP_SG_CORE_API bool isStandardPointLight( const LightSourceSharedPtr & lightSource );

      DP_SG_CORE_API LightSourceSharedPtr createStandardSpotLight( const dp::math::Vec3f & position = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                 , const dp::math::Vec3f & direction = dp::math::Vec3f( 0.0f, 0.0f, -1.0f )
                                                                 , const dp::math::Vec3f & ambient = dp::math::Vec3f( 0.0f, 0.0f, 0.0f )
                                                                 , const dp::math::Vec3f & diffuse = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                 , const dp::math::Vec3f & specular = dp::math::Vec3f( 1.0f, 1.0f, 1.0f )
                                                                 , const boost::array<float,3> & attenuations = dp::util::makeArray( 1.0f, 0.0f, 0.0f )
                                                                 , float exponent = 0.0f
                                                                 , float cutoff = 45.0f );
      DP_SG_CORE_API bool isStandardSpotLight( const LightSourceSharedPtr & lightSource );


      inline bool LightSource::isEnabled() const
      {
        return( m_enabled );
      }

      inline bool LightSource::isShadowCasting() const
      {
        return( m_shadowCasting );
      }

      inline void LightSource::setEnabled( bool enable )
      {
        if ( m_enabled != enable )
        {
          m_enabled = enable;
          notify( PropertyEvent( this, PID_Enabled ) );
        }
      }

      inline void LightSource::setShadowCasting( bool cast )
      {
        if ( m_shadowCasting != cast )
        {
          m_shadowCasting = cast;
          notify( PropertyEvent( this, PID_ShadowCasting ) );
        }
      }

      inline const EffectDataSharedPtr & LightSource::getLightEffect() const
      {
        return( m_lightEffect );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
