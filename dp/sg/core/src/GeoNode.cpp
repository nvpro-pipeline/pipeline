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


#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( GeoNode )
        DERIVE_STATIC_PROPERTIES( GeoNode, Node );
      END_REFLECTION_INFO

      GeoNodeSharedPtr GeoNode::create()
      {
        return( std::shared_ptr<GeoNode>( new GeoNode() ) );
      }

      HandledObjectSharedPtr GeoNode::clone() const
      {
        return( std::shared_ptr<GeoNode>( new GeoNode( *this ) ) );
      }

      GeoNode::GeoNode()
      {
        m_objectCode = OC_GEONODE; 
      }

      GeoNode::GeoNode( const GeoNode& rhs )
      : Node(rhs) // copy the base class part
      {
        m_objectCode = OC_GEONODE;

        if ( rhs.m_materialEffect )
        {
          m_materialEffect = rhs.m_materialEffect.clone();
          m_materialEffect->addOwner( this );
        }
        if ( rhs.m_primitive )
        {
          m_primitive = rhs.m_primitive.clone();
          m_primitive->addOwner( this );
        }
      }

      GeoNode::~GeoNode(void)
      {
        if ( m_materialEffect )
        {
          m_materialEffect->removeOwner( this );
        }
        if ( m_primitive )
        {
          m_primitive->removeOwner( this );
        }
      }

      void GeoNode::setMaterialEffect( const EffectDataSharedPtr & materialEffect )
      {
        DP_ASSERT( !materialEffect || ( materialEffect->getEffectSpec()->getType() == fx::EffectSpec::EST_PIPELINE ) );

        if ( m_materialEffect != materialEffect )
        {
          if ( m_materialEffect )
          {
            m_materialEffect->removeOwner( this );
          }
          m_materialEffect = materialEffect;
          if ( m_materialEffect )
          {
            m_materialEffect->addOwner( this );
          }

          notify( Event( this, Event::EFFECT_DATA_CHANGED ) );
        }
      }

      void GeoNode::setPrimitive( const PrimitiveSharedPtr & primitive )
      {
        if ( m_primitive != primitive )
        {
          if ( m_primitive )
          {
            m_primitive->removeOwner( this );
          }
          m_primitive = primitive;
          if ( m_primitive )
          {
            m_primitive->addOwner( this );
          }
          // GeoNode-Change
          markDirty( NVSG_BOUNDING_VOLUMES );
          notify( Event( this, Event::PRIMITIVE_CHANGED) );
        }
      }

      void GeoNode::clearTexCoords( unsigned int tu )
      {
        if ( m_primitive && m_primitive->getVertexAttributeSet() )
        {
          VertexAttributeSetSharedPtr const& vas = m_primitive->getVertexAttributeSet();
          vas->setEnabled( VertexAttributeSet::NVSG_TEXCOORD0 + tu, false );    // disable attribute!
          vas->removeVertexData( VertexAttributeSet::NVSG_TEXCOORD0 + tu );
          // Note: index cache stays valid, so no need to dismiss them
        }
      }

      GeoNode & GeoNode::operator=(const GeoNode & rhs)
      {
        if (&rhs != this)
        {
          Node::operator=(rhs);
          unsigned int dirtyBits = 0;
          if ( m_materialEffect != rhs.m_materialEffect )
          {
            if ( m_materialEffect )
            {
              m_materialEffect->removeOwner( this );
            }
            m_materialEffect = rhs.m_materialEffect.clone();
            if ( m_materialEffect )
            {
              m_materialEffect->addOwner( this );
            }
            notify( Event( this, Event::EFFECT_DATA_CHANGED ) );
          }
          if ( m_primitive != rhs.m_primitive )
          {
            if ( m_primitive )
            {
              m_primitive->removeOwner( this );
            }
            m_primitive = rhs.m_primitive.clone();
            if ( m_primitive )
            {
              m_primitive->addOwner( this );
            }
            markDirty( NVSG_BOUNDING_VOLUMES );
            notify( Event( this, Event::PRIMITIVE_CHANGED) );
          }
        }
        return *this;
      }

      bool GeoNode::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<GeoNode>() && Node::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          GeoNodeSharedPtr const& gn = object.staticCast<GeoNode>();
          if ( deepCompare )
          {
            equi = ( !!m_materialEffect == !!gn->m_materialEffect ) && ( !!m_primitive == !!gn->m_primitive );
            if ( equi && m_materialEffect )
            {
              equi = m_materialEffect->isEquivalent( gn->m_materialEffect, ignoreNames, true );
            }
            if ( equi && m_primitive )
            {
              equi = m_primitive->isEquivalent( gn->m_primitive, ignoreNames, true );
            }
          }
          else
          {
            equi = ( m_materialEffect == gn->m_materialEffect ) && ( m_primitive == gn->m_primitive );
          }
        }
        return( equi );
      }

      void GeoNode::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Node::feedHashGenerator( hg );
        if ( m_materialEffect )
        {
          hg.update( reinterpret_cast<const unsigned char *>(m_materialEffect.getWeakPtr()), sizeof(const EffectData *) );
        }
        if ( m_primitive )
        {
          hg.update( reinterpret_cast<const unsigned char *>(m_primitive.getWeakPtr()), sizeof(const Primitive *) );
        }
      }

      void GeoNode::generateTangentSpace( unsigned int tc, unsigned int tg, unsigned int bn, bool overwrite )
      {
        if ( m_primitive )
        {
          m_primitive->generateTangentSpace( tc, tg, bn, overwrite );
        }
      }

      void GeoNode::generateTexCoords( TextureCoordType tct, unsigned int tc, bool overwrite )
      {
        if ( m_primitive )
        {
          m_primitive->generateTexCoords( tct, tc, overwrite );
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp
