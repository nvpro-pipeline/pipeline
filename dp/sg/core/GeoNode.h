// Copyright NVIDIA Corporation 2002-2007
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
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/Node.h> // base class definition

// additional dependencies
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/ConstIterator.h>
#include <vector>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Leaf Node to maintain geometry.
        *  \par Namespace: dp::sg::core
        *  \remarks
        *  A GeoNode represents a leaf Node containing a Primitive and a StateSet.
        *  \sa Node, Primitive, Scene, StateSet */
      class GeoNode : public Node
      {
        public:
          DP_SG_CORE_API static GeoNodeSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~GeoNode();

        public:
          class Event : public core::Event
          {
          public:
            enum Type
            {
                PRIMITIVE_CHANGED
              , EFFECT_DATA_CHANGED
            };

            Event( GeoNode const* geoNode, Type type)
              : core::Event( core::Event::GEONODE )
              , m_geoNode( geoNode )
              , m_type( type )
            {
            }

            GeoNode const* getGeoNode() const { return m_geoNode; }
            Type           getType() const { return m_type; }

          private:
            GeoNode const* m_geoNode;
            Type           m_type;
          };
  
        public:
          const EffectDataSharedPtr & getMaterialEffect() const;
          DP_SG_CORE_API void setMaterialEffect( const EffectDataSharedPtr & materialEffect );

          /*! \brief Get the Primitive of this GeoNode
            *  \return The Primitive of this GeoNode.
            *  \sa setPrimitive */
          const PrimitiveSharedPtr & getPrimitive() const;

          /*! \brief Set the Primitive of this GeoNode
            *  \param drawable The Primitive to set.
            *  \sa getPrimitive */
          DP_SG_CORE_API void setPrimitive( const PrimitiveSharedPtr & primitive );


          /*! \brief Generates vertex normals 
            *  \param overwrite An optional flag to indicate whether or not to overwrite existing vertex normals. 
            *  \return \c true, if normals could be generated, otherwise \c false.
            *  The default is to overwrite existing vertex normals. */
          DP_SG_CORE_API bool generateNormals( bool overwrite = true );

          /*! \brief Generates tangents and binormals for all geometries contained in the indicated GeoNode
            *  \param tc Addresses the vertex attribute to hold the input texture coordinates used to calculate the tangent space. 
            *  By default, input coordinates are taken from the VertexAttributeSet::NVSG_TEXCOORD0. 
            *  \param tg Addresses the vertex attribute where to output the calculated tangents. 
            *  By default tangents are written to VertexAttributeSet::NVSG_TANGENT, 
            *  which is aligned to the TANGENT binding semantic used by Cg for varying vertex shader input. 
            *  \param bn Addresses the vertex attribute to output the calculated binormals. 
            *  By default binormals are written to VertexAttributeSet::NVSG_BINORMAL, 
            *  which is aligned to the BINORMAL binding semantic used by Cg for varying vertex shader input. 
            *  \param overwrite An optional flag to indicate whether or not to overwrite existing vertex data in the output vertex
            *  attributes \a tg, and \a bn. The default is to overwrite existing vertex data.
            *  \remarks The function iterates through all geometries contained in the indicated GeoNode, and calculates
            *  tangents and binormals from the specified input 2D texture coordinates and vertex normals, 
            *  which are required to be defined already for the contained geometries. The calculated tangents
            *  and binormals are written to the specified output vertex attributes. If the specified output vertex 
            *  attributes already contain data, this data gets lost, if the \a overwrite flag is set. 
            *  If the \a overwrite flag is not set, tangents and binormals are only written to the indicated output
            *  vertex attributes, if these are empty at the time of calling. */  
          DP_SG_CORE_API void generateTangentSpace( unsigned int tc = VertexAttributeSet::NVSG_TEXCOORD0
                                            , unsigned int tg = VertexAttributeSet::NVSG_TANGENT
                                            , unsigned int bn = VertexAttributeSet::NVSG_BINORMAL
                                            , bool overwrite = true );

          /*! \brief Generates 2D texture coordinates 
            *  \param type Desired texture coordinate type. Accepted are TCT_CYLINDRICAL, TCT_PLANAR, and TCT_SPHERICAL. 
            *  \param texcoords Addresses the vertex attribute where to output the generated texture coords. 
            *  VertexAttributeSet::NVSG_TEXCOORD0 - VertexAttributeSet::NVSG_TEXCOORD7 are allowed identifiers.
            *  By default texture coords are written to VertexAttributeSet::NVSG_TEXCOORD0, 
            *  \param overwrite An optional flag indicating whether or not to overwrite existing vertex data in the output
            *  vertex attribute \a tc. The default is to overwrite existing vertex data. */
          DP_SG_CORE_API void generateTexCoords( TextureCoordType type
                                          , unsigned int texcoords = VertexAttributeSet::NVSG_TEXCOORD0
                                          , bool overwrite = true );

          /*! \brief Clear texture coordinates on the Primitive. */
          DP_SG_CORE_API void clearTexCoords( unsigned int tu = 0 );

          /*! \brief Test for equivalence with an other GeoNode.
            *  \param p            A reference to the constant GeoNode to test for equivalence with.
            *  \param ignoreNames  Optional parameter to ignore the names of the objects; default is \c
            *  true.
            *  \param deepCompare  Optional parameter to perform a deep comparison; default is \c false.
            *  \return \c true if the GeoNode \a p is equivalent to \c this, otherwise \c false.
            *  \remarks If \a p and \c this are equivalent as Node, and both hold a Primitive and/or a StateSet,
            *  those are tested for equivalence. If \a deepCompare is false, they are considered to be equivalent if
            *  they are the same pointers. Otherwise a full equivalence test is performed.
            *  \note The behavior is undefined if \a p is not a GeoNode nor derived from one.
            *  \sa Node */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          /*! \brief Assignment operator
            *  \param rhs A reference to the constant GeoNode to copy from
            *  \return A reference to the assigned GeoNode
            *  \remarks The assignment operator calls the assignment operator of Node and copies the Primitive and the
            *  StateSet.
            *  \sa Node, Primitive, StateSet */
          DP_SG_CORE_API GeoNode & operator=(const GeoNode & rhs);

          REFLECTION_INFO_API( DP_SG_CORE_API, GeoNode );
        protected:
          /*! \brief Default-constructs a GeoNode.
            */
          DP_SG_CORE_API GeoNode();

          /*! \brief Constructs a GeoNode as a copy of another GeoNode.
            */
          DP_SG_CORE_API GeoNode( const GeoNode& rhs );
 
          /*! \brief Interface to calculate the bounding box of this GeoNode.
            *  \return The bounding box of this GeoNode
            *  \remarks This function is called by the framework to determine the
            *  actual bounding box of this GeoNode.
            *  \sa invalidateBoundingVolumes */
          DP_SG_CORE_API virtual dp::math::Box3f calculateBoundingBox() const;

          /*! \brief Interface to calculate the bounding sphere of this GeoNode.
            *  \return A sphere that contains the complete GeoNode.
            *  \remarks This function is called by the framework to determine a sphere that completely
            *  contains the GeoNode.
            *  \sa invalidateBoundingVolumes */
          DP_SG_CORE_API virtual dp::math::Sphere3f calculateBoundingSphere() const;

          /*! \brief Feed the data of this object into the provied HashGenerator.
            *  \param hg The HashGenerator to update with the data of this object.
            *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          EffectDataSharedPtr m_materialEffect;
          PrimitiveSharedPtr  m_primitive;
      };

      // - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlines
      // - - - - - - - - - - - - - - - - - - - - - - - - - 

      inline const EffectDataSharedPtr & GeoNode::getMaterialEffect() const
      {
        return( m_materialEffect );
      }

      inline const PrimitiveSharedPtr & GeoNode::getPrimitive() const
      {
        return( m_primitive );
      }

      inline dp::math::Box3f GeoNode::calculateBoundingBox() const
      {
        return( m_primitive ? m_primitive->getBoundingBox() : dp::math::Box3f() );
      }

      inline dp::math::Sphere3f GeoNode::calculateBoundingSphere() const
      {
        return( m_primitive ? m_primitive->getBoundingSphere() : dp::math::Sphere3f() );
      }

      inline bool GeoNode::generateNormals( bool overwrite )
      {
        return( m_primitive ? m_primitive->generateNormals( overwrite ) : false );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

