// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/OptimizeTraverser.h>
#include <dp/sg/core/Primitive.h>
#include <dp/util/Flags.h>

#include <list>
#include <vector>
#include <utility>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /*! \brief OptimizeTraverser that can unify identical objects in a scene.
       *  \remarks The UnifyTraverser can unify \link dp::sg::core::GeoNode GeoNodes, \endlink \link dp::sg::core::Group
       *  Groups, \endlink \link dp::sg::core::LOD LODs, \endlink \link dp::sg::core::StateAttribute StateAttributes,
       *  \endlink \link dp::sg::core::StateSet StateSets, \link dp::sg::core::VertexAttributeSet VertexAttributeSet, \endlink and
       *  vertices. Unifying in this context means, if there are two separate but identical objects in the scene, each
       *  reference to one of them is replaced by a reference of the other. That way, after unifying, there will be no
       *  separate but identical objects in the scene.
       *  For vertices, unification is performed within \link dp::sg::core::Primitive Primitives. \endlink\n
       *  The types of objects to unify can be selected by \link UnifyTraverser::setUnifyTargets
       *  setUnifyTargets. \endlink By default, each object type listed above is unified.\n
       *  The accepted epsilon used in comparing the components of vertices can be set by
       *  \link UnifyTraverser::setEpsilon setEpsilon. \endlink By default, epsilon is FLT_EPSILON.\n
       *  As with every OptimizeTraverser, identical objects with different names can be considered to
       *  be equal. This can be set with \link OptimizeTraverser::setIgnoreNames setIgnoreNames. \endlink
       *  By default, this is set to \c true.\n
       *  After unifying a scene, the UnifyTraverser can be queried if the latest traversal did modify
       *  the scene, using \link OptimizeTraverser::getTreeModified getTreeModified. \endlink
       *  \sa OptimizeTraverser */
      class UnifyTraverser : public OptimizeTraverser
      {
        public:
          enum class Target
          {
            GEONODE               = BIT0,   //!< UnifyTarget dp::sg::core::GeoNode: unify identical GeoNode objects into one.
            GROUP                 = BIT1,   //!< UnifyTarget dp::sg::core::Group: unify identical Group objects into one.
            INDEX_SET             = BIT2,   //!< UnifyTarget dp::sg::core::IndexSet: unify identical IndexSet objects into one.
            LOD                   = BIT3,   //!< UnifyTarget dp::sg::core::LOD: unify identical LOD objects into one.
            PARAMETER_GROUP_DATA  = BIT4,   //!< UnifyTarget dp::sg::core::ParameterGroupData: unify identical ParameterGroupData objects into one.
            PIPELINE_DATA         = BIT5,   //!< UnifyTarget dp::sg::core::PipelineData: unify identical EffectData objects into one.
            PRIMITIVE             = BIT6,   //!< UnifyTarget dp::sg::core::Primitive: unify identical Primitive objects into one.
            SAMPLER               = BIT7,   //!< UnifyTarget dp::sg::core::Sampler: unify identical Sampler objects into one.
            TEXTURE               = BIT8,  //!< UnifyTarget dp::sg::core::Texture: unify identical Texture objects into one.
            TRAFO_ANIMATION       = BIT9,  //!< UnifyTarget dp::sg::core::Animation<dp::math::Trafo>: unify identical Animations on Trafo into one.
            VERTEX_ATTRIBUTE_SET  = BIT10,  //!< UnifyTarget dp::sg::core::VertexAttributeSet: unify identical VertexAttributeSet objects into one.
            VERTICES              = BIT11,  //!< UnifyTarget Vertices: unify identical Vertices (with an epsilon) into one.
            ALL                   = ( GEONODE | GROUP | INDEX_SET | LOD | PARAMETER_GROUP_DATA | PIPELINE_DATA | PRIMITIVE
                                     | SAMPLER | TEXTURE | TRAFO_ANIMATION | VERTEX_ATTRIBUTE_SET | VERTICES )
          };                                 //!< Enum to specify the object types to unify.

          typedef dp::util::Flags<Target> TargetMask;

        public:
          /*! \brief Default constructor of an UnifyTraverser.
           *  \remarks Creates a UnifyTraverser with an epsilon for vertex components comparison of
           *  FLT_EPSILON, and the targets to unify set to all.
           *  \sa setEpsilon, setUnifyTargets */
          DP_SG_ALGORITHM_API UnifyTraverser( void );

          /*! \brief Destructor */
          DP_SG_ALGORITHM_API virtual ~UnifyTraverser( void );

          /*! \brief Get the bitmask describing the targets to unify.
           *  \return A bitmask describing the targets to unify. */
          DP_SG_ALGORITHM_API TargetMask getUnifyTargets() const;

          /*! \brief Set the bitmask describing the targets to unify.
           *  \param mask The bitmask describing the targets to unify. */
          DP_SG_ALGORITHM_API void setUnifyTargets( TargetMask mask );

          /*! \brief Get the epsilon used for compares in vertex unification.
           *  \return The epsilon used on component of a vertex to determine equality. */
          DP_SG_ALGORITHM_API float getEpsilon() const;

          /*! \brief Set the epsilon used for compares in vertex unification.
           *  \param eps The epsilon used on component of a vertex to determine equality.
           *  \note The unification of vertices undefined, if eps is not positive. */
          DP_SG_ALGORITHM_API void setEpsilon( float eps );

          REFLECTION_INFO_API( DP_SG_ALGORITHM_API, UnifyTraverser );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( UnifyTargets );
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( Epsilon );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          /*! \brief Overload of the \link OptimizeTraverser::doApply doApply \endlink method.
           *  \remarks After scene traversal, temporarily allocated storage is freed again. */
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Overload of the \link ExclusiveTraverser::handleBillboard Billboard \endlink method.
           *  \param billboard A pointer to the write-locked \link dp::sg::core::Billboard Billboard \endlink to handle.
           *  \remarks After traversal of the Billboard, identical children are unified. */
          DP_SG_ALGORITHM_API virtual void handleBillboard( dp::sg::core::Billboard * billboard );

          /*! \brief Overload of the \link ExclusiveTraverser::handleGroup Group \endlink method.
           *  \param group A pointer to the write-locked \link dp::sg::core::Group Group \endlink to handle.
           *  \remarks After traversal of the Group, identical children are unified. */
          DP_SG_ALGORITHM_API virtual void handleGroup( dp::sg::core::Group * group );

          /*! \brief Overload of the \link ExclusiveTraverser::handleLOD LOD \endlink method.
           *  \param lod A pointer to the write-locked \link dp::sg::core::LOD LOD \endlink to handle.
           *  \remarks After traversal of the LOD as a Group, identical children are unified. */
          DP_SG_ALGORITHM_API virtual void handleLOD( dp::sg::core::LOD * lod );

          /*! \brief Overload of the \link ExclusiveTraverser::handleSwitch Switch \endlink method.
           *  \param swtch A pointer to the write-locked \link dp::sg::core::Switch Switch \endlink to handle.
           *  \remarks After traversal of the Switch as a Group, identical children are unified. */
          DP_SG_ALGORITHM_API virtual void handleSwitch( dp::sg::core::Switch * swtch );

          /*! \brief Overload of the \link ExclusiveTraverser::handleTransform Transform \endlink method.
           *  \param transform A pointer to the write-locked \link dp::sg::core::Transform Transform \endlink to handle.
           *  \remarks After traversal of the Transform, identical children are unified. */
          DP_SG_ALGORITHM_API virtual void handleTransform( dp::sg::core::Transform * transform );


          /*! \brief Overload of the \link ExclusiveTraverser::handleGeoNode handleGeoNode \endlink method.
           *  \param gnode A pointer to the write-locked \link dp::sg::core::GeoNode GeoNode \endlink to handle.
           *  \remarks After traversing \a gnode, it's \link dp::sg::core::StateSet StateSets \endlink are
           *  unified, if requested. */
          DP_SG_ALGORITHM_API virtual void handleGeoNode( dp::sg::core::GeoNode * gnode );

          /*! \brief Overload of the \link ExclusiveTraverser::handlePrimitive handlePrimitive \endlink method.
           *  \param primitive A pointer to the write-locked \link dp::sg::core::Primitive Primitive \endlink to handle.
           *  \remarks After traversing \a primitive, it is unified if unification of \link
           *  dp::sg::core::Primitive Primitive \endlink is requested. */
          DP_SG_ALGORITHM_API virtual void handlePrimitive( dp::sg::core::Primitive * primitive );

          /*! \brief Overload of the \link ExclusiveTraverser::handleVertexAttributeSet handleVertexAttributeSet \endlink method.
           *  \param vas A pointer to the write-locked \link dp::sg::core::VertexAttributeSet VertexAttributeSet
           *  \endlink to handle.
           *  \remarks After traversing \a vas, the vertices inside are unified if unification of
           *  vertices is requested. */
          DP_SG_ALGORITHM_API virtual void handleVertexAttributeSet( dp::sg::core::VertexAttributeSet * vas );

          DP_SG_ALGORITHM_API virtual void handleLightSource( dp::sg::core::LightSource * p );

          DP_SG_ALGORITHM_API virtual void handleParameterGroupData( dp::sg::core::ParameterGroupData * p );

          DP_SG_ALGORITHM_API virtual void handlePipelineData( dp::sg::core::PipelineData * p );

          DP_SG_ALGORITHM_API virtual void handleSampler( dp::sg::core::Sampler * p );

        private:
          void checkPrimitive( std::multimap<dp::util::HashKey,dp::sg::core::PrimitiveSharedPtr> & v, dp::sg::core::Primitive * p );
          void unifyChildren( dp::sg::core::Group *p );
          void unifyGeoNodes( dp::sg::core::Group *p );
          void unifyGroups( dp::sg::core::Group *p );
          void unifyIndexSets( dp::sg::core::Primitive *p );
          void unifyLODs( dp::sg::core::Group *p );
          const dp::sg::core::PipelineDataSharedPtr & unifyPipelineData( const dp::sg::core::PipelineDataSharedPtr & pipelineData );
          void unifyStateSet( dp::sg::core::GeoNode *p );
          void unifyVertexAttributeSet( dp::sg::core::Primitive *p );

        private:
          // map an old VAS to a new one and the corresponding mapping of indices
          struct VASReplacement
          {
            VASReplacement()
            {}

            VASReplacement( dp::sg::core::VertexAttributeSetSharedPtr const& vas, std::vector<unsigned int> const& indexMap )
              : m_vas(vas)
              , m_indexMap(indexMap)
            {}

            dp::sg::core::VertexAttributeSetSharedPtr m_vas;
            std::vector<unsigned int>                 m_indexMap;
          };
          typedef std::map<dp::sg::core::VertexAttributeSetSharedPtr,VASReplacement> VASReplacementMap;

        private:
          float                                                                       m_epsilon;
          std::multimap<dp::util::HashKey,dp::sg::core::GeoNodeSharedPtr>             m_geoNodes;
          std::multimap<dp::util::HashKey,dp::sg::core::GroupSharedPtr>               m_groups;
          std::multimap<dp::util::HashKey,dp::sg::core::IndexSetSharedPtr>            m_indexSets;
          std::vector<dp::sg::core::LODSharedPtr>                                     m_LODs;
          std::set<const void *>                                                      m_objects;      //!< A set of pointers to hold all objects already encountered.
          std::multimap<dp::util::HashKey,dp::sg::core::ParameterGroupDataSharedPtr>  m_parameterGroupData;
          std::multimap<dp::util::HashKey,dp::sg::core::PipelineDataSharedPtr>        m_pipelineData;
          std::map<dp::sg::core::PrimitiveType,std::multimap<dp::util::HashKey,dp::sg::core::PrimitiveSharedPtr> >  m_primitives;
          dp::sg::core::PrimitiveSharedPtr                                            m_replacementPrimitive;
          std::multimap<dp::util::HashKey,dp::sg::core::SamplerSharedPtr>             m_samplers;
          std::multimap<dp::util::HashKey,dp::sg::core::TextureSharedPtr>             m_textures;
          TargetMask                                                                  m_unifyTargets;
          VASReplacementMap                                                           m_vasReplacements;
          std::multimap<dp::util::HashKey,dp::sg::core::VertexAttributeSetSharedPtr>  m_vertexAttributeSets;
      };

      inline UnifyTraverser::TargetMask UnifyTraverser::getUnifyTargets() const
      {
        return( m_unifyTargets );
      }

      inline void UnifyTraverser::setUnifyTargets( TargetMask mask )
      {
        if ( mask != m_unifyTargets )
        {
          m_unifyTargets = mask;
          notify( PropertyEvent( this, PID_UnifyTargets ) );
        }
      }

      inline float UnifyTraverser::getEpsilon() const
      {
        return( m_epsilon );
      }

      inline void UnifyTraverser::setEpsilon( float eps )
      {
        DP_ASSERT( 0 < eps );
        if ( m_epsilon != eps )
        {
          m_epsilon = eps;
          notify( PropertyEvent( this, PID_Epsilon ) );
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp


namespace dp
{
  namespace util
  {
    /*! \brief Specialization of the TypedPropertyEnum template for type UnifyTraverser::TargetMask. */
    template <> struct TypedPropertyEnum<dp::sg::algorithm::UnifyTraverser::TargetMask>
    {
      enum { type = Property::Type::UINT };
    };
  }
}
