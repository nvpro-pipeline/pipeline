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
/** \file */

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/Traverser.h>
#include <dp/fx/EffectSpec.h>

#include <map>
#include <set>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /*! \brief Base class of the analyzing results determined with the AnalyzeTraverser.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks The AnalyzeResult holds the \a objectCode of the object that this result refers to,
       *  and the number of objects that have been encountered in the scene graph.
       *  \sa AnalyzeTraverser, CombinableResult, DenormalizedNormalsResult, EmptyResult,
       *  EquivalenceResult, IdentityResult, MissingResult, NullNormalsResult, RedundantIndexSetResult,
       *  SingleChildResult, ShortStrippedResult, UnusedVerticesResult */
      class AnalyzeResult
      {
        public:
          AnalyzeResult() : objectCount(0) {}

          /*! \brief Virtual Destructor */
          virtual ~AnalyzeResult() {}

        public:
          dp::sg::core::ObjectCode objectCode;   //!< The code of objects, this result refers to.
          unsigned int             objectCount;  //!< The number of objects of this type.
      };

      /*! \brief AnalyzeResult indicating that a number of Objects can be combined.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks Reports if, and how many, combinable Objects are in the tree. It tells how many
       *  objects of the type \a objectCode are combinable, and by how many objects they could be
       *  replaced with. Currently, the following objects are checked for combinability: dp::sg::core::GeoNode,
       *  and dp::sg::core::IndexSet.
       *  \sa AnalyzeResult AnalyzeTraverser */
      class CombinableResult : public AnalyzeResult
      {
        public:
          virtual ~CombinableResult() {};

        public:
          unsigned int  combinableCount;  //!< The number of combinable objects of this type.
          unsigned int  reducedCount;     //!< The number of objects of this type, if combinable objects would be combined.
      };

      /*! \brief AnalyzeResult indicating a VertexAttributeSet with denormalized normals.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks Reports if there is a VertexAttributeSet with denormalized normals.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class DenormalizedNormalsResult : public AnalyzeResult
      {
        public:
          virtual ~DenormalizedNormalsResult() {}

        public:
          unsigned int  denormalizedNormalsCount;   //!< The number of objects with denormalized normals.
          unsigned int  denormalizedNormalsNumber;  //!< The number of denormalized normals in those objects.
      };

      /*! \brief AnalyzeResult indicating an empty object.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks All empty object in the tree (e.g. Groups without children, ...) are reported with an EmptyResult.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class EmptyResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~EmptyResult() {}

        public:
          unsigned int  emptyCount;   //!< The number of empty objects of this type.
      };

      /*! \brief AnalyzeResult indicating equivalent objects.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks All equivalent objects in the tree (e.g. StateAttributes with the same settings)
       *  are reported with an EquivalenceResult.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class EquivalenceResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~EquivalenceResult() {}

        public:
          unsigned int  equivalentCount;  //!< The number of equivalent objects of this type.
          unsigned int  reducedCount;     //!< The number of objects of this type, if equivalent objects would be combined.
      };

      /*! \brief AnalyzeResult indicating identity transformations.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks All Transform objects in the tree that are in fact identity transforms are reported
       *  with an IdentityResult.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class IdentityResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~IdentityResult() {}

        public:
          unsigned int  identityCount;  //!< The number of identity Transforms.
      };

      /*! \brief AnalyzeResult indicating non-complete objects.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks All Objects with missing information (e.g. an AnimatedTransform without an
       *  Animation) are reported with a MissingResult.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class MissingResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~MissingResult()  {}

        public:
          unsigned int  missingCount;   //!< The number of objects with missing information.
      };

      /*! \brief AnalyzeResult indicating a VertexAttributeSet with null normals.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks Reports if there is a VertexAttributeSet with null normals.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class NullNormalsResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~NullNormalsResult() {}

        public:
          unsigned int  nullNormalsCount;   //!< The number of objects with null normals.
          unsigned int  nullNormalsNumber;  //!< The number of null normals in those objects.
      };

      /*! \brief AnalyzeResult indicating redundant IndexSets
       *  \par Namespace: dp::sg::algorithm
       *  \remarks An IndexSet that holds all indices between the smallest and the largest just once are reported with
       *  a RedundantIndexSetResult.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class RedundantIndexSetResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~RedundantIndexSetResult() {}

        public:
          unsigned int redundantIndexSetCount;    //!< The number of redundant index sets.
      };

      /*! \brief AnalyzeResult indicating LODs with redundant ranges
       *  \par Namespace: dp::sg::algorithm
       *  \remarks Two ranges of an LOD are redundant, if they are the same.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class RedundantLODRangesResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~RedundantLODRangesResult() {}

        public:
          unsigned int  lodRanges;          //!< The number of LOD ranges.
          unsigned int  redundantLODs;      //!< The number of LODs with redundant ranges.
          unsigned int  redundantLODRanges; //!< The number of redundant LOD ranges.
      };

      /*! \brief AnalyzeResult indicating a Group with a single child.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks All Group (and Group-derived) objects with only one child are reported with a
       *  SingleChildResult.
       *  \sa AnalyzeResult, AnalyzeTraverser */
      class SingleChildResult : public AnalyzeResult
      {
        public:
          /*! \brief Virtual Destructor */
          virtual ~SingleChildResult()  {}

        public:
          unsigned int  singleChildCount;   //!< The number of objects with a single child.
      };

      /*! \brief AnalyzeResult indicating a Primitive of type PRIMITIVE_QUAD_STRIPS, PRIMITIVE_TRIANGLE_FANS, or
       *  PRIMITIVE_TRIANGLE_STRIPS with very short strips.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks All QuadStrips with strips of length up to four, and all TriFans and TriStrips
       *  with fans or strips of length up to three are reported.
       *  \sa AnalyzeResult, AnalyzeTraverser, Primitive */
      class ShortStrippedResult : public AnalyzeResult
      {
        public:
          virtual ~ShortStrippedResult() {}

        public:
          unsigned int  shortStrippedCount;    //!< The number of objects with short strips.
          unsigned int  shortStrippedNumber;   //!< The number of short strips in those objects.
      };

      /*! \brief AnalyzeResult indicating unused vertices.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks: All VertexAttributeSets holding unused vertices are reported.
       *  \sa AnalyzeResult, AnalyzeTraverser, VertexAttributeSet */
      class UnusedVerticesResult : public AnalyzeResult
      {
        public:
          UnusedVerticesResult()
            : AnalyzeResult()
            , unusedVerticesCount(0)
          {
            objectCode = dp::sg::core::OC_VERTEX_ATTRIBUTE_SET;
          }
          virtual ~UnusedVerticesResult() {}

        public:
          unsigned int  unusedVerticesCount;    //!< The number of unused vertices.
      };

      /*! \brief Traverser that analyzes a tree and reports about potential deficiencies.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks The AnalyzeTraverser is a scene graph analyzing tool. It can give you hints on
       *  potential problems in your scene graph.
       *  \par Example
       *  To get the results of the AnalyzeTraverser on a given Scene, do something like that:
       *  \code
       *    AnalyzeTraverser analyzeTraverser;
       *    analyzeTraverser.apply( pScene );
       *    std::vector<AnalyzeResult *> results;
       *    unsigned int numberOfResults = analyzeTraverser.getAnalysis( results );
       *    for ( unsigned int i=0 ; i<numberOfResults ; i++ )
       *    {
       *      if ( dynamic_cast<EmptyResult *>( results[i] ) != NULL )
       *      {
       *        EmptyResult * emptyResult = dynamic_cast<EmptyResult *>( results[i] );
       *        // handle empty results
       *      }
       *      else if ( dynamic_cast<EquivalenceResult *>( results[i] ) != NULL )
       *      {
       *        EquivalenceResult * equivalenceResult = dynamic_cast<EquivalenceResult *>( results[i] );
       *        // handle equivalence results
       *      }
       *      else if ( ... )
       *      // handle other known results
       *      // ...
       *      else
       *      {
       *        // handle any unknown AnalyzeResult
       *      }
       *      // delete the result after interpretation
       *      delete results[i];
       *    }
       *  \endcode
       *  \sa AnalyzeResult, Traverser */
      class AnalyzeTraverser : public SharedTraverser
      {
        public:
          /*! \brief Default Constructor */
          DP_SG_ALGORITHM_API AnalyzeTraverser( void );

          /*! \brief Destructor */
          DP_SG_ALGORITHM_API virtual ~AnalyzeTraverser( void );

          /*! \brief Get the analysis results of the latest traversal.
           *  \param results A reference to the vector pointers to an AnalyzeResult object to fill.
           *  \return The number of AnalyzeResult objects generated.
           *  \remarks On return, each element of the vector \a results holds a pointer to an
           *  AnalyzeResult. Those objects are owned by the requester, then. There can be multiple
           *  AnalyzeResult objects of the same type, but referring to different types of objects in
           *  the Scene.
           *  \sa AnalyzeResult */
          DP_SG_ALGORITHM_API unsigned int getAnalysis( std::vector<AnalyzeResult*> &results ) const;

        protected:
          /*! \brief Override of the doApply method.
           *  \param root A pointer to the read-locked root node of the tree to analyze.
           *  \remarks The doApply method is the traversal entry point of a Traverser. The local data
           *  is cleared and Traverser::doApply() is called to start traversal.
           *  \sa dp::sg::core::Scene, Traverser */
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Analyze a dp::sg::core::Billboard.
           *  \param p A pointer to the read-locked dp::sg::core::Billboard to analyze.
           *  \remarks A dp::sg::core::Billboard is tested for emptiness (no children), and equivalence with
           *  any previously encountered dp::sg::core::Billboard.
           *  \sa dp::sg::core::Billboard, EmptyResult, EquivalenceResult */
          DP_SG_ALGORITHM_API virtual void handleBillboard( const dp::sg::core::Billboard * p );

          DP_SG_ALGORITHM_API virtual void handleEffectData( const dp::sg::core::EffectData * p );

          /*! \brief Analyze a dp::sg::core::GeoNode.
           *  \param p A pointer to the read-locked dp::sg::core::GeoNode to analyze.
           *  \remarks A dp::sg::core::GeoNode is tested for emptiness (no geometries) and for equivalence
           *  with any previously encountered dp::sg::core::GeoNode.
           *  \sa dp::sg::core::GeoNode, EmptyResult, EquivalenceResult */
          DP_SG_ALGORITHM_API virtual void handleGeoNode( const dp::sg::core::GeoNode * p );

          /*! \brief Analyze a dp::sg::core::Group.
           *  \param p A pointer to the read-locked dp::sg::core::Group to analyze.
           *  \remarks A dp::sg::core::Group is tested for emptiness (no children), for equivalence with any
           *  previously encountered dp::sg::core::Group, and for holding a single child only.
           *  \sa dp::sg::core::Group, EmptyResult, EquivalenceResult, SingleChildResult */
          DP_SG_ALGORITHM_API virtual void handleGroup( const dp::sg::core::Group * p );

          /*! \brief Analyze an dp::sg::core::IndexSet.
           *  \param p A pointer to the read-locked IndexSet being traversed.
           *  \remarks An IndexSet is tested for emptiness and for equivalence with any previously encountered IndexSet .
           *  \sa dp::sg::core::EmptyResult, EquivalenceResult */
          DP_SG_ALGORITHM_API virtual void handleIndexSet( const dp::sg::core::IndexSet * p );

          /*! \brief Analyze a dp::sg::core::LOD.
           *  \param p A pointer to the read-locked dp::sg::core::LOD to analyze.
           *  \remarks A dp::sg::core::LOD is tested for emptiness (no children), for equivalence with any
           *  previously encountered dp::sg::core::LOD, and for holding a single child only.
           *  \sa dp::sg::core::LOD, EmptyResult, EquivalenceResult, RedundantLODRangesResult, SingleChildResult */
          DP_SG_ALGORITHM_API virtual void handleLOD( const dp::sg::core::LOD * p );

          DP_SG_ALGORITHM_API virtual void handleParameterGroupData( const dp::sg::core::ParameterGroupData * p );

          DP_SG_ALGORITHM_API virtual void handleSampler( const dp::sg::core::Sampler * p );

          /*! \brief Analyze a dp::sg::core::Switch.
           *  \param p A pointer to the read-locked dp::sg::core::Switch to analyze.
           *  \remarks A dp::sg::core::Switch is tested for emptiness (no children) and for equivalence with
           *  any previously encountered dp::sg::core::Switch.
           *  \sa dp::sg::core::Switch, EmptyResult, EquivalenceResult */
          DP_SG_ALGORITHM_API virtual void handleSwitch( const dp::sg::core::Switch * p );

          /*! \brief Analyze a dp::sg::core::Transform.
           *  \param p A pointer to the read-locked dp::sg::core::Transform to analyze.
           *  \remarks A dp::sg::core::Transform is tested for emptiness (no children), for equivalence with
           *  any previously encountered dp::sg::core::Transform, and for being the identity transform.
           *  \sa dp::sg::core::Transform, EmptyResult, EquivalenceResult, IdentityResult */
          DP_SG_ALGORITHM_API virtual void handleTransform( const dp::sg::core::Transform * p );

          /*! \brief Analyze a dp::sg::core::VertexAttributeSet.
           *  \param p A pointer to the read-locked dp::sg::core::VertexAttributeSet to analyze.
           *  \remarks A dp::sg::core::VertexAttributeSet is tested for emptiness (no vertices), for
           *  equivalence with any previously encountered dp::sg::core::VertexAttributeSet, and for holding
           *  denormalized normals.
           *  \sa dp::sg::core::VertexAttributeSet, EmptyResult, EquivalenceResult, DenormalizedNormalsResult,
           *  NullNormalsResult */
          DP_SG_ALGORITHM_API virtual void handleVertexAttributeSet( const dp::sg::core::VertexAttributeSet * p );

          /*! \brief Analyze an dp::sg::core::Primitive.
           *  \param p A poitner to the read-locked dp::sg::core::Primitive to analyze.
           *  \remarks An dp::sg::core::Primitive is tested for combinability, emptiness, equivalence containing a
           *  redundant dp::sg::core::IndexSet, and holding a VertexAttributeSet with unused vertices.
           *  \sa CombinableResult, EmptyResult, EquivalenceResult, RedundantIndexSetResult, UnusedVerticesResult */
          DP_SG_ALGORITHM_API virtual void traversePrimitive( const dp::sg::core::Primitive * p );

        private:
          class EquivalenceInfo
          {
            public :
              std::vector<dp::sg::core::ObjectSharedPtr>  uniqueObjects;
              std::set<dp::sg::core::ObjectSharedPtr>     equivalentObjects;
          };

          class EffectSpecEquivalenceInfo
          {
            public:
              std::vector<dp::fx::EffectSpecSharedPtr>  uniqueSpecs;
              std::set<dp::fx::EffectSpecSharedPtr>     equivalentSpecs;
          };

          class ParameterGroupSpecEquivalenceInfo
          {
            public:
              std::vector<dp::fx::ParameterGroupSpecSharedPtr>  uniqueSpecs;
              std::set<dp::fx::ParameterGroupSpecSharedPtr>     equivalentSpecs;
          };

          typedef std::map<dp::sg::core::VertexAttributeSetSharedPtr,std::set<unsigned int> > VertexUseMap;

        private:
          void analyzeCombinable( const dp::sg::core::Group * p );
          void analyzeEmpty( const dp::sg::core::Object * p, unsigned int numberOfElements );
          void analyzeEquivalent( const dp::sg::core::Object *p );
          void analyzeEquivalentEffectSpec( const dp::fx::EffectSpecSharedPtr & p );
          void analyzeEquivalentParameterGroupSpec( const dp::fx::ParameterGroupSpecSharedPtr & p );
          void analyzeNormalsNormalized( dp::sg::core::Buffer::ConstIterator<dp::math::Vec3f>::Type normals, unsigned int non
                                       , unsigned int &nullNormals, unsigned int &denormalizedNormals );
          void analyzeMissing( const dp::sg::core::Object * p, dp::sg::core::TextureSharedPtr const& ptr );
          void analyzeRedundantIndexSet( const dp::sg::core::IndexSetSharedPtr & p, unsigned int offset, unsigned int count );
          void analyzeRedundantLODRanges( const dp::sg::core::LOD * p );
          void analyzeSingleChild( const dp::sg::core::Group * p );
          void analyzeUnusedVertices( const dp::sg::core::IndexSetSharedPtr & isSP
                                    , const dp::sg::core::VertexAttributeSetSharedPtr & vasSP
                                    , unsigned int offset, unsigned int count );
          bool isChildOfCurrentLOD( const dp::sg::core::Object * p );
          bool isToBeHandled( const dp::sg::core::Object *p );
          void testVertexAttributeSet( const dp::sg::core::VertexAttributeSet * p );

        private:
          std::map<unsigned int,unsigned int>                                       m_countMap;
          std::stack<std::vector<const dp::sg::core::GeoNode*> >                    m_combinableGeoNodes;
          std::map<dp::sg::core::ObjectCode,std::pair<unsigned int,unsigned int> >  m_combinableInfo;
          const dp::sg::core::LOD *                                                 m_currentLOD;
          std::pair<unsigned int,unsigned int>                                      m_denormalizedNormalsVAS;
          std::map<dp::sg::core::ObjectCode, unsigned int>                          m_emptyMap;
          EffectSpecEquivalenceInfo                                                 m_effectSpecEquivalenceInfo;
          std::map<dp::sg::core::ObjectCode, EquivalenceInfo>                       m_equivalenceMap;
          unsigned int                                                              m_identityCount;
          unsigned int                                                              m_lodRanges;
          std::map<dp::sg::core::ObjectCode, unsigned int>                          m_missingMap;
          std::pair<unsigned int,unsigned int>                                      m_nullNormalsLIVAAD;
          std::pair<unsigned int,unsigned int>                                      m_nullNormalsVAS;
          ParameterGroupSpecEquivalenceInfo                                         m_parameterGroupSpecEquivalenceInfo;
          unsigned int                                                              m_redundantIndexSets;
          unsigned int                                                              m_redundantLODs;
          unsigned int                                                              m_redundantLODRanges;
          std::set<const dp::sg::core::Object *>                                    m_sharedObjects;
          std::map<dp::sg::core::ObjectCode, std::pair<unsigned int,unsigned int> > m_shortStripped;
          std::map<dp::sg::core::ObjectCode, unsigned int>                          m_singleChildMap;
          VertexUseMap                                                              m_vertexUseMap;
      };

    } // namespace algorithm
  } // namespace sp
} // namespace dp

  //! Output an analysis summary. 
DP_SG_ALGORITHM_API std::ostream& operator<<( std::ostream& os, const dp::sg::algorithm::AnalyzeTraverser& obj );
