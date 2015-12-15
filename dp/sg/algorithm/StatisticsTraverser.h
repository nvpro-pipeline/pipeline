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


#pragma once
/** \file */

#include <map>
#include <ostream>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <dp/fx/EffectSpec.h>
#include <dp/math/Trafo.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/algorithm/Traverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /*! \brief Base class of all statistics classes. */
      class StatisticsBase
      {
        public:
          StatisticsBase()  : m_count(0), m_referenced(0), m_instanced(0)  {}

        public:
          /*! \brief Checks if an object is encountered for the first time in the current traversal.
           *  \param p The pointer to the object.
           *  \return \c true, if this object has not been encountered before in the current traversal, otherwise \c false. */
          bool  firstEncounter( const void *p, bool inInstance );

        public:
          unsigned int                        m_count;        //!< Counts the number of occurences of objects of a specific type.
          unsigned int                        m_instanced;    //!< Counts the number of instances of objects of a specific type.
          std::set<const void *>              m_objects;      //!< A set of pointers to hold all objects already encountered.
          unsigned int                        m_referenced;   //!< Counts the number of references of objects of a specific type.
      };

      /*! \brief Class to hold statistics information about Object objects. */
      class StatObject : public StatisticsBase
      {
        public:
          StatObject() : m_assembled(0) {}

        public:
          unsigned int                        m_assembled;
      };

      /*! \brief Class to hold statistics information about Camera objects. */
      class StatCamera : public StatObject
      {
      };

      /*! \brief Class to hold statistics information about MatrixCamera objects. */
      class StatMatrixCamera : public StatCamera
      {
      };

      /*! \brief Class to hold statistics information about FrustumCamera objects. */
      class StatFrustumCamera : public StatCamera
      {
      };

      /*! \brief Class to hold statistics information about ParallelCamera objects. */
      class StatParallelCamera : public StatFrustumCamera
      {
      };

      /*! \brief Class to hold statistics information about PerspectiveCamera objects. */
      class StatPerspectiveCamera : public StatFrustumCamera
      {
      };

      /*! \brief Class to hold statistics information about IndexSet objects. */
      class StatIndexSet : public StatObject
      {
        public:
          std::map<dp::DataType, unsigned int> m_dataTypes;
          std::map<unsigned int, unsigned int> m_primitiveRestartIndices;
          std::map<unsigned int, unsigned int> m_numberOfIndices;
      };

      /*! \brief Class to hold statistics information about Primitive objects. */
      class StatPrimitive : public StatObject
      {
        public:
          StatPrimitive()
            : m_indexed(0), m_arrays(0), m_patches(0), m_lineStripAdj(0) , m_linesAdj(0) , m_lineSegments(0), m_triStripAdj(0)
            , m_trisAdj(0) , m_polygons(0) , m_lineLoops(0) , m_quads(0) , m_quadStrips(0) , m_tris(0) , m_triStrips(0)
            , m_triFans(0) , m_lines(0) , m_lineStrips(0) , m_points(0) , m_faces(0)
            , m_pointsPrimitives(0), m_lineStripPrimitives(0), m_linesPrimitives(0), m_lineLoopPrimitives(0), m_linesAdjacencyPrimitives(0)
            , m_lineStripAdjacencyPrimitives(0), m_triangleStripPrimitives(0), m_triangleFanPrimitives(0), m_trianglesPrimitives(0)
            , m_quadStripPrimitives(0), m_quadsPrimitives(0), m_polygonPrimitives(0), m_trianglesAdjacencyPrimitives(0)
            , m_triangleStripAdjacencyPrimitives(0), m_patchesPrimitives(0)
          {}

        public:
          unsigned int m_indexed;         //!< Counts the number of Primitive objects with indices.
          unsigned int m_arrays;          //!< Counts the number of Primitive objects without indices.
          unsigned int m_patches;         //!< Counts the number of Primitive objects of type PrimitiveType::PATCHES
          unsigned int m_lineStripAdj;    //!< Counts the number of Primitive objects of type PrimitiveType::LINE_STRIP_ADJACENCY
          unsigned int m_linesAdj;        //!< Counts the number of Primitive objects of type PrimitiveType::LINES_ADJACENCY
          unsigned int m_lineSegments;    //!< Counts the number of lines segments in all encountered Primitive objects.
          unsigned int m_triStripAdj;     //!< Counts the number of Primitive objects of type PrimitiveType::TRIANGLE_STRIP_ADJACENCY
          unsigned int m_trisAdj;         //!< Counts the number of Primitive objects of type PrimitiveType::TRIANGLES_ADJACENCY
          unsigned int m_polygons;        //!< Counts the number of Primitive objects of type PrimitiveType::POLYGON
          unsigned int m_lineLoops;       //!< Counts the number of Primitive objects of type PrimitiveType::LINE_LOOP
          unsigned int m_quads;           //!< Counts the number of Primitive objects of type PrimitiveType::QUADS
          unsigned int m_quadStrips;      //!< Counts the number of Primitive objects of type PrimitiveType::QUAD_STRIP
          unsigned int m_tris;            //!< Counts the number of Primitive objects of type PrimitiveType::TRIANGLES
          unsigned int m_triStrips;       //!< Counts the number of Primitive objects of type PrimitiveType::TRIANGLE_STRIP
          unsigned int m_triFans;         //!< Counts the number of Primitive objects of type PrimitiveType::TRIANGLE_FAN
          unsigned int m_lines;           //!< Counts the number of Primitive objects of type PrimitiveType::LINES
          unsigned int m_lineStrips;      //!< Counts the number of Primitive objects of type PrimitiveType::STRIPS
          unsigned int m_points;          //!< Counts the number of Primitive objects of type PrimitiveType::POINTS
          unsigned int m_faces;           //!< Counts the nubmer of faces in all encountered Primitive objects.
          unsigned int m_pointsPrimitives;
          unsigned int m_lineStripPrimitives;
          unsigned int m_linesPrimitives;
          unsigned int m_lineLoopPrimitives;
          unsigned int m_linesAdjacencyPrimitives;
          unsigned int m_lineStripAdjacencyPrimitives;
          unsigned int m_triangleStripPrimitives;
          unsigned int m_triangleFanPrimitives;
          unsigned int m_trianglesPrimitives;
          unsigned int m_quadStripPrimitives;
          unsigned int m_quadsPrimitives;
          unsigned int m_polygonPrimitives;
          unsigned int m_trianglesAdjacencyPrimitives;
          unsigned int m_triangleStripAdjacencyPrimitives;
          unsigned int m_patchesPrimitives;
      };

      /*! \brief Class to hold statistics information about VertexAttributeSet objects. */
      class StatVertexAttributeSet : public StatObject
      {
        public:
          StatVertexAttributeSet()  : m_numberOfVertices(0), m_numberOfNormaled(0), m_numberOfNormals(0), m_numberOfTextured(0)
                                    , m_numberOfTextureUnits(0), m_numberOfTextureDimensions(0), m_numberOfColored(0)
                                    , m_numberOfColors(0), m_numberOfSecondaryColored(0), m_numberOfSecondaryColors(0)
                                    , m_numberOfFogged(0), m_numberOfFogCoords(0)
          {
            for ( int i=0 ; i<8 ; i++ )
            {
              m_numberOfTextures[i] = 0;
            }
          }

        public:
          unsigned int  m_numberOfVertices;           //!< Counts the number of vertices.
          unsigned int  m_numberOfNormaled;           //!< Counts the number of VertexAttributeSet objects with normals.
          unsigned int  m_numberOfNormals;            //!< Counts the number of normals.
          unsigned int  m_numberOfTextured;           //!< Counts the number of VertexAttributeSet objects with texture coordinates.
          unsigned int  m_numberOfTextureUnits;       //!< Counts the number of used texture units.
          unsigned int  m_numberOfTextureDimensions;  //!< Counts the number of texture dimensions.
          unsigned int  m_numberOfTextures[8];        //!< Counts the number of texture coordinates per texture unit.
          unsigned int  m_numberOfColored;            //!< Counts the number of VertexAttributeSet objects with colors.
          unsigned int  m_numberOfColors;             //!< Counts the number of colors.
          unsigned int  m_numberOfSecondaryColored;   //!< Counts the number of VertexAttributeSet objects with secondary colors.
          unsigned int  m_numberOfSecondaryColors;    //!< Counts the number of secondary colors.
          unsigned int  m_numberOfFogged;             //!< Counts the number of VertexAttributeSet objects with fog coordinates.
          unsigned int  m_numberOfFogCoords;          //!< Counts the number of fog coordinates.
      };

      class StatSampler : public StatObject
      {
        public:
          StatSampler() : m_numberSamplerStates(0), m_numberTextures(0) {}

        public:
          unsigned int m_numberSamplerStates;
          unsigned int m_numberTextures;
      };

      /*! \brief Class to hold statistics information about Node objects. */
      class StatNode : public StatObject
      {
      };

      /*! \brief Class to hold statistics information about Group objects. */
      class StatGroup : public StatNode
      {
        public:
          StatGroup() : m_numberOfChildren(0) {}

        public:
          unsigned int                        m_numberOfChildren;   //!< Counts the number of children.
          std::map<unsigned int,unsigned int> m_childrenHistogram;  //!< Counts the number of Group objects per number of children.
      };

      /*! \brief Class to hold statistics information about LOD objects. */
      class StatLOD : public StatGroup
      {
      };

      /*! \brief Class to hold statistics information about Billboard objects. */
      class StatBillboard : public StatGroup
      {
      };

      /*! \brief Class to hold statistics information about Transform objects. */
      class StatTransform : public StatGroup
      {
      };

      /*! \brief Class to hold statistics information about Switch objects. */
      class StatSwitch : public StatGroup
      {
      };

      /*! \brief Class to hold statistics information about LightSource objects. */
      class StatLightSource : public StatNode
      {
      };

      /*! \brief Class to hold statistics information about GeoNode objects. */
      class StatGeoNode : public StatNode
      {
      };

      /*! \brief Class to hold statistics information about Texture objects. */
      class StatTexture : public StatisticsBase
      {
        public:
          StatTexture() : m_numImages(0), m_sumOfSizes(0) {}

        public:
          unsigned int                m_numImages;    //!< Counts the number of images.
          std::map<int,unsigned int>  m_widths;       //!< Counts the number of Texture objects per width.
          std::map<int,unsigned int>  m_heights;      //!< Counts the number of Texture objects per height.
          std::map<int,unsigned int>  m_depths;       //!< Counts the number of Texture objects per depth.
          std::map<int,unsigned int>  m_sizes;        //!< Counts the number of Texture objects per size.
          unsigned int                m_sumOfSizes;   //!< Sum of all the sizes.
      };

      class StatEffectData : public StatObject
      {
        public:
          std::map<dp::fx::EffectSpec::Type,unsigned int> m_effectSpecTypeHistogram;          //!< Counts the number of EffectData per EffectSpec::Type
          std::map<unsigned int,unsigned int>             m_parameterGroupDataSizeHistogram;  //!< Counts the number of EffectData per ParameterGroupData
      };

      class StatParameterGroupData : public StatObject
      {
        public:
          std::map<unsigned int,unsigned int> m_dataSizeHistogram;        //!< Counts the number of ParameterGroupData per data size
          std::map<unsigned int,unsigned int> m_numParameterHistogram;    //!< Counts the number of ParameterGroupData per number of Parameters
      };

      /*! \brief Class to hold statistics information about all objects. */
      class Statistics
      {
        public:
          StatBillboard           m_statBillboard;                    //!< Statistics for Billboard objects.
          StatGeoNode             m_statGeoNode;                      //!< Statistics for GeoNode objects.
          StatGroup               m_statGroup;                        //!< Statistics for Group objects.
          StatIndexSet            m_statIndexSet;                     //!< Statistics for IndexSet objects.
          StatLOD                 m_statLOD;                          //!< Statistics for LOD objects.
          StatMatrixCamera        m_statMatrixCamera;                 //!< Statistics for MatrixCamera objects.
          StatParallelCamera      m_statParallelCamera;               //!< Statistics for ParallelCamera objects.
          StatParameterGroupData  m_statParameterGroupData;           //!< Statistics for ParameterGroupData objects.
          StatPerspectiveCamera   m_statPerspectiveCamera;            //!< Statistics for PerspectiveCamera objects.
          StatEffectData          m_statPipelineData;                 //!< Statistics for PipelineData objects.
          StatPrimitive           m_statPrimitives;                   //!< Statistics for Primitives objects.
          StatPrimitive           m_statPrimitiveInstances;           //!< Statistics for Primitive instances.
          StatSampler             m_statSampler;                      //!< Statistics for Sampler objects.
          StatSwitch              m_statSwitch;                       //!< Statistics for Switch objects.
          StatTexture             m_statTexture;                      //!< Statistics for Texture objects.
          StatTransform           m_statTransform;                    //!< Statistics for Transform objects.
          StatVertexAttributeSet  m_statVertexAttributeSet;           //!< Statistics for VertexAttributeSet objects.
          StatVertexAttributeSet  m_statVertexAttributeSetInstances;  //!< Statistics for VertexAttributeSet instances.
      };

      //! Traverser to record some statistics of a scene.
      class StatisticsTraverser : public SharedTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API StatisticsTraverser(void);

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~StatisticsTraverser(void);

          //! Get a constant pointer to the statistics results.
          DP_SG_ALGORITHM_API const Statistics  * getStatistics( void ) const;

          //! Record statistics of a Camera.
          /** Just records the statistics of an Object object.  */
          DP_SG_ALGORITHM_API void  statCamera( const dp::sg::core::Camera *p, StatCamera &stats );

          //! Record statistics of a FrustumCamera.
          /** Just records the statistics of an Object object.  */
          DP_SG_ALGORITHM_API void  statFrustumCamera( const dp::sg::core::FrustumCamera *p, StatFrustumCamera &stats );

          //! Record statistics of a Group.
          /** Records the number of children and traverses them. Then the statistics of a Node are recorded.  */
          DP_SG_ALGORITHM_API void  statGroup( const dp::sg::core::Group *p, StatGroup &stats );

          //! Record statistics of a LightSource.
          /** Just records the statistics of a Node.  */
          DP_SG_ALGORITHM_API void  statLightSource( const dp::sg::core::LightSource *p, StatLightSource &stats );

          //! Record statistics of a Node.
          /** Just records the statistics of an Object. */
          DP_SG_ALGORITHM_API void  statNode( const dp::sg::core::Node *p, StatNode &stats );

          //! Record statistics of an Object.
          /** Does nothing. */
          DP_SG_ALGORITHM_API void  statObject( const dp::sg::core::Object *p, StatObject &stats );

          //! Record statistics of a Primitive.
          /** Records the statistics of the VertexAttributeSet. Then the statistics of an Object are recorded. */
          DP_SG_ALGORITHM_API void  statPrimitive( const dp::sg::core::Primitive *p, StatPrimitive &stats );

          //! Record statistics of a Transform.
          /** Just records the statistics of a Group. */
          DP_SG_ALGORITHM_API void statTransform( const dp::sg::core::Transform *p, StatTransform &stats );

          //! Record statistics of a VertexAttributeSet.
          /** Records the number of vertices, the number of Primitive objects with normals, the number of
            * normals, the number of textured Primitive objects, the number of used texture units, the
            * total dimension of the textures, the total number of texture coordinates, the number of colored
            * Primitive object, the total number of colors, the number of secondary colored Primitive
            * objects, the total number of secondary colors, the number of fogged Primitive objects, and
            * the total number of fog coordinates. Then the statistics of an Object are recorded. */
          DP_SG_ALGORITHM_API void statVertexAttributeSet( const dp::sg::core::VertexAttributeSet *p, StatVertexAttributeSet &stats );

        protected:
#if !defined(NDEBUG)
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );
#endif

          //--  Functions implemented from Traverser --
          //! Handle a Billboard object.
          /** The number of Billboard objects are count. Then the statistics of a Group is recorded.  */
          DP_SG_ALGORITHM_API virtual void  handleBillboard( const dp::sg::core::Billboard *p     //!<  Billboard to handle
                                       );

          //! Handle a GeoNode object.
          /** The number of GeoNode objects and the total number of geometries are count. After having traversed the geometries,
            * the statistics of a Node is recorded. */
          DP_SG_ALGORITHM_API virtual void  handleGeoNode( const dp::sg::core::GeoNode *p            //!<  GeoNode to handle
                                            );

          //! Handle a Group object.
          /** The number of Group objects are count. */
          DP_SG_ALGORITHM_API virtual void  handleGroup( const dp::sg::core::Group *p                 //!<  Group to handle
                                   );

          //! Handle an IndexSet object.
          /** A histogram  */
          DP_SG_ALGORITHM_API virtual void handleIndexSet( const dp::sg::core::IndexSet * p );

          //! Handle a LOD object.
          /** The number of LOD objects is counted. Then the statistics of a Group is recorded.  */
          DP_SG_ALGORITHM_API virtual void  handleLOD( const dp::sg::core::LOD *p                //!<  LOD to save
                                 );

          //! Handle a MatrixCamera object.
          /** The number of MatrixCamera objects is counted. Then the statistics of a Camera is recoreded.  */
          DP_SG_ALGORITHM_API virtual void  handleMatrixCamera( const dp::sg::core::MatrixCamera *p   //!<  MatrixCamera to handle
                                            );

          //! Handle a ParallelCamera object.
          /** The number of ParallelCamera objects is counted. Then the statistics of a Camera is recoreded.  */
          DP_SG_ALGORITHM_API virtual void  handleParallelCamera( const dp::sg::core::ParallelCamera *p     //!<  ParallelCamera to handle
                                            );

          DP_SG_ALGORITHM_API virtual void handleParameterGroupData( const dp::sg::core::ParameterGroupData * p );

          //! Handle a PerspectiveCamera object.
          /** The number of PerspectiveCamera objects is counted. Then the statistics of a Camera is recoreded.  */
          DP_SG_ALGORITHM_API virtual void  handlePerspectiveCamera( const dp::sg::core::PerspectiveCamera *p   //!<  PerspectiveCamera to handle
                                            );

          DP_SG_ALGORITHM_API virtual void handlePipelineData( const dp::sg::core::PipelineData * p );

          //! Handle a Primitive object.
          /** The number of Primitive objects are counted. */
          DP_SG_ALGORITHM_API virtual void  handlePrimitive( const dp::sg::core::Primitive *p               //!<  Primitive to handle
                                    );

          DP_SG_ALGORITHM_API virtual void handleSampler( const dp::sg::core::Sampler * p );

          //! Handle a Switch object.
          /** The number of Switch objects is count. Then the statistics of a Group is recoreded. */
          DP_SG_ALGORITHM_API virtual void  handleSwitch( const dp::sg::core::Switch *p             //!<  Switch to handle
                                            );

          //! Handle a Transform object.
          /** The number of Transform objects are count. Then the statistics of a Group is recorded.  */
          DP_SG_ALGORITHM_API virtual void  handleTransform( const dp::sg::core::Transform *p          //!<  Transform to handle
                                            );

          DP_SG_ALGORITHM_API virtual void  handleVertexAttributeSet( const dp::sg::core::VertexAttributeSet *vas );

        private:
          unsigned int  m_instanceCount;
          Statistics  * m_statistics;
      };

      //! Output a statistics summary.
      DP_SG_ALGORITHM_API std::ostream& operator<<( std::ostream& os, const StatisticsTraverser& obj );

      inline const Statistics * StatisticsTraverser::getStatistics( void ) const
      {
        return( m_statistics );
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
