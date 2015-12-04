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


#include <dp/math/Trafo.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::pair;
using std::map;

std::basic_ostream<char,std::char_traits<char>> & operator<<( std::basic_ostream<char,std::char_traits<char>> & ss, dp::DataType const dt )
{
  switch( dt )
  {
    case dp::DataType::UNSIGNED_INT_8 :
      ss << "UNSIGNED_INT_8";
      break;
    case dp::DataType::UNSIGNED_INT_16 :
      ss << "UNSIGNED_INT_16";
      break;
    case dp::DataType::UNSIGNED_INT_32 :
      ss << "UNSIGNED_INT_32";
      break;
    case dp::DataType::UNSIGNED_INT_64 :
      ss << "UNSIGNED_INT_64";
      break;
    case dp::DataType::INT_8 :
      ss << "INT_8";
      break;
    case dp::DataType::INT_16 :
      ss << "INT_16";
      break;
    case dp::DataType::INT_32 :
      ss << "INT_32";
      break;
    case dp::DataType::INT_64 :
      ss << "INT_64";
      break;
    case dp::DataType::FLOAT_16 :
      ss << "FLOAT_16";
      break;
    case dp::DataType::FLOAT_32 :
      ss << "FLOAT_32";
      break;
    case dp::DataType::FLOAT_64 :
      ss << "FLOAT_64";
      break;
    case dp::DataType::UNKNOWN :
      ss << "unknown";
      break;
    case dp::DataType::NUM_DATATYPES :
    default :
      DP_ASSERT( false );
  }
  return( ss );
}

template <class T1, class T2>
std::string toString( const T1& name, const T2& value, size_t depth=4, size_t nameWidth=20, size_t valueWidth=10 )
{
  std::stringstream ss;
  ss.setf(std::ios::adjustfield, std::ios::left);
  for( size_t i = 0; i < depth; ++i )
  {
    ss << " ";
  }
  ss << std::setw(nameWidth) << name << ":";
  ss.setf(std::ios::adjustfield, std::ios::right);
  ss << std::setw(valueWidth) << value << "\n";
  return ss.str();
}

template <class T1, class T2, class T3>
std::string histogramToString( const T1& name, const std::map<T2,T3>& histogram )
{
  std::string s;
  s += toString(name, "");
  typename std::map<T2,T3>::const_iterator it, it_end = histogram.end();
  for( it = histogram.begin(); it != it_end; ++it )
  {
    s += toString(it->first, it->second, 8, 16, 10);
  }
  return s;
}

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

    bool  StatisticsBase::firstEncounter( const void *p, bool inInstance )
    {
      bool ok = false;
      if ( inInstance )
      {
        DP_ASSERT( m_objects.find( p ) != m_objects.end() );
        m_instanced++;
      }
      else
      {
        m_referenced++;
        pair<std::set<const void *>::iterator,bool> result = m_objects.insert( p );
        ok = result.second;
        if ( ok )
        {
          m_count++;
        }
      }
      return( ok );
    }

    StatisticsTraverser::StatisticsTraverser(void)
    : m_instanceCount(0)
    , m_statistics(new Statistics)
    {
    }

    StatisticsTraverser::~StatisticsTraverser(void)
    {
      if ( m_statistics )
      {
        delete m_statistics;
      }
    }

    void  StatisticsTraverser::statCamera( const Camera *p, StatCamera &stats )
    {
      statObject( (const Object *)p, stats );
    }

    void  StatisticsTraverser::statFrustumCamera( const FrustumCamera *p, StatFrustumCamera &stats )
    {
      statCamera( (const Camera *)p, stats );
    }

    void  StatisticsTraverser::statPrimitive( const Primitive *p, StatPrimitive &stats )
    {
      statObject( (const Object *)p, stats );

      if( p->isIndexed() )
      {
        stats.m_indexed++;
      }
      else
      {
        stats.m_arrays++;
      }

      unsigned int pcount = p->getNumberOfPrimitives();
      unsigned int fcount = p->getNumberOfFaces();

      switch( p->getPrimitiveType() )
      {
        case PRIMITIVE_POINTS:
          stats.m_points += pcount;
          stats.m_pointsPrimitives++;
          break;
        case PRIMITIVE_LINE_STRIP:
          stats.m_lineStrips += pcount;
          stats.m_lineSegments += p->getElementCount() - pcount - p->getNumberOfPrimitiveRestarts();
          stats.m_lineStripPrimitives++;
          break;
        case PRIMITIVE_LINES:
          stats.m_lines += pcount;
          stats.m_lineSegments += pcount;
          stats.m_linesPrimitives++;
          break;
        case PRIMITIVE_LINE_LOOP:
          stats.m_lineLoops += pcount;
          stats.m_lineSegments += p->getElementCount() - p->getNumberOfPrimitiveRestarts();
          stats.m_lineLoopPrimitives++;
          break;
        case PRIMITIVE_LINES_ADJACENCY:
          stats.m_linesAdj += pcount;
          stats.m_lineSegments += pcount;
          stats.m_linesAdjacencyPrimitives++;
          break;
        case PRIMITIVE_LINE_STRIP_ADJACENCY:
          stats.m_lineStripAdj += pcount;
          stats.m_lineSegments += p->getElementCount() - pcount - p->getNumberOfPrimitiveRestarts();
          stats.m_lineStripAdjacencyPrimitives++;
          break;
        case PRIMITIVE_TRIANGLE_STRIP:
          stats.m_triStrips += pcount;
          stats.m_faces += fcount;
          stats.m_triangleStripPrimitives++;
          break;
        case PRIMITIVE_TRIANGLE_FAN:
          stats.m_triFans += pcount;
          stats.m_faces += fcount;
          stats.m_triangleFanPrimitives++;
          break;
        case PRIMITIVE_TRIANGLES:
          stats.m_tris += pcount;
          stats.m_faces += fcount;
          stats.m_trianglesPrimitives++;
          break;
        case PRIMITIVE_QUAD_STRIP:
          stats.m_quadStrips += pcount;
          stats.m_faces += fcount;
          stats.m_quadStripPrimitives++;
          break;
        case PRIMITIVE_QUADS:
          stats.m_quads += pcount;
          stats.m_faces += fcount;
          stats.m_quadsPrimitives++;
          break;
        case PRIMITIVE_POLYGON:
          stats.m_polygons += pcount;
          stats.m_faces += fcount;
          stats.m_polygonPrimitives++;
          break;
        case PRIMITIVE_TRIANGLES_ADJACENCY:
          stats.m_trisAdj += pcount;
          stats.m_faces += fcount;
          stats.m_trianglesAdjacencyPrimitives++;
          break;
        case PRIMITIVE_TRIANGLE_STRIP_ADJACENCY:
          stats.m_triStripAdj += pcount;
          stats.m_faces += fcount;
          stats.m_triangleStripAdjacencyPrimitives++;
          break;
        case PRIMITIVE_PATCHES:
          stats.m_patches += pcount;
          stats.m_faces += fcount;
          stats.m_patchesPrimitives++;
          break;
      }
    }

    void  StatisticsTraverser::statVertexAttributeSet( const VertexAttributeSet *p, StatVertexAttributeSet &stats )
    {
      stats.m_numberOfVertices += p->getNumberOfVertices();
      if ( p->getNumberOfNormals() )
      {
        stats.m_numberOfNormaled++;
        stats.m_numberOfNormals += p->getNumberOfNormals();
      }
      {
        unsigned int numTexUnits = stats.m_numberOfTextureUnits;
        for ( unsigned int i=0 ; i<8 ; i++ )
        {
          if ( p->getNumberOfTexCoords( i ) )
          {
            stats.m_numberOfTextureUnits++;
            stats.m_numberOfTextureDimensions += p->getSizeOfTexCoords( i );
            stats.m_numberOfTextures[i] += p->getNumberOfTexCoords( i );
          }
        }
        if ( numTexUnits != stats.m_numberOfTextureUnits )
        {
          stats.m_numberOfTextured++;
        }
      }
      if ( p->getNumberOfColors() )
      {
        stats.m_numberOfColored++;
        stats.m_numberOfColors += p->getNumberOfColors();
      }
      if ( p->getNumberOfSecondaryColors() )
      {
        stats.m_numberOfSecondaryColored++;
        stats.m_numberOfSecondaryColors += p->getNumberOfSecondaryColors();
      }
      if ( p->getNumberOfFogCoords() )
      {
        stats.m_numberOfFogged++;
        stats.m_numberOfFogCoords += p->getNumberOfFogCoords();
      }
      statObject( (const Object *)p, stats );
    }

    void  StatisticsTraverser::statGroup( const Group *p, StatGroup &stats )
    {
      stats.m_numberOfChildren += p->getNumberOfChildren();
      stats.m_childrenHistogram[p->getNumberOfChildren()]++;
      statNode( (const Node *)p, stats );
    }

    void  StatisticsTraverser::statLightSource( const LightSource *p, StatLightSource &stats )
    {
      statNode( (const Node *)p, stats );
    }

    void  StatisticsTraverser::statNode( const Node *p, StatNode &stats )
    {
      statObject( (const Object *)p, stats );
    }

    void  StatisticsTraverser::statObject( const Object *p, StatObject &stats )
    {
      if ( p->getHints() & Object::DP_SG_HINT_ASSEMBLY )
      {
        stats.m_assembled++;
      }
    }

    void  StatisticsTraverser::statTransform( const Transform *p, StatTransform &stats )
    {
      statGroup( (const Group *)p, stats );
    }

    #if !defined(NDEBUG)
    void  StatisticsTraverser::doApply( const NodeSharedPtr &root )
    {
      DP_ASSERT( m_instanceCount == 0 );
      SharedTraverser::doApply( root );
      DP_ASSERT( m_instanceCount == 0 );
    }
    #endif

    void  StatisticsTraverser::handleBillboard( const Billboard *p )
    {
      if ( m_statistics->m_statBillboard.firstEncounter( p, !!m_instanceCount ) )
      {
        statGroup( ( const Group *) p, m_statistics->m_statBillboard );
        SharedTraverser::handleBillboard(p);
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleBillboard(p);
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleIndexSet( const IndexSet * p )
    {
      if ( m_statistics->m_statIndexSet.firstEncounter( p, !!m_instanceCount ) )
      {
        m_statistics->m_statIndexSet.m_dataTypes[p->getIndexDataType()]++;
        m_statistics->m_statIndexSet.m_primitiveRestartIndices[p->getPrimitiveRestartIndex()]++;
        m_statistics->m_statIndexSet.m_numberOfIndices[p->getNumberOfIndices()]++;
        statObject( (const Object *)p, m_statistics->m_statIndexSet );
        SharedTraverser::handleIndexSet( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleIndexSet( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleGeoNode( const GeoNode *p )
    {
      if ( m_statistics->m_statGeoNode.firstEncounter( p, !!m_instanceCount ) )
      {
        statNode( (const Node *)p, m_statistics->m_statGeoNode );
        SharedTraverser::handleGeoNode(p);
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleGeoNode( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleGroup( const Group *p )
    {
      if ( m_statistics->m_statGroup.firstEncounter( p, !!m_instanceCount ) )
      {
        statGroup( ( const Group *) p, m_statistics->m_statGroup );
        SharedTraverser::handleGroup( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleGroup( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleLOD( const LOD *p )
    {
      if ( m_statistics->m_statLOD.firstEncounter( p, !!m_instanceCount ) )
      {
        statGroup( (const Group *)p, m_statistics->m_statLOD );
        SharedTraverser::handleLOD(p);
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleLOD( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleMatrixCamera( const MatrixCamera *p )
    {
      if ( m_statistics->m_statMatrixCamera.firstEncounter( p, !!m_instanceCount ) )
      {
        statCamera( (const Camera *)p, m_statistics->m_statMatrixCamera );
        SharedTraverser::handleMatrixCamera( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleMatrixCamera( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleParallelCamera( const ParallelCamera *p )
    {
      if ( m_statistics->m_statParallelCamera.firstEncounter( p, !!m_instanceCount ) )
      {
        statFrustumCamera( (const FrustumCamera *)p, m_statistics->m_statParallelCamera );
        SharedTraverser::handleParallelCamera( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleParallelCamera( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handlePerspectiveCamera( const PerspectiveCamera *p )
    {
      if ( m_statistics->m_statPerspectiveCamera.firstEncounter( p, !!m_instanceCount ) )
      {
        statFrustumCamera( (const FrustumCamera *)p, m_statistics->m_statPerspectiveCamera );
        SharedTraverser::handlePerspectiveCamera( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handlePerspectiveCamera( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleParameterGroupData( const ParameterGroupData *p )
    {
      if ( m_statistics->m_statParameterGroupData.firstEncounter( p, !!m_instanceCount ) )
      {
        statObject( (const ParameterGroupData *)p, m_statistics->m_statParameterGroupData);
        const dp::fx::ParameterGroupSpecSharedPtr & spec = p->getParameterGroupSpec();
        m_statistics->m_statParameterGroupData.m_dataSizeHistogram[spec->getDataSize()]++;
        m_statistics->m_statParameterGroupData.m_numParameterHistogram[spec->getNumberOfParameterSpecs()]++;
        SharedTraverser::handleParameterGroupData( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleParameterGroupData( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handlePipelineData( const dp::sg::core::PipelineData *p )
    {
      if ( m_statistics->m_statPipelineData.firstEncounter( p, !!m_instanceCount ) )
      {
        statObject( (const Object *)p, m_statistics->m_statPipelineData );
        m_statistics->m_statPipelineData.m_effectSpecTypeHistogram[p->getEffectSpec()->getType()]++;
        m_statistics->m_statPipelineData.m_parameterGroupDataSizeHistogram[p->getNumberOfParameterGroupData()]++;
        SharedTraverser::handlePipelineData( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handlePipelineData( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handlePrimitive( const Primitive *p )
    {
      if ( m_statistics->m_statPrimitives.firstEncounter( p, !!m_instanceCount ) )
      {
        statPrimitive( p, m_statistics->m_statPrimitives );
        SharedTraverser::handlePrimitive( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handlePrimitive( p );
        m_instanceCount--;
      }
      statPrimitive( p, m_statistics->m_statPrimitiveInstances );
    }

    void  StatisticsTraverser::handleSampler( const Sampler * p )
    {
      if ( m_statistics->m_statSampler.firstEncounter( p, !!m_instanceCount ) )
      {
        statObject( p, m_statistics->m_statSampler );
        if ( p->getTexture() )
        {
          m_statistics->m_statSampler.m_numberTextures++;
        }
        SharedTraverser::handleSampler( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleSampler( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleSwitch( const Switch *p )
    {
      if ( m_statistics->m_statSwitch.firstEncounter( p, !!m_instanceCount ) )
      {
        statGroup( ( const Group *) p, m_statistics->m_statSwitch );
        SharedTraverser::handleSwitch(p);
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleSwitch( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleTransform( const Transform *p )
    {
      if ( m_statistics->m_statTransform.firstEncounter( p, !!m_instanceCount ) )
      {
        statGroup( ( const Group *) p, m_statistics->m_statTransform );
        SharedTraverser::handleTransform(p);
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleTransform( p );
        m_instanceCount--;
      }
    }

    void  StatisticsTraverser::handleVertexAttributeSet( const VertexAttributeSet *p )
    {
      if ( m_statistics->m_statVertexAttributeSet.firstEncounter( p, !!m_instanceCount ) )
      {
        statVertexAttributeSet( p, m_statistics->m_statVertexAttributeSet );
        SharedTraverser::handleVertexAttributeSet( p );
      }
      else
      {
        m_instanceCount++;
        SharedTraverser::handleVertexAttributeSet( p );
        m_instanceCount--;
      }
      statVertexAttributeSet( p, m_statistics->m_statVertexAttributeSetInstances );
    }


      std::ostream& operator<<( std::ostream& os, const StatisticsBase& obj )
      {
        os << toString("Count", obj.m_count);
        os << toString("Referenced", obj.m_referenced);
        os << toString("Instanced", obj.m_instanced);
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatNode& obj )
      {
        os <<(StatObject)obj;
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatObject& obj )
      {
        os << (StatisticsBase)obj;
        if ( obj.m_assembled )
        {
          os << toString("Assembled", obj.m_assembled );
        }
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatIndexSet& obj )
      {
        os << (StatObject)obj;
        os << histogramToString("DataTypes", obj.m_dataTypes);
        os << histogramToString("PrimitiveRestartIndices", obj.m_primitiveRestartIndices);
        os << histogramToString("NumberOfIndices", obj.m_numberOfIndices);
        unsigned int sum = 0;
        unsigned int setCount = 0;
        unsigned int median = ~0;
        map<unsigned int,unsigned int> binnedIndices;
        for ( map<unsigned int, unsigned int>::const_iterator it = obj.m_numberOfIndices.begin() ; it != obj.m_numberOfIndices.end() ; ++it )
        {
          sum += it->first * it->second;
          binnedIndices[static_cast<unsigned int>(ceil(log2(static_cast<double>(it->first))))] += it->second;
          setCount += it->second;
          if ( ( obj.m_count <= 2 * setCount ) && ( median == ~0 ) )
          {
            // not absolutely correct! With obj.m_count even, we might have to calcalate the mean of the two center values !
            median = it->first;
          }
        }
        os << toString("  Sum", sum );
        os << toString("  Average", (float)sum / obj.m_count);
        os << toString("  Median", median);
        os << toString("BinnedIndices", "");
        for( map<unsigned int,unsigned int>::const_iterator it = binnedIndices.begin(); it != binnedIndices.end(); ++it )
        {
          os << toString(static_cast<unsigned int>(exp2(static_cast<double>(it->first))), it->second, 8, 16, 10);
        }
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatVertexAttributeSet& obj )
      {
        os << (StatObject)obj;
        os << toString("Vertices", obj.m_numberOfVertices);
        os << toString("Normaled", obj.m_numberOfNormaled);
        os << toString("Normals", obj.m_numberOfNormals);
        os << toString("Textured", obj.m_numberOfTextured);
        os << toString("TextureUnits", obj.m_numberOfTextureUnits);
        os << toString("TextureDimensions", obj.m_numberOfTextureDimensions);
        for ( int i=0 ; i<8 ; i++ )
        {
          std::stringstream ss;
          ss << i;
          os << toString("Textures["+ss.str()+"]", obj.m_numberOfTextures[i]);
        }
        os << toString("Colored", obj.m_numberOfColored);
        os << toString("Colors", obj.m_numberOfColors);
        os << toString("SecondaryColored", obj.m_numberOfSecondaryColored);
        os << toString("SecondaryColors", obj.m_numberOfSecondaryColors);
        os << toString("Fogged", obj.m_numberOfFogged);
        os << toString("FogCoords", obj.m_numberOfFogCoords);
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatPrimitive& obj )
      {
        os << (StatObject)obj;
        os << toString("Primitives", obj.m_count);
        os << toString("Indexed", obj.m_indexed);
        os << toString("Arrays", obj.m_arrays);

        os << toString("Points", obj.m_pointsPrimitives);
        os << toString("Line Strips", obj.m_lineStripPrimitives);
        os << toString("Line Loops", obj.m_lineLoopPrimitives);
        os << toString("Lines", obj.m_linesPrimitives);
        os << toString("Triangle Strips", obj.m_triangleStripPrimitives);
        os << toString("Triangle Fans", obj.m_triangleFanPrimitives);
        os << toString("Triangles", obj.m_trianglesPrimitives);
        os << toString("Quad Strips", obj.m_quadStripPrimitives);
        os << toString("Quads", obj.m_quadsPrimitives);
        os << toString("Polygons", obj.m_polygonPrimitives);
        os << toString("Triangles Adjacency", obj.m_trianglesAdjacencyPrimitives);
        os << toString("Triangle Strips Adjacency", obj.m_triangleStripAdjacencyPrimitives);
        os << toString("Lines Adjacency", obj.m_linesAdjacencyPrimitives);
        os << toString("Line Strips Adjacency", obj.m_lineStripAdjacencyPrimitives);
        os << toString("Patches", obj.m_patchesPrimitives);

        os << toString("Total Faces", obj.m_faces);
        os << toString("Total Line Segments", obj.m_lineSegments);
        os << toString("Total Points", obj.m_points );
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatGroup& obj )
      {
        os << (StatNode)obj;
        os << toString("Children", obj.m_numberOfChildren);
        os << histogramToString("NumberOfChildrenHistogram", obj.m_childrenHistogram);
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatParameterGroupData& obj )
      {
        os << (StatObject)obj;
        os << histogramToString("dataSizeHistogram", obj.m_dataSizeHistogram);
        os << histogramToString("numParameterHistogram", obj.m_numParameterHistogram);
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatEffectData& obj )
      {
        os << (StatObject)obj;
        os << histogramToString("effectSpecTypeHistogram", obj.m_effectSpecTypeHistogram);
        os << histogramToString("parameterGroupDataSizeHistogram", obj.m_parameterGroupDataSizeHistogram);
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatTexture& obj )
      {
        os << (StatisticsBase)obj;
        os << toString("Images", obj.m_numImages);
        os << histogramToString("WidthHistogram", obj.m_widths);
        os << histogramToString("HeightHistogram", obj.m_heights);
        os << histogramToString("DepthHistogram", obj.m_depths);
        os << histogramToString("SizeHistogram", obj.m_sizes);
        os << toString("Sum of sizes", obj.m_sumOfSizes);
        return os;
      }

      std::ostream& operator<<( std::ostream& os, const StatisticsTraverser& obj )
      {
        if ( obj.getStatistics()->m_statBillboard.m_count != 0 )
        {
          os << "\nBillboard\n";
          os << obj.getStatistics()->m_statBillboard;
        }
        if ( obj.getStatistics()->m_statGeoNode.m_count != 0 )
        {
          os << "\nGeoNode\n";
          os << obj.getStatistics()->m_statGeoNode;
        }
        if ( obj.getStatistics()->m_statGroup.m_count != 0 )
        {
          os << "\nGroup\n";
          os << obj.getStatistics()->m_statGroup;
        }
        if ( obj.getStatistics()->m_statIndexSet.m_count != 0 )
        {
          os << "\nIndexSet\n";
          os << obj.getStatistics()->m_statIndexSet;
        }
        if ( obj.getStatistics()->m_statLOD.m_count != 0 )
        {
          os << "\nLOD\n";
          os << obj.getStatistics()->m_statLOD;
        }
        if ( obj.getStatistics()->m_statMatrixCamera.m_count != 0 )
        {
          os << "\nMatrixCamera\n";
          os << obj.getStatistics()->m_statMatrixCamera;
        }
        if ( obj.getStatistics()->m_statParallelCamera.m_count != 0 )
        {
          os << "\nParallelCamera\n";
          os << obj.getStatistics()->m_statParallelCamera;
        }
        if ( obj.getStatistics()->m_statParameterGroupData.m_count != 0 )
        {
          os << "\nParameterGroupData\n";
          os << obj.getStatistics()->m_statParameterGroupData;
        }
        if ( obj.getStatistics()->m_statPipelineData.m_count != 0 )
        {
          os << "\nPipelineData\n";
          os << obj.getStatistics()->m_statPipelineData;
        }
        if ( obj.getStatistics()->m_statPerspectiveCamera.m_count != 0 )
        {
          os << "\nPerspectiveCamera\n";
          os << obj.getStatistics()->m_statPerspectiveCamera;
        }
        if ( obj.getStatistics()->m_statPrimitives.m_count != 0 )
        {
          os << "\nPrimitives\n";
          os << obj.getStatistics()->m_statPrimitives;
        }
        if ( obj.getStatistics()->m_statPrimitiveInstances.m_count != 0 )
        {
          os << "\nPrimitiveInstances\n";
          os << obj.getStatistics()->m_statPrimitiveInstances;
        }
        if ( obj.getStatistics()->m_statSwitch.m_count != 0 )
        {
          os << "\nSwitch\n";
          os << obj.getStatistics()->m_statSwitch;
        }
        if ( obj.getStatistics()->m_statTexture.m_count != 0 )
        {
          os << "\nTexture\n";
          os << obj.getStatistics()->m_statTexture;
        }
        if ( obj.getStatistics()->m_statTransform.m_count != 0 )
        {
          os << "\nTransform\n";
          os << obj.getStatistics()->m_statTransform;
        }
        if ( obj.getStatistics()->m_statVertexAttributeSet.m_count != 0 )
        {
          os << "\nVertexAttributeSet\n";
          os << obj.getStatistics()->m_statVertexAttributeSet;
        }
        if ( obj.getStatistics()->m_statVertexAttributeSetInstances.m_count != 0 )
        {
          os << "\nVertexAttributeSetInstances\n";
          os << obj.getStatistics()->m_statVertexAttributeSetInstances;
        }
        return os;
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
