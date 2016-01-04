// Copyright (c) 2013-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/generator/Terrain.h>
#include <dp/util/Image.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/DP.h>


namespace dp
{
  namespace sg
  {
    namespace generator
    {

      // TODO where to put this?
      template <typename T>
      struct Clamped2DArrayAccessor
      {
        Clamped2DArrayAccessor( size_t width, size_t height, T const* data )
          : m_width( width )
          , m_height( height )
          , m_data( data )
        {
        }

        T operator()( int x, int y )
        {
          if ( x < 0 )
          {
            x = 0;
          }
          if ( y < 0 )
          {
            y = 0;
          }
          if ( x >= m_width )
          {
            x = int( m_width - 1);
          }
          if ( y >= m_height )
          {
            y = int( m_height - 1 );
          }

          return m_data[m_width * y + x];
        }
      private:
        size_t          m_width;
        size_t          m_height;
        T const * const m_data;
      };

      // terrain geometry mode
      template <typename HeightType>
      std::vector<dp::math::Vec3f> generateTerrainVertices( dp::util::ImageSharedPtr const & heightMap
                                                          , dp::math::Vec3f const & resolution, dp::math::Vec3f const & offset )
      {
        std::vector<dp::math::Vec3f> vertices;
        Clamped2DArrayAccessor<HeightType> cpa( heightMap->getWidth(), heightMap->getHeight(), reinterpret_cast<HeightType const*>(heightMap->getLayerData(0,0))) ;

        vertices.reserve( heightMap->getHeight() * heightMap->getWidth() );
        for ( int y = 0; y < int( heightMap->getHeight() ) ; ++y )
        {
          for ( int x = 0; x < int( heightMap->getWidth() ) ; ++x )
          {
            vertices.push_back( dp::math::Vec3f(float(x) * resolution[0], float(y) * resolution[1], float(cpa(x,y)) * resolution[2] ) + offset );
          }
        }
        return vertices;
      }

      std::vector<dp::math::Vec3f> generateTerrainVertices( dp::util::ImageSharedPtr const & heightMap
                                                          , dp::math::Vec3f const & resolution, dp::math::Vec3f const & offset )
      {
        DP_ASSERT( heightMap->getPixelFormat() == dp::PixelFormat::LUMINANCE );
        switch( heightMap->getDataType() )
        {
        case dp::DataType::UNSIGNED_INT_8:
          return generateTerrainVertices<uint8_t>( heightMap, resolution, offset );
        case dp::DataType::UNSIGNED_INT_16:
          return generateTerrainVertices<uint16_t>( heightMap, resolution, offset );
        case dp::DataType::UNSIGNED_INT_32:
          return generateTerrainVertices<uint32_t>( heightMap, resolution, offset );
        case dp::DataType::INT_8:
          return generateTerrainVertices<int8_t>( heightMap, resolution, offset );
        case dp::DataType::INT_16:
          return generateTerrainVertices<int16_t>( heightMap, resolution, offset );
        case dp::DataType::INT_32:
          return generateTerrainVertices<int32_t>( heightMap, resolution, offset );
        case dp::DataType::FLOAT_32:
          return generateTerrainVertices<int32_t>( heightMap, resolution, offset );
        case dp::DataType::FLOAT_64:
          return generateTerrainVertices<int64_t>( heightMap, resolution, offset );
        default:
          DP_ASSERT( !"Unknown heightmap format" );
        }
        return std::vector<dp::math::Vec3f>();
      }

      dp::sg::core::GeoNodeSharedPtr generateTerrainVertices( std::string const & filenameHeightMap, std::string const & filenameColorMap
                                                            , dp::math::Vec3f const & resolution, dp::math::Vec3f const & offset )
      {
        dp::util::ImageSharedPtr heightMap = dp::util::imageFromFile( filenameHeightMap );
        DP_ASSERT(heightMap);

        dp::sg::core::TextureHostSharedPtr colorMap = dp::sg::io::loadTextureHost( filenameColorMap );

        std::vector<dp::math::Vec3f> colors;

        dp::sg::core::GeoNodeSharedPtr geoNode = dp::sg::core::GeoNode::create();
        dp::sg::core::PrimitiveSharedPtr primitive = dp::sg::core::Primitive::create( dp::sg::core::PrimitiveType::TRIANGLES );
        dp::sg::core::VertexAttributeSetSharedPtr vertexAttributeset = dp::sg::core::VertexAttributeSet::create();
        dp::sg::core::IndexSetSharedPtr indexSet = dp::sg::core::IndexSet::create();

        {
          // generate vertices & normals
          std::vector<dp::math::Vec3f> vertices = generateTerrainVertices( heightMap, resolution, offset );
          vertexAttributeset->setVertices( 0, &vertices[0], dp::checked_cast<unsigned int>(vertices.size()), true );

          Clamped2DArrayAccessor<dp::math::Vec3f> cpa( heightMap->getWidth(), heightMap->getHeight(), &vertices[0]);

          std::vector<dp::math::Vec3f> normals;
          normals.reserve( heightMap->getHeight() * heightMap->getWidth() );
          for ( int y = 0; y < int( heightMap->getHeight() ) ; ++y )
          {
            for ( int x = 0; x < int( heightMap->getWidth() ) ; ++x )
            {
              dp::math::Vec3f dx( cpa( x + 1, y     ) - cpa( x - 1, y     ) );
              dp::math::Vec3f dy( cpa( x    , y + 1 ) - cpa( x    , y - 1 ) );
              dp::math::Vec3f cross( dx ^ dy );
              cross.normalize();
              normals.push_back( cross );
            }
          }

          vertexAttributeset->setNormals( 0, &normals[0], dp::checked_cast<unsigned int>(normals.size()), true );
        }

        if ( colorMap )
        {
          // generate uvs
          std::vector<dp::math::Vec2f> uv;
          uv.reserve( heightMap->getHeight() * heightMap->getWidth() );
          for ( int y = 0; y < int( heightMap->getHeight() ) ; ++y )
          {
            float v = float(y) / float( heightMap->getHeight() - 1);
            for ( int x = 0; x < int( heightMap->getWidth() ) ; ++x )
            {
              float u = float(x) / float( heightMap->getWidth() - 1);
              uv.push_back( dp::math::Vec2f(u, v) );
            }
          }
          vertexAttributeset->setTexCoords( 0, 0, &uv[0], dp::checked_cast<unsigned int>(uv.size()), true );
        }

        {
          // generate indices
          std::vector<dp::math::Vec3ui> indices;
          for ( int y = 0; y < int( heightMap->getHeight() - 1 ) ; ++y )
          {
            for ( int x = 0; x < int( heightMap->getWidth() - 1 ) ; ++x )
            {
              uint32_t base = dp::checked_cast<uint32_t>(y * heightMap->getWidth() + x);
              indices.push_back( dp::math::Vec3ui( base, base + 1, base + uint32_t(heightMap->getWidth()) + 1 ) );
              indices.push_back( dp::math::Vec3ui( base + uint32_t(heightMap->getWidth()) + 1, base + uint32_t(heightMap->getWidth()), base ) );
            }
          }
          indexSet->setData( &indices[0], uint32_t(indices.size() * 3) );
        }

        primitive->setIndexSet(indexSet);
        primitive->setVertexAttributeSet(vertexAttributeset);

        geoNode->setPrimitive(primitive);

        dp::sg::core::PipelineDataSharedPtr pipelineData = dp::sg::core::createStandardMaterialData();

        // attach colorMap if loading it
        if ( colorMap )
        {
          dp::sg::core::SamplerSharedPtr sampler = dp::sg::core::Sampler::create( colorMap );
          sampler->setName( "textureMap" );
          sampler->setMagFilterMode( dp::sg::core::TextureMagFilterMode::LINEAR );
          sampler->setMinFilterMode( dp::sg::core::TextureMinFilterMode::LINEAR );

          pipelineData->setParameterGroupData( dp::sg::core::createStandardTextureParameterData( sampler ) );
        }

        geoNode->setMaterialPipeline( pipelineData );

        return geoNode;
      }

      /** \brief A GeoNode which describes a terrain **/
      class TerrainNode : public dp::sg::core::GeoNode
      {
      public:
        enum class Mode { VERTEX, GEOMETRY, TESSELLATION };
      public:
        TerrainNode( dp::sg::core::TextureHostSharedPtr const & heightMap, dp::sg::core::TextureHostSharedPtr const & colorMap
                    , dp::math::Vec3f const & resolution, dp::math::Vec3f const & offset, Mode mode);

      protected:


        DP_SG_GENERATOR_API virtual dp::math::Box3f calculateBoundingBox() const;
        DP_SG_GENERATOR_API virtual dp::math::Sphere3f calculateBoundingSphere() const;

        dp::math::Vec3f m_scale;
        dp::math::Box3f m_boundingBox;
      };

      TerrainNode::TerrainNode( dp::sg::core::TextureHostSharedPtr const & heightMap, dp::sg::core::TextureHostSharedPtr const & colorMap
                              , dp::math::Vec3f const & resolution, dp::math::Vec3f const & offset, Mode mode )
      {
        dp::fx::EffectLibrary::instance()->loadEffects( "terrain.xml", dp::util::FileFinder( dp::home() + "/media/effects/xml" ) );

        std::string effectName;
        dp::sg::core::PrimitiveType primitiveType = dp::sg::core::PrimitiveType::TRIANGLES;
        //size_t verticesPerTexel = 6;
        float verticesPerTexel = 6;
        switch ( mode )
        {
        case Mode::VERTEX:
          effectName = "terrain";
          primitiveType = dp::sg::core::PrimitiveType::TRIANGLES;
          verticesPerTexel = 6;
          break;
        case Mode::GEOMETRY:
          effectName = "terrain_geometry";
          primitiveType = dp::sg::core::PrimitiveType::POINTS;
          verticesPerTexel = 1;
          break;
        case Mode::TESSELLATION:
          effectName = "terrain_tessellation";
          primitiveType = dp::sg::core::PrimitiveType::PATCHES;
          verticesPerTexel = 1.0f / (4.0f * 4.0f);
          //verticesPerTexel = 1.0f / (64.0f * 64.0f);
          break;
        default:
          throw std::runtime_error("Unsupported TerrainMode");
          break;
        }

        float height; // scale due to the int->[0...1] float conversion
        switch ( heightMap->getType() )
        {
        case dp::sg::core::Image::PixelDataType::UNSIGNED_BYTE:
          height = 255.0f;
          break;
        case dp::sg::core::Image::PixelDataType::UNSIGNED_SHORT:
          height = 65535.0f;
          break;
        case dp::sg::core::Image::PixelDataType::UNSIGNED_INT:
          height = 4294967295.0f;
          break;
        case dp::sg::core::Image::PixelDataType::FLOAT16:
        case dp::sg::core::Image::PixelDataType::FLOAT32:
        case dp::sg::core::Image::PixelDataType::BYTE:
        case dp::sg::core::Image::PixelDataType::SHORT:
        case dp::sg::core::Image::PixelDataType::INT:
          DP_ASSERT( !"those are currently not supported.");
          break;
        }

        // compute bounding box
        dp::math::Vec3f realResolution( resolution );
        realResolution[2] *= height;
        m_boundingBox = dp::math::Box3f( dp::math::Vec3f(0.0f, 0.0f, 0.0f) + offset
                                       , dp::math::Vec3f( heightMap->getWidth() * realResolution[0] , heightMap->getHeight() * realResolution[1], realResolution[2] ) + offset );

        // setup vertex attributes
        // generate vertices for now. It's stupid that one has to create dummy vertices to determine the #elements for glDrawArrays.
        size_t numRects = ( (heightMap->getWidth() - 1) * (heightMap->getHeight() - 1) );

        // this is a dummy required to render the required amount of vertices.
        dp::sg::core::VertexAttribute va;
        dp::sg::core::BufferHostSharedPtr buffer = dp::sg::core::BufferHost::create();
        buffer->setSize(1); // currently it's necessary to have at least one byte in the buffer for other parts of the pipeline.

        va.setData( 3, dp::DataType::FLOAT_32, buffer, 0, 0, (unsigned int)(verticesPerTexel * numRects) );
        dp::sg::core::VertexAttributeSetSharedPtr vertexAttributeset = dp::sg::core::VertexAttributeSet::create();
        vertexAttributeset->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::POSITION, va);

        // setup primtive
        dp::sg::core::PrimitiveSharedPtr primitive = dp::sg::core::Primitive::create( primitiveType );
        //primitive->setIndexSet(indexSet);
        primitive->setVertexAttributeSet(vertexAttributeset);

        dp::sg::core::SamplerSharedPtr sampler = dp::sg::core::Sampler::create( heightMap );
        sampler->setName( "heightMap" );
        sampler->setMagFilterMode( dp::sg::core::TextureMagFilterMode::LINEAR );
        sampler->setMinFilterMode( dp::sg::core::TextureMinFilterMode::LINEAR );
        sampler->setTexture( heightMap );

        dp::sg::core::PipelineDataSharedPtr pipelineData = dp::sg::core::PipelineData::create( dp::fx::EffectLibrary::instance()->getEffectData(effectName) );

        dp::sg::core::ParameterGroupDataSharedPtr pgd = pipelineData->findParameterGroupData( std::string("terrain_parameters") );
        pgd->setParameter( "heightMap", sampler );
        pgd->setParameter( "resolution", realResolution );
        pgd->setParameter( "offset", offset );

        // setup GeoNode
        setPrimitive(primitive);

        // attach colorMap if loading it
        if ( colorMap )
        {
          dp::sg::core::SamplerSharedPtr sampler = dp::sg::core::Sampler::create( colorMap );
          sampler->setName( "textureMap" );
          sampler->setMagFilterMode( dp::sg::core::TextureMagFilterMode::LINEAR );
          sampler->setMinFilterMode( dp::sg::core::TextureMinFilterMode::LINEAR );

          pipelineData->setParameterGroupData( dp::sg::core::createStandardTextureParameterData( sampler ) );
        }

        setMaterialPipeline( pipelineData );

      }

      dp::math::Box3f TerrainNode::calculateBoundingBox() const
      {
        return m_boundingBox;
      }

      dp::math::Sphere3f TerrainNode::calculateBoundingSphere() const
      {
        return dp::math::Sphere3f( m_boundingBox.getCenter(), dp::math::length(m_boundingBox.getSize()) / 2 );
      }


      // terrain factory
      dp::sg::core::GeoNodeSharedPtr generateTerrain( std::string const& filenameHeightMap, std::string const& filenameColorMap
                                                    , dp::math::Vec3f const & resolution, dp::math::Vec3f const & offset )
      {
        if ( false ) // vertices terrain mode is for testing only atm
        {
          dp::fx::EffectLibrary::instance()->loadEffects( "terrain.xml", dp::util::FileFinder( dp::home() + "/media/effects/xml" ) );
          return generateTerrainVertices( filenameHeightMap, filenameColorMap, resolution, offset );
        }
        else
        {
          dp::sg::core::TextureHostSharedPtr heightMap = dp::sg::io::loadTextureHost( filenameHeightMap );
          dp::sg::core::TextureHostSharedPtr colorMap = dp::sg::io::loadTextureHost( filenameColorMap );

          return( std::make_shared<TerrainNode>( heightMap, colorMap, resolution, offset, TerrainNode::Mode::TESSELLATION ) );
        }
      }

    } // namespace generator
  } // namespace sg
} // namespace dp
