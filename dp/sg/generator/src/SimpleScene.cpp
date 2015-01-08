// Copyright NVIDIA Corporation 2012-2015
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


#include <dp/sg/generator/SimpleScene.h>

#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Transform.h>

#include <dp/sg/generator/MeshGenerator.h>

#include <dp/math/math.h>
#include <dp/math/Vecnt.h>
#include <dp/fx/EffectLibrary.h>

using namespace dp::math;
using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace generator
    {

      SimpleScene::SimpleScene()
      {  
        m_sceneHandle = Scene::create();

        m_sceneHandle->setBackColor(  Vec4f( 71.0f/255.0f, 111.0f/255.0f, 0.0f, 1.0f ) );

        dp::fx::EffectLibrary::instance()->loadEffects( "SimpleScene.xml" );

        // Create cube (or other shape)
        //m_primitive = dp::sg::generator::createQuadSet(1,1);
        //m_primitive = dp::sg::generator::createQuadStrip(10);
        //m_primitive = dp::sg::generator::createTriSet(1,1);
        //m_primitive = dp::sg::generator::createTriFan(50);
        //m_primitive = dp::sg::generator::createTriStrip(10,1);
        m_primitive = dp::sg::generator::createCube();
        //m_primitive = dp::sg::generator::createTetrahedron();
        //m_primitive = dp::sg::generator::createOctahedron();
        //m_primitive = dp::sg::generator::createDodecahedron();
        //m_primitive = dp::sg::generator::createIcosahedron();
        //m_primitive = dp::sg::generator::createSphere(32,16);
        //m_primitive = dp::sg::generator::createTorus(64,32);
        //m_primitive = dp::sg::generator::createTessellatedPlane(1);
        //m_primitive = dp::sg::generator::createTessellatedBox(10);

        char const* const names[] =
        {
            "White Object"
          , "Red Object"
          , "Green Object"
          , "Blue Object"
        };

        char const* const materials[] =
        {
            "phong_white"
          , "phong_red"
          , "phong_green"
          , "phong_blue"
        };

        // Create four GeoNodes.
        for ( int i=0 ; i<4 ; i++ )
        {
          m_geoNodeHandle[i] = GeoNode::create();
          m_geoNodeHandle[i]->setPrimitive( m_primitive );
          m_geoNodeHandle[i]->setName( names[i] );
          setEffectData( i, materials[i] );
        }

        // Create four transforms. Cube coordinates are in the range [-1, 1], set them 3 units apart.
        m_transformHandle[0] = dp::sg::generator::createTransform( m_geoNodeHandle[0] );
        m_transformHandle[0]->setName( "White Object Transform" );

        m_transformHandle[1] = dp::sg::generator::createTransform( m_geoNodeHandle[1], Vec3f( 3.0f, 0.0f, 0.0f ) );
        m_transformHandle[1]->setName( "Red Object Transform" );

        m_transformHandle[2] = dp::sg::generator::createTransform( m_geoNodeHandle[2], Vec3f( 0.0f, 3.0f, 0.0f ), Quatf( Vec3f( 0.0f, 1.0f, 0.0f ), 10.0f) );
        m_transformHandle[2]->setName( "Green Object Transform" );

        m_transformHandle[3] = dp::sg::generator::createTransform( m_geoNodeHandle[3], Vec3f( 0.0f, 0.0f, 3.0f ), Quatf( Vec3f( 0.0f, 0.0f, 1.0f ), 20.0f) );
        m_transformHandle[3]->setName( "Blue Object Transform" );

        // Create the root
        GroupSharedPtr groupHdl = Group::create();
        for ( int i=0 ; i<4 ; i++ )
        {
          groupHdl->addChild( m_transformHandle[i] );
        }
        groupHdl->setName( "Root Node" );

        m_sceneHandle->setRootNode( groupHdl );
      }

      SimpleScene::~SimpleScene()
      {
      }

      void SimpleScene::setEffectData( size_t index, const std::string& effectData )
      {
        DP_ASSERT( index < sizeof dp::util::array( m_effectHandle ) );

        dp::fx::EffectDataSharedPtr fxEffectData = dp::fx::EffectLibrary::instance()->getEffectData( effectData );
        DP_ASSERT( fxEffectData );

        m_effectHandle[index] = EffectData::create( fxEffectData );
        m_geoNodeHandle[index]->setMaterialEffect( m_effectHandle[index] );
      }

    } // namespace generator
  } // namespace sg
} // namespace dp
