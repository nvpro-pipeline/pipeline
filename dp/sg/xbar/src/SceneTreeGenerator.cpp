// Copyright NVIDIA Corporation 2010-2012
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


#include <dp/sg/xbar/inc/SceneTreeGenerator.h>

#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>

using namespace dp::sg::core;

using std::vector;

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      SceneTreeGenerator::SceneTreeGenerator( SceneTreeSharedPtr const& sceneTree )
        : m_sceneTree( sceneTree )
      {
        setTraversalMaskOverride( ~0 );

        m_generatorState = GeneratorState::create( m_sceneTree.getSharedPtr() );
      }

      void SceneTreeGenerator::doApply( const dp::sg::core::NodeSharedPtr & root )
      {
        SharedTraverser::doApply( root );
      }

      void SceneTreeGenerator::setCurrentObjectTreeData( ObjectTreeIndex parentIndex, ObjectTreeIndex siblingIndex )
      {
        m_generatorState->setCurrentObjectTreeData( parentIndex, siblingIndex );
      }

      void  SceneTreeGenerator::addClipPlane( const ClipPlaneWeakPtr& clipPlane )
      {
        ClipPlaneInstanceSharedPtr instance( ClipPlaneInstance::create() );
        instance->m_clipPlane = clipPlane.getSharedPtr();
        instance->m_transformIndex = m_sceneTree->getObjectTreeNode(m_generatorState->getParentObjectIndex()).m_transform;
        m_generatorState->addClipPlane( instance );
      }

      bool SceneTreeGenerator::preTraverseGroup( const dp::sg::core::Group *p )
      {
        bool ok = SharedTraverser::preTraverseGroup(p);

        if( ok )
        {
          switch ( p->getObjectCode() )
          {
          case OC_LOD:
            m_generatorState->pushObject( p->getSharedPtr<LOD>() );
            break;
          case OC_SWITCH:
            m_generatorState->pushObject( p->getSharedPtr<Switch>() );
            break;
          default:
            m_generatorState->pushObject( p->getSharedPtr<Group>() );
          }

          // add group's clip planes to clip planes stack
          if( p->getNumberOfClipPlanes() )
          {
            m_generatorState->pushClipPlaneSet();

            Group::ClipPlaneConstIterator it_end = p->endClipPlanes();
            for (Group::ClipPlaneConstIterator it = p->beginClipPlanes(); it != it_end; ++it)
            {
              ClipPlaneInstanceSharedPtr instance(ClipPlaneInstance::create());
              instance->m_clipPlane = *it;
              instance->m_transformIndex = m_sceneTree->getObjectTreeNode(m_generatorState->getParentObjectIndex()).m_transform;
              m_generatorState->addClipPlane( instance );
            }
          }

        }

        return ok;
      }

      void SceneTreeGenerator::postTraverseGroup( const dp::sg::core::Group *p )
      {
        m_generatorState->popObject();
      }

      void SceneTreeGenerator::handleLOD( const dp::sg::core::LOD *p )
      {
        if( preTraverseGroup(p) )
        {
          Group::ChildrenConstIterator gcci = p->beginChildren();
          for ( unsigned int i=0 ; gcci != p->endChildren() ; ++gcci, ++i )
          {
            traverseObject( *gcci );
          }

          postTraverseGroup(p);
        }
      }

      void SceneTreeGenerator::handleSwitch( const dp::sg::core::Switch *p )
      {  
        if( preTraverseGroup(p) )
        {
          // TODO: API to switch Switch collecting completely off?
          bool collectAllChildren = true;

          // Either collect all children or only the active ones. If the switch is not 
          // flagged dynamic, the RL will be rebuilt on a switch mask change
          Group::ChildrenConstIterator gcci = p->beginChildren();
          for ( unsigned int i=0 ; gcci != p->endChildren() ; ++gcci, ++i )
          {
            if( collectAllChildren || p->isActive( i ) )
            {
              traverseObject( *gcci );
            }
          }

          postTraverseGroup(p);
        }
      }

      void SceneTreeGenerator::handleGeoNode( const dp::sg::core::GeoNode *p )
      {
        m_generatorState->addGeoNode( p->getSharedPtr<GeoNode>() );

        // stop traversal here, we only need the geonode position
      }

      void SceneTreeGenerator::handleLightSource( const LightSource * p )
      {
        m_generatorState->addLightSource( p->getSharedPtr<LightSource>() );

        // stop traversal here, we only need the light source position
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
