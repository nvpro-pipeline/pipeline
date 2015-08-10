// Copyright NVIDIA Corporation 2011-2012
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


#include <dp/sg/xbar/DrawableManager.h>
#include <dp/sg/xbar/inc/GeoNodeObserver.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/core/GeoNode.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      DrawableManager::HandleData::~HandleData()
      {
      }

      dp::util::SharedPtr<DrawableManager::HandleData> DrawableManager::HandleData::create()
      {
        return(std::shared_ptr<DrawableManager::HandleData>(new DrawableManager::HandleData()));
      }


      class SceneTreeObserver : public dp::util::Observer
      {
      public:
        SceneTreeObserver( DrawableManager * drawableManager );
        virtual void onNotify( dp::util::Event const & event, dp::util::Payload * payload );
        virtual void onDestroyed( dp::util::Subject const & subject, dp::util::Payload * payload );

      private:
        DrawableManager* m_drawableManager;
      };

      SceneTreeObserver::SceneTreeObserver( DrawableManager *drawableManager )
        : m_drawableManager( drawableManager )
      {

      }

      void SceneTreeObserver::onNotify( dp::util::Event const & event, dp::util::Payload * payload )
      {
        SceneTree::Event const& eventSceneTree = static_cast<SceneTree::Event const&>( event );
        switch ( eventSceneTree.getType() )
        {
        case SceneTree::Event::Object:
          {
            SceneTree::EventObject const& eventObject = static_cast<SceneTree::EventObject const&>( eventSceneTree );
            ObjectTreeNode const &node = eventObject.getNode();
            DP_ASSERT( node.m_object.getSharedPtr() );
            dp::sg::core::GeoNodeSharedPtr geoNode = node.m_object.getSharedPtr().dynamicCast<dp::sg::core::GeoNode>();
            DP_ASSERT( geoNode );

            if ( m_drawableManager->m_dis.size() != m_drawableManager->m_sceneTree->getObjectTree().size() )
            {
              m_drawableManager->m_dis.resize( m_drawableManager->m_sceneTree->getObjectTree().size() );
            }

            switch ( eventObject.getSubType() )
            {
            case SceneTree::EventObject::Added:
              DP_ASSERT( !m_drawableManager->m_dis[eventObject.getIndex()] );

              // TODO there're two locations which execute this code, unify with as function
              m_drawableManager->m_dis[eventObject.getIndex()] = m_drawableManager->addDrawableInstance( geoNode.getWeakPtr(), eventObject.getIndex() ); // TODO, don't pass geonode?
              m_drawableManager->setDrawableInstanceActive( m_drawableManager->m_dis[eventObject.getIndex()], node.m_worldActive );
              m_drawableManager->m_geoNodeObserver->attach( geoNode, eventObject.getIndex() );
              break;
            case SceneTree::EventObject::Removed:
              DP_ASSERT( m_drawableManager->m_dis[eventObject.getIndex()] );
              m_drawableManager->m_geoNodeObserver->detach( eventObject.getIndex() );
              m_drawableManager->removeDrawableInstance( m_drawableManager->m_dis[eventObject.getIndex()] );
              m_drawableManager->m_dis[eventObject.getIndex()].reset();
              break;
            case SceneTree::EventObject::Changed:
              DP_ASSERT(!"removed");
            case SceneTree::EventObject::ActiveChanged:
              DP_ASSERT( m_drawableManager->m_dis[eventObject.getIndex()] );
              m_drawableManager->setDrawableInstanceActive( m_drawableManager->m_dis[eventObject.getIndex()], node.m_worldActive );
              break;
            case SceneTree::EventObject::TraversalMaskChanged:
              DP_ASSERT( m_drawableManager->m_dis[eventObject.getIndex()] );
              m_drawableManager->setDrawableInstanceTraversalMask( m_drawableManager->m_dis[eventObject.getIndex()], node.m_worldMask );
              break;
            }
          }
          break;
        }
      }

      void SceneTreeObserver::onDestroyed( dp::util::Subject const & subject, dp::util::Payload * payload )
      {
        DP_ASSERT( !"should never be called" );
      }

      /************************************************************************/
      /* DrawableManager                                                      */
      /************************************************************************/
      DrawableManager::DrawableManager( )
      {
        m_sceneTreeObserver.reset( new SceneTreeObserver( this ) );
      }

      DrawableManager::~DrawableManager()
      {
        if (m_sceneTree) {
          m_sceneTree->detach( m_sceneTreeObserver.get() );
        }
      }

      void DrawableManager::setSceneTree( SceneTreeSharedPtr const & sceneTree )
      {
        if ( sceneTree != m_sceneTree )
        {
          if ( m_sceneTree )
          {
            detachSceneTree();
          }
          m_sceneTree = sceneTree;

          // Notify derived classes that the SceneTree has changed. They might need to reinitialize
          // resources and deallocate resources required for the old SceneTree.
          // This needs to be done before adding the new GeoNodes from the new SceneTree. Otherwise
          // resources which belong to the new SceneTree would be deleted too.
          onSceneTreeChanged();
          if( m_sceneTree )
          {
            attachSceneTree();
          }
        }
      }

      void DrawableManager::attachSceneTree()
      {
        class Visitor
        {
        public:
          struct Data {};

          Visitor( DrawableManager * drawableManager, ObjectTree const &objectTree )
            : m_drawableManager( drawableManager )
            , m_objectTree( objectTree )
          {

          }

          bool preTraverse( ObjectTreeIndex index, Data const & data )
          {
            if ( m_objectTree[index].m_isDrawable )
            {
              ObjectTreeNode const &node = m_objectTree[index];
              DP_ASSERT( node.m_object.getSharedPtr() );
              dp::sg::core::GeoNodeSharedPtr geoNode = node.m_object.getSharedPtr().dynamicCast<dp::sg::core::GeoNode>();
              DP_ASSERT( geoNode );

              m_drawableManager->m_dis[index] = m_drawableManager->addDrawableInstance( geoNode.getWeakPtr(), index );
              m_drawableManager->setDrawableInstanceActive( m_drawableManager->m_dis[index], node.m_worldActive );
              m_drawableManager->m_geoNodeObserver->attach( geoNode, index );
            }
            return true;
          }

          void postTraverse( TransformTreeIndex index, const Data& data )
          {
          }

        private:
          ObjectTree const & m_objectTree;
          DrawableManager  * m_drawableManager;
        };

        // allocate enough space for di mapping
        m_dis.resize( getSceneTree()->getObjectTree().size() );

        // create GeoNodeObserver
        m_geoNodeObserver = GeoNodeObserver::create( m_sceneTree );

        dp::sg::xbar::PreOrderTreeTraverser<ObjectTree, Visitor> p;
        Visitor v( this, m_sceneTree->getObjectTree() );
        p.traverse( m_sceneTree->getObjectTree(), v );

        m_sceneTree->attach( m_sceneTreeObserver.get() );
      }

      void DrawableManager::detachSceneTree()
      {
        /************************************************************************/
        /* Visitor class for ObjectTree                                         */
        /************************************************************************/
        class Visitor
        {
        public:
          struct Data {};

          Visitor( DrawableManager * drawableManager, ObjectTree const &objectTree )
            : m_drawableManager( drawableManager )
            , m_objectTree( objectTree )
          {

          }

          bool preTraverse( ObjectTreeIndex index, Data const & data )
          {
            if ( m_objectTree[index].m_isDrawable && m_drawableManager->m_dis[index])
            {
              // TODO is is possible that m_dis[index] is null?
              //DP_ASSERT( m_drawableManager->m_dis[eventObject.getIndex()] );
              m_drawableManager->m_geoNodeObserver->detach( index );
              m_drawableManager->removeDrawableInstance( m_drawableManager->m_dis[index] );
              m_drawableManager->m_dis[index].reset();
            }
            return true;
          }

          void postTraverse( TransformTreeIndex index, const Data& data )
          {
          }

        private:
          ObjectTree const & m_objectTree;
          DrawableManager  * m_drawableManager;
        };

        m_sceneTree->detach( m_sceneTreeObserver.get() );

        m_sceneTree->getObjectTree();
        dp::sg::xbar::PreOrderTreeTraverser<ObjectTree, Visitor> p;
        Visitor v( this, m_sceneTree->getObjectTree() );
        p.traverse( m_sceneTree->getObjectTree(), v );

        m_geoNodeObserver.reset();
      }

      void DrawableManager::update()
      {
        ObjectTreeIndexSet dirtyGeoNodes;
        m_geoNodeObserver->popDirtyGeoNodes(dirtyGeoNodes);
        for ( ObjectTreeIndexSet::const_iterator it = dirtyGeoNodes.begin(); it != dirtyGeoNodes.end(); ++it )
        {
          ObjectTreeIndex index = *it;
          ObjectTreeNode node = m_sceneTree->getObjectTreeNode(index);

          DP_ASSERT( m_dis[index] );
          // Remove/Add to change GeometryInstance
          removeDrawableInstance( m_dis[index] );
          DP_ASSERT( node.m_object.getSharedPtr() );
          m_dis[index] = addDrawableInstance( node.m_object.getSharedPtr().dynamicCast<dp::sg::core::GeoNode>().getWeakPtr(), index );    // TODO, don't pass geonode?
          setDrawableInstanceActive( m_dis[index], node.m_worldActive );
          break;
        }
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
