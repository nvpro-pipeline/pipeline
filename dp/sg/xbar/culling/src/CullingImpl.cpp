// Copyright NVIDIA Corporation 2013
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


#include <dp/sg/xbar/culling/inc/CullingImpl.h>
#include <dp/sg/xbar/culling/inc/ResultImpl.h>
#include <dp/sg/core/GeoNode.h>

#include <dp/culling/cpu/Manager.h>
#include <dp/culling/opengl/Manager.h>

#include <stdexcept>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      namespace culling
      {
        CullingImplSharedPtr CullingImpl::create( SceneTreeSharedPtr const & sceneTree, dp::culling::Mode cullingMode )
        {
          return( std::shared_ptr<CullingImpl>( new CullingImpl( sceneTree, cullingMode ) ) );
        }

        CullingImpl::CullingImpl( SceneTreeSharedPtr const & sceneTree, dp::culling::Mode cullingMode )
          : m_sceneTree( sceneTree )
        {
          switch ( cullingMode )
          {
          case dp::culling::MODE_CPU:
            m_culling.reset(dp::culling::cpu::Manager::create());
            break;
          case dp::culling::MODE_OPENGL_COMPUTE:
            m_culling.reset(dp::culling::opengl::Manager::create());
            break;
          default:
            std::cerr << "unknown culling mode, falling back to CPU version" << std::endl;
            m_culling.reset(dp::culling::cpu::Manager::create());
          }
          m_cullingGroup = m_culling->groupCreate();

          // traverse SceneTree to get the initial list of Objects
          class Visitor
          {
          public:
            struct Data {};

            Visitor( CullingImpl * culling, ObjectTree const &objectTree )
              : m_culling( culling )
              , m_objectTree( objectTree )
            {

            }

            bool preTraverse( ObjectTreeIndex index, Data const & data )
            {
              if ( m_objectTree[index].m_isDrawable )
              {
                m_culling->addObject( index );
              }
              return true;
            }

            void postTraverse( TransformTreeIndex index, const Data& data )
            {
            }

          private:
            ObjectTree const & m_objectTree;
            CullingImpl      * m_culling;
          };

          dp::sg::xbar::PreOrderTreeTraverser<ObjectTree, Visitor> p;
          Visitor v( this, m_sceneTree->getObjectTree() );
          p.traverse( m_sceneTree->getObjectTree(), v );

          // and attach to SceneTree get update events
          m_sceneTree->attach( this );
        }

        CullingImpl::~CullingImpl()
        {
          m_cullingGroup.reset();
          m_sceneTree->detach( this );
        }

        dp::sg::xbar::culling::ResultSharedPtr CullingImpl::resultCreate()
        {
          return( ResultImpl::create( m_culling->groupCreateResult( m_cullingGroup ) ) );
        }

        bool CullingImpl::resultIsVisible( ResultSharedPtr const & result, ObjectTreeIndex objectTreeIndex ) const
        {
          return( m_culling->resultObjectIsVisible( result.staticCast<ResultImpl>()->getResult(), m_objects[objectTreeIndex] ) );
        }

        std::vector<dp::sg::xbar::ObjectTreeIndex> const & CullingImpl::resultGetChangedIndices( ResultSharedPtr const & result ) const
        {
          return( result.staticCast<ResultImpl>()->getChanged() );
        }

        void CullingImpl::cull( ResultSharedPtr const & result, dp::math::Mat44f const & world2ViewProjection )
        {
          ResultImplSharedPtr const & resultImpl = result.staticCast<ResultImpl>();
          dp::sg::xbar::TransformTree const & transformTree = m_sceneTree->getTransformTree();
          m_culling->groupSetMatrices( m_cullingGroup, &transformTree[0].m_worldMatrix, transformTree.size(), sizeof( transformTree[0]) );
          m_culling->cull( m_cullingGroup, resultImpl->getResult(), world2ViewProjection );

          std::vector<dp::culling::ObjectSharedPtr> const & changedObjects = m_culling->resultGetChanged( resultImpl->getResult() );
          std::vector<ObjectTreeIndex> & changedIndices = resultImpl->getChanged();

          // clear the previous list of indices and get new the indices from the culling result
          changedIndices.clear();
          for ( size_t index = 0;index < changedObjects.size(); ++index )
          {
            PayloadSharedPtr const & p = m_culling->objectGetUserData(changedObjects[index]).staticCast<Payload>();

            changedIndices.push_back( p->getObjectTreeIndex() );
          }
        }

        dp::math::Box3f CullingImpl::getBoundingBox()
        {
          dp::sg::xbar::TransformTree const & transformTree = m_sceneTree->getTransformTree();
          m_culling->groupSetMatrices( m_cullingGroup, &transformTree[0].m_worldMatrix, transformTree.size(), sizeof( transformTree[0]) );
          return m_culling->getBoundingBox( m_cullingGroup );
        }

        void CullingImpl::updateBoundingBox( ObjectTreeIndex objectTreeIndex )
        {
          dp::sg::core::GeoNodeWeakPtr geoNodeWeakPtr = dp::util::weakPtr_cast<dp::sg::core::GeoNode>(m_sceneTree->getObjectTreeNode( objectTreeIndex ).m_object);
          dp::sg::core::PrimitiveSharedPtr primitive = geoNodeWeakPtr->getPrimitive();
          m_culling->objectSetBoundingBox( m_objects[objectTreeIndex], geoNodeWeakPtr->getBoundingBox() );
        }

        void CullingImpl::addObject( ObjectTreeIndex index )
        {
          DP_ASSERT( index < m_sceneTree->getObjectTree().size() );

          // ensure that the mapping table from the SceneTree to the objects of the culling module is large enough
          if ( m_objects.size() != m_sceneTree->getObjectTree().size() )
          {
            m_objects.resize( m_sceneTree->getObjectTree().size() );
          }

          DP_ASSERT( !m_objects[index] && "there's already a culling object at the given position" );

          // create a new culling object and add it to the culling group
          ObjectTreeNode const &node = m_sceneTree->getObjectTreeNode( index );
          m_objects[index] = m_culling->objectCreate( Payload::create( index ) );
          m_culling->groupAddObject( m_cullingGroup, m_objects[index] );
          m_culling->objectSetTransformIndex( m_objects[index], node.m_transformIndex );
          updateBoundingBox( index );
        }

        void CullingImpl::onNotify( dp::util::Event const & event, dp::util::Payload * payload )
        {
          SceneTree::Event const& eventSceneTree = static_cast<SceneTree::Event const&>( event );
          switch ( eventSceneTree.getType() )
          {
          case SceneTree::Event::Object:
            {
              // get objects from event
              SceneTree::EventObject const& eventObject = static_cast<SceneTree::EventObject const&>( eventSceneTree );
              ObjectTreeIndex index = eventObject.getIndex();
              ObjectTreeNode const &node = eventObject.getNode();

              switch ( eventObject.getSubType() )
              {
              case SceneTree::EventObject::Added:
                addObject( index );
                break;

              case SceneTree::EventObject::Removed:
                DP_ASSERT( m_objects[eventObject.getIndex()] && "culling object for the given object has already been destroyed" );
                m_culling->groupRemoveObject( m_cullingGroup, m_objects[index] );
                m_objects[index].reset();
                break;

              case SceneTree::EventObject::Changed:
                DP_ASSERT( m_objects[eventObject.getIndex()]  && "no culling object available for the given index" );
                updateBoundingBox( index );
                // TODO update bounding box!
                break;

              default:
                break;
              }
            }
            break;

          case dp::sg::xbar::SceneTree::Event::Transform:
            // transform have changed. Notify culling about the change.
            m_culling->groupMatrixChanged( m_cullingGroup, static_cast<dp::sg::xbar::SceneTree::EventTransform const&>( event ).getIndex() );
            break;

          }
        }

        void CullingImpl::onDestroyed( dp::util::Subject const & subject, dp::util::Payload * payload )
        {
          throw std::logic_error("Unexpected event.");
        }

      } // namespace culling
    } // namespace xbar
  } // namespace sg
} // namespace dp
