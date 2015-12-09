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


#include <dp/sg/algorithm/Traverser.h>
// dp/sg/core objects
#include <dp/sg/core/Object.h>
// superior types
#include <dp/sg/core/Scene.h>
// cameras
#include <dp/sg/core/MatrixCamera.h>
#include <dp/sg/core/ParallelCamera.h>
#include <dp/sg/core/PerspectiveCamera.h>
// nodes
// ... groups
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
// ... lights
#include <dp/sg/core/LightSource.h>
// ... geometry node
#include <dp/sg/core/GeoNode.h>
// primitives
#include <dp/sg/core/Primitive.h>
// states
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>

using namespace dp::math;
using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /************************************************************************/
      /* Traverser                                                            */
      /************************************************************************/
      Traverser::Traverser()
      : m_scene(NULL)
      , m_root(NULL)
      , m_viewState(NULL)
      , m_camera(NULL)
      , m_mftbl(static_cast<size_t>(ObjectCode::VERTEX_ATTRIBUTE_SET))
      , m_currentAttrib(~0)
      , m_traversalMask(~0)        // traverse all masks
      , m_traversalMaskOverride(0) // do not modify object's traversal mask
      {
      }

      Traverser::~Traverser()
      {
      }

      void Traverser::setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState )
      {
        m_viewState = viewState;
      }

      void Traverser::apply( )
      {
        if ( m_viewState )
        {
          m_camera = m_viewState->getCamera();
          m_scene = m_viewState->getScene();
          if (m_viewState->getScene())
          {
            m_root = m_scene->getRootNode();
          }

          if ( m_root && m_scene && preApply( m_root ) )
          {
            // start polymorphic traversal
            doApply( m_root );

            // let custom overrides perform some post-traversal work
            postApply( m_root );

          }

          m_root.reset();
          m_scene.reset();
          m_camera.reset();
        }
      }

      void Traverser::apply( dp::sg::ui::ViewStateSharedPtr const& viewState )
      {
        if ( viewState )
        {
          if ( viewState != m_viewState )
          {
            dp::sg::ui::ViewStateSharedPtr oldViewState = m_viewState;
            setViewState( viewState );
            apply();
            setViewState( oldViewState );
          }
          else
          {
            apply();
          }
        }
      }

      void Traverser::apply( const NodeSharedPtr & root )
      {
        // let custom overrides perform some pre-traversal work
        if ( root && preApply(root) )
        {
          if ( m_viewState )
          {
            m_camera = m_viewState->getCamera();
            m_scene = m_viewState->getScene();
          }
          m_root = root;

          // start polymorphic traversal
          doApply(root);

          // let custom overrides perform some post-traversal work
          postApply(root);

          m_root.reset();
          m_scene.reset();
          m_camera.reset();
        }
      }

      void Traverser::apply( const SceneSharedPtr & scene )
      {
        DP_ASSERT(scene);

        apply(scene->getRootNode());
      }

      bool Traverser::preApply( const NodeSharedPtr & root )
      {
        m_root = root;
        return true;
      }

      void Traverser::postApply( const NodeSharedPtr & root )
      {
      }

      bool Traverser::preTraverseGroup(const Group * grp)
      {
        return true;
      }

      void Traverser::postTraverseGroup(const Group * grp)
      {
        postTraverseObject( grp );
      }

      void Traverser::traverse(const Group * grp)
      {
        for ( Group::ChildrenConstIterator gcci = grp->beginChildren() ; gcci != grp->endChildren() ; ++gcci )
        {
          traverseObject( *gcci );
        }
      }

      void Traverser::traverse(const Switch * swtch)
      {
        Group::ChildrenConstIterator gcci = swtch->beginChildren();
        for ( unsigned int i=0 ; gcci != swtch->endChildren() ; ++gcci, ++i )
        {
          // traverse active children only
          if ( swtch->isActive( i ) )
          {
            traverseObject( *gcci );
          }
        }
      }

      void Traverser::traverseGeoNode(const GeoNode * gnode)
      {
        if ( gnode->getMaterialPipeline() )
        {
          traverseObject( gnode->getMaterialPipeline() );
        }
        if ( gnode->getPrimitive() )
        {
          traverseObject( gnode->getPrimitive() );
        }
        postTraverseObject( gnode );
      }

      void Traverser::traverseParameterGroupData( const ParameterGroupData * pgd )
      {
        const dp::fx::ParameterGroupSpecSharedPtr & pgs = pgd->getParameterGroupSpec();
        for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
        {
          if ( ( it->first.getType() & dp::fx::PT_POINTER_TYPE_MASK ) == dp::fx::PT_SAMPLER_PTR )
          {
            const SamplerSharedPtr & sampler = pgd->getParameter<SamplerSharedPtr>( it );
            if ( sampler )
            {
              traverseObject( sampler );
            }
          }
        }
      }

      void Traverser::traversePipelineData( const dp::sg::core::PipelineData * p )
      {
        m_currentTextureUnit = 0;
        const dp::fx::EffectSpecSharedPtr & es = p->getEffectSpec();
        for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
        {
          const ParameterGroupDataSharedPtr & parameterGroupData = p->getParameterGroupData( it );
          if ( parameterGroupData )
          {
            traverseObject( parameterGroupData );
            if ( (*it)->getName() == "standardTextureParameters" )
            {
              m_currentTextureUnit++;
            }
          }
        }
      }

      void Traverser::traversePrimitive(const Primitive * p)
      {
        postTraverseObject( p );
      }

      dp::sg::core::ObjectCode Traverser::getObjectTraversalCode(const Object * object)
      {
        dp::sg::core::ObjectCode oc = object->getObjectCode();
        DP_ASSERT(oc!=ObjectCode::INVALID);

        if ( !m_mftbl.testEntry(oc) )
        {
          dp::sg::core::ObjectCode orgOC = oc;
          do
          { // came across an unknown object
            // move up the object's hierarchy to find an appropriate handler
            oc = object->getHigherLevelObjectCode(oc);
            if ( ObjectCode::INVALID==oc )
            { // proceed immediately without handling the object
              return oc;
            }
          }
          while ( !m_mftbl.testEntry(oc) );

          // found an appropriate handler if we get here

          // don't loop again for this object - register the handler
          m_mftbl.addEntry(orgOC, m_mftbl[oc]);
        }
        return oc;
      }

      /************************************************************************/
      /* SharedTraverser                                                      */
      /************************************************************************/

      SharedTraverser::SharedTraverser()
      {
        addObjectHandler(ObjectCode::PARALLEL_CAMERA, &SharedTraverser::handleParallelCamera);
        addObjectHandler(ObjectCode::PERSPECTIVE_CAMERA, &SharedTraverser::handlePerspectiveCamera);
        addObjectHandler(ObjectCode::MATRIX_CAMERA, &SharedTraverser::handleMatrixCamera);

        // Group-derived
        addObjectHandler(ObjectCode::GROUP, &SharedTraverser::handleGroup);
        addObjectHandler(ObjectCode::TRANSFORM, &SharedTraverser::handleTransform);
        addObjectHandler(ObjectCode::LOD, &SharedTraverser::handleLOD);
        addObjectHandler(ObjectCode::SWITCH, &SharedTraverser::handleSwitch);
        addObjectHandler(ObjectCode::BILLBOARD, &SharedTraverser::handleBillboard);

        // LightSource
        addObjectHandler(ObjectCode::LIGHT_SOURCE, &SharedTraverser::handleLightSource);

        // GeoNode
        addObjectHandler(ObjectCode::GEO_NODE, &SharedTraverser::handleGeoNode);

        // Primitive
        addObjectHandler(ObjectCode::PRIMITIVE, &SharedTraverser::handlePrimitive);

        // ... single state attribs
        addObjectHandler(ObjectCode::PARAMETER_GROUP_DATA, &SharedTraverser::handleParameterGroupData);
        addObjectHandler(ObjectCode::PIPELINE_DATA, &SharedTraverser::handlePipelineData);
        addObjectHandler(ObjectCode::SAMPLER, &SharedTraverser::handleSampler);

        addObjectHandler(ObjectCode::INDEX_SET, &SharedTraverser::handleIndexSet);
        addObjectHandler(ObjectCode::VERTEX_ATTRIBUTE_SET, &SharedTraverser::handleVertexAttributeSet);
      }

      SharedTraverser::~SharedTraverser()
      {

      }

      void SharedTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT(root); // invalid root results in undefined behavior!

        if ( m_viewState && m_viewState->getCamera() ) // a ViewState and a camera is optional
        {
          traverseObject( m_viewState->getCamera() );
        }

        traverseObject(root);
      }

      void SharedTraverser::handleParallelCamera(const ParallelCamera * camera)
      {
        traverseFrustumCamera(camera);
      }

      void SharedTraverser::handlePerspectiveCamera(const PerspectiveCamera * camera)
      {
        traverseFrustumCamera(camera);
      }

      void SharedTraverser::handleMatrixCamera( const MatrixCamera * camera )
      {
        traverseCamera( camera );
      }

      void SharedTraverser::handleGroup(const Group * group)
      {
        traverseGroup(group);
      }

      void SharedTraverser::handleTransform(const Transform * trafo)
      {
        traverseGroup(trafo);
      }

      void SharedTraverser::handleLOD(const LOD * lod)
      {
        traverseGroup(lod);
      }

      void SharedTraverser::handleSwitch(const Switch * swtch)
      {
        traverseGroup(swtch);
      }

      void SharedTraverser::handleBillboard(const Billboard * billboard)
      {
        traverseGroup(billboard);
      }

      void SharedTraverser::handleLightSource( const LightSource * light )
      {
        traverseLightSource( light );
      }

      void SharedTraverser::handleGeoNode(const GeoNode * gnode)
      {
        traverseGeoNode(gnode);
      }

      void SharedTraverser::handlePrimitive(const Primitive * primitive)
      {
        traversePrimitive(primitive);
      }

      void SharedTraverser::handleParameterGroupData( const ParameterGroupData * pgd )
      {
        traverseParameterGroupData( pgd );
      }

      void SharedTraverser::handlePipelineData( const dp::sg::core::PipelineData * pd )
      {
        traversePipelineData( pd );
      }

      void SharedTraverser::handleSampler( const Sampler * p )
      {
      }

      void SharedTraverser::handleIndexSet(const IndexSet * iset )
      {
        postTraverseObject( iset );
      }

      void SharedTraverser::handleVertexAttributeSet(const VertexAttributeSet *vas)
      {
        postTraverseObject( vas );
      }

      void SharedTraverser::traverseCamera(const Camera * camera)
      {
        for ( Camera::HeadLightConstIterator hlci = camera->beginHeadLights() ; hlci != camera->endHeadLights() ; ++hlci )
        {
          traverseObject( *hlci );
        }
        postTraverseObject( camera );
      }

      void SharedTraverser::traverseFrustumCamera(const FrustumCamera * camera)
      {
        traverseCamera( camera );
      }

      void SharedTraverser::traverseLightSource(const LightSource * light)
      {
        if ( light->getLightPipeline() )
        {
          traverseObject( light->getLightPipeline() );
        }
        postTraverseObject( light );
      }

      void SharedTraverser::traversePrimitive(const Primitive * p)
      {
        if ( p->getIndexSet() )
        {
          traverseObject( p->getIndexSet() );
        }
        if ( p->getVertexAttributeSet() )
        {
          traverseObject( p->getVertexAttributeSet() );
        }
        postTraverseObject( p );
      }

      /************************************************************************/
      /* ExclusiveTraverser                                                   */
      /************************************************************************/
      ExclusiveTraverser::ExclusiveTraverser()
      : m_treeModified(false)
      {
        // Camera-derived
        addObjectHandler(ObjectCode::PARALLEL_CAMERA, &ExclusiveTraverser::handleParallelCamera);
        addObjectHandler(ObjectCode::PERSPECTIVE_CAMERA, &ExclusiveTraverser::handlePerspectiveCamera);
        addObjectHandler(ObjectCode::MATRIX_CAMERA, &ExclusiveTraverser::handleMatrixCamera);

        // Group-derived
        addObjectHandler(ObjectCode::GROUP, &ExclusiveTraverser::handleGroup);
        addObjectHandler(ObjectCode::TRANSFORM, &ExclusiveTraverser::handleTransform);
        addObjectHandler(ObjectCode::LOD, &ExclusiveTraverser::handleLOD);
        addObjectHandler(ObjectCode::SWITCH, &ExclusiveTraverser::handleSwitch);
        addObjectHandler(ObjectCode::BILLBOARD, &ExclusiveTraverser::handleBillboard);

        // LightSource
        addObjectHandler(ObjectCode::LIGHT_SOURCE, &ExclusiveTraverser::handleLightSource);

        // GeoNode
        addObjectHandler(ObjectCode::GEO_NODE, &ExclusiveTraverser::handleGeoNode);

        // Primitive
        addObjectHandler(ObjectCode::PRIMITIVE, &ExclusiveTraverser::handlePrimitive);

        // ... single state attribs
        addObjectHandler(ObjectCode::PARAMETER_GROUP_DATA, &ExclusiveTraverser::handleParameterGroupData);
        addObjectHandler(ObjectCode::PIPELINE_DATA, &ExclusiveTraverser::handlePipelineData);
        addObjectHandler(ObjectCode::SAMPLER, &ExclusiveTraverser::handleSampler);

        addObjectHandler(ObjectCode::INDEX_SET, &ExclusiveTraverser::handleIndexSet);
        addObjectHandler(ObjectCode::VERTEX_ATTRIBUTE_SET, &ExclusiveTraverser::handleVertexAttributeSet);
      }

      ExclusiveTraverser::~ExclusiveTraverser()
      {
      };

      void ExclusiveTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT(root); // invalid root results in undefined behavior!

        m_treeModified = false;
        if ( m_viewState && m_viewState->getCamera() ) // a ViewState is optional
        {
          // with a ViewState available, use the camera to traverse
          DP_ASSERT( m_viewState->getCamera() );
          traverseObject( m_viewState->getCamera() );
          traverseObject(root);
        }
        else
        {
          // just traverse the root without a camera
          traverseObject(root);
        }
      }

      void ExclusiveTraverser::handleParallelCamera(ParallelCamera * camera)
      {
        traverseFrustumCamera(camera);
      }

      void ExclusiveTraverser::handlePerspectiveCamera(PerspectiveCamera * camera)
      {
        traverseFrustumCamera(camera);
      }

      void ExclusiveTraverser::handleMatrixCamera(MatrixCamera * camera)
      {
        traverseCamera(camera);
      }

      void ExclusiveTraverser::handleGroup(Group * group)
      {
        traverseGroup(group);
      }

      void ExclusiveTraverser::handleTransform(Transform * trafo)
      {
        traverseGroup(trafo);
      }

      void ExclusiveTraverser::handleLOD(LOD * lod)
      {
        traverseGroup(lod);
      }

      void ExclusiveTraverser::handleSwitch(Switch * swtch)
      {
        traverseGroup(swtch);
      }

      void ExclusiveTraverser::handleBillboard(Billboard * billboard)
      {
        traverseGroup(billboard);
      }

      void ExclusiveTraverser::handleLightSource( LightSource * light )
      {
        traverseLightSource( light );
      }

      void ExclusiveTraverser::handleGeoNode(GeoNode * gnode)
      {
        traverseGeoNode(gnode);
      }

      void ExclusiveTraverser::handlePrimitive(Primitive * primitive)
      {
        traversePrimitive(primitive);
      }

      void ExclusiveTraverser::handleParameterGroupData( ParameterGroupData * pgd )
      {
        traverseParameterGroupData( pgd );
      }

      void ExclusiveTraverser::handlePipelineData( dp::sg::core::PipelineData * p )
      {
        traversePipelineData( p );
      }

      void ExclusiveTraverser::handleSampler( Sampler * p )
      {
      }

      void ExclusiveTraverser::handleIndexSet(IndexSet *iset)
      {
        postTraverseObject( iset );
      }

      void ExclusiveTraverser::handleVertexAttributeSet(VertexAttributeSet *vas)
      {
        postTraverseObject( vas );
      }

      void ExclusiveTraverser::traverseCamera(Camera * camera)
      {
        for ( Camera::HeadLightIterator hli = camera->beginHeadLights() ; hli != camera->endHeadLights() ; ++hli )
        {
          traverseObject( *hli );
        }
      }

      void ExclusiveTraverser::traverseFrustumCamera(FrustumCamera * camera)
      {
        traverseCamera( camera );
      }

      void ExclusiveTraverser::traverseLightSource(LightSource * light)
      {
        if ( light->getLightPipeline() )
        {
          traverseObject( light->getLightPipeline() );
        }
        postTraverseObject( light );
      }

      void ExclusiveTraverser::traversePrimitive(Primitive * p)
      {
        if ( p->getIndexSet() )
        {
          traverseObject( p->getIndexSet() );
        }
        if ( p->getVertexAttributeSet() )
        {
          traverseObject( p->getVertexAttributeSet() );
        }
        postTraverseObject( p );
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
