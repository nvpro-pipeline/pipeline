// Copyright NVIDIA Corporation 2002-2005
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


#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/IdentityToGroupTraverser.h>

using namespace dp::sg::core;

using std::pair;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      IdentityToGroupTraverser::IdentityToGroupTraverser( void )
      {
      }

      IdentityToGroupTraverser::~IdentityToGroupTraverser( void )
      {
      }

      void IdentityToGroupTraverser::postApply( const NodeSharedPtr & root )
      {
        OptimizeTraverser::postApply( root );

        if ( m_scene && root->getObjectCode() == OC_TRANSFORM )
        {
          TransformSharedPtr const& th = root.staticCast<Transform>();
          m_scene->setRootNode( createGroupFromTransform( th ) );
          setTreeModified();
        }
        m_objects.clear();
      }

      void IdentityToGroupTraverser::handleBillboard( Billboard *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleBillboard( p );
          replaceTransforms( p );
        }
      }

      void IdentityToGroupTraverser::handleGroup( Group *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGroup( p );
          replaceTransforms( p );
        }
      }

      void IdentityToGroupTraverser::handleLOD( LOD *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleLOD( p );
          replaceTransforms( p );
        }
      }

      void IdentityToGroupTraverser::handleSwitch( Switch *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleSwitch( p );
          replaceTransforms( p );
        }
      }

      void IdentityToGroupTraverser::handleTransform( Transform *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleTransform( p );
          replaceTransforms( p );
        }
      }

      GroupSharedPtr IdentityToGroupTraverser::createGroupFromTransform( const TransformSharedPtr & th )
      {
        GroupSharedPtr gh = Group::create();
        gh->setName( th->getName() );
        gh->setAnnotation( th->getAnnotation() );
        gh->setUserData( th->getUserData() );
        gh->setHints( th->getHints() );
        gh->setTraversalMask( th->getTraversalMask() );
        for ( Group::ClipPlaneIterator gcpci = th->beginClipPlanes() ; gcpci != th->endClipPlanes() ; ++gcpci )
        {
          gh->addClipPlane( *gcpci );
        }
        for ( Group::ChildrenIterator gcci = th->beginChildren() ; gcci != th->endChildren() ; ++gcci )
        {
          gh->addChild( *gcci );
        }
        return( gh );
      }

      bool IdentityToGroupTraverser::isTransformToReplace( const NodeSharedPtr & nh )
      {
        bool ok = false;
        if( nh.isPtrTo<Transform>() )
        {
          TransformSharedPtr const& t = nh.staticCast<Transform>();
          ok =    ( getIgnoreNames() || t->getName().empty() )
              &&  optimizationAllowed( t )
              &&  !t->isJoint()
              &&  isIdentity( t->getTrafo().getMatrix() );
        }
        return( ok );
      }

      void IdentityToGroupTraverser::replaceTransforms( Group *p )
      {
        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( isTransformToReplace( *gci ) )
          {
            GroupSharedPtr gh = createGroupFromTransform( gci->staticCast<Transform>() );
            p->replaceChild( gh, gci );
            setTreeModified();
          }
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
