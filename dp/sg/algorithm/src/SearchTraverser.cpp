// Copyright NVIDIA Corporation 2002-2011
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
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/MatrixCamera.h>
#include <dp/sg/core/ParallelCamera.h>
#include <dp/sg/core/Path.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/SearchTraverser.h>

#include <algorithm>
#include <vector>

using namespace dp::math;
using namespace dp::sg::core;

using std::for_each;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      DEFINE_STATIC_PROPERTY( SearchTraverser, ClassName );
      DEFINE_STATIC_PROPERTY( SearchTraverser, ObjectName );
      DEFINE_STATIC_PROPERTY( SearchTraverser, BaseClassSearch );

      BEGIN_REFLECTION_INFO( SearchTraverser )
        DERIVE_STATIC_PROPERTIES( SearchTraverser, SharedTraverser );
        INIT_STATIC_PROPERTY_RW( SearchTraverser, ClassName, std::string, SEMANTIC_VALUE, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( SearchTraverser, ObjectName, std::string, SEMANTIC_VALUE, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( SearchTraverser, BaseClassSearch, bool, SEMANTIC_VALUE, value, value );
      END_REFLECTION_INFO

      SearchTraverser::SearchTraverser(void)
      : m_currentPath(NULL)
      , m_objectPointer(NULL)
      , m_searchBaseClass( false )
      {
      }

      SearchTraverser::~SearchTraverser(void)
      {
      }

      void SearchTraverser::addItem( const Object * obj )
      {
        m_foundObjects.insert( obj );
        m_smartPaths.push_back( new Path( *m_currentPath ) );
      }

      void  SearchTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT( root );
        DP_ASSERT( m_currentPath == NULL );
  
        m_currentPath = new Path;
        m_foundObjects.clear();
        m_paths.clear();
        m_smartPaths.clear();
        m_results.clear();

        SharedTraverser::doApply( root );

        m_currentPath = NULL;
      }

      const vector<const Path*> & SearchTraverser::getPaths()
      {
        if ( m_paths.empty() && !m_smartPaths.empty() )
        {
          m_paths.reserve( m_smartPaths.size() );
          for ( vector<dp::util::SmartPtr<Path> >::const_iterator it = m_smartPaths.begin() ; it != m_smartPaths.end() ; ++it )
          {
            m_paths.push_back( it->get() );
          }
        }
        return( m_paths );
      }

      const vector<ObjectWeakPtr> & SearchTraverser::getResults()
      {
        if ( m_results.empty() && !m_foundObjects.empty() )
        {
          m_results.reserve( m_foundObjects.size() );
          for ( set<const Object *>::const_iterator it = m_foundObjects.begin() ; it != m_foundObjects.end() ; ++it )
          {
            m_results.push_back( dp::util::getWeakPtr<Object>(*it) );
          }
        }
        return( m_results );
      }

      void SearchTraverser::handleBillboard( const Billboard * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::Billboard" ) )
          {
            search( (const Group *) p );
          }
        }
        else if( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleBillboard( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleEffectData( const EffectData * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::EffectData" ) )
          {
            search( (const OwnedObject<Object> *) p);
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleEffectData( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleGeoNode( const GeoNode * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::GeoNode" ) )
          {
            search( (const Node *) p );
          }
        }
        else if( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleGeoNode( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleGroup( const Group * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          search( p );
        }
        else if( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleGroup( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleLOD( const LOD * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::LOD" ) )
          {
            search( (const Group *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleLOD( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleParameterGroupData( const ParameterGroupData * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::ParameterGroupData" ) )
          {
            search( (const OwnedObject<EffectData> *) p);
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleParameterGroupData( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleSampler( const Sampler * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::Sampler" ) )
          {
            search( (const OwnedObject<ParameterGroupData> *) p);
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleSampler( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleSwitch( const Switch * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::Switch" ) )
          {
            search( (const Group *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleGroup( p );    // traverse the Switch like a Group to search through all children
        m_currentPath->pop();
      }
                                    
      void SearchTraverser::handleTransform( const Transform * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          search( p );
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleTransform( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleLightSource( const LightSource * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::LightSource" ) )
          {
            search( (const Node *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleLightSource( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handlePrimitive( const Primitive * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::Primitive" ) )
          {
            search( (const OwnedObject<Object> *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handlePrimitive( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleParallelCamera( const ParallelCamera * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::ParallelCamera" ) )
          {
            search( (const FrustumCamera *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleParallelCamera( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handlePerspectiveCamera( const PerspectiveCamera * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::PerspectiveCamera" ) )
          {
            search( (const FrustumCamera *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handlePerspectiveCamera( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleMatrixCamera( const MatrixCamera * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::MatrixCamera" ) )
          {
            search( (const Camera *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem(p);
        }
        SharedTraverser::handleMatrixCamera( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleVertexAttributeSet( const VertexAttributeSet * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          search( p );
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem( p );
        }
        SharedTraverser::handleVertexAttributeSet( p );
        m_currentPath->pop();
      }

      void SearchTraverser::handleIndexSet( const IndexSet * p )
      {
        m_currentPath->push( p->getSharedPtr<Object>() );
        if ( ! m_objectPointer )
        {
          if ( searchObject( p, "class dp::sg::core::IndexSet" ) )
          {
            search( (const OwnedObject<Primitive> *) p );
          }
        }
        else if ( m_objectPointer == dp::util::getWeakPtr<Object>( p ) )
        {
          addItem( p );
        }
        SharedTraverser::handleIndexSet( p );
        m_currentPath->pop();
      }

      void  SearchTraverser::search( const Camera *p )
      {
        if ( searchObject( p,"class dp::sg::core::Camera" ))
        {
          search( (const Object *) p );
        }
      }

      void  SearchTraverser::search( const Group *p )
      {
        if ( searchObject( p,"class dp::sg::core::Group" ))
        {
          search( (const Node *) p );
        }
      }

      void  SearchTraverser::search( const LightSource *p )
      {
        if ( searchObject( p,"class dp::sg::core::LightSource" ))
        {
          search( (const Node *) p );
        }
      }

      void  SearchTraverser::search( const Node *p )
      {
        if ( searchObject( p,"class dp::sg::core::Node" ))
        {
          search( (const Object *) p );
        }
      }

      void  SearchTraverser::search( const Object *p )
      {
        searchObject(p,"class dp::sg::core::Object" );
      }

      void  SearchTraverser::search( const Transform *p )
      {
        if ( searchObject( p,"class dp::sg::core::Transform" ))
        {
          search( (const Group *) p );
        }
      }

      void SearchTraverser::search( const VertexAttributeSet * p )
      {
        if ( searchObject( p, "class dp::sg::core::VertexAttributeSet" ) )
        {
          search( (const OwnedObject<Primitive> *) p );
        }
      }

      void SearchTraverser::search( const OwnedObject<Primitive> * p )
      {
        if ( searchObject( p, "class dp::sg::core::OwnedObject<Primitive>" ) )
        {
          search( (const Object *) p );
        }
      }

      /* 
      Checks whether the node being traversed is the required node.
      Returns true/false depending upon whether base class needs to be searched or not.
      */
      bool SearchTraverser::searchObject( const Object * p, const std::string & classNameToHandle )
      {
        if ( !m_className.empty() )
        {
          if ( m_className == classNameToHandle )
          {
            if ( !m_objectName.empty() )
            {
              if ( m_objectName == p->getName() )
              {
                addItem(p);
              }
            }
            else
            {
              addItem(p);
            }
            return false;                       //no need to search for base class
          }
          else 
          {
            return m_searchBaseClass;           //search base class only if m_searchBaseClass is true
          }
        }
        else if ( !m_objectName.empty() )
        {
          if ( m_objectName == p->getName() )
          {
            addItem(p);
            return false;
          }
          else 
          {
            return m_searchBaseClass;
          }
        }
        else
        {
          return false;
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
