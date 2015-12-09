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


#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Object.h>

using namespace dp::math;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Scene, AmbientColor );
      DEFINE_STATIC_PROPERTY( Scene, BackColor );
      DEFINE_STATIC_PROPERTY( Scene, RootNode );

      BEGIN_REFLECTION_INFO( Scene )
        DERIVE_STATIC_PROPERTIES( Scene, Object );
        INIT_STATIC_PROPERTY_RW( Scene, AmbientColor,     Vec3f,   SEMANTIC_COLOR,      const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Scene, BackColor,        Vec4f,   SEMANTIC_COLOR,      const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW( Scene, RootNode, NodeSharedPtr,  SEMANTIC_OBJECT,      const_reference, const_reference );
      END_REFLECTION_INFO

      SceneSharedPtr Scene::create()
      {
        return( std::shared_ptr<Scene>( new Scene() ) );
      }

      HandledObjectSharedPtr Scene::clone() const
      {
        return( std::shared_ptr<Scene>( new Scene( *this ) ) );
      }

      Scene::Scene()
      : m_ambientColor(0.2f,0.2f,0.2f)
      , m_backColor(0.4f,0.4f,0.4f,1.0f)
      , m_backImage(NULL)
      , m_root(NULL)
      {
        m_objectCode = ObjectCode::SCENE;
      }

      Scene::~Scene(void)
      {
      }

      void Scene::setRootNode( const NodeSharedPtr & root )
      {
        if ( root != m_root )
        {
          m_root = root;
          notify( Event(this ) );
          notify( PropertyEvent( this, PID_RootNode ) );
       }
      }

      void Scene::setBackImage( const TextureHostSharedPtr & image )
      {
        if ( m_backImage != m_backImage )
        {
          m_backImage = image;
          notify( Event(this ) );
        }
      }

      Scene::CameraIterator Scene::addCamera( const CameraSharedPtr & camera )
      {
        CameraContainer::iterator cci = std::find( m_cameras.begin(), m_cameras.end(), camera );
        if ( cci == m_cameras.end() )
        {
          m_cameras.push_back( camera );
          cci = m_cameras.end() - 1;
        }
        return( CameraIterator( cci ) );
      }

      bool Scene::removeCamera( const CameraSharedPtr & camera )
      {
        CameraContainer::iterator cci = std::find( m_cameras.begin(), m_cameras.end(), camera );
        if ( cci != m_cameras.end() )
        {
          m_cameras.erase( cci );
          return( true );
        }
        return( false );
      }

      Scene::CameraIterator Scene::removeCamera( const CameraIterator & sci )
      {
        if ( sci.m_iter != m_cameras.end() )
        {
          return( CameraIterator( m_cameras.erase( sci.m_iter ) ) );
        }
        return( sci );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
