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


#include <dp/sg/core/Group.h>
#include <dp/sg/core/Path.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Transform.h>
#include <dp/math/Matmnt.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      PathSharedPtr Path::create()
      {
        return( std::shared_ptr<Path>( new Path() ) );
      }

      PathSharedPtr Path::create( PathSharedPtr const& rhs )
      {
        return( std::shared_ptr<Path>( new Path( rhs ) ) );
      }

      Path::Path()
      {
      }

      Path::Path( PathSharedPtr const& rhs)
      {
        m_path = rhs->m_path;
      }

      Path::~Path()
      {
        // clean up the node list
        m_path.clear();
      }

      void Path::push( const ObjectSharedPtr & object )
      {
        DP_ASSERT( object );

    #if !defined(NDEBUG)
        if ( !m_path.empty() )
        {
          GroupSharedPtr const& pGroup = m_path.back().dynamicCast<Group>();
          if ( pGroup )
          {
            DP_ASSERT( object.isPtrTo<Node>() );
            DP_ASSERT( pGroup->findChild( pGroup->beginChildren(), object.inplaceCast<Node>() ) != pGroup->endChildren() );
          }
        }
    #endif

        m_path.push_back( object );
      }

      void Path::truncate( unsigned int start )
      {
        DP_ASSERT(0 <= start && start < m_path.size());

        m_path.erase(m_path.begin() + start, m_path.end());
      }

      void Path::getModelToWorldMatrix( Mat44f & modelToWorld, Mat44f & worldToModel ) const
      {
        worldToModel = cIdentity44f;
        modelToWorld = cIdentity44f;

        if ( 0 < getLength() )
        {
          for ( unsigned int i=0 ; i<getLength()-1 ; i++ )
          {
            ObjectSharedPtr obj = getFromHead( i );
            if ( obj.isPtrTo<Transform>() )
            {
              Trafo const& trafo = obj.staticCast<Transform>()->getTrafo();
              worldToModel *= trafo.getInverse();
              modelToWorld = trafo.getMatrix() * modelToWorld;
            }
          }
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp
