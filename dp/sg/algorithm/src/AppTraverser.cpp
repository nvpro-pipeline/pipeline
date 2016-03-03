// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/algorithm/AppTraverser.h>
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Transform.h>

using namespace dp::math;
using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      AppTraverser::AppTraverser()
      : m_animationFrame(0)
      , m_forcedTraversal(false)
      {
      }

      AppTraverser::~AppTraverser()
      {
      }

      void AppTraverser::doApply( const dp::sg::core::NodeSharedPtr & root )
      {
        DP_ASSERT( m_viewState && "This traverser needs a valid ViewState. Use setViewState() prior calling apply()");
        DP_ASSERT( m_camera && "This traverser needs a valid camera as part of the ViewState" );
        DP_ASSERT( root );

        m_pLastViewState      = m_viewState; // no need to addref() since we only compare pointers
        m_pLastRoot           = root;

        if ( needsTraversal( root ) )
        {

          ExclusiveModelViewTraverser::doApply( root );

          m_forcedTraversal = false;    // make sure, the traversal isn't forced any more on the next apply
        }
      }

      bool AppTraverser::needsTraversal( const dp::sg::core::NodeSharedPtr & root ) const
      {
        DP_ASSERT( root );
        return( m_forcedTraversal );
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
