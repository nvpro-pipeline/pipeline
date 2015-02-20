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


#pragma once

#include <dp/fx/Config.h>
#include <dp/fx/EffectDefs.h>
#include <dp/sg/xbar/xbar.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/sg/xbar/Tree.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/ui/ViewState.h>
#include <boost/scoped_ptr.hpp>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      DEFINE_PTR_TYPES( GeoNodeObserver );

      class SceneTree;

      class SceneTreeObserver;
      class GeoNodeObserver;

      class DrawableManager
      {
      public:
        DP_SG_XBAR_API DrawableManager( );
        DP_SG_XBAR_API virtual ~DrawableManager();

        // A handle cannot be a simple unsigned int. It needs to be able to store more 
        // information like which rendergroup(s) the geonode belongs to. Of course having
        // an indirection would be possible too.
        DP_SG_XBAR_API class HandleData
        {
        public:
          DP_SG_XBAR_API static dp::util::SharedPtr<HandleData> create()
          {
            return( std::shared_ptr<HandleData>( new HandleData() ) );
          }

          DP_SG_XBAR_API virtual ~HandleData() {}

        protected:
          DP_SG_XBAR_API HandleData() {}
        };

        typedef dp::util::SharedPtr<HandleData> Handle;

        DP_SG_XBAR_API virtual void update( dp::sg::ui::ViewStateSharedPtr const& viewState ) = 0;
        DP_SG_XBAR_API virtual void update( dp::math::Vec2ui const & viewportSize ) = 0;

        DP_SG_XBAR_API virtual void setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & samper ) = 0;
        DP_SG_XBAR_API virtual const dp::sg::core::SamplerSharedPtr & getEnvironmentSampler() const = 0;

        DP_SG_XBAR_API virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const = 0;
  
        DP_SG_XBAR_API void setSceneTree( SceneTreeSharedPtr const & sceneTree );
        SceneTreeSharedPtr const & getSceneTree() const { return m_sceneTree; }

      protected:

        /** \brief Call this from update(ViewStateSharedPtr const&) to process all deferred events.
                   Later on this should be the only update function! **/
        DP_SG_XBAR_API void update();

        /** \brief Initialize handles for existing objects in SceneTree during construction. Call from constructor.**/
        void initializeHandles(); 

        /** \brief Get handle for given ObjectTreeIndex **/
        Handle const & getDrawableInstance( ObjectTreeIndex objectTreeIndex );

        DP_SG_XBAR_API virtual Handle addDrawableInstance( dp::sg::core::GeoNodeWeakPtr geoNode, ObjectTreeIndex objectTreeIndex ) = 0;
        DP_SG_XBAR_API virtual void removeDrawableInstance( Handle handle ) = 0;
        DP_SG_XBAR_API virtual void updateDrawableInstance( Handle handle ) = 0;
        DP_SG_XBAR_API virtual void setDrawableInstanceActive( Handle handle, bool visible ) = 0;
        DP_SG_XBAR_API virtual void setDrawableInstanceTraversalMask( Handle handle, dp::Uint32 traversalMask ) = 0;

      private:
        /** \brief Detach from current SceneTree. Called from setSceneTree. Calls removeDrawableInstance for all Drawables **/
        void detachSceneTree();

        /** \brief Attach to current SceneTree. Called from setSceneTree. Calls addDrawableInstance for all Drawables **/
        void attachSceneTree();

        DP_SG_XBAR_API virtual void onSceneTreeChanged() = 0;

        friend class SceneTreeObserver;

        dp::sg::xbar::SceneTreeSharedPtr m_sceneTree;
        boost::scoped_ptr<SceneTreeObserver> m_sceneTreeObserver;

        GeoNodeObserverSharedPtr m_geoNodeObserver; // observe GeoNodes for changes

        std::vector<Handle> m_dis;
      };

      inline DrawableManager::Handle const & DrawableManager::getDrawableInstance( ObjectTreeIndex objectTreeIndex ) 
      {
        DP_ASSERT(objectTreeIndex < m_dis.size());
        return m_dis[objectTreeIndex];
      }
      

    } // namespace xbar
  } // namespace sg
} // namespace dp
