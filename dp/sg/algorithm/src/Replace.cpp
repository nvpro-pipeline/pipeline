// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/algorithm/Replace.h>
#include <dp/sg/algorithm/Traverser.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Scene.h>
#include <dp/fx/EffectLibrary.h>

using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      namespace
      {
        /** \brief Replace EffectDatas using the given map.
        **/
        class ReplaceTraverser : public ExclusiveTraverser
        {
        public:
          static void replace( NodeSharedPtr const& node, ReplacementMapPipelineData const& replacements );

        protected:
          void handleGeoNode( GeoNode * geoNode );

        private:
          std::set<dp::sg::core::GeoNode*>  m_geoNodes;

          ReplaceTraverser( ReplacementMapPipelineData const& replacements );

          ReplacementMapPipelineData const& m_replacements;
        };

        ReplaceTraverser::ReplaceTraverser( ReplacementMapPipelineData const& replacements )
          : m_replacements( replacements )
        {
        }

        void ReplaceTraverser::replace( NodeSharedPtr const& node, ReplacementMapPipelineData const& replacements )
        {
          ReplaceTraverser traverser( replacements );
          traverser.apply( node );
        }

        void ReplaceTraverser::handleGeoNode( GeoNode* geoNode )
        {
          // handle each GeoNode only once to avoid cyclic swaps
          if ( m_geoNodes.insert( geoNode ).second == true )
          {
            dp::sg::core::PipelineDataSharedPtr const& pipelineData = geoNode->getMaterialPipeline();
            std::string pipelineName = pipelineData ? pipelineData->getName() : "";
            ReplacementMapPipelineData::const_iterator it = m_replacements.find( pipelineName );
            if ( it != m_replacements.end() )
            {
              geoNode->setMaterialPipeline( it->second );
            }
          }
        }

      } // namespace

      void replacePipelineData( NodeSharedPtr const& node, ReplacementMapPipelineData const& replacements )
      {
        ReplaceTraverser::replace( node, replacements );
      }

      void replacePipelineData( NodeSharedPtr const& node, ReplacementMapNames const& replacements )
      {
        ReplacementMapPipelineData newReplacements;
        for ( ReplacementMapNames::const_iterator it = replacements.begin(); it != replacements.end(); ++it )
        {
          newReplacements[it->first] = dp::sg::core::PipelineData::create( dp::fx::EffectLibrary::instance()->getEffectData( it->second ) );
          DP_ASSERT( newReplacements[it->first])
        }
        ReplaceTraverser::replace( node, newReplacements );
      }

      void replacePipelineData( SceneSharedPtr const& scene, ReplacementMapPipelineData const& replacements )
      {
        ReplaceTraverser::replace( scene->getRootNode(), replacements );
      }

      void replacePipelineData( SceneSharedPtr const& scene, ReplacementMapNames const& replacements )
      {
        replacePipelineData( scene->getRootNode(), replacements );
      }


    } // namespace algorithm
  } // namespace sg
} // namespace dp
