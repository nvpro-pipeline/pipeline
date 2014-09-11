// Copyright NVIDIA Corporation 2012
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


#include <dp/sg/algorithm/Optimize.h>

// optimize
#include <dp/sg/algorithm/AnalyzeTraverser.h>
#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/algorithm/EliminateTraverser.h>
#include <dp/sg/algorithm/IdentityToGroupTraverser.h>
#include <dp/sg/algorithm/NormalizeTraverser.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/algorithm/TriangulateTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>
#include <dp/sg/algorithm/VertexCacheOptimizeTraverser.h>

using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      void optimizeScene( const SceneSharedPtr & scene, bool ignoreNames, bool identityToGroup
        , unsigned int combineFlags, unsigned int eliminateFlags, unsigned int unifyFlags
        , float epsilon, bool optimizeVertexCache )
      {
        if ( identityToGroup )
        {
          dp::util::SmartPtr<IdentityToGroupTraverser> itgt( new IdentityToGroupTraverser );
          itgt->setIgnoreNames( ignoreNames );
          itgt->apply( scene );
        }

        //  loop over optimizers until nothing changed
#define UNDEFINED_TRAVERSER ~0
#define ELIMINATE_TRAVERSER 0
#define COMBINE_TRAVERSER   1
#define UNIFY_TRAVERSER     2
        int lastModifyingTraverser = UNDEFINED_TRAVERSER;

        bool modified;
        do 
        {
          modified = false;
          //  first eliminate redundant/degenerated objects
          if ( eliminateFlags )
          {
            if ( lastModifyingTraverser != ELIMINATE_TRAVERSER )
            {
              dp::util::SmartPtr<EliminateTraverser> et( new EliminateTraverser );
              et->setIgnoreNames( ignoreNames );
              et->setEliminateTargets( eliminateFlags );
              et->apply( scene );
              if ( et->getTreeModified() )
              {
                modified = true;
                lastModifyingTraverser = ELIMINATE_TRAVERSER;
              }
            }
            else
            {
              break;
            }
          }

          // second unify all equivalent objects
          if ( unifyFlags )
          {
            if ( lastModifyingTraverser != UNIFY_TRAVERSER )
            {
              dp::util::SmartPtr<UnifyTraverser> ut( new UnifyTraverser );
              ut->setIgnoreNames( ignoreNames );
              ut->setUnifyTargets( unifyFlags );
              if ( unifyFlags & UnifyTraverser::UT_VERTICES )
              {
                ut->setEpsilon( epsilon );
              }
              ut->apply( scene );
              if ( ut->getTreeModified() )
              {
                modified = true;
                lastModifyingTraverser = UNIFY_TRAVERSER;

                if ( unifyFlags & UnifyTraverser::UT_VERTICES )
                {
                  // after unifying vertices we need to re-normalize the normals
                  dp::util::SmartPtr<NormalizeTraverser> nt( new NormalizeTraverser );
                  nt->apply( scene );
                }
              }
            }
            else
            {
              break;
            }
          }

          // third combine compatible objects
          if ( combineFlags )
          {
            if ( lastModifyingTraverser != COMBINE_TRAVERSER )
            {
              dp::util::SmartPtr<CombineTraverser> ct( new CombineTraverser );
              ct->setIgnoreNames( ignoreNames );
              ct->setCombineTargets( combineFlags );
              ct->apply( scene );
              if ( ct->getTreeModified() )
              {
                modified = true;
                lastModifyingTraverser = COMBINE_TRAVERSER;
              }
            }
            else
            {
              break;
            }
          }
        } while( modified );

        if ( optimizeVertexCache )
        {
          dp::util::SmartPtr<VertexCacheOptimizeTraverser> vcot = new VertexCacheOptimizeTraverser;
          vcot->apply( scene );
        }
      }

      void optimizeForRaytracing( const SceneSharedPtr & scene )
      {
        bool ignoreNames = true;

        //  first some preprocessing optimizers
        //  -> no specific order here
        {
          dp::util::SmartPtr<IdentityToGroupTraverser> tr( new IdentityToGroupTraverser );
          tr->setIgnoreNames( ignoreNames );
          tr->apply( scene );
        }

        {
          dp::util::SmartPtr<DestrippingTraverser> tr( new DestrippingTraverser );
          tr->apply( scene );
        }

        {
          dp::util::SmartPtr<TriangulateTraverser> tr( new TriangulateTraverser );
          tr->apply( scene );
        }

        //  loop over optimizers until nothing changed
        bool modified = false;
        float vertexEpsilon = FLT_EPSILON;
        do
        {
          //  first eliminate redundant/degenerated objects
          {
            dp::util::SmartPtr<EliminateTraverser> tr( new EliminateTraverser );
            tr->setIgnoreNames( ignoreNames );
            tr->setEliminateTargets( EliminateTraverser::ET_ALL_TARGETS_MASK );
            tr->apply( scene );
            modified = tr->getTreeModified();
          }

          // second combine compatible objects
          {
            dp::util::SmartPtr<CombineTraverser> tr( new CombineTraverser );
            tr->setIgnoreNames( ignoreNames );
            tr->setCombineTargets( CombineTraverser::CT_ALL_TARGETS_MASK );
            tr->apply( scene );
            modified = tr->getTreeModified();
          }
        } while( modified );
      }

      void optimizeUnifyVertices( const SceneSharedPtr & scene )
      {
        unsigned int unifySelection = 0;
        unifySelection |= UnifyTraverser::UT_VERTICES;

        dp::util::SmartPtr<UnifyTraverser> tr( new UnifyTraverser );
        tr->setIgnoreNames( false );
        tr->setUnifyTargets( unifySelection );
        tr->setEpsilon( FLT_EPSILON );
        tr->apply( scene );

        // after unifying vertices we need to re-normalize the normals
        dp::util::SmartPtr<NormalizeTraverser> nt( new NormalizeTraverser );
        nt->apply( scene );
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
