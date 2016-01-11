// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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

// optimize
#include <dp/sg/algorithm/Config.h>
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

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /*! \brief Optimize the given scene specified by the given flags
       *  \param scene The scene which to optimize.
       *  \param ignoreNames If \c true, optimizing ingores names of objects.
       *  \param identityToGroup If \c true, the IdentityToGroupTraverser is executed.
       *  \param combineFlags The flags to use for the CombineTraverser.
       *  \param eliminateFlags The flags to use for the EliminateTraverser.
       *  \param unifyFlags The flags to use for the UnifyTraverser.
       *  \param epsilon The epsilon value to use to identify unique vertices while running the UnifyTraverser.
       **/
      DP_SG_ALGORITHM_API void optimizeScene( const dp::sg::core::SceneSharedPtr & scene, bool ignoreNames = true, bool identityToGroup = true
                                            , CombineTraverser::TargetMask combineFlags = CombineTraverser::Target::ALL
                                            , unsigned int eliminateFlags = EliminateTraverser::ET_ALL_TARGETS_MASK
                                            , unsigned int unifyFlags = UnifyTraverser::UT_ALL_TARGETS_MASK
                                            , float epsilon = FLT_EPSILON, bool optimizeVertexCache = true );

      /*! \brief optimize the given scene for optimal raytracing performance
       *  \param scene The Scene which is going to be optimized.
       **/
      DP_SG_ALGORITHM_API void optimizeForRaytracing( const dp::sg::core::SceneSharedPtr & scene );

      /*! \brief Merge nearby vertices. Note that this can be a very expensive operation.
       *  \param scene The Scene which is going to be optimized.
       **/
      DP_SG_ALGORITHM_API void optimizeUnifyVertices( const dp::sg::core::SceneSharedPtr & scene );

    } // namespace algorithm
  } // namespace sg
} // namespace dp