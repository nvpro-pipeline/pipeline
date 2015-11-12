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


#pragma once

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/core/CoreTypes.h>

#include <string>
#include <map>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      typedef std::map<std::string, dp::sg::core::PipelineDataSharedPtr>  ReplacementMapPipelineData;
      typedef std::map<std::string, std::string>                          ReplacementMapNames;

      /** \brief Replace PipelineData objects attached to dp::sg::core::Primitive and dp::sg::core::GeoNode
          \param scene The scene which PipelineData should be replaced
          \param replacements A map specifying replacements for a set of PipelineData names.
      **/
      DP_SG_ALGORITHM_API void replacePipelineData( dp::sg::core::SceneSharedPtr const& scene, ReplacementMapPipelineData const& replacements );

      /** \brief Replace PipelineData objects attached to dp::sg::core::Primitive and dp::sg::core::GeoNode.
          \param root The scene which PipelineData should be replaced
          \param replacements A map specifying replacements or a set of PipelineData names.
                 The replacements will be fetched using EffectLibrary::getPipelineData( name );
      **/
      DP_SG_ALGORITHM_API void replacePipelineData( dp::sg::core::SceneSharedPtr const& scene, ReplacementMapNames const& replacements );

      /** \brief Replace PipelineData objects attached to dp::sg::core::Primitive and dp::sg::core::GeoNode
          \param scene The root node of a Tree which PipelineData should be replaced
          \param replacements A map specifying replacements for a set of PipelineData names.
      **/
      DP_SG_ALGORITHM_API void replacePipelineData( dp::sg::core::NodeSharedPtr const& root, ReplacementMapPipelineData const& replacements );

      /** \brief Replace PipelineData objects attached to dp::sg::core::Primitive and dp::sg::core::GeoNode.
          \param root The root node of a TreeS which PipelineData should be replaced
          \param replacements A map specifying replacements or a set of PipelineData names.
                 The replacements will be fetched using EffectLibrary::getPipelineData( name );
      **/
      DP_SG_ALGORITHM_API void replacePipelineData( dp::sg::core::NodeSharedPtr const& root, ReplacementMapNames const& replacements );


    } // namespace algorithm
  } // namespace sg
} // namespace dp
