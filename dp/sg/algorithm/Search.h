// Copyright NVIDIA Corporation 2013
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
#include <dp/sg/algorithm/SearchTraverser.h>

#include <dp/sg/core/CoreTypes.h>

#include <vector>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /** \brief Find all nodes of a certain class below a node
          \param root The node the search shall start at
          \param name The class name that the search should look for
          \return A vector of weak pointers to all objects that were found during the search
      **/
      DP_SG_ALGORITHM_API const std::vector<dp::sg::core::ObjectWeakPtr> searchClass( const dp::sg::core::NodeSharedPtr& root, const std::string& name, bool baseClassSearch = false );

      /** \brief Find all nodes of a certain class below a node
          \param root The node the search shall start at
          \param name The class name that the search should look for
          \return A vector of paths to all objects that were found during the search
      **/
      DP_SG_ALGORITHM_API std::vector<dp::sg::core::PathSharedPtr> const searchClassPaths( const dp::sg::core::NodeSharedPtr& root, const std::string& name, bool baseClassSearch = false );

      /** \brief Determine whether a scene contains at least one light source
          \param scene The scene that should be searched for lights
          \return true if \a scene contains at least one light, false otherwise.
      **/
      DP_SG_ALGORITHM_API bool containsLight( dp::sg::core::SceneSharedPtr scene );


    } // namespace algorithm
  } // namespace sg
} // namespace dp