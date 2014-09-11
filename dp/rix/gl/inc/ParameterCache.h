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

#include <dp/rix/core/HandledObject.h>
#include <dp/rix/gl/inc/ContainerGL.h>
#include <dp/rix/gl/inc/ProgramPipelineGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      struct ParameterCacheStream;
      struct ParameterCacheBase;

      struct ContainerCacheEntry {};

      template <typename ParameterCacheType>
      class ParameterCache
      {
      public:
        /************************************************************************/
        /* Note: the following functions and structures do not have an actual   */
        /* implementation: There're here to document the interface of the       */
        /* ParameterCache template interface                                    */
        /************************************************************************/
        struct ContainerCacheEntry : public dp::rix::gl::ContainerCacheEntry {};
        struct Location {};

        /** \brief Clear all allocations in the cache **/
        void clear();

        /** \brief  Parameters for the given set of ContainerCacheEntries **/
        void renderParameters( dp::rix::gl::ContainerCacheEntry const* containerCacheEntry );

        /** update a given container **/
        // TODO location could know about container...
        void updateContainer( ContainerGLHandle container, Location& location );

        /** \brief begin with allocation process **/
        void allocationBegin();

        /** \brief allocate a Location for the container
            \return true if new allocation took place or false if there was an allocation before
        **/
        bool allocateContainer( ContainerGLHandle container );

        /** \brief remove given container from cache **/
        void removeContainer( ContainerGLHandle container );

        /** \brief end allocation process **/
        void allocationEnd();
      };

      struct ParameterStateBase
      {
        ProgramGLHandle       m_program;               // the program this ParameterState belongs to
        size_t                m_uniqueDescriptorIdx;   // index into the vector of descriptors
        size_t                m_numParameterObjects;   // number of ParameterObjects
        void*                 m_currentDataPtr;        // ptr to currently set data
      };

    } // namespace gl
  } // namespace rix
} // namespace dp
