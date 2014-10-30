// Copyright NVIDIA Corporation 2002-2011
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
/** \file */

#include <set>
#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/Traverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! Traverser that normalizes all normals.
      class NormalizeTraverser : public ExclusiveTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API NormalizeTraverser(void);

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~NormalizeTraverser(void);

          //! Get the index of the VertexAttributeSet that is to be normalized.
          /** The default index for normalizing is NORMAL. */
          DP_SG_ALGORITHM_API unsigned int getVertexAttributeIndex() const;

          //! Set the index of the VertexAttributeSet that is to be normalized.
          /** The default index for normalizing is NORMAL. */
          DP_SG_ALGORITHM_API void setVertexAttributeIndex( unsigned int attrib );

          REFLECTION_INFO_API( DP_SG_ALGORITHM_API, NormalizeTraverser );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( VertexAttributeIndex );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          //! Initiate traversal of the scene.
          DP_SG_ALGORITHM_API virtual void  doApply( const dp::sg::core::NodeSharedPtr & root );

          //! Normalize VertexAttributeSets.
          DP_SG_ALGORITHM_API virtual void handleVertexAttributeSet( dp::sg::core::VertexAttributeSet * p );

        private:
          template<unsigned char N, typename T>
            void  normalizeData( dp::sg::core::VertexAttributeSet * p );

        private:
          unsigned int                          m_attrib;
          std::set<const void *>                m_objects;      //!< A set of pointers to hold all objects already encountered.
      };

      inline unsigned int NormalizeTraverser::getVertexAttributeIndex() const
      {
        return( m_attrib );
      }

      inline void NormalizeTraverser::setVertexAttributeIndex( unsigned int attrib )
      {
        if ( attrib != m_attrib )
        {
          m_attrib = attrib;
          notify( PropertyEvent( this, PID_VertexAttributeIndex ) );
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
