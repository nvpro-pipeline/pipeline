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

#include <list>
#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/Traverser.h>
#include <dp/sg/core/Node.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! Base for Traversers that perform some optimizations on a scene.
      class OptimizeTraverser : public ExclusiveTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API OptimizeTraverser( void );

        public:
          //! Get the 'ignore names' flag.
          /** If the 'ignore names' flag is set, the names of the GeoNodes are ignored.
            * \return true if the names will be ignored, otherwise false */
          DP_SG_ALGORITHM_API bool getIgnoreNames( void ) const;

          //! Set the 'ignore names' flags.
          /** If the 'ignore names' flag is set, the names of the GeoNodes are ignored. */
          DP_SG_ALGORITHM_API void setIgnoreNames( bool ignore   //!<  set true to ignore names on comparison
                                      );
          REFLECTION_INFO_API( DP_SG_ALGORITHM_API, OptimizeTraverser );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( IgnoreNames );
          END_DECLARE_STATIC_PROPERTIES
        protected:
          //! returns whether we can/should optimize a given object.
          //  currently, if the node is not marked "DYNAMIC", and it doesn't have
          //  any callbacks registered, we say it is optimizable
          DP_SG_ALGORITHM_API virtual bool optimizationAllowed( dp::sg::core::ObjectSharedPtr const& obj );

          //! Protected destructor to prevent instantiation of a TreeOptimizeTraverser on stack.
          DP_SG_ALGORITHM_API virtual ~OptimizeTraverser( void );

          //! doApply override
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

        private:
          bool  m_ignoreNames;
      };

      inline bool OptimizeTraverser::getIgnoreNames( void ) const
      {
        return( m_ignoreNames );
      }

      inline void OptimizeTraverser::setIgnoreNames( bool ignore )
      {
        if ( m_ignoreNames != ignore )
        {
          m_ignoreNames = ignore;
          notify( PropertyEvent( this, PID_IgnoreNames ) );
        }
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
