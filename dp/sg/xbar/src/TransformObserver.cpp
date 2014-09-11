// Copyright NVIDIA Corporation 2010-2013
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


#include <dp/sg/xbar/inc/TransformObserver.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      TransformObserver::~TransformObserver()
      {
      }

      void TransformObserver::attach( dp::sg::core::TransformWeakPtr const & t, TransformTreeIndex index )
      {
        DP_ASSERT( m_indexMap.find( index ) == m_indexMap.end() );

        SmartDirtyPayload payload( new DirtyPayload( index ) );
        Observer<TransformTreeIndex>::attach( t, payload );

        payload->m_dirty = true;
        m_dirtyPayloads.push_back( payload.get() );
      }

      void TransformObserver::onNotify( const dp::util::Event &event, dp::util::Payload *payload )
      {
        switch ( event.getType() )
        {
        case dp::util::Event::PROPERTY:
          {
            dp::util::Reflection::PropertyEvent const& propertyEvent = static_cast<dp::util::Reflection::PropertyEvent const&>(event);

            if( propertyEvent.getPropertyId() == dp::sg::core::Transform::PID_Matrix )
            {
              DirtyPayload* p = static_cast< DirtyPayload* >( payload );

              if ( !p->m_dirty )
              {
                p->m_dirty = true;
                m_dirtyPayloads.push_back(p);
              }
            }
          }
          break;
        }
      }

      void TransformObserver::onDetach( TransformTreeIndex index )
      {
        for ( DirtyPayloads::iterator it = m_dirtyPayloads.begin(); it != m_dirtyPayloads.end(); ++it )
        {
          if ( (*it)->m_index == index )
          {
            *it = m_dirtyPayloads.back();
            m_dirtyPayloads.pop_back();
            break;
          }
        }
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
