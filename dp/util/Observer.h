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


#pragma once

#include <dp/util/Config.h>
#include <dp/util/SharedPtr.h>
#include <memory>
#include <vector>

namespace dp
{
  namespace util
  {
    class Subject;
    class Observer;

    class Payload : public std::enable_shared_from_this<Payload>
    {
    public:
      DP_UTIL_API virtual ~Payload();
    };

    DEFINE_PTR_TYPES( Payload );


    class Event
    {
    public:
      enum Type
      {
          GENERIC
        , PROPERTY
        , DP_SG_CORE
      };

      virtual~ Event() {}
      Type getType() const { return m_eventType; }

      Event( Type type = GENERIC )
        : m_eventType( type )
      {
      }

    private:
      Type m_eventType;
    };

    class Subject
    {
    public:
      // Do not copy the list of observers during copy/assignment.
      // The observers won't know about the 'new' attachment and thus
      // it cannot detach itself;
      Subject() {}
      Subject( Subject const& ) {}
      Subject& operator=( const Subject& /* rhs */ ) { return *this; }

      DP_UTIL_API virtual ~Subject();

      DP_UTIL_API void attach( Observer* observer, Payload * payload = nullptr );
      DP_UTIL_API void detach( Observer* observer, Payload * payload = nullptr  );
      DP_UTIL_API bool isAttached( Observer * observer, Payload * payload = nullptr ) const;

    protected:
      DP_UTIL_API void notify( const Event &event );

    private:
      typedef std::pair<Observer*, Payload*> ObserverEntry;
      typedef std::vector<ObserverEntry> Observers;

    private:
      Observers m_observers;
    };

    DEFINE_PTR_TYPES( Subject );


    class Observer
    {
    public:
      DP_UTIL_API virtual void onNotify( dp::util::Event const & event, dp::util::Payload * payload ) = 0;
      DP_UTIL_API virtual void onDestroyed( dp::util::Subject const & subject, dp::util::Payload * payload ) = 0;
    };


  } // namespace util
} // namespace dp
