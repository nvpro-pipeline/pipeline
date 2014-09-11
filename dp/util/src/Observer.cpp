// Copyright NVIDIA Corporation 2011
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


#include <dp/util/Observer.h>
#include <dp/util/DPAssert.h>

#include <algorithm>

namespace dp
{
  namespace util
  {

    Payload::~Payload()
    {
    }

    /************************************************************************/
    /* Subject                                                              */
    /************************************************************************/

    Subject::~Subject()
    {
      Observers::iterator it, itEnd = m_observers.end();

      for ( it = m_observers.begin(); it != itEnd;++it )
      {
        if( it->first )
        {
          it->first->onDestroyed( *this, it->second );
        }
      }
    }

    void Subject::attach( Observer *observer, Payload *payload )
    {
      m_observers.push_back( ObserverEntry(observer, payload) );
    }

    void Subject::detach( Observer *observer, Payload *payload )
    {
      Observers::iterator it = std::find(m_observers.begin(), m_observers.end(), std::make_pair(observer, payload) );
      if ( it != m_observers.end() )
      {
        // mark this entry as invalid instead of deleting it
        it->first = nullptr;
      }
    }

    bool Subject::isAttached( Observer * observer, Payload * payload ) const
    {
      return( std::find( m_observers.begin(), m_observers.end(), std::make_pair(observer, payload) ) != m_observers.end() );
    }

    void Subject::notify( const Event &event )
    {
      // notify the list ob observers, cleaning the list of observers set to nullptr on the way
      // don't clean the list first and notify then, this would mean iterating the list of observers twice
      if( !m_observers.empty() )
      {
        size_t idx;
        size_t last = m_observers.size() - 1;
        size_t deletedElements = 0;
         
        for ( idx = 0; idx <= last; ++idx )
        {
          if( !m_observers[idx].first )
          {
            // observer at idx was detached, find a valid observer beginning at the end of the vector
            while( !m_observers[last].first && last > idx )
            {
              --last;
              ++deletedElements;
            }

            if( last != idx )
            {
              // move the last valid observer to position idx
              m_observers[idx] = m_observers[last];

              // abandon the last observer
              --last;
              ++deletedElements;
            }
            else
            {
              ++deletedElements;
              // no valid observers left, leave the for loop
              break;
            }
          }

          m_observers[idx].first->onNotify( event, m_observers[idx].second );
        }

        if( deletedElements )
        {
          m_observers.resize( m_observers.size() - deletedElements );
        }
      }
    }

    /************************************************************************/
    /* SubjectTrackingObserver                                              */
    /************************************************************************/
    /*
    SubjectTrackingObserver::~SubjectTrackingObserver()
    {
      Subjects::iterator it;
      Subjects::iterator it_end = m_subjects.end();
      for( it = m_subjects.begin(); it != it_end; ++it )
      {
        it->m_subject->detach( this, it->m_payload );
      }
    }

    void SubjectTrackingObserver::addSubject( Subject& subject, Payload* payload )
    {
      payload->subjectPosition = m_subjects.size();
      m_subjects.push_back( SubjectEntry(&subject, payload) );

      subject.attach( this, payload );

      doAddSubject( subject, payload );
    }

    void SubjectTrackingObserver::removeSubject( Subject& subject, Payload* payload )
    {
      doRemoveSubject( subject, payload );

      subject.detach( this, payload );

      DP_ASSERT( m_subjects[payload->subjectPosition].m_subject == &subject );
      DP_ASSERT( m_subjects[payload->subjectPosition].m_payload == payload );
      removeSubjectEntry( payload->subjectPosition );
    }

    void SubjectTrackingObserver::onDestroyed( const Subject& subject, Payload* payload )
    {
      doOnDestroyed( subject, payload );

      DP_ASSERT( m_subjects[payload->subjectPosition].m_subject == &subject );
      DP_ASSERT( m_subjects[payload->subjectPosition].m_payload == payload );
      removeSubjectEntry( payload->subjectPosition );
    }

    void SubjectTrackingObserver::doAddSubject( Subject& subject, Payload* payload )
    {
    }

    void SubjectTrackingObserver::doRemoveSubject( Subject& subject, Payload* payload )
    {
    }

    void SubjectTrackingObserver::doOnDestroyed( const Subject& subject, Payload* payload )
    {
    }

    void SubjectTrackingObserver::removeSubjectEntry( size_t position )
    {
      // move the last subject of the vector to the subject's position, if subject is not the last
      if( position != m_subjects.size() - 1 )
      {
        // move entry
        m_subjects[position] = m_subjects.back();

        // update it's position
        m_subjects[position].m_payload->subjectPosition = position;
      }

      m_subjects.resize( m_subjects.size() - 1 );
    }
    */
  } // namespace util
} // namespace dp
