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


#include <dp/util/FrameProfiler.h>

#include <iostream>

namespace dp
{
  namespace util
  {
    /************************************************************************/
    /* FrameProfiler                                                        */
    /************************************************************************/
    FrameProfiler::FrameProfiler()
      : m_enabled( false )
    {
      m_lastDisplay.start();
    }

    void FrameProfiler::beginFrame()
    {
      m_entries.clear();
    }

    void FrameProfiler::setEnabled( bool enabled )
    {
      if ( m_enabled != enabled )
      {
        m_enabled = enabled;
        if ( m_enabled == false )
        {
          m_entries.clear();
        }
        else
        {
          m_lastDisplay.restart();
        }
      }
    }

    bool FrameProfiler::isEnabled() const
    {
      return m_enabled;
    }

    void FrameProfiler::addEntry( std::string const& key, double time )
    {
      if ( m_enabled )
      {
        Entry entry;
        entry.key = key;
        entry.time = time;
        m_entries.push_back( entry );
      }
    }

    void FrameProfiler::endFrame()
    {
      if ( m_enabled && m_lastDisplay.getTime() > 1.0 )
      {
        std::cout << "------frame-----" << std::endl;
        for ( std::vector<Entry>::iterator it = m_entries.begin(); it != m_entries.end(); ++it )
        {
          std::cout << it->key << " ";
          if ( it->time >= 1 )
          {
            std::cout << it->time << "s";
          }
          if ( it->time >= 1.0/1000.0)
          {
            std::cout << it->time * 1000.0 << "ms";
          }
          else
          {
            std::cout << it->time * 1000000.0 << "us";
          }
          std::cout << std::endl;
        }
        m_lastDisplay.restart();
        std::cout << "----endframe----" << std::endl;
      }
    }

    FrameProfiler& FrameProfiler::instance()
    {
      static FrameProfiler singleton;
      return singleton;
    }

    /************************************************************************/
    /* ProfileEntry                                                         */
    /************************************************************************/
    ProfileEntry::ProfileEntry( std::string const& key )
      : m_key( key )
    {
      m_timer.start();
    }

    ProfileEntry::~ProfileEntry()
    {
      FrameProfiler::instance().addEntry( m_key, m_timer.getTime() );
    }

  } // namespace util
} // namespace dp
