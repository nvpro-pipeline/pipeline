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


#include <test/testfw/core/Test.h>

#include <dp/util/File.h>
#include <dp/util/DynamicLibrary.h>

using namespace std;

namespace dp
{
  namespace testfw
  {
    namespace core
    {


      Test::Test()
      {
      }

      Test::~Test()
      {
      }

      bool Test::onInit()
      {
        return( true );
      }

      bool Test::onRunCheck( unsigned int i )
      {
        return( i == 0 );
      }

      bool Test::onRunInit( unsigned int i )
      {
        return( true );
      }

      bool Test::onRunClear( unsigned int i )
      {
        return( true );
      }

      bool Test::onClear()
      {
        return( true );
      }

      bool Test::allowMeasurement()
      {
        return( false );
      }

      static std::string emptyString;

      const std::string & Test::getDescriptionOnInit()
      {
        return( emptyString );
      }

      const std::string & Test::getDescriptionOnRunInit( unsigned int i )
      {
        m_curRunVal = util::to_string(i);
        return( m_curRunVal );
      }

      const std::string & Test::getDescriptionOnRun( unsigned int i )
      {
        return( emptyString );
      }

      const std::string & Test::getDescriptionOnRunClear( unsigned int i )
      {
        return( emptyString );
      }

      const std::string & Test::getDescriptionOnClear()
      {
        return( emptyString );
      }

      bool Test::run( MeasurementFunctor & mf, const std::string& name )
      {
        mf.start( this, allowMeasurement(), name, emptyString );
        bool succeeded = mf.callOnInit();
        if ( succeeded )
        {
          for ( unsigned int i=0 ; mf.callOnRunCheck( i ) ; i++ )
          {
            bool ok = mf.callOnRunInit( i );
            if ( ok )
            {
              ok = mf.callOnRun( i );
              if ( ! mf.callOnRunClear( i ) )
              {
                ok = false;
              }
            }
            if ( ! ok )
            {
              succeeded = false;
            }
          }
          if ( ! mf.callOnClear() )
          {
            succeeded = false;
          }
        }
        mf.stop();

        return( succeeded );
      }

      bool Test::option( const std::vector<std::string>& options )
      {
        return false;
      }

    } //namespace core
  } //namespace testfw
} //namespace dp
