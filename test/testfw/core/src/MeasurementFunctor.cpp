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
#include <test/testfw/core/MeasurementFunctor.h>
#include <dp/util/File.h>
#include <dp/util/DPAssert.h>
#include <iostream>

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      MeasurementFunctor::MeasurementFunctor()
        : m_test( nullptr )
      {
      }

      MeasurementFunctor::~MeasurementFunctor()
      {
      }

      bool MeasurementFunctor::option( const std::vector<std::string>& optionString )
      {
        return true;
      }

      void MeasurementFunctor::start( Test * test, bool doMeasurement, const std::string & name, const std::string & description )
      {
        DP_ASSERT( test );
        m_test = test;
        m_curTestName = name;
      }

      bool MeasurementFunctor::callOnInit()
      {
        DP_ASSERT( m_test );
        return( m_test->onInit() );
      }

      bool MeasurementFunctor::callOnRunCheck( unsigned int i )
      {
        DP_ASSERT( m_test );
        return( m_test->onRunCheck( i ) );
      }

      bool MeasurementFunctor::callOnRunInit( unsigned int i )
      {
        DP_ASSERT( m_test );
        return( m_test->onRunInit( i ) );
      }

      bool MeasurementFunctor::callOnRun( unsigned int i )
      {
        DP_ASSERT( m_test );
        return( m_test->onRun( i ) );
      }

      bool MeasurementFunctor::callOnRunClear( unsigned int i )
      {
        DP_ASSERT( m_test );
        return( m_test->onRunClear( i ) );
      }

      bool MeasurementFunctor::callOnClear()
      {
        DP_ASSERT( m_test );
        return( m_test->onClear() );
      }

      void MeasurementFunctor::stop()
      {
      }

    } // namespace core
  } // namespace testfw
} // namespace dp
