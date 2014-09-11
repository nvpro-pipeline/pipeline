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


#pragma once

#include <test/testfw/core/inc/Config.h>
#include <string>
#include <vector>

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      class Test;

      // MeasurementFunctor provides the interface to measure the separate steps of a Test
      class MeasurementFunctor
      {
        public:
          DPTCORE_API MeasurementFunctor();
          DPTCORE_API virtual ~MeasurementFunctor();

          /*! \brief get a pointer to the Test.
           *  \returns A pointer to the Test. */
          Test * getTest() const { return( m_test ); }

          /*! \brief get the current test name.
           *  \returns returns the current test name. */
          const std::string & getCurTestName() const { return( m_curTestName ); }

          /*! \brief Start the measurement of a Test.
           *  \param test The Test to measure.
           *  \param doMeasurement Flag to indicate if this test should be measured.
           *  \param name The name of the Test.
           *  \param description The description of the Test.
           *  \note The base implementation just stores the Test for later usage. */
          DPTCORE_API virtual void start( Test * test, bool doMeasurement, const std::string & name, const std::string & description );

          /*! \brief Do the measurement work around calling Test::onInit()
           *  \returns The result of Test::onInit().
           *  \note The base implementation just calls Test::onInit and returns its result. */
          DPTCORE_API virtual bool callOnInit();

          /*! \brief Do the measurement work around calling Test::onRunCheck()
           *  \returns The result of Test::onRunCheck().
           *  \note The base implementation just calls Test::onRunCheck and returns its result. */
          DPTCORE_API virtual bool callOnRunCheck( unsigned int i );

          /*! \brief Do the measurement work around calling Test::onRunInit()
           *  \returns The result of Test::onRunInit().
           *  \note The base implementation just calls Test::onRunInit and returns its result. */
          DPTCORE_API virtual bool callOnRunInit( unsigned int i );

          /*! \brief Do the measurement work around calling Test::onRun()
           *  \returns The result of Test::onRun().
           *  \note The base implementation just calls Test::onRun and returns its result. */
          DPTCORE_API virtual bool callOnRun( unsigned int i );

          /*! \brief Do the measurement work around calling Test::onRunClear()
           *  \returns The result of Test::onRunClear().
           *  \note The base implementation just calls Test::onRunClear and returns its result. */
          DPTCORE_API virtual bool callOnRunClear( unsigned int i );

          /*! \brief Do the measurement work around calling Test::onClear()
           *  \returns The result of Test::onClear().
           *  \note The base implementation just calls Test::onClear and returns its result. */
          DPTCORE_API virtual bool callOnClear();

          /*! \brief Stops the measurement of a Test. */
          DPTCORE_API virtual void stop();

          //Optional derivation defined functions
          DPTCORE_API virtual bool option( const std::vector<std::string>& optionString );

        private:
          Test* m_test;
          std::string m_curTestName;
      };

    } //namespace core
  } //namespace testfw
} //namespace dp
