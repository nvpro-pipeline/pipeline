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
#include <test/testfw/core/MeasurementFunctor.h>
#include <string>
#include <vector>

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      // template to determine #elements of an static array, usage sizeof array( yourarray );
      template< typename T, size_t N> char(&array(T(&)[N]))[N];

      class Test
      {
      public:
        DPTCORE_API Test();
        DPTCORE_API virtual ~Test();

      public:
          /*! \brief Virtual function called to initialize the Test.
           *  \return \c true if initialization was successful, otherwise \c false.
           *  \note The base implementation just returns \c true. */
          DPTCORE_API virtual bool onInit();

          /*! \brief Virtual function to check for the end of the sequence of test runs.
           *  \param i The index of the test to be run.
           *  \return \c true if the check was successful, otherwise \c false.
           *  \note The base implementation returns \c true for \a i == 0, otherwise \c false */
          DPTCORE_API virtual bool onRunCheck( unsigned int i );

          /*! \brief Virtual function to initialize one test run out of a sequence of test runs.
           *  \param i The index of the test to initialize.
           *  \return \c true if the test was initialized successfully, otherwise \c false.
           *  \note The base implementation always returns \c true. */
          DPTCORE_API virtual bool onRunInit( unsigned int i );

          /*! \brief Pure virtual function to run on test run out of a sequence of test runs.
           *  \param i The index of the test to run.
           *  \return \c true if the test was run successfully, otherwise \c false.
           *  \note There is no base implementation. */
          DPTCORE_API virtual bool onRun( unsigned int i = 0 ) = 0;

          /*! \brief Virtual function to clear one test run out of a sequence of test runs.
           *  \param i The index of the test run to clear.
           *  \return \c true if the test was cleared successfully, otherwise \c false.
           *  \note The base implementation always returns \c true. */
          DPTCORE_API virtual bool onRunClear( unsigned int i );

          /*! \brief Virtual function to clear the test.
           *  \return \c true if the test was cleared successfully, otherwise \c false.
           *  \note The base implementation just returns \c true. */
          DPTCORE_API virtual bool onClear();


          /*! \brief Check if measurement is allowed for this test.
           *  \return \c true if measurement is allowed for this test, otherwise \c false.
           *  \note The base implementation just returns \c false. */
          DPTCORE_API virtual bool allowMeasurement();

          /*! \brief Get the description for the test initialization.
           *  \return The description for the test initialization.
           *  \note The base implementation returns an empty string. */
          DPTCORE_API virtual const std::string & getDescriptionOnInit();

          /*! \brief Get the description for the initialization of one test run out of a sequence of test runs.
           *  \param i The index of the test run.
           *  \note The base implementation returns an empty string . */
          DPTCORE_API virtual const std::string & getDescriptionOnRunInit( unsigned int i );

          /*! \brief Get the description for the test run out of a sequence of test runs.
           *  \param i The index of the test run.
           *  \note The base implementation returns an empty string . */
          DPTCORE_API virtual const std::string & getDescriptionOnRun( unsigned int i );

          /*! \brief Get the description for the clearing of one test run out of a sequence of test runs.
           *  \param i The index of the test run.
           *  \note The base implementation returns an empty string . */
          DPTCORE_API virtual const std::string & getDescriptionOnRunClear( unsigned int i );

          /*! \brief Get the description for the test clearing.
           *  \return The description for the test clearing.
           *  \note The base implementation returns an empty string. */
          DPTCORE_API virtual const std::string & getDescriptionOnClear();
          
          /*! \brief This function can be overriden to process user defined command line options.
           *  \param optionString The unparsed options propagated from DPT.
           *  \return This function should return false only if there was input so invalid that the program must terminate. */
          DPTCORE_API virtual bool option( const std::vector<std::string>& optionString );
          
          /*! \brief Launch the test.
           *  \param mf The measurement functor to use in the test launch.
           *  \param name The test name to use in the launch.
           *  \return Returns true if the test passed, otherwise returns false. */
          DPTCORE_API virtual bool run( MeasurementFunctor & mf, const std::string& name );

      private:
        std::string m_curRunVal;

      };
    } //namespace core
  } //namespace testfw
} //namespace dp
