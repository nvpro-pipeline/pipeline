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

#include <test/testfw/core/Test.h>

#include <dp/util/Timer.h>

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      class MeasurementFunctorTimer : public MeasurementFunctor
      {
      public:
        DPTCORE_API MeasurementFunctorTimer();
        DPTCORE_API virtual ~MeasurementFunctorTimer();

        //Optional derivation defined functions
        DPTCORE_API virtual bool option( const std::vector<std::string>& optionString );

        /*! \brief Take the time spend in Test::onInit()
         *  \returns The result of Test::onInit(). */
        DPTCORE_API virtual bool callOnInit();

        /*! \brief Take the time spend in Test::onRunInit()
         *  \returns The result of Test::onRunInit(). */
        DPTCORE_API virtual bool callOnRunInit( unsigned int i );

        /*! \brief Take the time spend in Test::onRun()
         *  \returns The result of Test::onRun(). */
        DPTCORE_API virtual bool callOnRun( unsigned int i );

        /*! \brief Take the time spend in Test::onRunClear()
         *  \returns The result of Test::onRunClear(). */
        DPTCORE_API virtual bool callOnRunClear( unsigned int i );

        /*! \brief Take the time spend in Test::onClear() and produce some output file(s) with all the measurements.
         *  \returns The result of Test::onClear(). */
        DPTCORE_API virtual bool callOnClear();

      protected:
        DPTCORE_API void OutputPlotDataCSV();
        DPTCORE_API void OutputPlotDataHTML();

      protected:
        std::string m_resultsDir;
        std::string m_resultsFilenamePrefix;
        std::string m_resultsFilenameSuffix;

      private:
        util::Timer m_timer;
        bool m_outputCSV;

        std::pair<std::string, double> m_initTime;
        std::vector< std::pair<std::string, double> > m_runInitTime;
        std::vector< std::pair<std::string, double> > m_runTime;
        std::vector< std::pair<std::string, double> > m_runClearTime;
        std::pair<std::string, double> m_clearTime;
      };

    } // namespace core
  } // namespace testfw
} // namespace dp
