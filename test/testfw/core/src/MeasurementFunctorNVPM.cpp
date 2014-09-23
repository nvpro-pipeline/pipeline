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


#include <test/testfw/core/MeasurementFunctorNVPM.h>
#include <dp/util/File.h>
#include <dp/util/DPAssert.h>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#if defined(HAVE_NVPMAPI)

using namespace dp::util;
using std::pair;
using std::string;
using std::vector;

namespace options = boost::program_options;

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      MeasurementFunctorNVPM::MeasurementFunctorNVPM()
        : m_resultsDir( util::getCurrentPath() + "\\dpt\\result" )
        , m_resultsFilenamePrefix("")
        , m_resultsFilenameSuffix("")
      {
      }

      MeasurementFunctorNVPM::~MeasurementFunctorNVPM()
      {
      }

      bool MeasurementFunctorNVPM::option( const vector<string>& optionString )
      {
        options::options_description od("Usage: DPTApp");
        od.add_options()
          ( "counterFilter", options::value< std::vector<std::string> >()->composing()->multitoken(), "tests to run" )
          ( "resultsDir", options::value<std::string>(), "Directory to dump the results to" )
          ( "resultsFilenamePrefix", options::value<std::string>(), "Prefix to add to the result output file" )
          ( "resultsFilenameSuffix", options::value<std::string>(), "Suffix to add to the result output file" )
          ;

        options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>( optionString ).options( od ).allow_unregistered().run();

        options::variables_map optsMap;
        options::store( parsedOpts, optsMap );

        // Counter Filter

        if( optsMap["counterFilter"].empty() )
        {
          std::cerr << "Warning: no counter filter was specified. All counters will be recorded for this run.\n";
        }
        else if( !optsMap.count("counterFilter") )
        {
          std::cerr << "Warning: one or more filters must follow the --counterFilter flag. All counters will be recorded for this run.\n";
        }
        else
        {
          m_counterFilters = optsMap["counterFilter"].as< std::vector<std::string> >();
        }

        // Results Dir
        if( optsMap["resultsDir"].empty() )
        {
          std::cerr << "Warning: No result directory was specified\n";
        }
        else
        {
          m_resultsDir = optsMap["resultsDir"].as<std::string>();
        }
        if ( !dp::util::directoryExists( m_resultsDir ) )
        {
          if ( ! dp::util::createDirectory( m_resultsDir ) )
          {
            std::cerr << "Unable to create directory <" << m_resultsDir << ">\n";
            return( false );
          }
        }

        // Result Filename Prefix

        if( optsMap["resultsFilenamePrefix"].empty() )
        {
          std::cerr << "No result filename prefix was specified. A filename prefix won't be used for this run\n";
        }
        else
        {
          std::string prefix( optsMap["resultsFilenamePrefix"].as<std::string>() );
          m_resultsFilenamePrefix = prefix;
        }

        // Result Filename Suffix

        if( optsMap["resultsFilenameSuffix"].empty() )
        {
          std::cerr << "No result filename suffix was specified. A filename suffix won't be used for this run\n";
        }
        else
        {
          std::string suffix( optsMap["resultsFilenameSuffix"].as<std::string>() );
          m_resultsFilenamePrefix = suffix;
        }

        return true;
      }

      bool MeasurementFunctorNVPM::callOnInit()
      {
        DP_ASSERT( dynamic_cast<TestRender*>( getTest() ) );

        if ( !m_nvPerfMon.get() )
        {
          m_nvPerfMon.reset( new NVPerfMon( m_counterFilters ) );
        }
        DP_ASSERT( m_nvPerfMon.get() );
        return( m_nvPerfMon->init() && MeasurementFunctor::callOnInit() );
      }

      bool MeasurementFunctorNVPM::callOnRunInit( unsigned int i )
      {
        DP_ASSERT( m_nvPerfMon.get() );
        m_nvPerfMon->beginExperiment();
        return( MeasurementFunctor::callOnRunInit( i ) );
      }

      bool MeasurementFunctorNVPM::callOnRun( unsigned int i )
      {
        DP_ASSERT( m_nvPerfMon.get() );

        bool ok = true;
        while ( ok && !m_nvPerfMon->finishedExperiment() )
        {
          m_nvPerfMon->beginPass();
          ok = MeasurementFunctor::callOnRun( i );
          static_cast<TestRender*>(getTest())->getBackend()->finish();
          m_nvPerfMon->endPass();
        }
        return( ok );
      }

      bool MeasurementFunctorNVPM::callOnRunClear( unsigned int i )
      {
        DP_ASSERT( m_nvPerfMon.get() );
        bool ok = MeasurementFunctor::callOnRunClear( i );
        m_counterResults.resize( m_counterResults.size() + 1 );
        m_nvPerfMon->endExperiment( m_counterResults.back() );

        return( ok );
      }

      bool MeasurementFunctorNVPM::callOnClear()
      {
        OutputPlotDataCSV();
        return( MeasurementFunctor::callOnClear() );
      }

      DPTCORE_API void MeasurementFunctorNVPM::OutputPlotDataCSV()
      {
        Test * test = getTest();
        DP_ASSERT( test );
        std::string fileName = m_resultsDir + "//" + m_resultsFilenamePrefix + getCurTestName() + m_resultsFilenameSuffix + "_run.csv";
        std::ofstream of( fileName.c_str() );
        if ( of.fail() )
        {
          std::cerr << "Warning: Could not open file <" << fileName << "> !!\n";
        }
        else
        {
          of << "# Generated by dp::testfw::core::MeasurementFunctorNVPM for test <" << test->getDescriptionOnInit() << ">\n\n";
          of << "CounterName";
          unsigned int nResults = checked_cast<unsigned int>(m_counterResults.size());
          for ( unsigned int i=0 ; i<nResults ; i++ )
          {
            of << ";" << test->getDescriptionOnRunInit( i );
          }
          of << std::endl;

          const vector<NVPMCounterID> & counterIDs = m_nvPerfMon->getCounterIDs();
          const NVPerfMon::CounterIDToSpecMap & counterToSpecMap = m_nvPerfMon->getCounterIDToSpecMap();
          for ( size_t i=0 ; i<counterIDs.size() ; i++ )
          {
            NVPerfMon::CounterIDToSpecMap::const_iterator it = counterToSpecMap.find( counterIDs[i] );
            DP_ASSERT( it != counterToSpecMap.end() );
            of << it->second.name;
            if ( it->second.percentage )
            {
              of << " (%)";
              for ( unsigned int j=0 ; j<nResults ; j++ )
              {
                of << ";" << m_counterResults[j][i].percent;
              }
            }
            else
            {
              for ( unsigned int j=0 ; j<nResults ; j++ )
              {
                of << ";" << m_counterResults[j][i].count;
              }
            }
            of << std::endl;
          }
        }
      }

    } // namespace core
  } // namespace testfw
} // namespace dp

// HAVE_NVPMAPI
#endif
