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


#include <test/testfw/app/App.h>
#include <test/testfw/core/MeasurementFunctorGoldImage.h>
#include <test/testfw/core/MeasurementFunctorNVPM.h>
#include <test/testfw/core/MeasurementFunctorTimer.h>

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/program_options.hpp>

#include <dp/util/File.h>

namespace options = boost::program_options;

using namespace std;

namespace dp
{
  namespace testfw
  {
    namespace app
    {

      App::App(int argc, char** argv)
        : m_mf(nullptr)
        , m_failedTests(0)
      {

        if( !parseCommandLineArguments(argc, argv) )
        {
          return;
        }

        if( !initTests() )
        {
          return;
        }

        test();

      }

      App::~App()
      {
        delete m_mf;
        m_tests.clear();
      }

      int App::getFailedTests()
      {
        if(m_failedTests)
        {
          cout << m_failedTests << " test" << (m_failedTests > 1 ? "s have " : " has ") << "failed\n";
        }
        else
        {
          cout << "All the tests succeeded\n";
        }

        return m_failedTests;
      }

      bool App::parseCommandLineArguments(int argc, char** argv)
      {
        options::options_description od("Usage: DPTApp");
        od.add_options()
          ( "tests", options::value< std::vector<std::string> >()->composing()->multitoken(), "tests to run" )
          ( "mf", options::value<std::string>(), "MeasurementFunctor to use while testing" )
          ( "help", "show help" )
          ;
        
        options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>(argc, argv).options( od ).allow_unregistered().run();

        options::variables_map optsMap;
        options::store( parsedOpts, optsMap );

        if( !optsMap["help"].empty() )
        {
          printHelp();
          printHelpTestRender();
          printHelpTestGoldImage();
          printHelpTestBasicBenchmark();
          printHelpMeasurementFunctorNVPM();

          return false;
        }

        if( optsMap["tests"].empty() )
        {
          std::cerr << "Error: tests need to be specified. Use --help flag for more information.\n";
          return false;
        }
        else if( !optsMap.count("tests") )
        {
          std::cerr << "Error: one or more test or test modules need to follow the --tests option\n";
          return false;
        }
        else
        {
          m_testFilters = optsMap["tests"].as< std::vector<std::string> >();
        }


        if( !optsMap["mf"].empty() )
        {
          std::string mf( optsMap["mf"].as<std::string>() );
          transform(mf.begin(), mf.end(), mf.begin(), ::tolower);

          if( mf.compare("goldimage") == 0 )
          {
            m_mf = new core::MeasurementFunctorGoldImage();
          }
#if defined(HAVE_NVPMAPI)
          else if( mf.compare("perfmon") == 0 )
          {
            m_mf = new core::MeasurementFunctorNVPM();
          }
#endif
          else if( mf.compare("timer") == 0 )
          {
            m_mf = new core::MeasurementFunctorTimer();
          }
          else if( mf.compare("default") == 0 )
          {
            m_mf = new core::MeasurementFunctor();
          }
          else
          {
            cerr << "Warning: invalid argument for --mf flag. Use --help flag for more information.\n";
            cerr << "Using default measurement functor.";
            m_mf = new core::MeasurementFunctor();
          }
        }
        else
        {
          cerr << "Warning: A measurement functor should be specified\n";
          cerr << "Using default measurement functor.";
          m_mf = new core::MeasurementFunctor();
        }

        m_userOptions = options::collect_unrecognized( parsedOpts.options, options::include_positional );

        if(argc == 1)
        {
          cout << "Use --help flag for instructions\n";
          return false;
        }

        return true;
      }

      bool App::initTests()
      {
        bool success = true;

        if( m_testFilters.empty() )
        {
          std::cerr << "Error: test modules must be specified with the --tests option\n";
          return false;
        }

        std::string modulePath = dp::util::getModulePath();
        size_t nTests = m_testFilters.size();
        for(size_t i = 0; i < nTests; i++)
        {
          if( !core::findTests(modulePath, m_testFilters[i], m_tests) )
          {
            success = false;
            cerr << "File(s) could not be found: " << modulePath << "\\" << m_testFilters[i] << '\n';
          }
        }

        return success;
      }

      void App::test()
      {
        if( !m_mf )
        {
          m_mf = new core::MeasurementFunctor();
        }

        for( vector<core::TestObject>::iterator testIt = m_tests.begin(); testIt != m_tests.end(); testIt++ )
        {
          switch( (*testIt).run( *m_mf, m_userOptions ) )
          {
          case core::TestReturnFlag::FAILED:
            cout << (*testIt).getName() << " has failed\n";
            ++m_failedTests;
            break;
          case core::TestReturnFlag::INVALID_INPUT:
            cout << "invalid input for test: " << (*testIt).getName() << "\n";
            break;
          }
        }
      }

      void App::printHelp()
      {
        cout << "DPTApp is used to launch devtech platform tests (binary\n";
        cout << "files with extension .dptest) in various modes\n\n";
        cout << "Standard arguments\n";
        cout << "--tests        Specifies the tests to be launched. This flag\n";
        cout << "               is followed by a list of tests to run. Each\n";
        cout << "               item must contain the test library name followed by a\n";
        cout << "               forward slash and the specific test to launch. If only\n";
        cout << "               a test library name is specified, then all the tests\n";
        cout << "               contained in will be launched.\n\n";
        cout << "--mf           Specifies the measurement functor to be used\n";
        cout << "               to run test(s). This flag is followed by the\n";
        cout << "               name of a measurement functor. Currently only\n";
        cout << "               GoldImage, PerfMon, Timer, and default are supported.\n\n";
        cout << "--help         Prints our information on how to use DPTapp\n\n";
        printHelpTestRender();
        printHelpTestGoldImage();
        printHelpTestBasicBenchmark();
        printHelpMeasurementFunctorNVPM();
      }

      void App::printHelpTestRender()
      {
        cout << "Render test arguments\n";
        cout << "--renderer     Specifies the renderer to be used for running\n";
        cout << "               the test(s). This flag is followed by the\n";
        cout << "               file name of the renderer's binary file.\n";
        cout << "               Currently only DPTRiXGL.rdr is supported.\n\n";
        cout << "--width        Specifies the width of the render target to\n";
        cout << "               be used for the test. This flag is followed\n";
        cout << "               by the amount of pixels for the width of the\n";
        cout << "               render target\n\n";
        cout << "--height       Specifies the height of the render target to\n";
        cout << "               be used for the test. This flag is followed\n";
        cout << "               by the amount of pixels for the height of the\n";
        cout << "               render target\n\n";
      }

      void App::printHelpTestGoldImage()
      {
        cout << "Gold Image test arguments\n";
        cout << "--gold         Specifies weather or not the test run should\n";
        cout << "               generate gold images. The presence of this\n";
        cout << "               flag indicates that gold images should be\n";
        cout << "               generated. If this flag is not specified then\n";
        cout << "               gold image comparison will be carried out.\n\n";
        cout << "--goldDir      Specifies the directory where gold images\n";
        cout << "               will be saved. This flag is followed by this\n";
        cout << "               directory.\n\n";
        cout << "--imageDir     Specifies the directory where screenshots\n";
        cout << "               will be saved. This flag is followed by this\n";
        cout << "               directory.\n\n";
        cout << "Example usage to launch all tests whose names start with\n";
        cout << """feature_"" in gold image mode to generate gold images:\n";
        cout << "dptapp --tests feature_* --mf goldImage --gold --goldDir c:/tmp/test/gold --renderer RiXGL.rdr --width 1024 --height 768\n\n";
      }

      void App::printHelpTestBasicBenchmark()
      {
        cout << "Benchmark test arguments\n";
        cout << "--resultsDir             Specifies the directory where\n";
        cout << "                         performance results will be saved.\n";
        cout << "                         This flag is followed by this directory.\n\n";
        cout << "--resultsFilenamePrefix  Specifies the prefix for the name\n";
        cout << "                         of the performance results file.\n";
        cout << "                         This flag is followed by this\n";
        cout << "                         string.\n\n";
        cout << "--resultsFilenameSuffix  Specifies the suffix for the name\n";
        cout << "                         of the performance results file.\n";
        cout << "                         This flag is followed by this\n";
        cout << "                         string.\n\n";
        cout << "--formattedOutputType    Specifies the type of format for\n";
        cout << "                         the performance results. Currently\n";
        cout << "                         only csv and html are supported.\n\n";
        cout << "Example usage to launch all tests whose names start with\n";
        cout << """benchmark_"" in timer mode:\n";
        cout << "dptapp --tests benchmark_* --mf Timer --resultsDir c:/tmp/test/results --renderer RiXGL.rdr --width 1024 --height 768\n\n";
      }

      void App::printHelpMeasurementFunctorNVPM()
      {
        cout << "MeasurementFunctorNVPM arguments\n";
        cout << "--counterFilter\n";
        cout << "    Specifies the filter strings to select counters.\n";
        cout << "    Any number of strings can be specified after this flag.\n";
        cout << "    From all available counters those that include at least one of the filter strings are selected.\n";
        cout << "    If this flag is omitted, or no filters follow this flag, all counters are selected.\n";
        cout << "    Example usage to launch all tests with 'OGL' or 'Bottleneck' in its name:\n";
        cout << "    dptapp --mf perfmon --counterFilter OGL Bottleneck\n";
      }

    } //namespace app
  } //namespace testfw
} //namespace dp
