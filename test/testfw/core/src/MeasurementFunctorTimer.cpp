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


#include <test/testfw/core/MeasurementFunctorTimer.h>
#include <dp/util/File.h>
#include <dp/util/DPAssert.h>
#include <iostream>

#include <boost/program_options.hpp>

namespace options = boost::program_options;

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      MeasurementFunctorTimer::MeasurementFunctorTimer()
        : m_resultsDir( util::getCurrentPath() + "\\dpt\\result" )
        , m_resultsFilenamePrefix("")
        , m_resultsFilenameSuffix("")
        , m_outputCSV(false)
      {
      }

      MeasurementFunctorTimer::~MeasurementFunctorTimer()
      {
      }

      bool MeasurementFunctorTimer::option( const std::vector<std::string>& optionString )
      {
        options::options_description od("Usage: DPTApp");
        od.add_options()
          ( "formattedOutputType", options::value<std::string>(), "Output format" )
          ( "resultsDir", options::value<std::string>(), "Directory to dump the results to" )
          ( "resultsFilenamePrefix", options::value<std::string>(), "Prefix to add to the result output file" )
          ( "resultsFilenameSuffix", options::value<std::string>(), "Suffix to add to the result output file" )
          ;

        options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>( optionString ).options( od ).allow_unregistered().run();

        options::variables_map optsMap;
        options::store( parsedOpts, optsMap );

        // Formatted output type

        if( optsMap["formattedOutputType"].empty() )
        {
          std::cerr << "Warning: No formatted output type was specified. Using html\n";
          m_outputCSV = false;
        }
        else
        {
          std::string fot( optsMap["formattedOutputType"].as<std::string>() );

          if( fot.compare("csv") == 0 )
          {
            m_outputCSV = true;
          }
          else if( fot.compare("html") == 0 )
          {
            m_outputCSV = false;
          }
          else
          {
            std::cerr << "Warning: Invalid argument: " << fot << "\n";
            std::cerr << "Currently only 'csv' and 'html' are supported. Default is html." << "\n";
            m_outputCSV = false;
          }
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

      bool MeasurementFunctorTimer::callOnInit()
      {
        m_timer.restart();
        bool ok = MeasurementFunctor::callOnInit();
        m_timer.stop();

        m_initTime = make_pair( getTest()->getDescriptionOnInit(), m_timer.getTime() );

        return( ok );
      }

      bool MeasurementFunctorTimer::callOnRunInit( unsigned int i )
      {
        m_timer.restart();
        bool ok = MeasurementFunctor::callOnRunInit( i );
        m_timer.stop();

        m_runInitTime.push_back( make_pair( getTest()->getDescriptionOnRunInit( i ), m_timer.getTime() ) );

        return( ok );
      }

      bool MeasurementFunctorTimer::callOnRun( unsigned int i )
      {
        m_timer.restart();
        bool ok = MeasurementFunctor::callOnRun( i );
        m_timer.stop();

        m_runTime.push_back( make_pair( getTest()->getDescriptionOnRun( i ), m_timer.getTime() ) );

        return( ok );
      }

      bool MeasurementFunctorTimer::callOnRunClear( unsigned int i )
      {
        m_timer.restart();
        bool ok = MeasurementFunctor::callOnRunClear( i );
        m_timer.stop();

        m_runClearTime.push_back( make_pair( getTest()->getDescriptionOnRunClear( i ), m_timer.getTime() ) );

        return( ok );
      }

      bool MeasurementFunctorTimer::callOnClear()
      {
        m_timer.restart();
        bool ok = MeasurementFunctor::callOnClear();
        m_timer.stop();

        m_clearTime = make_pair( getTest()->getDescriptionOnClear(), m_timer.getTime() );

        DP_ASSERT( dp::util::directoryExists( m_resultsDir ) );   // has been checked/created in option()
        if ( m_outputCSV )
        {
          OutputPlotDataCSV();
        }
        else
        {
          OutputPlotDataHTML();
        }

        m_runInitTime.clear();
        m_runTime.clear();
        m_runClearTime.clear();

        return( ok );
      }

      DPTCORE_API void MeasurementFunctorTimer::OutputPlotDataCSV()
      {
        const std::string & name = getCurTestName();
        FILE* fRunInitOut = fopen( (m_resultsDir + "//" + m_resultsFilenamePrefix + name + m_resultsFilenameSuffix + "_runInit.csv").c_str(), "wt" );
        FILE* fRunOut = fopen( (m_resultsDir + "//" + m_resultsFilenamePrefix + name + m_resultsFilenameSuffix + "_run.csv").c_str(), "wt" );
        FILE* fRunClearOut = fopen( (m_resultsDir + "//" + m_resultsFilenamePrefix + name + m_resultsFilenameSuffix + "_runClear.csv").c_str(), "wt" );

        fprintf(fRunInitOut, "%s", "# Generated by dp::testfw::core::MeasurementFunctorTimer\n\n");
        fprintf(fRunOut, "%s", "# Generated by dp::testfw::core::MeasurementFunctorTimer\n\n");
        fprintf(fRunClearOut, "%s", "# Generated by dp::testfw::core::MeasurementFunctorTimer\n\n");

        for(size_t i = 0; i < m_runInitTime.size(); i++)
        {
          fprintf( fRunInitOut, "%s;%g\n", m_runInitTime[i].first.c_str(), m_runInitTime[i].second );
          fprintf( fRunOut, "%s;%g\n", m_runTime[i].first.c_str(), m_runTime[i].second );
          fprintf( fRunClearOut, "%s;%g\n", m_runClearTime[i].first.c_str(), m_runClearTime[i].second );
        }

        fclose(fRunInitOut);
        fclose(fRunOut);
        fclose(fRunClearOut);
      }

      DPTCORE_API void MeasurementFunctorTimer::OutputPlotDataHTML()
      {
        FILE* fout = fopen( (m_resultsDir + "//" + m_resultsFilenamePrefix + getCurTestName() + m_resultsFilenameSuffix + ".html").c_str(), "wt" );

        //header
        fprintf(fout, "<html>\n<head>\n<script type=%ctext/javascript%c src=%chttps://www.google.com/jsapi%c></script>\n", '"', '"', '"', '"');
        fprintf(fout, "<script type=%ctext/javascript%c>\ngoogle.load(%cvisualization%c, %c1%c, {packages:[%ccorechart%c]});\n", '"', '"', '"', '"', '"', '"', '"', '"');
        fprintf(fout, "%s", "google.setOnLoadCallback(drawChart);\nfunction drawChart() {\n");
        
        //OnRunInit
        fprintf(fout, "%s", "var dataOnRunInit = new google.visualization.DataTable();\n");
        fprintf(fout, "%s", "dataOnRunInit.addColumn('number', 'desc');\n");
        fprintf(fout, "%s", "dataOnRunInit.addColumn('number', 'Time');\n");
        fprintf(fout, "%s", "dataOnRunInit.addRows([\n");
        for(size_t i = 0; i < m_runInitTime.size(); i++)
        {
          fprintf(fout, "[%d, %g],\n", atoi( m_runInitTime[i].first.c_str() ), m_runInitTime[i].second );
        }
        fprintf(fout, "%s", "]);\n");

        fprintf(fout, "%s", "var optionsOnRunInit = {\n");
        fprintf(fout, "%s", "title: 'OnRunInit',\n");
        fprintf(fout, "%s", "hAxis: {title: 'desc'},\n");
        fprintf(fout, "%s", "vAxis: {title: 'Time'},\n");
        fprintf(fout, "%s", "legend: 'none'\n");
        fprintf(fout, "%s", "};\n");

        //OnRun
        fprintf(fout, "%s", "var dataOnRun = new google.visualization.DataTable();\n");
        fprintf(fout, "%s", "dataOnRun.addColumn('number', 'desc');\n");
        fprintf(fout, "%s", "dataOnRun.addColumn('number', 'Time');\n");
        fprintf(fout, "%s", "dataOnRun.addRows([\n");
        for(size_t i = 0; i < m_runTime.size(); i++)
        {
          fprintf(fout, "[%d, %g],\n", atoi( m_runTime[i].first.c_str() ), m_runTime[i].second );
        }
        fprintf(fout, "%s", "]);\n");

        fprintf(fout, "%s", "var optionsOnRun = {\n");
        fprintf(fout, "%s", "title: 'OnRun',\n");
        fprintf(fout, "%s", "hAxis: {title: 'desc'},\n");
        fprintf(fout, "%s", "vAxis: {title: 'Time'},\n");
        fprintf(fout, "%s", "legend: 'none'\n");
        fprintf(fout, "%s", "};\n");

        //OnRunClear
        fprintf(fout, "%s", "var dataOnRunClear = new google.visualization.DataTable();\n");
        fprintf(fout, "%s", "dataOnRunClear.addColumn('number', 'desc');\n");
        fprintf(fout, "%s", "dataOnRunClear.addColumn('number', 'Time');\n");
        fprintf(fout, "%s", "dataOnRunClear.addRows([\n");
        for(size_t i = 0; i < m_runClearTime.size(); i++)
        {
          fprintf(fout, "[%d, %g],\n", atoi( m_runClearTime[i].first.c_str() ), m_runClearTime[i].second );
        }
        fprintf(fout, "%s", "]);\n");

        fprintf(fout, "%s", "var optionsOnRunClear = {\n");
        fprintf(fout, "%s", "title: 'OnRunClear',\n");
        fprintf(fout, "%s", "hAxis: {title: 'desc'},\n");
        fprintf(fout, "%s", "vAxis: {title: 'Time'},\n");
        fprintf(fout, "%s", "legend: 'none'\n");
        fprintf(fout, "%s", "};\n");

        //instantiate charts
        fprintf(fout, "%s", "var chartOnInitRun = new google.visualization.ScatterChart(document.getElementById('chart_onRunInit'));\n");
        fprintf(fout, "%s", "chartOnInitRun.draw(dataOnRunInit, optionsOnRunInit);\n");


        fprintf(fout, "%s", "var chartOnRun = new google.visualization.ScatterChart(document.getElementById('chart_onRun'));\n");
        fprintf(fout, "%s", "chartOnRun.draw(dataOnRun, optionsOnRun);\n");


        fprintf(fout, "%s", "var chartOnRunClear = new google.visualization.ScatterChart(document.getElementById('chart_onRunClear'));\n");
        fprintf(fout, "%s", "chartOnRunClear.draw(dataOnRunClear, optionsOnRunClear);\n");

        //footer
        fprintf(fout, "%s", "}\n");
        fprintf(fout, "%s", "</script>\n");
        fprintf(fout, "%s", "</head>\n");
        fprintf(fout, "%s", "<body>\n");

          //html in the charts
        fprintf(fout, "<div id=%cchart_onRunInit%c style=%cwidth: 900px; height: 500px;%c></div>\n", '"', '"', '"', '"');
        fprintf(fout, "<div id=%cchart_onRun%c style=%cwidth: 900px; height: 500px;%c></div>\n", '"', '"', '"', '"');
        fprintf(fout, "<div id=%cchart_onRunClear%c style=%cwidth: 900px; height: 500px;%c></div>\n", '"', '"', '"', '"');


        fprintf(fout, "%s", "</body>\n");
        fprintf(fout, "%s", "</html>\n");

        fclose(fout);

      }

    } // namespace core
  } // namespace testfw
} // namespace dp
